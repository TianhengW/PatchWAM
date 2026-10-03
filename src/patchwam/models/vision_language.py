"""Local Qwen3-VL with pooled history and optional LoRA.

Accepts current/past head-camera images only; future targets are excluded.
"""

import math
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class LowRankResidualLinear(nn.Module):
    """Frozen linear map with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 128, dropout: float = 0.05):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid low-rank adapter settings")
        self.base = base.requires_grad_(False)
        self.residual_in = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.residual_out = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.kaiming_uniform_(self.residual_in.weight, a=math.sqrt(5))
        nn.init.zeros_(self.residual_out.weight)
        self.dropout = nn.Dropout(dropout)
        self.multiplier = alpha / rank

    def forward(self, value):
        return self.base(value) + self.residual_out(self.residual_in(self.dropout(value))) * self.multiplier


def pool_visual_grid(tokens: torch.Tensor, height: int, width: int, size: int = 4):
    """Average-pool raster tokens without learned weights."""
    if size < 1 or tokens.ndim != 2 or tokens.shape[0] != height * width:
        raise ValueError("Visual tokens must match the declared raster grid")
    raster = tokens.T.reshape(1, tokens.shape[-1], height, width)
    return F.adaptive_avg_pool2d(raster, (size, size))[0].flatten(1).T


class VisualInstructionEncoder(nn.Module):
    """Encode instructions and the last-prompt summary.

    Frozen vision uses unaugmented RGB [-1,1]: current [B,3,H,W] and
    nearest-first history [B,K,3,H,W].
    Invalid history is omitted; past grids and DeepStack features are pooled.
    """

    requires_images = True

    def __init__(
        self, model_path=None, *, encoder=None, processor=None, dtype=torch.bfloat16,
        trainable_adapters=False, lora_rank=64, lora_alpha=128, lora_dropout=0.05,
        instruction_layers=(9, 18, 27), summary_layers=(18, 27, 36),
        history_pool_size=4, image_size=448, auxiliary_weight=0.1, label_smoothing=0.1,
        require_subtasks=False, gradient_checkpointing=False,
    ):
        super().__init__()
        if (encoder is None) != (processor is None):
            raise ValueError("Supply both an encoder and a processor, or a local model path")
        if encoder is None:
            if model_path is None or not Path(model_path).is_dir():
                raise FileNotFoundError("A local Qwen3-VL Hugging Face directory is required")
            try:
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            except ImportError as exc:
                raise ImportError("Qwen3-VL conditioning requires Transformers >=4.57") from exc
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, local_files_only=True, trust_remote_code=False,
                attn_implementation="sdpa",
            )
        self.encoder, self.processor = encoder, processor
        self.instruction_layers = tuple(instruction_layers)
        self.summary_layers = tuple(summary_layers)
        self.history_pool_size, self.image_size = int(history_pool_size), int(image_size)
        self.auxiliary_weight, self.label_smoothing = float(auxiliary_weight), float(label_smoothing)
        self.require_subtasks = bool(require_subtasks)
        self.lora_rank, self.lora_alpha, self.lora_dropout = int(lora_rank), float(lora_alpha), float(lora_dropout)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.history_pool_size < 1 or self.image_size < 1:
            raise ValueError("Visual image and pooling sizes must be positive")
        if not math.isfinite(self.auxiliary_weight) or self.auxiliary_weight < 0 or not 0 <= self.label_smoothing < 1:
            raise ValueError("Invalid auxiliary language objective")
        config = encoder.config.text_config
        layers = self.instruction_layers + self.summary_layers
        if not self.instruction_layers or len(self.instruction_layers) != len(self.summary_layers) or min(layers) < 1 or max(layers) > config.num_hidden_layers:
            raise ValueError("The configured language readout layers are unavailable")
        self.feature_dim = config.hidden_size * len(self.instruction_layers)
        encoder.requires_grad_(False)
        self.trainable_adapters = bool(trainable_adapters)
        if self.trainable_adapters:
            self.adapter_count = self._attach_adapters(lora_rank, lora_alpha, lora_dropout)
            if not self.adapter_count:
                raise ValueError("No supported language attention/feed-forward projections were found")
            if gradient_checkpointing:
                encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.adapter_count = 0
        self.train(False)

    def checkpoint_contract(self):
        return {
            "kind": "causal_qwen3_vl", "instruction_layers": self.instruction_layers,
            "summary_layers": self.summary_layers, "history_pool_size": self.history_pool_size,
            "image_size": self.image_size, "trainable_adapters": self.trainable_adapters,
            "lora": [self.lora_rank, self.lora_alpha, self.lora_dropout],
            "auxiliary_weight": self.auxiliary_weight, "label_smoothing": self.label_smoothing,
            "require_subtasks": self.require_subtasks, "gradient_checkpointing": self.gradient_checkpointing,
            "prompt_order": "oldest_valid_history_current_instruction_assistant_prefix",
        }

    def _attach_adapters(self, rank, alpha, dropout):
        projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        language = self.encoder.model.language_model
        replacements = [(name, module) for name, module in language.named_modules()
                        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in projections]
        for name, module in replacements:
            parent_name, _, attribute = name.rpartition(".")
            parent = language.get_submodule(parent_name) if parent_name else language
            setattr(parent, attribute, LowRankResidualLinear(module, rank, alpha, dropout))
        return len(replacements)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if self.trainable_adapters:
            self.encoder.model.language_model.train(mode)
        self.encoder.model.visual.eval()
        return self

    def _image(self, value):
        if value.ndim != 3 or value.shape[0] != 3 or not torch.isfinite(value).all():
            raise ValueError("VL observations must be finite CHW RGB tensors")
        if value.min() < -1.001 or value.max() > 1.001:
            raise ValueError("VL RGB observations must be in [-1,1]")
        image = F.interpolate(value.detach().float().cpu().unsqueeze(0), size=(self.image_size,) * 2,
                              mode="bilinear", align_corners=False, antialias=True)[0]
        pixels = ((image.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(pixels)

    def _prepare_sample(self, instruction, images, pooled, subtask):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": instruction}]
        formatted = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        tokenizer = self.processor.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("VL conditioning requires a fast tokenizer with instruction offsets")
        encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
        instruction_start = formatted.rfind(instruction)
        if instruction_start < 0:
            raise ValueError("The chat template changed the instruction text")
        instruction_end = instruction_start + len(instruction)
        pixels = self.processor.image_processor(images=images, return_tensors="pt", do_resize=False)
        core = self.encoder.model
        parameter = next(core.visual.parameters())
        grid = pixels["image_grid_thw"].to(parameter.device)
        with torch.no_grad():
            image_features, deep_features = core.get_image_features(pixels["pixel_values"].to(parameter), grid)
        if len(image_features) != len(images):
            raise ValueError("The vision encoder returned a different image count")
        merge = core.visual.spatial_merge_size
        counts = (grid.prod(-1) // merge**2).tolist()
        deep_split = [list(torch.split(features, counts)) for features in deep_features]
        adjusted_grid = grid.clone()
        selected_features, selected_deep = [], [[] for _ in deep_features]
        for index, (features, pool) in enumerate(zip(image_features, pooled)):
            temporal, height, width = grid[index].tolist()
            height, width = height // merge, width // merge
            if temporal != 1 or features.shape[0] != height * width:
                raise ValueError("VL conditioning accepts separate image frames, not video clips")
            if pool:
                features = pool_visual_grid(features, height, width, self.history_pool_size)
                adjusted_grid[index, 1:] = self.history_pool_size * merge
            selected_features.append(features)
            for layer, split in enumerate(deep_split):
                selected_deep[layer].append(pool_visual_grid(split[index], height, width, self.history_pool_size) if pool else split[index])
        image_token = self.encoder.config.image_token_id
        expanded_ids, instruction_positions = [], []
        image_index = 0
        for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
            if token == image_token:
                if image_index >= len(selected_features):
                    raise ValueError("The instruction contains reserved image placeholder tokens")
                expanded_ids.extend([token] * len(selected_features[image_index]))
                image_index += 1
            else:
                if start < instruction_end and end > instruction_start:
                    instruction_positions.append(len(expanded_ids))
                expanded_ids.append(token)
        if image_index != len(images) or not instruction_positions:
            raise ValueError("The prompt did not preserve its visual or instruction placeholders")
        prompt_length = len(expanded_ids)
        target_ids = []
        if subtask is not None:
            if not isinstance(subtask, str) or not subtask.strip():
                raise ValueError("Subtask supervision must be a nonempty sentence")
            target_ids = tokenizer(subtask, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                target_ids.append(tokenizer.eos_token_id)
            if image_token in target_ids:
                raise ValueError("Subtask labels cannot contain reserved image placeholder tokens")
        input_ids = torch.tensor([expanded_ids + target_ids], device=parameter.device, dtype=torch.long)
        embeddings = core.get_input_embeddings()(input_ids)
        visual_positions = input_ids == image_token
        embeddings = embeddings.masked_scatter(visual_positions[..., None], torch.cat(selected_features).to(embeddings))
        positions, _ = core.get_rope_index(input_ids, adjusted_grid, attention_mask=torch.ones_like(input_ids))
        output = core.language_model(
            inputs_embeds=embeddings, attention_mask=torch.ones_like(input_ids), position_ids=positions,
            visual_pos_masks=visual_positions, deepstack_visual_embeds=[torch.cat(parts).to(embeddings) for parts in selected_deep],
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        hidden = output.hidden_states
        instruction_hidden = torch.cat([hidden[layer][0, instruction_positions] for layer in self.instruction_layers], -1)
        summary_hidden = torch.cat([hidden[layer][0, prompt_length - 1:prompt_length] for layer in self.summary_layers], -1)
        auxiliary = None
        if target_ids:
            logits = self.encoder.lm_head(hidden[-1][:, prompt_length - 1:-1]).float()
            targets = input_ids[:, prompt_length:]
            auxiliary = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten(), label_smoothing=self.label_smoothing)
        return torch.cat((instruction_hidden, summary_hidden)), auxiliary

    def forward(self, prompts, *, current, history=None, history_valid=None, subtasks=None):
        if current.ndim != 4 or current.shape[0] != len(prompts):
            raise ValueError("VL current observations must be [B,3,H,W], one per prompt")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("VL instructions must be nonempty strings")
        if history is None:
            history = current.new_empty((len(prompts), 0, *current.shape[1:]))
        if history.ndim != 5 or history.shape[0] != len(prompts) or history.shape[2] != 3:
            raise ValueError("VL history must be [B,K,3,H,W]")
        if history_valid is None:
            if history.shape[1]:
                raise ValueError("Past VL observations require an explicit validity mask")
            history_valid = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        if history_valid.shape != history.shape[:2]:
            raise ValueError("VL history validity must match [B,K]")
        if subtasks is not None and len(subtasks) != len(prompts):
            raise ValueError("Provide one subtask sentence per sample")
        if self.training and self.require_subtasks and subtasks is None:
            raise ValueError("This VL training contract requires subtask sentence labels")
        features, auxiliary = [], []
        for index, instruction in enumerate(prompts):
            # Causal order: past images, current image, instruction.
            slots = torch.nonzero(history_valid[index].bool(), as_tuple=False).flatten().tolist()[::-1]
            images = [self._image(history[index, slot]) for slot in slots] + [self._image(current[index])]
            feature, loss = self._prepare_sample(instruction, images, [True] * len(slots) + [False],
                                                 None if subtasks is None else subtasks[index])
            features.append(feature)
            if loss is not None:
                auxiliary.append(loss)
        length = max(len(feature) for feature in features)
        tokens = torch.stack([F.pad(feature, (0, 0, 0, length - len(feature))) for feature in features])
        valid = torch.stack([torch.arange(length, device=tokens.device) < len(feature) for feature in features])
        result = {"text_tokens": tokens, "text_valid": valid}
        if auxiliary:
            result["auxiliary_loss"] = torch.stack(auxiliary).mean()
        return result
