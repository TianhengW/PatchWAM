"""Prefix-masked adapter for external Black Forest Labs FLUX.2 weights.

Supports Klein base 4B/9B at revision 50fe5162777813d869182b139e83b10743caef15.
Native weight loading is strict; standard image-generation attention differs.
"""

import hashlib
import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import flatten_image_latents, raster_coordinates
from .policy import PatchFlowPolicy


def _asset_file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    identity = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    # Use metadata for large weights to avoid rereading gigabytes.
    if stat.st_size <= 8 * 1024 * 1024 and path.suffix not in {".safetensors", ".bin", ".pt"}:
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def _asset_identity(transformer_path, autoencoder_path, text_encoder_path, modules) -> dict[str, Any]:
    text_root = Path(text_encoder_path)
    text_files = sorted(path for path in text_root.rglob("*") if path.is_file() and
                        not any(part.startswith(".") for part in path.relative_to(text_root).parts))
    if not text_files:
        raise ValueError("The local text encoder directory contains no model assets")
    return {
        "transformer": _asset_file_identity(Path(transformer_path)),
        "autoencoder": _asset_file_identity(Path(autoencoder_path)),
        "text_encoder": [_asset_file_identity(path) for path in text_files],
        "source": [_asset_file_identity(Path(module.__file__)) for module in modules],
    }


class OfficialFluxDenoiser(nn.Module):
    """Masked execution with unchanged official parameter names."""

    def __init__(self, transformer: nn.Module, *, gradient_checkpointing: bool = False):
        super().__init__()
        model_api = importlib.import_module("flux2.model")
        self._rotate = model_api.apply_rope
        self._embed_time = model_api.timestep_embedding
        required = (
            "img_in", "txt_in", "time_in", "pe_embedder", "double_blocks", "single_blocks",
            "double_stream_modulation_img", "double_stream_modulation_txt",
            "single_stream_modulation", "final_layer",
        )
        missing = [name for name in required if not hasattr(transformer, name)]
        if missing or getattr(transformer, "use_guidance_embed", False):
            raise ValueError(f"Expected the official Klein base layer contract; missing={missing}")
        self.transformer = transformer
        self.gradient_checkpointing = gradient_checkpointing
        self.representation_dim = transformer.hidden_size
        self.representation_layers = len(transformer.double_blocks) + len(transformer.single_blocks)

    @staticmethod
    def _split_qkv_heads(projected: Tensor, count: int) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(projected.unflatten(-1, (3, count, -1)).permute(2, 0, 3, 1, 4).unbind(0))

    def _attend(self, query: Tensor, key: Tensor, value: Tensor, position: Tensor, visibility: Tensor) -> Tensor:
        query, key = self._rotate(query, key, position)
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=visibility)
        return result.transpose(1, 2).flatten(-2)

    def _paired_block(self, block, text, image, modulation_text, modulation_image, position, visibility):
        ports = (("txt", text, modulation_text), ("img", image, modulation_image))
        projections = []
        for name, states, (attention_mod, _) in ports:
            shift, scale, _ = attention_mod
            normalized = getattr(block, name + "_norm1")(states)
            attention = getattr(block, name + "_attn")
            q, k, v = self._split_qkv_heads(attention.qkv(normalized * (1 + scale) + shift), block.num_heads)
            q, k = attention.norm(q, k, v)
            projections.append((q, k, v))
        q, k, v = (torch.cat(parts, dim=2) for parts in zip(*projections))
        mixed = self._attend(q, k, v, position, visibility)
        outputs = []
        for (name, states, (attention_mod, mlp_mod)), update in zip(ports, mixed.split((text.shape[1], image.shape[1]), 1)):
            states = states + attention_mod[2] * getattr(block, name + "_attn").proj(update)
            shift, scale, gate = mlp_mod
            normalized = getattr(block, name + "_norm2")(states)
            outputs.append(states + gate * getattr(block, name + "_mlp")(normalized * (1 + scale) + shift))
        return tuple(outputs)

    def _unified_block(self, block, states, modulation, position, visibility):
        shift, scale, gate = modulation
        projected = block.linear1(block.pre_norm(states) * (1 + scale) + shift)
        attention_input, feedforward_input = projected.split(
            (3 * block.hidden_size, block.mlp_hidden_dim * block.mlp_mult_factor), dim=-1,
        )
        q, k, v = self._split_qkv_heads(attention_input, block.num_heads)
        q, k = block.norm(q, k, v)
        attended = self._attend(q, k, v, position, visibility)
        joint_update = torch.cat((attended, block.mlp_act(feedforward_input)), -1)
        return states + gate * block.linear2(joint_update)

    def forward(
        self, reference: Tensor, noisy: Tensor, sigma: Tensor, context: Tensor,
        *, reference_ids: Tensor, noisy_ids: Tensor, context_ids: Tensor, visibility: Tensor,
        token_sigma: Tensor | None = None, representation_layer: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if representation_layer is not None and not 1 <= representation_layer <= self.representation_layers:
            raise ValueError("representation_layer must be a 1-based transformer layer")
        core = self.transformer
        images = core.img_in(torch.cat((reference, noisy), 1))
        text = core.txt_in(context)
        # The official embedding scales sigma by 1000.
        time = core.time_in(self._embed_time(sigma.to(images), 256))
        image_time = single_time = final_time = time
        if token_sigma is not None:
            if token_sigma.shape != noisy.shape[:2]:
                raise ValueError("token_sigma must match the noisy token shape [B,N]")
            generated_time = core.time_in(self._embed_time(token_sigma.to(images).flatten(), 256)).reshape(*token_sigma.shape, -1)
            image_time = torch.cat((time[:, None].expand(-1, reference.shape[1], -1), generated_time), 1)
            single_time = torch.cat((time[:, None].expand(-1, text.shape[1], -1), image_time), 1)
            final_time = generated_time
        text_mod = core.double_stream_modulation_txt(time)
        image_mod = core.double_stream_modulation_img(image_time)
        single_mod, _ = core.single_stream_modulation(single_time)
        image_positions = core.pe_embedder(torch.cat((reference_ids, noisy_ids), 1))
        text_positions = core.pe_embedder(context_ids)
        position = torch.cat((text_positions, image_positions), dim=2)
        checkpointed = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        features = None
        for index, block in enumerate(core.double_blocks, 1):
            # Bind this layer for checkpoint recomputation.
            def execute(t, i, block=block):
                return self._paired_block(block, t, i, text_mod, image_mod, position, visibility)

            text, images = checkpoint(execute, text, images, use_reentrant=False) if checkpointed else execute(text, images)
            if index == representation_layer:
                features = images[:, reference.shape[1]:]
        states = torch.cat((text, images), 1)
        for index, block in enumerate(core.single_blocks, len(core.double_blocks) + 1):
            def execute(s, block=block):
                return self._unified_block(block, s, single_mod, position, visibility)

            states = checkpoint(execute, states, use_reentrant=False) if checkpointed else execute(states)
            if index == representation_layer:
                features = states[:, text.shape[1] + reference.shape[1]:]
        generated = states[:, text.shape[1] + reference.shape[1]:]
        velocity = core.final_layer(generated, final_time)
        return velocity if representation_layer is None else (velocity, features)


class FluxAssetPolicy(nn.Module):
    """Frozen image encoders and optional trainable VL adapters.

    Train video: [B,3,2,H,W] current/future RGB in [-1,1]; infer: [B,3,1,H,W]
    or [B,3,H,W]. Cached text may replace prompts. Decode normalized actions
    with the training data processor before controller execution.
    """

    def __init__(self, policy: PatchFlowPolicy, autoencoder: nn.Module, text_encoder: nn.Module,
                 *, history_pool_size: int = 4):
        super().__init__()
        self.policy, self.autoencoder, self.text_encoder = policy, autoencoder, text_encoder
        if history_pool_size < 1:
            raise ValueError("History pooling size must be positive")
        self.history_pool_size = int(history_pool_size)
        self.autoencoder.requires_grad_(False).eval()
        if not getattr(self.text_encoder, "requires_images", False):
            self.text_encoder.requires_grad_(False).eval()

    def checkpoint_contract(self):
        contract = getattr(self.text_encoder, "checkpoint_contract", None)
        return {"history_pool_size": self.history_pool_size,
                "text_encoder": contract() if callable(contract) else type(self.text_encoder).__qualname__}

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        if getattr(self.text_encoder, "requires_images", False):
            self.text_encoder.train(mode)
        else:
            self.text_encoder.eval()
        return self

    def _encode(self, batch: Mapping[str, Any], *, target: bool) -> dict[str, Any]:
        cache_formats = batch.get("text_cache_format")
        if cache_formats is not None:
            cache_formats = [cache_formats] if isinstance(cache_formats, str) else cache_formats
            if getattr(self.text_encoder, "requires_images", False):
                raise ValueError("VL conditioning requires current/past images rather than a text-only cache")
            if any(value != "qwen3_flux2" for value in cache_formats) or "text_tokens" not in batch:
                raise ValueError("This policy requires a native Qwen3 FLUX.2 text cache")
        result = dict(batch)
        if "reference_tokens" in batch:
            current = batch["reference_tokens"]
            if current.ndim != 3:
                raise ValueError("reference_tokens must be [B,R,token_dim]")
            parameter = next(self.policy.parameters(), None)
            current = current if parameter is None else current.to(parameter)
            result["reference_tokens"] = current
            b = current.shape[0]
            channels = current.shape[-1]
            if target and "future_tokens" not in batch:
                raise ValueError("Cached training observations require future_tokens")
            if "future_tokens" in batch:
                result["future_tokens"] = batch["future_tokens"].to(current)
            for name in ("reference_ids", "future_ids"):
                if name in batch:
                    result[name] = batch[name].to(device=current.device)
        else:
            video = batch.get("camera_video", batch.get("video"))
            if video is None:
                raise KeyError("A video or separate camera_video observation is required")
            if video.ndim == 4:
                video = video.unsqueeze(2)
            if video.ndim == 5:
                video = video.unsqueeze(1)
            if video.ndim != 6 or video.shape[2] != 3 or video.shape[3] < (2 if target else 1):
                raise ValueError("Observation video must be [B,V,3,T,H,W], with T>=2 for training")
            b, views = video.shape[:2]
            if not 1 <= views <= 10:
                raise ValueError("Separate camera observations support one to ten views")
            encoder_parameter = next(self.autoencoder.parameters())
            with torch.no_grad():
                current = self.autoencoder.encode(video[:, :, :, 0].flatten(0, 1).to(encoder_parameter))
            _, channels, height, width = current.shape
            if channels != self.policy.codec.token_dim:
                raise ValueError("Official AE must produce already packed 128-channel latents")
            result["reference_tokens"] = flatten_image_latents(current).reshape(b, views * height * width, channels)
            result["reference_ids"] = torch.cat([raster_coordinates(b, height, width, group=10 + view, device=current.device) for view in range(views)], 1)
            result["future_ids"] = torch.cat([raster_coordinates(b, height, width, group=view, device=current.device) for view in range(views)], 1)
            if target:
                with torch.no_grad():
                    future = self.autoencoder.encode(video[:, :, :, -1].flatten(0, 1).to(encoder_parameter))
                if future.shape != current.shape:
                    raise ValueError("Current and future raster shapes differ")
                result["future_tokens"] = flatten_image_latents(future).reshape(b, views * height * width, channels)
        history = batch.get("history_video")
        if history is not None:
            if history.ndim != 5 or history.shape[0] != b or history.shape[2] != 3:
                raise ValueError("World-model history must be [B,K,3,H,W]")
            valid = batch.get("history_valid")
            if valid is None or valid.shape != history.shape[:2]:
                raise ValueError("History observations require an explicit [B,K] validity mask")
            if history.shape[1]:
                if "reference_ids" not in result:
                    raise ValueError("Cached observations need raster reference_ids before adding history")
                valid = valid.to(device=current.device, dtype=torch.bool)
                safe_history = torch.where(valid[..., None, None, None].to(history.device), history, 0)
                with torch.no_grad():
                    encoded_history = self.autoencoder.encode(safe_history.flatten(0, 1).to(next(self.autoencoder.parameters())))
                    pooled = F.adaptive_avg_pool2d(encoded_history, (self.history_pool_size,) * 2)
                history_tokens = flatten_image_latents(pooled).reshape(b, history.shape[1] * self.history_pool_size**2, channels)
                token_valid = valid.repeat_interleave(self.history_pool_size**2, dim=1)
                history_tokens = history_tokens.masked_fill(~token_valid[..., None], 0)
                history_ids = torch.cat([raster_coordinates(b, self.history_pool_size, self.history_pool_size, group=-(slot + 1), device=current.device) for slot in range(history.shape[1])], 1)
                reference_valid = result.get("reference_valid", torch.ones(result["reference_tokens"].shape[:2], dtype=torch.bool, device=current.device))
                result["reference_valid"] = torch.cat((reference_valid.to(current.device), token_valid), 1)
                result["reference_tokens"] = torch.cat((result["reference_tokens"], history_tokens.to(result["reference_tokens"])), 1)
                result["reference_ids"] = torch.cat((result["reference_ids"], history_ids), 1)
        # Pooled history is context; predict only the current raster.
        if "future_ids" in result:
            result["sampling_future_length"] = result["future_ids"].shape[1]
        text = batch.get("text_tokens")
        if text is None:
            prompt = batch.get("prompt")
            if not isinstance(prompt, (list, tuple)) or len(prompt) != b:
                raise ValueError("prompt must contain one instruction per sample")
            if getattr(self.text_encoder, "requires_images", False):
                vl_current = batch.get("vl_current")
                if vl_current is None:
                    raise ValueError("VL conditioning requires an unaugmented vl_current observation")
                encoded = self.text_encoder(
                    list(prompt), current=vl_current, history=batch.get("vl_history"),
                    history_valid=batch.get("vl_history_valid"),
                    subtasks=batch.get("subtask") if target else None,
                )
            else:
                encoded = self.text_encoder(list(prompt))
            if isinstance(encoded, tuple):
                text, result["text_valid"] = encoded
            elif isinstance(encoded, Mapping):
                text = encoded["text_tokens"]
                result["text_valid"] = encoded["text_valid"]
                if "auxiliary_loss" in encoded:
                    result["language_auxiliary_loss"] = encoded["auxiliary_loss"]
            else:
                text = encoded
        elif getattr(self.text_encoder, "trainable_adapters", False):
            raise ValueError("Trainable VL adapters cannot be bypassed with cached text features")
        result["text_tokens"] = text.to(current)
        if result["text_tokens"].ndim != 3 or result["text_tokens"].shape[0] != b or result["text_tokens"].shape[-1] != self.policy.text_dim:
            raise ValueError(f"text_tokens must be [B,L,{self.policy.text_dim}]")
        if "text_valid" in result:
            result["text_valid"] = result["text_valid"].to(device=current.device, dtype=torch.bool)
        return result

    def forward(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        encoded = self._encode(batch, target=True)
        result = self.policy(encoded, **kwargs)
        if "language_auxiliary_loss" in encoded and "loss" in result:
            result["loss_subtask"] = encoded["language_auxiliary_loss"]
            result["loss"] = result["loss"] + self.text_encoder.auxiliary_weight * result["loss_subtask"]
        return result

    def loss(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        return self(batch, **kwargs)

    @torch.no_grad()
    def sample_actions(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        encoded = self._encode(batch, target=False)
        kwargs.setdefault("future_length", encoded.get("sampling_future_length"))
        return self.policy.sample_actions(encoded, **kwargs)

    @torch.no_grad()
    def sample_actions_from_future(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        encoded = self._encode({key: value for key, value in batch.items() if key != "subtask"}, target=True)
        return self.policy.sample_actions_from_future(encoded, **kwargs)

    @torch.no_grad()
    def sample_future_from_actions(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        encoded = self._encode(batch, target=False)
        kwargs.setdefault("future_length", encoded.get("sampling_future_length"))
        return self.policy.sample_future_from_actions(encoded, **kwargs)

    @classmethod
    def from_local_assets(
        cls, transformer_path: str, autoencoder_path: str, text_encoder_path: str,
        *, action_dim: int = 14, proprio_dim: int | None = 14, variant: str = "klein-base-4b",
        flux_source: str | None = None, device: str = "cuda", dtype: str = "bfloat16",
        context_length: int = 128, gradient_checkpointing: bool = False,
        shift: float = 5, video_weight: float = 0.5, action_weight: float = 1,
        action_scale: float = 1, isolate_actions: bool = False, text_encoder_kind: str = "qwen3",
        history_pool_size: int = 4, vl_trainable_adapters: bool = False,
        vl_lora_rank: int = 64, vl_lora_alpha: float = 128, vl_lora_dropout: float = 0.05,
        vl_instruction_layers=(9, 18, 27), vl_summary_layers=(18, 27, 36),
        vl_history_pool_size: int = 4, vl_image_size: int = 448,
        vl_auxiliary_weight: float = 0.1, vl_label_smoothing: float = 0.1,
        vl_require_subtasks: bool = False, vl_gradient_checkpointing: bool = True,
        **policy_options,
    ) -> "FluxAssetPolicy":
        """Load local assets without downloads or remote code."""
        if variant not in {"klein-base-4b", "klein-base-9b"}:
            raise ValueError("variant must be klein-base-4b or klein-base-9b")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported torch dtype")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        for path in (transformer_path, autoencoder_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if not Path(text_encoder_path).is_dir():
            raise FileNotFoundError("text_encoder_path must be a local Hugging Face model directory")
        package_root = None
        if flux_source is not None:
            package_root = Path(flux_source).expanduser().resolve() / "src"
            if not (package_root / "flux2" / "model.py").is_file():
                raise FileNotFoundError("flux_source must be the official FLUX.2 repository root")
            sys.path.insert(0, str(package_root))
        try:
            model_api = importlib.import_module("flux2.model")
            ae_api = importlib.import_module("flux2.autoencoder")
            text_api = importlib.import_module("flux2.text_encoder")
        except ImportError as exc:
            raise ImportError("Install the official black-forest-labs/flux2 package or supply flux_source") from exc
        if package_root is not None and any(
            not Path(module.__file__).resolve().is_relative_to(package_root / "flux2")
            for module in (model_api, ae_api, text_api)
        ):
            raise RuntimeError("Another FLUX.2 source is already imported; start a fresh process for flux_source")
        from safetensors.torch import load_file

        asset_signature = _asset_identity(transformer_path, autoencoder_path, text_encoder_path,
                                          (model_api, ae_api, text_api))
        precision = getattr(torch, dtype)
        config = model_api.Klein4BParams() if variant.endswith("4b") else model_api.Klein9BParams()
        with torch.device("meta"):
            transformer = model_api.Flux2(config).to(dtype=precision)
            autoencoder = ae_api.AutoEncoder(ae_api.AutoEncoderParams()).to(dtype=precision)
        transformer.load_state_dict(load_file(transformer_path, device=device), strict=True, assign=True)
        autoencoder.load_state_dict(load_file(autoencoder_path, device=device), strict=True, assign=True)
        transformer, autoencoder = transformer.to(device=device, dtype=precision), autoencoder.to(device=device, dtype=precision)
        if text_encoder_kind == "qwen3":
            text_encoder = LocalInstructionEncoder(text_encoder_path, context_length=context_length, dtype=precision).to(device)
        elif text_encoder_kind == "qwen3_vl":
            from .vision_language import VisualInstructionEncoder
            text_encoder = VisualInstructionEncoder(
                text_encoder_path, dtype=precision, trainable_adapters=vl_trainable_adapters,
                lora_rank=vl_lora_rank, lora_alpha=vl_lora_alpha, lora_dropout=vl_lora_dropout,
                instruction_layers=vl_instruction_layers, summary_layers=vl_summary_layers,
                history_pool_size=vl_history_pool_size, image_size=vl_image_size,
                auxiliary_weight=vl_auxiliary_weight, label_smoothing=vl_label_smoothing,
                require_subtasks=vl_require_subtasks,
                gradient_checkpointing=vl_gradient_checkpointing,
            ).to(device)
        else:
            raise ValueError("text_encoder_kind must be qwen3 or qwen3_vl")
        if text_encoder.feature_dim != config.context_in_dim:
            raise ValueError(f"The text encoder provides {text_encoder.feature_dim} features; {variant} requires {config.context_in_dim}")
        denoiser = OfficialFluxDenoiser(transformer, gradient_checkpointing=gradient_checkpointing)
        policy = PatchFlowPolicy(
            denoiser, action_dim=action_dim, text_dim=config.context_in_dim, proprio_dim=proprio_dim,
            action_scale=action_scale, shift=shift, video_weight=video_weight, action_weight=action_weight,
            isolate_actions=isolate_actions,
            **policy_options,
        ).to(device=device, dtype=precision)
        if asset_signature != _asset_identity(transformer_path, autoencoder_path, text_encoder_path,
                                              (model_api, ae_api, text_api)):
            raise RuntimeError("Local model assets changed while they were being loaded")
        result = cls(policy, autoencoder, text_encoder, history_pool_size=history_pool_size)
        result.asset_signature = asset_signature
        return result


class LocalInstructionEncoder(nn.Module):
    """Local Qwen3 embeddings using official FLUX.2 readout layers."""

    def __init__(self, model_path: str, *, context_length: int = 128, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True, trust_remote_code=False,
        )
        # Use the external package's readout layers.
        self.layers = tuple(importlib.import_module("flux2.text_encoder").OUTPUT_LAYERS_QWEN3)
        if not self.layers or min(self.layers) < 0 or max(self.layers) > self.encoder.config.num_hidden_layers:
            raise ValueError("The text encoder does not provide the required hidden-state layers")
        self.feature_dim = self.encoder.config.hidden_size * len(self.layers)
        self.context_length = context_length

    def checkpoint_contract(self):
        return {"kind": "qwen3_text", "layers": self.layers, "context_length": self.context_length}

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(
            formatted, padding="max_length", truncation=True,
            max_length=self.context_length, return_tensors="pt",
        ).to(next(self.encoder.parameters()).device)
        outputs = self.encoder(**inputs, output_hidden_states=True, use_cache=False)
        features = torch.cat([outputs.hidden_states[index] for index in self.layers], dim=-1)
        return features, inputs["attention_mask"].bool()
