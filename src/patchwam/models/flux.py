# SPDX-License-Identifier: Apache-2.0
"""Masked execution adapter for the official Black Forest Labs FLUX.2 weights.

FLUX.2 remains an external dependency, not a vendored backbone. This adapter
uses its learned layers, rotary implementation, and timestep embedding while
providing PatchWAM's prefix visibility. The official default image-generation
attention is different and must not be substituted for this masked execution.

Supported upstream layer contract: black-forest-labs/flux2, src/flux2/model.py,
Klein4BParams / Klein9BParams. CPU micro-model integration was tested against
commit 50fe5162777813d869182b139e83b10743caef15. Checkpoint load is strict;
older research policy payloads require an explicit conversion.
"""

import importlib
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import flatten_image_latents, raster_coordinates
from .policy import PatchFlowPolicy


class OfficialFluxDenoiser(nn.Module):
    """Keep upstream weight names intact and replace only execution orchestration."""

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

    @staticmethod
    def _heads(projected: Tensor, count: int) -> tuple[Tensor, Tensor, Tensor]:
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
            q, k, v = self._heads(attention.qkv(normalized * (1 + scale) + shift), block.num_heads)
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
        q, k, v = self._heads(attention_input, block.num_heads)
        q, k = block.norm(q, k, v)
        attended = self._attend(q, k, v, position, visibility)
        joint_update = torch.cat((attended, block.mlp_act(feedforward_input)), -1)
        return states + gate * block.linear2(joint_update)

    def forward(
        self, reference: Tensor, noisy: Tensor, sigma: Tensor, context: Tensor,
        *, reference_ids: Tensor, noisy_ids: Tensor, context_ids: Tensor, visibility: Tensor,
    ) -> Tensor:
        core = self.transformer
        images = core.img_in(torch.cat((reference, noisy), 1))
        text = core.txt_in(context)
        # The official function multiplies sigma by 1000 internally.
        time = core.time_in(self._embed_time(sigma.to(images), 256))
        text_mod = core.double_stream_modulation_txt(time)
        image_mod = core.double_stream_modulation_img(time)
        single_mod, _ = core.single_stream_modulation(time)
        image_positions = core.pe_embedder(torch.cat((reference_ids, noisy_ids), 1))
        text_positions = core.pe_embedder(context_ids)
        position = torch.cat((text_positions, image_positions), dim=2)
        checkpointed = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in core.double_blocks:
            # Bind the layer in the callable, so recomputation never uses a later layer.
            def execute(t, i, block=block):
                return self._paired_block(block, t, i, text_mod, image_mod, position, visibility)

            text, images = checkpoint(execute, text, images, use_reentrant=False) if checkpointed else execute(text, images)
        states = torch.cat((text, images), 1)
        for block in core.single_blocks:
            def execute(s, block=block):
                return self._unified_block(block, s, single_mod, position, visibility)

            states = checkpoint(execute, states, use_reentrant=False) if checkpointed else execute(states)
        generated = states[:, text.shape[1] + reference.shape[1]:]
        return core.final_layer(generated, time)


class FluxAssetPolicy(nn.Module):
    """Frozen official encoders around the trainable latent policy.

    Raw dataset video is [B,3,2,H,W] in [-1,1], with current/future frames.
    Inference accepts [B,3,1,H,W] or [B,3,H,W]. Cached text_tokens may replace
    prompts. The returned action is normalized; a controller must invert the
    same dataset transform used during training.
    """

    def __init__(self, policy: PatchFlowPolicy, autoencoder: nn.Module, text_encoder: nn.Module):
        super().__init__()
        self.policy, self.autoencoder, self.text_encoder = policy, autoencoder, text_encoder
        for asset in (self.autoencoder, self.text_encoder):
            asset.requires_grad_(False)
            asset.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def _encode(self, batch: Mapping[str, Any], *, target: bool) -> dict[str, Any]:
        if "reference_tokens" in batch:
            return dict(batch)
        video = batch["video"]
        if video.ndim == 4:
            video = video.unsqueeze(2)
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < (2 if target else 1):
            raise ValueError("video must be [B,3,T,H,W], with T>=2 for training")
        encoder_parameter = next(self.autoencoder.parameters())
        current = self.autoencoder.encode(video[:, :, 0].to(encoder_parameter))
        b, channels, height, width = current.shape
        if channels != self.policy.codec.token_dim:
            raise ValueError("Official AE must produce already packed 128-channel latents")
        result = dict(batch)
        result["reference_tokens"] = flatten_image_latents(current)
        result["reference_ids"] = raster_coordinates(b, height, width, group=10, device=current.device)
        result["future_ids"] = raster_coordinates(b, height, width, group=0, device=current.device)
        if target:
            future = self.autoencoder.encode(video[:, :, -1].to(encoder_parameter))
            if future.shape != current.shape:
                raise ValueError("Current and future raster shapes differ")
            result["future_tokens"] = flatten_image_latents(future)
        text = batch.get("text_tokens")
        if text is None:
            prompt = batch.get("prompt")
            if not isinstance(prompt, (list, tuple)) or len(prompt) != b:
                raise ValueError("prompt must contain one instruction per sample")
            encoded = self.text_encoder(list(prompt))
            if isinstance(encoded, tuple):
                text, result["text_valid"] = encoded
            else:
                text = encoded
        result["text_tokens"] = text.to(current)
        return result

    def forward(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        return self.policy(self._encode(batch, target=True), **kwargs)

    def loss(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        return self(batch, **kwargs)

    @torch.no_grad()
    def sample_actions(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        return self.policy.sample_actions(self._encode(batch, target=False), **kwargs)

    @classmethod
    def from_local_assets(
        cls, transformer_path: str, autoencoder_path: str, text_encoder_path: str,
        *, action_dim: int = 14, proprio_dim: int | None = 14, variant: str = "klein-base-4b",
        flux_source: str | None = None, device: str = "cuda", dtype: str = "bfloat16",
        context_length: int = 128, gradient_checkpointing: bool = False,
        shift: float = 5, video_weight: float = 0.5, action_weight: float = 1,
        action_scale: float = 1, isolate_actions: bool = False,
    ) -> "FluxAssetPolicy":
        """Load explicit local assets. No model download or remote-code execution."""
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
        if flux_source is not None:
            package_root = Path(flux_source).expanduser().resolve() / "src"
            if not (package_root / "flux2" / "model.py").is_file():
                raise FileNotFoundError("flux_source must be the official FLUX.2 repository root")
            sys.path.insert(0, str(package_root))
        try:
            model_api = importlib.import_module("flux2.model")
            ae_api = importlib.import_module("flux2.autoencoder")
        except ImportError as exc:
            raise ImportError("Install the official black-forest-labs/flux2 package or supply flux_source") from exc
        from safetensors.torch import load_file

        precision = getattr(torch, dtype)
        config = model_api.Klein4BParams() if variant.endswith("4b") else model_api.Klein9BParams()
        with torch.device("meta"):
            transformer = model_api.Flux2(config).to(dtype=precision)
            autoencoder = ae_api.AutoEncoder(ae_api.AutoEncoderParams()).to(dtype=precision)
        transformer.load_state_dict(load_file(transformer_path, device=device), strict=True, assign=True)
        autoencoder.load_state_dict(load_file(autoencoder_path, device=device), strict=True, assign=True)
        transformer, autoencoder = transformer.to(device=device, dtype=precision), autoencoder.to(device=device, dtype=precision)
        text_encoder = LocalInstructionEncoder(text_encoder_path, context_length=context_length, dtype=precision).to(device)
        denoiser = OfficialFluxDenoiser(transformer, gradient_checkpointing=gradient_checkpointing)
        policy = PatchFlowPolicy(
            denoiser, action_dim=action_dim, text_dim=config.context_in_dim, proprio_dim=proprio_dim,
            action_scale=action_scale, shift=shift, video_weight=video_weight, action_weight=action_weight,
            isolate_actions=isolate_actions,
        ).to(device=device, dtype=precision)
        return cls(policy, autoencoder, text_encoder)


class LocalInstructionEncoder(nn.Module):
    """Local Qwen3 instruction embeddings with the official FLUX.2 layer readout."""

    def __init__(self, model_path: str, *, context_length: int = 128, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True, trust_remote_code=False,
        )
        # Externally supplied upstream package owns the feature-layer convention.
        self.layers = tuple(importlib.import_module("flux2.text_encoder").OUTPUT_LAYERS_QWEN3)
        self.context_length = context_length

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
