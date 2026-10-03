# SPDX-License-Identifier: Apache-2.0
"""Explicit weight import and compact checkpoints for frozen-encoder policies."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def import_research_weights(model, path):
    """Import a full single-stream checkpoint without changing training state.

    This maps names and checks complete tensor coverage. It does not establish
    numerical or closed-loop parity. Expert, LoRA, and partial checkpoints must
    be converted separately rather than silently dropping their parameters.
    """
    policy = getattr(model, "policy", model)
    if not hasattr(policy.denoiser, "transformer"):
        raise TypeError("Research weight import requires the official FLUX.2 adapter")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_format", "full") != "full":
        raise ValueError("Only full, unmerged research checkpoint payloads are supported")
    if "concept_bottleneck" in payload:
        raise ValueError("Reasoner/expert checkpoint requires a separately implemented architecture")
    source = payload.get("mot")
    if not isinstance(source, dict):
        raise ValueError("Expected a research checkpoint with a mot weight dictionary")
    prefix = "mixtures.video.transformer."
    imported = {}
    aliases = []
    for key, value in source.items():
        if not key.startswith(prefix):
            alias_prefix = "mixtures.video."
            if key.startswith((alias_prefix + "double_blocks.", alias_prefix + "single_blocks.")):
                aliases.append((key, value))
                continue
            raise ValueError(f"Unsupported checkpoint component: {key}")
        imported["denoiser.transformer." + key[len(prefix):]] = value
    for key, value in aliases:
        canonical = prefix + key[len("mixtures.video."):]
        if canonical not in source or not torch.equal(value, source[canonical]):
            raise ValueError(f"Conflicting or incomplete checkpoint alias: {key}")
    for key, value in payload.get("proprio_encoder", {}).items():
        imported["state_projection." + key] = value
    expected = policy.state_dict()
    if imported.keys() != expected.keys():
        missing = sorted(expected.keys() - imported.keys())[:10]
        extra = sorted(imported.keys() - expected.keys())[:10]
        raise ValueError(f"Incomplete weight mapping: missing={missing}, unexpected={extra}")
    for name, tensor in imported.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"Weight shape differs for {name}: {tensor.shape} vs {expected[name].shape}")
    policy.load_state_dict(imported, strict=True)
    return {"source_step": payload.get("step"), "mapped_tensors": len(imported)}


def register_compact_policy_state(accelerator):
    """Store the trainable policy once; frozen assets are reconstructed from config."""
    def compact(model):
        model = accelerator.unwrap_model(model)
        return model if all(hasattr(model, key) for key in ("policy", "autoencoder", "text_encoder")) else None

    def save_hook(models, weights, output_dir):
        for index in range(len(models) - 1, -1, -1):
            model = compact(models[index])
            if model is not None:
                if accelerator.is_main_process:
                    state = {key: tensor.detach().cpu().contiguous() for key, tensor in model.policy.state_dict().items()}
                    save_file(state, str(Path(output_dir) / f"policy_{index}.safetensors"))
                weights.pop(index)

    def load_hook(models, input_dir):
        for index in range(len(models) - 1, -1, -1):
            model = compact(models[index])
            if model is not None:
                model.policy.load_state_dict(load_file(str(Path(input_dir) / f"policy_{index}.safetensors")), strict=True)
                models.pop(index)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
