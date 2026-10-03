# SPDX-License-Identifier: Apache-2.0
"""Native policy weights and compact checkpoints for frozen-encoder policies."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_policy_weights(model, path):
    """Load a complete native safetensors policy without changing training state."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("Policy weights must use the native .safetensors format")
    policy = getattr(model, "policy", model)
    weights = load_file(str(path))
    expected = policy.state_dict()
    if weights.keys() != expected.keys():
        missing = sorted(expected.keys() - weights.keys())[:10]
        extra = sorted(weights.keys() - expected.keys())[:10]
        raise ValueError(f"Incomplete policy weights: missing={missing}, unexpected={extra}")
    for name, tensor in weights.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"Weight shape differs for {name}: {tensor.shape} vs {expected[name].shape}")
        if expected[name].is_floating_point() and (
            not tensor.is_floating_point() or not torch.isfinite(tensor.to(dtype=expected[name].dtype)).all()
        ):
            raise ValueError(f"Weight values are invalid for {name}")
    policy.load_state_dict(weights, strict=True)
    return {"loaded_tensors": len(weights)}


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
                load_policy_weights(model, Path(input_dir) / f"policy_{index}.safetensors")
                models.pop(index)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
