"""Policy weights and compact training checkpoints."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _policy_layout(model):
    policy = getattr(model, "policy", model)
    named = getattr(model, "named_parameters", None)
    extra = (
        {}
        if policy is model or named is None
        else {
            name: tensor
            for name, tensor in named()
            if tensor.requires_grad and not name.startswith("policy.")
        }
    )
    return policy, extra


def policy_weight_state(model):
    """Collect policy state and trainable encoder adapters."""
    policy, extra = _policy_layout(model)
    state = policy.state_dict()
    if extra:
        state = {"policy." + key: tensor for key, tensor in state.items()}
        state.update({key: tensor.detach() for key, tensor in extra.items()})
    return state


def checkpoint_parameter_keys(model):
    """Map model parameters to checkpoint keys."""
    policy, extra = _policy_layout(model)
    target = model if hasattr(model, "named_parameters") else policy
    result = {}
    for name, tensor in target.named_parameters(remove_duplicate=False):
        if policy is model or target is policy:
            result[name] = name
        elif name.startswith("policy."):
            result[name] = name if extra else name.removeprefix("policy.")
        elif name in extra:
            result[name] = name
    return result


def load_policy_weights(model, path):
    """Load native weights without changing optimizer state."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("Policy weights must use the native .safetensors format")
    policy = getattr(model, "policy", model)
    weights = load_file(str(path))
    expected = policy_weight_state(model)
    if weights.keys() != expected.keys():
        missing = sorted(expected.keys() - weights.keys())[:10]
        extra = sorted(weights.keys() - expected.keys())[:10]
        raise ValueError(f"Incomplete policy weights: missing={missing}, unexpected={extra}")
    for name, tensor in weights.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"Weight shape differs for {name}: {tensor.shape} vs {expected[name].shape}"
            )
        if expected[name].is_floating_point() and (
            not tensor.is_floating_point()
            or not torch.isfinite(tensor.to(dtype=expected[name].dtype)).all()
        ):
            raise ValueError(f"Weight values are invalid for {name}")
    _, extra = _policy_layout(model)
    policy_weights = (
        {
            key.removeprefix("policy."): value
            for key, value in weights.items()
            if key.startswith("policy.")
        }
        if extra
        else weights
    )
    policy.load_state_dict(policy_weights, strict=True)
    with torch.no_grad():
        for name, parameter in extra.items():
            parameter.copy_(weights[name])
    return {"loaded_tensors": len(weights)}


def register_compact_policy_state(accelerator):
    """Save trainable weights; reload frozen assets from config."""

    def compact(model):
        model = accelerator.unwrap_model(model)
        return (
            model
            if all(hasattr(model, key) for key in ("policy", "autoencoder", "text_encoder"))
            else None
        )

    def save_hook(models, weights, output_dir):
        for index in range(len(models) - 1, -1, -1):
            model = compact(models[index])
            if model is not None:
                if accelerator.is_main_process:
                    state = {
                        key: tensor.detach().cpu().clone().contiguous()
                        for key, tensor in policy_weight_state(model).items()
                    }
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


def save_policy_average(average, model, path):
    """Export EMA weights separately from training state."""
    save_file(
        average.native_weights(model),
        str(path),
        metadata={
            "weight_variant": "ema",
            "successful_updates": str(average.updates),
        },
    )
