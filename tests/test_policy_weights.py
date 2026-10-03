"""Native policy initialization and frozen-asset checkpoint coverage."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from patchwam.checkpoints import load_policy_weights, register_compact_policy_state
from patchwam.models import make_tiny_policy


def weight_copy(policy):
    return {key: value.detach().clone().contiguous() for key, value in policy.state_dict().items()}


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_policy_weights_load_completely(tmp_path, wrapped):
    source, destination = make_tiny_policy(), make_tiny_policy()
    path = tmp_path / "policy.safetensors"
    save_file(weight_copy(source), str(path))
    model = SimpleNamespace(policy=destination) if wrapped else destination
    assert load_policy_weights(model, path) == {"loaded_tensors": len(source.state_dict())}
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["shape", "missing", "unexpected"])
def test_invalid_policy_weights_do_not_change_any_parameters(tmp_path, damage):
    source, destination = make_tiny_policy(), make_tiny_policy()
    before = weight_copy(destination)
    weights = weight_copy(source)
    key = next(iter(weights))
    if damage == "shape":
        weights[key] = torch.ones(1)
    elif damage == "missing":
        del weights[key]
    else:
        weights["unrecognized.weight"] = torch.ones(1)
    path = tmp_path / "invalid.safetensors"
    save_file(weights, str(path))
    with pytest.raises(ValueError):
        load_policy_weights(destination, path)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("extension", [".pt", ".bin"])
def test_policy_initialization_requires_safetensors(tmp_path, extension):
    with pytest.raises(ValueError, match="native .safetensors"):
        load_policy_weights(make_tiny_policy(), tmp_path / ("weights" + extension))


class HookHarness:
    is_main_process = True

    @staticmethod
    def unwrap_model(model):
        return model

    def register_save_state_pre_hook(self, hook):
        self.save = hook

    def register_load_state_pre_hook(self, hook):
        self.load = hook


def test_compact_hook_roundtrip_excludes_frozen_encoder_weights(tmp_path):
    policy = make_tiny_policy()
    wrapper = SimpleNamespace(policy=policy, autoencoder=nn.Linear(4, 4), text_encoder=nn.Linear(4, 4))
    accelerator = HookHarness()
    register_compact_policy_state(accelerator)
    expected = weight_copy(policy)
    asset_before = wrapper.autoencoder.weight.detach().clone()
    weights = [{"sentinel": torch.tensor(1)}]
    accelerator.save([wrapper], weights, str(tmp_path))
    assert weights == []
    saved = load_file(str(tmp_path / "policy_0.safetensors"))
    assert saved.keys() == expected.keys()
    for name, tensor in saved.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    models = [wrapper]
    accelerator.load(models, str(tmp_path))
    assert models == []
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    torch.testing.assert_close(wrapper.autoencoder.weight, asset_before, rtol=0, atol=0)
