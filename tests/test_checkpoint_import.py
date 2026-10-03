"""Strict migration checks using real, micro-size official FLUX.2 modules."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchwam.checkpoints import import_research_weights, register_compact_policy_state
from patchwam.models import OfficialFluxDenoiser, PatchFlowPolicy, make_tiny_policy


def micro_policy():
    api = pytest.importorskip("flux2.model")
    parameters = api.Klein4BParams(
        hidden_size=64, num_heads=4, axes_dim=[4, 4, 4, 4], context_in_dim=32,
        depth=1, depth_single_blocks=1, mlp_ratio=2,
    )
    return PatchFlowPolicy(OfficialFluxDenoiser(api.Flux2(parameters)), action_dim=14, text_dim=32, proprio_dim=14)


def archived_payload(policy, *, aliases=True):
    # The historical wrapper registered both the model and its layer lists.
    class ArchivedVisualContainer(nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer
            if aliases:
                self.double_blocks = transformer.double_blocks
                self.single_blocks = transformer.single_blocks

    container = nn.Module()
    container.mixtures = nn.ModuleDict({"video": ArchivedVisualContainer(policy.denoiser.transformer)})
    return {
        "mot": {key: value.detach().clone() for key, value in container.state_dict().items()},
        "proprio_encoder": {key: value.detach().clone() for key, value in policy.state_projection.state_dict().items()},
        "step": 123,
    }


@pytest.mark.parametrize("aliases", [False, True])
def test_full_micro_official_checkpoint_imports_complete_weights(tmp_path, aliases):
    source = micro_policy()
    destination = micro_policy()
    path = tmp_path / "research.pt"
    torch.save(archived_payload(source, aliases=aliases), path)
    result = import_research_weights(destination, path)
    assert result == {"source_step": 123, "mapped_tensors": len(source.state_dict())}
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["expert", "shape", "missing", "alias", "partial"])
def test_unsupported_or_inconsistent_checkpoint_is_rejected_before_any_weight_changes(tmp_path, damage):
    source = micro_policy()
    destination = micro_policy()
    before = {key: value.clone() for key, value in destination.state_dict().items()}
    payload = archived_payload(source)
    canonical = "mixtures.video.transformer.img_in.weight"
    if damage == "expert":
        payload["mot"]["mixtures.action.token_input.weight"] = torch.ones(4, 4)
    elif damage == "shape":
        payload["mot"][canonical] = torch.ones(1, 1)
    elif damage == "missing":
        del payload["mot"][canonical]
    elif damage == "alias":
        key = next(key for key in payload["mot"] if key.startswith("mixtures.video.double_blocks."))
        payload["mot"][key] = payload["mot"][key] + 1
    else:
        payload["checkpoint_format"] = "trainable_only"
    path = tmp_path / "damaged.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        import_research_weights(destination, path)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


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
    from safetensors.torch import load_file

    policy = make_tiny_policy()
    wrapper = SimpleNamespace(policy=policy, autoencoder=nn.Linear(4, 4), text_encoder=nn.Linear(4, 4))
    accelerator = HookHarness()
    register_compact_policy_state(accelerator)
    expected = {key: value.clone() for key, value in policy.state_dict().items()}
    asset_before = wrapper.autoencoder.weight.detach().clone()
    weights = [{"sentinel": torch.tensor(1)}]
    accelerator.save([wrapper], weights, str(tmp_path))
    assert weights == []
    saved = load_file(str(tmp_path / "policy_0.safetensors"))
    assert saved.keys() == expected.keys()
    assert not any("autoencoder" in key or "text_encoder" in key for key in saved)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    models = [wrapper]
    accelerator.load(models, str(tmp_path))
    assert models == []
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    torch.testing.assert_close(wrapper.autoencoder.weight, asset_before, rtol=0, atol=0)
