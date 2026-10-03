"""Optional integration tests against the external official FLUX.2 package.

No pretrained checkpoint is needed: micro-size official modules exercise the
same execution/weight contract. These tests do not establish checkpoint parity,
CUDA kernel parity, performance, or closed-loop success rate.
"""

import copy

import pytest
import torch

from patchwam.models import OfficialFluxDenoiser, joint_visibility, sequence_coordinates

api = pytest.importorskip("flux2.model", reason="Install the official FLUX.2 dependency for its adapter tests")


def official_core():
    config = api.Klein4BParams(
        context_in_dim=32, hidden_size=64, num_heads=4, depth=2, depth_single_blocks=2,
        axes_dim=[4, 4, 4, 4], mlp_ratio=2,
    )
    return api.Flux2(config)


def inputs(reference_length=3):
    return {
        "reference": torch.randn(2, reference_length, 128),
        "noisy": torch.randn(2, 6, 128), "context": torch.randn(2, 4, 32),
        "sigma": torch.tensor([0.3, 0.7]),
        "reference_ids": sequence_coordinates(2, reference_length, group=10),
        "noisy_ids": sequence_coordinates(2, 6, group=20),
        "context_ids": sequence_coordinates(2, 4),
        "visibility": joint_visibility(4, reference_length, 4, 2),
    }


def test_adapter_matches_official_weights_when_visibility_is_fully_open():
    core = official_core()
    adapter = OfficialFluxDenoiser(core)
    data = inputs(0)
    data["visibility"] = torch.ones(1, 1, 10, 10, dtype=torch.bool)
    adapted = adapter(**data)
    native = core(data["noisy"], data["noisy_ids"], data["sigma"], data["context"], data["context_ids"], None)
    torch.testing.assert_close(adapted, native, atol=3e-6, rtol=3e-6)


def test_masked_official_execution_has_finite_gradients_and_stable_prefix():
    adapter = OfficialFluxDenoiser(official_core())
    data = inputs()
    captured = []
    original = adapter._unified_block

    def capture(*args):
        result = original(*args)
        captured.append(result.detach())
        return result

    adapter._unified_block = capture
    baseline = adapter(**data)
    baseline.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    adapter(**dict(data, noisy=data["noisy"] + 100))
    torch.testing.assert_close(captured[1][:, :7], captured[3][:, :7], atol=0, rtol=0)
    assert not torch.allclose(captured[1][:, 7:], captured[3][:, 7:])


def test_checkpointed_gradients_match_direct_execution():
    direct = OfficialFluxDenoiser(official_core())
    recomputed = OfficialFluxDenoiser(copy.deepcopy(direct.transformer), gradient_checkpointing=True)
    data = inputs()
    direct(**data).square().mean().backward()
    recomputed(**data).square().mean().backward()
    for a, b in zip(direct.parameters(), recomputed.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
