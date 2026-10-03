"""Official FLUX.2 micro-module tests without pretrained weights."""

import copy

import pytest
import torch
from safetensors.torch import save_file

from patchwam.models import (
    FluxAssetPolicy,
    OfficialFluxDenoiser,
    joint_visibility,
    sequence_coordinates,
)
from patchwam.models.flux import LocalInstructionEncoder

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


def test_local_instruction_encoder_reads_official_qwen_layers_and_keeps_padding(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=28,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4,
    )
    transformers.Qwen3ForCausalLM(config).save_pretrained(tmp_path)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False and kwargs["add_generation_prompt"] is True
            return [entry[0]["content"] for entry in messages]

        def __call__(self, formatted, **kwargs):
            assert kwargs["max_length"] == 6 and kwargs["padding"] == "max_length"
            return transformers.BatchEncoding({
                "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [2, 3, 4, 5, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]]),
            })

    tokenizer = Tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    encoder = LocalInstructionEncoder(str(tmp_path), context_length=6, dtype=torch.float32).eval()
    assert encoder.feature_dim == 24
    features, mask = encoder(["move", "lift"])
    inputs = tokenizer(["move", "lift"], max_length=6, padding="max_length")
    native = encoder.encoder(**inputs, output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(features, torch.cat([native.hidden_states[index] for index in (9, 18, 27)], -1))
    torch.testing.assert_close(mask, inputs["attention_mask"].bool())


def test_local_asset_factory_loads_real_micro_weights_and_trains_without_frozen_gradients(tmp_path, monkeypatch):
    import importlib

    transformers = pytest.importorskip("transformers")
    ae_api = importlib.import_module("flux2.autoencoder")
    config = api.Klein4BParams(
        context_in_dim=24, hidden_size=64, num_heads=4, depth=1, depth_single_blocks=1,
        axes_dim=[4, 4, 4, 4], mlp_ratio=2,
    )
    ae_config = ae_api.AutoEncoderParams(resolution=16, ch=32, ch_mult=[1, 1], num_res_blocks=1)
    transformer, autoencoder = api.Flux2(config), ae_api.AutoEncoder(ae_config)
    transformer_path, ae_path = tmp_path / "transformer.safetensors", tmp_path / "ae.safetensors"
    save_file(transformer.state_dict(), str(transformer_path))
    save_file(autoencoder.state_dict(), str(ae_path))
    text_path = tmp_path / "text"
    transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=28,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4,
    )).save_pretrained(text_path)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [entry[0]["content"] for entry in messages]

        def __call__(self, formatted, **kwargs):
            return transformers.BatchEncoding({
                "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0]]).expand(len(formatted), -1),
                "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0]]).expand(len(formatted), -1),
            })

    monkeypatch.setattr(api, "Klein4BParams", lambda: config)
    monkeypatch.setattr(ae_api, "AutoEncoderParams", lambda: ae_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    policy = FluxAssetPolicy.from_local_assets(
        str(transformer_path), str(ae_path), str(text_path), device="cpu", dtype="float32",
        context_length=6, gradient_checkpointing=True,
    )
    for name, value in policy.policy.denoiser.transformer.state_dict().items():
        torch.testing.assert_close(value, transformer.state_dict()[name], rtol=0, atol=0)
    assert policy.asset_signature["autoencoder"]["path"] == str(ae_path.resolve())
    sample = {"video": torch.randn(2, 3, 2, 16, 16), "prompt": ["move", "lift"],
              "action": torch.randn(2, 16, 14), "proprio": torch.randn(2, 14)}
    policy.train()
    loss = policy(sample)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.policy.state_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.autoencoder.parameters())
    assert all(p.grad is None for p in policy.text_encoder.parameters())
    assert torch.isfinite(policy.sample_actions(sample, steps=2)["action"]).all()
