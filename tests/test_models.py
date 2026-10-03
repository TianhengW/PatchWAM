import math

import pytest
import torch
from torch import nn

from patchwam.models import (
    FluxAssetPolicy,
    PatchFlowPolicy,
    RepeatedActionCodec,
    ShiftedFlow,
    joint_visibility,
    make_tiny_policy,
    raster_coordinates,
    sequence_coordinates,
)
from patchwam.models.flow import masked_sample_mse
from patchwam.models.geometry import flatten_image_latents, restore_image_latents


@pytest.mark.parametrize("action_dim", [7, 14, 80, 128])
def test_codec_is_a_left_inverse_and_has_no_parameters(action_dim):
    codec = RepeatedActionCodec(action_dim, scale=0.75)
    actions = torch.randn(2, 16, action_dim, requires_grad=True)
    tokens = codec.encode(actions)
    assert tokens.shape == (2, 16, 128)
    torch.testing.assert_close(codec.decode(tokens), actions)
    assert torch.count_nonzero(tokens[..., codec.occupied:]) == 0
    codec.decode(tokens).sum().backward()
    torch.testing.assert_close(actions.grad, torch.ones_like(actions))


def test_codec_decodes_least_squares_coordinates():
    codec = RepeatedActionCodec(14)
    tokens = torch.randn(2, 16, 128)
    residual = tokens - codec.encode(codec.decode(tokens))
    torch.testing.assert_close(codec.decode(residual), torch.zeros(2, 16, 14), atol=2e-7, rtol=0)


def test_prefix_visibility_and_isolation():
    mask = joint_visibility(3, 4, 5, 2)[0, 0]
    assert mask[:7, :7].all() and not mask[:7, 7:].any()
    assert mask[7:].all()
    isolated = joint_visibility(3, 4, 5, 2, isolate_actions=True)[0, 0]
    assert not isolated[7:12, 12:].any() and not isolated[12:, 7:12].any()
    assert isolated[7:, :7].all()
    padded = joint_visibility(3, 4, 5, 2, text_valid=torch.tensor([[True, False, True]]))[0, 0]
    assert not padded[:, 1].any() and padded[:, 0].all()


def test_raster_and_group_coordinates_preserve_the_paper_layout():
    reference = raster_coordinates(2, 2, 3, group=10)
    assert reference.shape == (2, 6, 4)
    torch.testing.assert_close(reference[0, :, 1], torch.tensor([0, 0, 0, 1, 1, 1]).float())
    torch.testing.assert_close(reference[0, :, 2], torch.tensor([0, 1, 2, 0, 1, 2]).float())
    assert (reference[..., 0] == 10).all()
    actions = sequence_coordinates(2, 16, group=20)
    assert (actions[..., 0] == 20).all()
    torch.testing.assert_close(actions[0, :, 3], torch.arange(16).float())
    latent = torch.randn(2, 128, 2, 3)
    torch.testing.assert_close(restore_image_latents(flatten_image_latents(latent), 2, 3), latent)


def test_shift_weight_and_solver_endpoints():
    flow = ShiftedFlow(5)
    tau = (torch.arange(40000).double() + 0.5) / 40000
    sigma = flow.warp(tau)
    torch.testing.assert_close(flow.warp(torch.tensor([0.0, 1 / 6, 1.0])), torch.tensor([0.0, 0.5, 1.0]))
    assert abs(flow.weight(sigma).mean().item() - 1) < 1e-6
    assert flow.weight(torch.tensor([0.0, 1.0])).sum() == 0
    schedule = flow.schedule(10)
    assert schedule[0] == 1 and schedule[-1] == 0 and (schedule.diff() < 0).all()
    assert schedule.diff().sum() == -1
    raw_peak = 1 - math.exp(-0.5)
    assert abs(flow.weight(torch.tensor(0.5)).item() - raw_peak / flow.weight_normalizer) < 1e-6


def test_masked_mse_ignores_invalid_coordinates_and_empty_samples():
    prediction = torch.tensor([[[2.0, float("nan")]], [[float("nan"), float("nan")]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[[True, False]], [[False, False]]])
    loss = masked_sample_mse(prediction, target, valid)
    torch.testing.assert_close(loss, torch.tensor([4.0, 0.0]))
    loss.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[4.0, 0.0]], [[0.0, 0.0]]]))


def batch():
    return {
        "reference_tokens": torch.randn(2, 4, 128), "future_tokens": torch.randn(2, 4, 128),
        "text_tokens": torch.randn(2, 3, 32), "action": torch.randn(2, 16, 14),
        "proprio": torch.randn(2, 16, 14),
    }


def test_shared_model_trains_and_samples_joint_tokens():
    model = make_tiny_policy()
    sample = batch()
    sample["action_is_pad"] = torch.zeros(2, 16, dtype=torch.bool)
    sample["action_is_pad"][0, -2:] = True
    sample["action_dim_is_pad"] = torch.zeros(2, 14, dtype=torch.bool)
    sample["action_dim_is_pad"][:, -1] = True
    losses = model(sample, generator=torch.Generator().manual_seed(7))
    assert torch.isfinite(losses["loss"]) and losses["loss"] > 0
    torch.testing.assert_close(losses["loss"], 0.5 * losses["loss_video"] + losses["loss_action"])
    losses["loss"].backward()
    assert model.state_projection.weight.grad.abs().sum() > 0
    assert model.denoiser.token_input.weight.grad.abs().sum() > 0
    assert model.denoiser.token_output.weight.grad.abs().sum() > 0
    sampled = model.sample_actions(sample, steps=3, generator=torch.Generator().manual_seed(9))
    assert sampled["action"].shape == (2, 16, 14)
    assert sampled["future_tokens"].shape == (2, 4, 128)
    assert torch.isfinite(sampled["action"]).all()


def test_invalid_labels_are_sanitized_before_encoding_and_do_not_affect_loss():
    model = make_tiny_policy()
    sample = batch()
    sample["action_dim_is_pad"] = torch.zeros(2, 16, 14, dtype=torch.bool)
    sample["action_dim_is_pad"][..., -1] = True
    sample["action_is_pad"] = torch.zeros(2, 16, dtype=torch.bool)
    sample["action_is_pad"][:, -3:] = True
    damaged = dict(sample)
    damaged["action"] = sample["action"].clone()
    damaged["action"][..., -1] = float("nan")
    damaged["action"][:, -3:] = float("nan")
    original = model(sample, generator=torch.Generator().manual_seed(10))
    corrupt = model(damaged, generator=torch.Generator().manual_seed(10))
    for key in original:
        torch.testing.assert_close(original[key], corrupt[key])


def test_invalid_text_padding_is_inert_in_loss_and_gradients():
    model = make_tiny_policy()
    sample = batch()
    sample["text_valid"] = torch.tensor([[True, False, True], [False, True, True]])
    damaged = dict(sample, text_tokens=sample["text_tokens"].clone())
    damaged["text_tokens"][~sample["text_valid"]] = float("nan")
    baseline = model(sample, generator=torch.Generator().manual_seed(10))
    losses = model(damaged, generator=torch.Generator().manual_seed(10))
    for name in baseline:
        torch.testing.assert_close(baseline[name], losses[name], rtol=0, atol=0)
    losses["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_tiny_factory_honors_custom_token_width():
    model = make_tiny_policy(token_dim=64)
    sample = batch()
    sample["reference_tokens"] = torch.randn(2, 4, 64)
    sample["future_tokens"] = torch.randn(2, 4, 64)
    loss = model(sample)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.sample_actions(sample, steps=2)["future_tokens"].shape == (2, 4, 64)


def test_clean_prefix_does_not_depend_on_generated_tokens():
    model = make_tiny_policy()
    sample = batch()
    observed = []
    hook = model.denoiser.blocks[-1].register_forward_hook(lambda module, args, output: observed.append(output.detach()))
    noise = torch.randn(2, 4, 128)
    action_noise = torch.randn(2, 16, 128)
    model(sample, sigma=torch.tensor([0.6, 0.6]), future_noise=noise, action_noise=action_noise)
    model(sample, sigma=torch.tensor([0.6, 0.6]), future_noise=noise + 100, action_noise=action_noise - 100)
    hook.remove()
    # Three text + one state + four reference tokens form the closed prefix.
    torch.testing.assert_close(observed[0][:, :8], observed[1][:, :8], rtol=0, atol=0)
    assert not torch.allclose(observed[0][:, 8:], observed[1][:, 8:])


class RecordingOracle(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity
        self.seen = []

    def forward(self, reference, noisy, sigma, context, **kwargs):
        self.seen.append((noisy.detach().clone(), sigma.detach().clone()))
        return self.velocity


def test_shared_noise_level_and_exact_velocity_target():
    sample = batch()
    codec = RepeatedActionCodec(14)
    image_noise = torch.randn_like(sample["future_tokens"])
    action_noise = torch.randn(2, 16, 128)
    clean = torch.cat((sample["future_tokens"], codec.encode(sample["action"])), 1)
    noise = torch.cat((image_noise, action_noise), 1)
    oracle = RecordingOracle(noise - clean)
    model = PatchFlowPolicy(oracle, action_dim=14, text_dim=32)
    sigma = torch.tensor([0.3, 0.7])
    losses = model(sample, sigma=sigma, future_noise=image_noise, action_noise=action_noise)
    torch.testing.assert_close(losses["loss"], torch.tensor(0.0))
    torch.testing.assert_close(oracle.seen[0][0], torch.lerp(clean, noise, sigma[:, None, None]))
    torch.testing.assert_close(oracle.seen[0][1], sigma)


def test_descending_euler_recovers_an_exact_linear_flow():
    sample = batch()
    codec = RepeatedActionCodec(14)
    x = torch.randn_like(sample["future_tokens"])
    u = torch.randn(2, 16, 128)
    target = codec.encode(sample["action"])
    oracle = RecordingOracle(torch.cat((x - sample["future_tokens"], u - target), 1))
    policy = PatchFlowPolicy(oracle, action_dim=14, text_dim=32)
    prediction = policy.sample_actions(sample, steps=10, initial_future=x, initial_action=u)
    torch.testing.assert_close(prediction["action"], sample["action"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(prediction["future_tokens"], sample["future_tokens"], atol=1e-6, rtol=1e-6)
    assert len(oracle.seen) == 10


class MockImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 128, 1)

    def encode(self, images):
        return self.projection(torch.nn.functional.adaptive_avg_pool2d(images, (2, 2)))


class MockTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(3, 32))

    def forward(self, prompts):
        return self.embeddings.unsqueeze(0).expand(len(prompts), -1, -1)


def test_raw_video_wrapper_freezes_assets_and_uses_the_current_frame_at_inference():
    wrapper = FluxAssetPolicy(make_tiny_policy(), MockImageEncoder(), MockTextEncoder())
    wrapper.train()
    assert wrapper.policy.training and not wrapper.autoencoder.training and not wrapper.text_encoder.training
    assert not any(p.requires_grad for p in wrapper.autoencoder.parameters())
    sample = {"video": torch.randn(2, 3, 2, 16, 16), "prompt": ["move", "lift"],
              "action": torch.randn(2, 16, 14), "proprio": torch.randn(2, 14)}
    losses = wrapper(sample)
    losses["loss"].backward()
    assert wrapper.policy.state_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in wrapper.autoencoder.parameters())
    observation = dict(sample, video=sample["video"][:, :, 0])
    assert wrapper.sample_actions(observation, steps=2)["action"].shape == (2, 16, 14)


def test_cached_observations_are_cast_to_policy_and_retain_text_padding():
    wrapper = FluxAssetPolicy(make_tiny_policy(), MockImageEncoder(), MockTextEncoder()).double()
    sample = batch()
    sample["text_valid"] = torch.tensor([[True, False, True], [True, True, False]])
    sample["text_tokens"][~sample["text_valid"]] = float("nan")
    encoded = wrapper._encode(sample, target=True)
    assert encoded["reference_tokens"].dtype == torch.float64
    assert encoded["future_tokens"].dtype == torch.float64
    assert encoded["text_tokens"].dtype == torch.float64
    torch.testing.assert_close(encoded["text_valid"], sample["text_valid"])
    assert torch.isfinite(wrapper(sample)["loss"])
    assert wrapper.sample_actions(sample, steps=2)["action"].shape == (2, 16, 14)


@pytest.mark.parametrize("cache_format", ["t5", "qwen2_5_vl"])
def test_incompatible_text_cache_cannot_silently_fall_back_to_prompt_encoding(cache_format):
    wrapper = FluxAssetPolicy(make_tiny_policy(), MockImageEncoder(), MockTextEncoder())
    sample = dict(batch(), text_cache_format=[cache_format, cache_format])
    with pytest.raises(ValueError, match="native Qwen3"):
        wrapper(sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise host observation masks")
def test_raw_host_observation_masks_work_with_cuda_encoders():
    wrapper = FluxAssetPolicy(make_tiny_policy(), MockImageEncoder(), MockTextEncoder()).cuda()
    sample = {"video": torch.randn(2, 3, 2, 16, 16), "prompt": ["move", "lift"],
              "action": torch.randn(2, 16, 14), "proprio": torch.randn(2, 14),
              "action_dim_is_pad": torch.zeros(2, 14, dtype=torch.bool),
              "action_is_pad": torch.zeros(2, 16, dtype=torch.bool)}
    losses = wrapper(sample)
    assert losses["loss"].is_cuda and torch.isfinite(losses["loss"])
