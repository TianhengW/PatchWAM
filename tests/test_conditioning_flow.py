"""Conditioning, structured flow, and detached-teacher contracts."""

import copy

import pytest
import torch

from patchwam.engine import OptimizationRun, RunSettings
from patchwam.models import make_tiny_policy
from patchwam.models.flow import ShiftedFlow, masked_sample_mse, weighted_masked_sample_mse
from patchwam.models.geometry import joint_visibility, sequence_coordinates
from patchwam.testing import TensorExamples


def batch():
    generator = torch.Generator().manual_seed(77)
    return {key: torch.stack([TensorExamples(horizon=2)[i][key] for i in range(2)])
            for key in TensorExamples(horizon=2)[0]} | {
        "future_noise": torch.randn(2, 4, 128, generator=generator),
        "action_noise": torch.randn(2, 2, 128, generator=generator),
    }


def policy(**options):
    torch.manual_seed(11)
    return make_tiny_policy(width=32, heads=4, depth=4, **options)


def noise_arguments(data):
    return {"sigma": torch.tensor([0.3, 0.7]), "future_noise": data["future_noise"],
            "action_noise": data["action_noise"]}


def attach_teacher(model):
    teacher = copy.deepcopy(model)
    model.attach_self_flow_teacher(teacher)
    return teacher


def test_cfg_identity_zero_and_separate_action_scale():
    model, data = policy(), batch()
    model.eval()
    future, action = data["future_noise"], data["action_noise"]
    kwargs = {"steps": 1, "horizon": 2, "initial_future": future, "initial_action": action}
    baseline = model.sample_actions(data, **kwargs)
    conditional = model.sample_actions(data, guidance_scale=1, **kwargs)
    for key in baseline:
        torch.testing.assert_close(baseline[key], conditional[key], rtol=0, atol=0)
    vx, vu = model._velocity(data, future, action, torch.ones(2), condition_drop_mask=torch.ones(2).bool())
    unconditioned = model.sample_actions(data, guidance_scale=0, **kwargs)
    torch.testing.assert_close(unconditioned["future_tokens"], future - vx)
    torch.testing.assert_close(unconditioned["action"], model.codec.decode(action - vu))
    mixed = model.sample_actions(data, guidance_scale=0, action_guidance_scale=1, **kwargs)
    torch.testing.assert_close(mixed["future_tokens"], unconditioned["future_tokens"])
    torch.testing.assert_close(mixed["action"], baseline["action"])


def test_language_dropout_removes_language_without_removing_state_or_image():
    model, data = policy(condition_dropout=1), batch()
    data["text_tokens"].requires_grad_()
    data["proprio"].requires_grad_()
    data["reference_tokens"].requires_grad_()
    result = model(data, **noise_arguments(data))
    result["loss"].backward()
    assert data["text_tokens"].grad.abs().sum() == 0
    assert data["proprio"].grad.abs().sum() > 0
    assert data["reference_tokens"].grad.abs().sum() > 0
    changed = dict(data, text_tokens=torch.full_like(data["text_tokens"], float("nan")))
    torch.testing.assert_close(model(changed, **noise_arguments(data))["loss"], result["loss"], rtol=0, atol=0)


def test_invalid_history_values_and_positions_do_not_leak():
    model, data = policy(), batch()
    data["reference_valid"] = torch.tensor([[True, True, False, False]]).expand(2, -1)
    data["reference_ids"] = sequence_coordinates(2, 4, group=10)
    original = model(data, **noise_arguments(data))["loss"]
    changed = dict(data, reference_tokens=data["reference_tokens"].clone(), reference_ids=data["reference_ids"].clone())
    changed["reference_tokens"][:, 2:] = float("nan")
    changed["reference_ids"][:, 2:] = float("nan")
    torch.testing.assert_close(model(changed, **noise_arguments(data))["loss"], original, rtol=0, atol=0)
    visibility = joint_visibility(3, 4, 4, 2, reference_valid=data["reference_valid"])
    assert not visibility[..., 5:7].any()
    assert not visibility[..., :7, 7:].any()


def test_dual_time_perturbation_and_cleaner_teacher():
    flow = ShiftedFlow()
    first, token_time, teacher_time = flow.dual_timesteps(
        2, 2, 2, first=torch.tensor([0.2, 0.8]), second=torch.tensor([0.7, 0.3]),
        mask=torch.tensor([[True, False, False, True], [False, True, True, False]]),
    )
    torch.testing.assert_close(first, torch.tensor([0.2, 0.8]))
    torch.testing.assert_close(token_time, torch.tensor([[0.7, 0.2, 0.2, 0.7], [0.8, 0.3, 0.3, 0.8]]))
    torch.testing.assert_close(teacher_time, torch.tensor([0.2, 0.3]))
    clean = torch.ones(2, 4, 3)
    noisy, target = flow.perturb(clean, torch.zeros_like(clean), token_time)
    torch.testing.assert_close(noisy, (1 - token_time[..., None]).expand_as(clean))
    torch.testing.assert_close(target, -clean)


def test_structured_plane_and_both_clean_edges():
    flow = ShiftedFlow()
    _, levels, teacher = flow.dual_timesteps(
        4, 2, 2, structured=True, first=torch.full((4,), 0.3), second=torch.full((4,), 0.8),
        mask=torch.zeros(4, 4).bool(), branches=torch.arange(4),
    )
    torch.testing.assert_close(levels, torch.tensor([[0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.8, 0.8],
                                                    [0, 0, 0.8, 0.8], [0.3, 0.3, 0, 0]]))
    torch.testing.assert_close(teacher, torch.tensor([0.3, 0.3, 0, 0]))


def test_weighting_matches_historical_grid_and_shared_time():
    flow = ShiftedFlow(normalization="endpoint_1000")
    grid = torch.arange(1, 1001, dtype=torch.float64) / 1000
    warped = 5 * grid / (1 + 4 * grid)
    raw = torch.exp(-2 * (warped - 0.5).square())
    torch.testing.assert_close(torch.tensor(flow.weight_normalizer), (raw - raw.min()).mean().float())
    data = batch()
    weights = flow.weight(torch.tensor([0.3, 0.7]))
    expected = weights * masked_sample_mse(data["future_noise"], data["future_tokens"])
    result = weighted_masked_sample_mse(data["future_noise"], data["future_tokens"], weights[:, None].expand(2, 4))
    torch.testing.assert_close(result, expected)


def test_teacher_is_detached_and_student_projection_learns():
    model, data = policy(self_flow_variant=1), batch()
    teacher = attach_teacher(model)
    model(data, second_sigma=torch.tensor([0.7, 0.2]),
          timestep_mask=torch.tensor([[True, False, True, False, True, False]]).expand(2, -1),
          **noise_arguments(data))["loss"].backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.representation_projection.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.denoiser.parameters())
    assert not any(key.startswith("_self_flow_teacher") for key in model.state_dict())


def test_self_flow_requires_teacher_and_feature_layers():
    with pytest.raises(ValueError, match="student layer"):
        make_tiny_policy(depth=1, self_flow_variant=1)
    with pytest.raises(RuntimeError, match="EMA teacher"):
        policy(self_flow_variant=1)(batch())


@pytest.mark.parametrize("branch,empty_loss", [(2, "loss_video"), (3, "loss_action")])
def test_clean_edge_is_conditioning_and_has_no_flow_loss(branch, empty_loss):
    model, data = policy(self_flow_variant=2), batch()
    attach_teacher(model)
    result = model(data, structured_branches=torch.full((2,), branch),
                   second_sigma=torch.tensor([0.7, 0.2]), **noise_arguments(data))
    assert result[empty_loss] == 0
    assert torch.isfinite(result["loss"])


def test_withheld_warmup_labels_have_no_input_or_gradient_path():
    model, data = policy(self_flow_variant=3), batch()
    attach_teacher(model)
    data["action"].requires_grad_()
    kwargs = noise_arguments(data) | {"pseudo_label_mask": torch.ones(2).bool(),
                                     "second_sigma": torch.tensor([0.7, 0.2]),
                                     "structured_branches": torch.zeros(2, dtype=torch.long)}
    # Fix the mask to isolate label effects from RNG.
    kwargs["timestep_mask"] = torch.ones(2, 6).bool()
    result = model(data, **kwargs)
    changed = dict(data, action=data["action"] + 100)
    torch.testing.assert_close(model(changed, **kwargs)["loss"], result["loss"], rtol=0, atol=0)
    result["loss"].backward()
    assert data["action"].grad.abs().sum() == 0
    assert result["loss_action"] == 0


def test_pseudo_labels_use_real_future_and_replace_every_label_path():
    model, data = policy(self_flow_variant=3, self_flow_label_warmup=0), batch()
    detached_teacher = copy.deepcopy(model)
    generated = torch.full_like(data["action"], 2, requires_grad=True)

    def teacher(inputs, **kwargs):
        if kwargs.get("teacher_sampling"):
            assert "action" not in inputs
            torch.testing.assert_close(inputs["future_tokens"], data["future_tokens"])
            return {"action": generated}
        return detached_teacher(inputs, **kwargs)

    model.attach_self_flow_teacher(teacher)
    data["action"].requires_grad_()
    kwargs = noise_arguments(data) | {"pseudo_label_mask": torch.ones(2).bool(),
                                     "second_sigma": torch.tensor([0.7, 0.2]),
                                     "structured_branches": torch.ones(2, dtype=torch.long)}
    result = model(data, **kwargs)
    changed = dict(data, action=data["action"] + 100)
    torch.testing.assert_close(model(changed, **kwargs)["loss"], result["loss"], rtol=0, atol=0)
    assert model(changed, **kwargs)["pseudo_mse"] > result["pseudo_mse"]
    result["loss"].backward()
    assert data["action"].grad.abs().sum() == 0
    assert generated.grad is None


def test_conditional_sampling_keeps_the_observed_modality_fixed():
    model, data = policy(self_flow_variant=2), batch()
    inverse = model.sample_actions_from_future(data, horizon=2, steps=2)
    assert inverse["future_tokens"] is data["future_tokens"]
    forward = model.sample_future_from_actions(data, future_length=4, steps=2)
    assert forward["action"] is data["action"]
    assert inverse["action"].shape == (2, 2, 14)
    assert forward["future_tokens"].shape == (2, 4, 128)


def test_history_sampling_uses_current_future_raster_length():
    model, data = policy(), batch()
    data["future_ids"] = sequence_coordinates(2, 2)
    assert model.sample_actions(data, steps=1)["future_tokens"].shape[1] == 2


def test_official_flux_token_modulation_and_checkpointed_gradients():
    api = pytest.importorskip("flux2.model")
    from patchwam.models import OfficialFluxDenoiser, PatchFlowPolicy

    core = api.Flux2(api.Klein4BParams(context_in_dim=32, hidden_size=64, num_heads=4,
                                     depth=2, depth_single_blocks=2, axes_dim=[4, 4, 4, 4], mlp_ratio=2))
    direct = OfficialFluxDenoiser(core)
    recomputed = OfficialFluxDenoiser(copy.deepcopy(core), gradient_checkpointing=True)
    data = batch()
    noisy = torch.cat((data["future_noise"], data["action_noise"]), 1)
    kwargs = {"reference_ids": sequence_coordinates(2, 4, group=10),
              "noisy_ids": sequence_coordinates(2, 6, group=20),
              "context_ids": sequence_coordinates(2, 3), "visibility": joint_visibility(3, 4, 4, 2)}
    times = torch.tensor([0.3, 0.7])
    shared = direct(data["reference_tokens"], noisy, times, data["text_tokens"], **kwargs)
    uniform = direct(data["reference_tokens"], noisy, times, data["text_tokens"],
                     token_sigma=times[:, None].expand(2, 6), **kwargs)
    torch.testing.assert_close(shared, uniform, atol=3e-6, rtol=3e-6)
    token_times = torch.tensor([[0.1, 0.8, 0.1, 0.8, 0, 0.8]]).expand(2, -1)
    predictions = []
    for denoiser in (direct, recomputed):
        output, features = denoiser(data["reference_tokens"], noisy, times, data["text_tokens"],
                                    token_sigma=token_times, representation_layer=2, **kwargs)
        assert features.shape == (2, 6, 64)
        (output.square().mean() + features.square().mean()).backward()
        predictions.append(output)
    torch.testing.assert_close(*predictions, atol=1e-6, rtol=1e-5)
    for left, right in zip(direct.parameters(), recomputed.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)

    # Check policy and teacher across both conditioning modes.
    model = PatchFlowPolicy(direct, action_dim=14, text_dim=32, proprio_dim=14, self_flow_variant=2)
    attach_teacher(model)
    result = model(data, structured_branches=torch.tensor([2, 3]),
                   second_sigma=torch.tensor([0.7, 0.2]), **noise_arguments(data))
    result["loss"].backward()
    assert torch.isfinite(result["loss_representation"])


def test_self_flow_teacher_and_label_warmup_continue_with_engine(tmp_path):
    def run(directory):
        return OptimizationRun(
            policy(self_flow_variant=3, self_flow_label_warmup=1,
                   self_flow_pseudo_label_fraction=1, self_flow_sampling_steps=1),
            TensorExamples(length=8, horizon=2),
            RunSettings(output_dir=str(directory), epochs=1, max_updates=3, batch_size=2,
                        accumulation=1, cpu=True, precision="no", workers=0, ema_decay=0.9,
                        checkpoint_every=1, log_every=1),
        )

    full = run(tmp_path / "full")
    full.train()
    resumed = run(tmp_path / "resumed")
    resumed.restore(tmp_path / "full" / "step_0000001")
    assert resumed.accelerator.unwrap_model(resumed.model)._optimizer_updates == 1
    resumed.train()
    assert resumed.updates == resumed.ema.updates == 3
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
    for name, value in full.ema.shadows.items():
        torch.testing.assert_close(resumed.ema.shadows[name], value, rtol=0, atol=0)
