import copy

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from patchwam.averaging import PolicyWeightAverage
from patchwam.checkpoints import load_policy_weights, save_policy_average
from patchwam.engine import OptimizationRun, RunSettings
from patchwam.models import FluxAssetPolicy, make_tiny_policy
from patchwam.testing import TensorExamples


def test_fixed_decay_and_copy_warmup_use_fp32_shadows():
    model = nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)
    model.weight.data.fill_(1)
    average = PolicyWeightAverage(model, decay=0.5, warmup_updates=2)
    assert average.shadows["weight"].dtype == torch.float32
    for value, expected in [(3, 3), (5, 5), (9, 7), (11, 9)]:
        model.weight.data.fill_(value)
        average.update(model)
        assert average.shadows["weight"].item() == expected
    assert average.updates == 4


@pytest.mark.parametrize("decay", [-0.1, 1, float("nan"), float("inf")])
def test_invalid_decay_is_rejected(decay):
    with pytest.raises(ValueError, match="decay"):
        PolicyWeightAverage(nn.Linear(1, 1), decay=decay)


def test_stateless_teacher_preserves_student_autograd_buffers_and_modes():
    class BufferedPrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)
            self.dropout = nn.Dropout(0.5)
            self.register_buffer("calls", torch.tensor(0))

        def forward(self, value):
            self.calls.add_(1)
            return self.linear(self.dropout(value))

    model = BufferedPrediction()
    model.linear.weight.data.fill_(1)
    average = PolicyWeightAverage(model, decay=0.5)
    model.linear.weight.data.fill_(3)
    model.train()
    model.dropout.eval()
    value = torch.tensor([[2.0]], requires_grad=True)
    student = model(value)
    version = model.linear.weight._version
    teacher = average.call(model, value)
    assert teacher.item() == 2 and not teacher.requires_grad
    assert model.linear.weight._version == version and model.linear.weight.item() == 3
    assert model.calls.item() == 1
    assert model.training and not model.dropout.training
    student.backward()
    assert value.grad.item() == 3 and model.linear.weight.grad.item() == 2


def test_evaluation_context_restores_weights_after_exception():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    average = PolicyWeightAverage(model)
    model.weight.data.fill_(4)
    with pytest.raises(RuntimeError, match="interrupted evaluation"), average.apply_to(model):
        assert not model.training and model.weight.item() == 1
        raise RuntimeError("interrupted evaluation")
    assert model.training and model.weight.item() == 4


def test_teacher_can_average_denoiser_while_sharing_live_conditioning():
    class ConditionedPrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_projection = nn.Linear(1, 1, bias=False)
            self.denoiser = nn.Linear(1, 1, bias=False)

        def forward(self, value):
            return self.denoiser(self.state_projection(value))

    model = ConditionedPrediction()
    model.state_projection.weight.data.fill_(1)
    model.denoiser.weight.data.fill_(1)
    average = PolicyWeightAverage(model)
    model.state_projection.weight.data.fill_(4)
    model.denoiser.weight.data.fill_(3)
    value = torch.ones(1, 1)
    assert average.call(model, value).item() == 1
    assert average.call(model, value, parameter_prefixes=("denoiser.",)).item() == 4
    assert model(value).item() == 12
    with pytest.raises(ValueError, match="match no"):
        average.call(model, value, parameter_prefixes=("missing.",))


def test_public_update_rejects_nonfinite_parameters_before_mutating_shadows():
    model = nn.Linear(1, 1)
    average = PolicyWeightAverage(model)
    before = average.state_dict()
    model.bias.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        average.update(model)
    assert average.updates == 0
    for name, value in before["weights"].items():
        torch.testing.assert_close(average.shadows[name], value, rtol=0, atol=0)


class AdapterAssets(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 1).requires_grad_(False)
        self.residual_in = nn.Linear(1, 1, bias=False)


class FrozenAssetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(1, 1, bias=False)
        self.autoencoder = nn.Linear(1, 1).requires_grad_(False)
        self.text_encoder = AdapterAssets()


def test_native_export_averages_policy_and_adapters_and_excludes_frozen_assets(tmp_path):
    model = FrozenAssetWrapper()
    model.policy.weight.data.fill_(1)
    model.text_encoder.residual_in.weight.data.fill_(2)
    average = PolicyWeightAverage(model, decay=0.5)
    assert set(average.shadows) == {"policy.weight", "text_encoder.residual_in.weight"}
    model.policy.weight.data.fill_(3)
    model.text_encoder.residual_in.weight.data.fill_(6)
    average.update(model)
    path = tmp_path / "ema_policy.safetensors"
    save_policy_average(average, model, path)
    weights = load_file(str(path))
    assert set(weights) == {"policy.weight", "text_encoder.residual_in.weight"}
    assert weights["policy.weight"].item() == 2
    assert weights["text_encoder.residual_in.weight"].item() == 4
    with safe_open(path, framework="pt") as saved:
        assert saved.metadata()["weight_variant"] == "ema"
    destination = FrozenAssetWrapper()
    frozen_before = destination.autoencoder.weight.detach().clone()
    load_policy_weights(destination, path)
    assert destination.policy.weight.item() == 2
    assert destination.text_encoder.residual_in.weight.item() == 4
    torch.testing.assert_close(destination.autoencoder.weight, frozen_before, rtol=0, atol=0)


def test_export_retains_complete_policy_state_and_tied_weight_aliases(tmp_path):
    class SharedPrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(1, 1, bias=False)
            self.second = self.first
            self.register_buffer("constant", torch.tensor(7.0))

    model = SharedPrediction()
    model.first.weight.data.fill_(1)
    average = PolicyWeightAverage(model, decay=0.5)
    model.first.weight.data.fill_(3)
    average.update(model)
    path = tmp_path / "ema_policy.safetensors"
    save_policy_average(average, model, path)
    weights = load_file(str(path))
    assert weights["constant"].item() == 7
    assert weights["first.weight"].item() == weights["second.weight"].item() == 2
    load_policy_weights(SharedPrediction(), path)


def test_corrupt_average_state_is_rejected_atomically():
    model = nn.Linear(1, 1)
    average = PolicyWeightAverage(model)
    state = copy.deepcopy(average.state_dict())
    state["weights"]["bias"].fill_(float("nan"))
    before = average.state_dict()
    with pytest.raises(ValueError, match="EMA weight"):
        average.load_state_dict(state)
    assert average.updates == 0
    for name, tensor in before["weights"].items():
        torch.testing.assert_close(average.shadows[name], tensor, rtol=0, atol=0)


class NoisyObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 128)
        self.optimizer_updates = -1

    def set_optimizer_updates(self, count):
        self.optimizer_updates = count

    def forward(self, batch):
        value = batch["reference_tokens"] + torch.randn_like(batch["reference_tokens"]) * 0.1
        return {"loss": (self.layer(value) - batch["future_tokens"]).square().mean()}


def run_settings(directory, **changes):
    return RunSettings(
        **(
            {
                "output_dir": str(directory),
                "epochs": 2,
                "max_updates": 4,
                "batch_size": 2,
                "accumulation": 2,
                "precision": "no",
                "cpu": True,
                "workers": 0,
                "checkpoint_every": 2,
                "log_every": 1,
                "ema_decay": 0.5,
            }
            | changes
        )
    )


def make_run(directory, **changes):
    torch.manual_seed(17)
    return OptimizationRun(
        NoisyObjective(), TensorExamples(length=16), run_settings(directory, **changes)
    )


def test_checkpoint_continuation_restores_average_and_native_export_exactly(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    baseline.train()
    interrupted = make_run(tmp_path / "interrupted")
    checkpoint = interrupted.checkpoint

    class Interrupted(Exception):
        pass

    def stop_after_save(epoch, next_batch, **kwargs):
        checkpoint(epoch, next_batch, **kwargs)
        raise Interrupted

    interrupted.checkpoint = stop_after_save
    with pytest.raises(Interrupted):
        interrupted.train()
    assert interrupted.ema.updates == 2
    resumed = make_run(tmp_path / "resumed")
    resumed.restore(tmp_path / "interrupted" / "step_0000002")
    assert resumed.ema.updates == 2
    assert resumed.model.optimizer_updates == 2
    resumed.train()
    assert resumed.updates == resumed.ema.updates == baseline.ema.updates == 4
    assert resumed.model.optimizer_updates == 4
    for name, value in baseline.ema.shadows.items():
        torch.testing.assert_close(resumed.ema.shadows[name], value, rtol=0, atol=0)
    expected = load_file(str(tmp_path / "baseline" / "step_0000004" / "ema_policy.safetensors"))
    actual = load_file(str(tmp_path / "resumed" / "step_0000004" / "ema_policy.safetensors"))
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)


def test_failed_loss_does_not_advance_average(tmp_path):
    run = make_run(tmp_path)
    before = run.ema.state_dict()
    run.model.forward = lambda _: {"loss": run.model.layer.weight.sum() * float("nan")}
    with pytest.raises(FloatingPointError):
        run.train()
    assert run.updates == run.ema.updates == 0
    for name, value in before["weights"].items():
        torch.testing.assert_close(run.ema.shadows[name], value, rtol=0, atol=0)


def test_skipped_optimizer_steps_do_not_advance_average(tmp_path):
    run = make_run(tmp_path, epochs=1, max_updates=None, accumulation=1)
    before = run.ema.state_dict()
    run.optimizer.step = lambda: setattr(run.optimizer, "_is_overflow", True)
    checkpoint = run.train()
    assert run.updates == run.ema.updates == 0
    assert checkpoint.endswith("step_0000000_exhausted")
    for name, value in before["weights"].items():
        torch.testing.assert_close(run.ema.shadows[name], value, rtol=0, atol=0)


def test_default_training_leaves_ema_disabled(tmp_path):
    run = make_run(tmp_path, ema_decay=None)
    run.train()
    assert run.ema is None
    assert not (tmp_path / "step_0000004" / "ema_policy.safetensors").exists()


class SmallImageAsset(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 128, 1)

    def encode(self, images):
        return torch.nn.functional.adaptive_avg_pool2d(self.projection(images), (2, 2))


class TrainableVisualConditioning(nn.Module):
    requires_images = True
    trainable_adapters = True
    auxiliary_weight = 0.1

    def __init__(self):
        super().__init__()
        self.residual = nn.Linear(3, 16)
        self.calls = 0

    def forward(self, prompts, *, current, history, history_valid, subtasks):
        self.calls += 1
        features = self.residual(current.mean((-2, -1)))[:, None].expand(-1, 3, -1)
        return {
            "text_tokens": features,
            "text_valid": torch.ones(features.shape[:2], dtype=torch.bool),
            "auxiliary_loss": features.square().mean(),
        }

    def checkpoint_contract(self):
        return {"type": "trainable_visual_fixture", "auxiliary_weight": self.auxiliary_weight}


class VisualHistoryExamples(TensorExamples):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        for name in ("reference_tokens", "future_tokens", "text_tokens"):
            sample.pop(name)
        generator = torch.Generator().manual_seed(2000 + index)
        sample.update(
            {
                "video": torch.randn(3, 2, 4, 4, generator=generator).tanh(),
                "history_video": torch.randn(1, 3, 4, 4, generator=generator).tanh(),
                "history_valid": torch.ones(1, dtype=torch.bool),
                "vl_current": torch.randn(3, 4, 4, generator=generator).tanh(),
                "vl_history": torch.randn(1, 3, 4, 4, generator=generator).tanh(),
                "vl_history_valid": torch.ones(1, dtype=torch.bool),
                "prompt": "move the block",
                "subtask": "reach the block",
            }
        )
        return sample


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_combined_visual_history_and_self_flow_teacher_resume_without_reencoding(tmp_path, variant):
    def create(directory, **policy_changes):
        torch.manual_seed(71)
        model = FluxAssetPolicy(
            make_tiny_policy(
                text_dim=16,
                width=32,
                depth=2,
                self_flow_variant=variant,
                self_flow_label_warmup=1,
                self_flow_pseudo_label_fraction=1,
                self_flow_sampling_steps=2,
                condition_dropout=0.2,
                **policy_changes,
            ),
            SmallImageAsset(),
            TrainableVisualConditioning(),
            history_pool_size=2,
        )
        return OptimizationRun(
            model,
            VisualHistoryExamples(length=8, text_dim=16),
            run_settings(directory, epochs=1, accumulation=1),
        )

    baseline = create(tmp_path / "baseline")
    baseline.train()
    assert baseline.model.text_encoder.calls == 4
    assert baseline.model.policy._optimizer_updates == baseline.ema.updates == 4
    interrupted = create(tmp_path / "interrupted")
    checkpoint = interrupted.checkpoint

    class Interrupted(Exception):
        pass

    def stop_after_save(epoch, next_batch, **kwargs):
        checkpoint(epoch, next_batch, **kwargs)
        raise Interrupted

    interrupted.checkpoint = stop_after_save
    with pytest.raises(Interrupted):
        interrupted.train()
    resumed = create(tmp_path / "resumed")
    resumed.restore(tmp_path / "interrupted" / "step_0000002")
    assert resumed.model.policy._optimizer_updates == 2
    resumed.train()
    assert resumed.model.text_encoder.calls == 2
    for name, value in baseline.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
    for name, value in baseline.ema.shadows.items():
        torch.testing.assert_close(resumed.ema.shadows[name], value, rtol=0, atol=0)
    changed = create(tmp_path / "changed", self_flow_representation_weight=0.4)
    with pytest.raises(ValueError, match="unchanged"):
        changed.restore(tmp_path / "interrupted" / "step_0000002")
