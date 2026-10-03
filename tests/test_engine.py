import json

import pytest
import torch
from torch import nn

from patchwam.engine import OptimizationRun, RunSettings
from patchwam.testing import TensorExamples


class RegressionObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 128)

    def forward(self, batch):
        target = batch["future_tokens"]
        noisy = batch["reference_tokens"] + torch.randn_like(target) * 0.1
        return {"loss": (self.layer(noisy) - target).square().mean()}


def settings(directory, **changes):
    values = {
        "output_dir": str(directory), "epochs": 2, "max_updates": 4, "batch_size": 2,
        "accumulation": 2, "precision": "no", "cpu": True, "workers": 0,
        "checkpoint_every": 2, "log_every": 1,
    }
    return RunSettings(**(values | changes))


def make_run(directory, **changes):
    torch.manual_seed(17)
    return OptimizationRun(RegressionObjective(), TensorExamples(length=16), settings(directory, **changes))


def test_interrupted_training_restores_model_optimizer_scheduler_and_rng(tmp_path):
    baseline = make_run(tmp_path / "baseline")
    baseline.train()
    expected = {key: value.clone() for key, value in baseline.model.state_dict().items()}

    interrupted = make_run(tmp_path / "interrupted")
    checkpoint = interrupted.checkpoint

    class Interrupted(Exception):
        pass

    def stop_after_save(epoch, next_batch):
        checkpoint(epoch, next_batch)
        raise Interrupted

    interrupted.checkpoint = stop_after_save
    with pytest.raises(Interrupted):
        interrupted.train()

    resumed = make_run(tmp_path / "resumed")
    resumed.restore(tmp_path / "interrupted" / "step_0000002")
    resumed.train()
    assert resumed.updates == baseline.updates == 4
    assert resumed.scheduler.state_dict() == baseline.scheduler.state_dict()
    for key, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_partial_accumulation_at_epoch_end_is_saved(tmp_path):
    torch.manual_seed(17)
    run = OptimizationRun(
        RegressionObjective(), TensorExamples(length=10),
        settings(tmp_path, epochs=1, max_updates=None),
    )
    # Five microbatches leave a partial final group.
    final = run.train()
    assert run.updates == 3
    metadata = json.loads((tmp_path / "step_0000003" / "cursor.json").read_text())
    assert metadata["epoch"] == 1 and metadata["next_batch"] == 0
    assert final.endswith("step_0000003")


class ConstantGradientObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return {"loss": self.weight, "sample_mean": batch["future_tokens"].mean()}


def test_partial_accumulation_uses_actual_group_size_and_logs_group_mean(tmp_path):
    run = OptimizationRun(
        ConstantGradientObjective(), TensorExamples(length=10),
        settings(tmp_path, epochs=1, max_updates=None, gradient_clip=0),
    )
    gradients, microbatch_metrics = [], []
    step = run.optimizer.optimizer.step
    forward = run.model.forward

    def record_step(*args, **kwargs):
        gradients.append(run.model.weight.grad.detach().item())
        return step(*args, **kwargs)

    def record_forward(batch):
        result = forward(batch)
        microbatch_metrics.append(result["sample_mean"].item())
        return result

    run.optimizer.optimizer.step = record_step
    run.model.forward = record_forward
    run.train()
    assert gradients == pytest.approx([1.0, 1.0, 1.0])
    logs = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    expected = [sum(microbatch_metrics[i:i + 2]) / len(microbatch_metrics[i:i + 2])
                for i in range(0, 5, 2)]
    assert [row["sample_mean"] for row in logs] == pytest.approx(expected)


@pytest.mark.parametrize("gradient_clip", [0, 1])
def test_finite_loss_with_nonfinite_gradient_does_not_update_parameters(tmp_path, gradient_clip):
    class SingularObjective(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, _batch):
            return {"loss": self.weight.sqrt()}

    run = OptimizationRun(SingularObjective(), TensorExamples(length=16),
                          settings(tmp_path, gradient_clip=gradient_clip))
    with pytest.raises(FloatingPointError, match="gradient"):
        run.train()
    assert run.updates == 0
    assert run.model.weight.item() == 0.0


def test_nonfinite_loss_does_not_update_parameters(tmp_path):
    run = make_run(tmp_path)
    original = {key: value.clone() for key, value in run.model.state_dict().items()}

    def invalid(_batch):
        return {"loss": run.model.layer.weight.sum() * float("nan")}

    run.model.forward = invalid
    with pytest.raises(FloatingPointError):
        run.train()
    assert run.updates == 0
    for key, value in run.model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_resume_rejects_changed_batch_size(tmp_path):
    original = make_run(tmp_path / "original")
    original.train()
    changed = make_run(tmp_path / "changed", batch_size=4)
    with pytest.raises(ValueError, match="unchanged batch"):
        changed.restore(tmp_path / "original" / "step_0000004")


def test_checkpoint_io_failure_does_not_publish_a_complete_checkpoint(tmp_path, monkeypatch):
    run = make_run(tmp_path)

    def fail_save(*_args, **_kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr(run.accelerator, "save_state", fail_save)
    with pytest.raises(OSError, match="disk write"):
        run.checkpoint(0, 0)
    assert not (tmp_path / "step_0000000").exists()
    assert run.last_checkpoint is None


def test_existing_settings_are_preserved_when_constructing_an_incompatible_run(tmp_path):
    make_run(tmp_path)
    before = (tmp_path / "settings.json").read_text()
    make_run(tmp_path, batch_size=4)
    assert (tmp_path / "settings.json").read_text() == before


def test_exhausted_training_saves_cursor_after_skipped_update(tmp_path, monkeypatch):
    run = OptimizationRun(
        ConstantGradientObjective(), TensorExamples(length=4),
        settings(tmp_path, epochs=1, max_updates=2, accumulation=1, checkpoint_every=1),
    )
    step, calls = run.optimizer.step, 0

    def skip_last(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            run.optimizer._is_overflow = True
        else:
            step(*args, **kwargs)

    monkeypatch.setattr(run.optimizer, "step", skip_last)
    final = run.train()
    metadata = json.loads((tmp_path / "step_0000001_exhausted" / "cursor.json").read_text())
    assert metadata["epoch"] == 1 and metadata["next_batch"] == 0 and metadata["exhausted"]
    assert run.updates == 1 < run.total_updates
    assert final.endswith("step_0000001_exhausted")


@pytest.mark.parametrize("damage", [{"updates": -1}, {"epoch": 100}, {"next_batch": 1}])
def test_resume_rejects_invalid_cursor_before_loading_state(tmp_path, monkeypatch, damage):
    original = make_run(tmp_path / "original")
    path = original.checkpoint(0, 0)
    metadata = json.loads((path / "cursor.json").read_text())
    (path / "cursor.json").write_text(json.dumps(metadata | damage))
    resumed = make_run(tmp_path / "resumed")
    loaded = []
    monkeypatch.setattr(resumed.accelerator, "load_state", lambda *_: loaded.append(True))
    with pytest.raises(ValueError, match="cursor|boundary"):
        resumed.restore(path)
    assert loaded == []


def test_scheduler_matches_linear_warmup_cosine_floor_contract(tmp_path):
    run = make_run(tmp_path, epochs=25, max_updates=100, warmup_updates=5)
    reference_optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=1e-4)
    reference = torch.optim.lr_scheduler.SequentialLR(
        reference_optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(reference_optimizer, start_factor=1/5, total_iters=5),
            torch.optim.lr_scheduler.CosineAnnealingLR(reference_optimizer, T_max=95, eta_min=1e-6),
        ],
        milestones=[5],
    )
    for _ in range(100):
        assert run.scheduler.get_last_lr()[0] == pytest.approx(reference.get_last_lr()[0])
        run.optimizer.step()
        run.scheduler.step()
        reference_optimizer.step()
        reference.step()
