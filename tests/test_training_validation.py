import pytest
import torch
from safetensors.torch import save_file

from patchwam.validation import compare_weights, validate_training


def test_checkpoint_comparison_checks_values_shapes_and_finiteness(tmp_path):
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    save_file({"weight": torch.ones(3)}, first)
    save_file({"weight": torch.ones(3)}, second)
    assert compare_weights(str(first), str(second))["max_abs_difference"] == 0
    save_file({"weight": torch.full((3,), 1.01)}, second)
    with pytest.raises(ValueError, match="differ"):
        compare_weights(str(first), str(second))
    compare_weights(str(first), str(second), atol=0.02)
    save_file({"weight": torch.full((3,), float("nan"))}, second)
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_weights(str(first), str(second))


def test_validation_trains_and_resumes_in_separate_processes(tmp_path):
    report = validate_training("configs/smoke.yaml", output=str(tmp_path / "validation"), allow_cpu=True)
    assert report["status"] == "passed"
    assert report["baseline"]["updates"] == report["resumed"]["updates"] == 2
    assert report["comparisons"]["model.safetensors"]["max_abs_difference"] == 0
    with pytest.raises(FileExistsError):
        validate_training("configs/smoke.yaml", output=str(tmp_path / "validation"), allow_cpu=True)


def test_full_model_validation_rejects_cpu_recipe(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_training("configs/smoke.yaml", output=str(tmp_path / "validation"))
