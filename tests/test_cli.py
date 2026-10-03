import hashlib
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from patchwam import cli
from patchwam.configuration import runtime_model_specification


def test_eager_asset_loading_uses_rank_device_without_mutating_saved_config():
    specification = OmegaConf.create({"_target_": "example.policy", "device": "cuda"})
    bound = runtime_model_specification(specification, torch.device("cuda:3"))
    assert bound.device == "cuda:3"
    assert specification.device == "cuda"


@pytest.mark.parametrize("requested, runtime", [("cuda:0", "cuda:1"), ("cuda", "cpu")])
def test_model_device_conflicts_are_rejected_before_loading_assets(requested, runtime):
    specification = OmegaConf.create({"device": requested})
    with pytest.raises(ValueError, match="device|CUDA"):
        runtime_model_specification(specification, torch.device(runtime))


def test_cli_selects_process_device_before_loading_model_and_hashes_portable_config(
    tmp_path, monkeypatch,
):
    import accelerate

    from patchwam import configuration, engine

    path = tmp_path / "configuration.yaml"
    config = OmegaConf.create({
        "model": {"_target_": "example.model", "device": "cuda"},
        "data": {"_target_": "example.data"},
        "training": {"output_dir": str(tmp_path / "run"), "precision": "no", "workers": 0},
    })
    OmegaConf.save(config, path)
    output = tmp_path / "run"
    output.mkdir()
    saved_configuration = output / "configuration.yaml"
    saved_configuration.write_text("original run evidence\n")
    events, recorded = [], {}

    def select_runtime(**kwargs):
        events.append("runtime")
        return SimpleNamespace(device=torch.device("cuda:2"))

    def construct(specification):
        events.append(specification._target_)
        if specification._target_ == "example.model":
            assert specification.device == "cuda:2"
        return object()

    class FakeRun:
        def __init__(self, model, dataset, settings):
            recorded["settings"] = settings
            self.accelerator = SimpleNamespace(is_main_process=False)

        def train(self):
            return "complete"

        def _on_main(self, _operation, _callback):
            events.append("record configuration")
            _callback()

    monkeypatch.setattr(accelerate, "PartialState", select_runtime)
    monkeypatch.setattr(configuration, "construct", construct)
    monkeypatch.setattr(engine, "OptimizationRun", FakeRun)
    monkeypatch.setattr("sys.argv", ["patchwam", "train", "--config", str(path)])
    cli.main()
    assert events == ["runtime", "example.model", "example.data", "record configuration"]
    assert saved_configuration.read_text() == "original run evidence\n"
    assert recorded["settings"].model_config_hash == hashlib.sha256(
        OmegaConf.to_yaml(config.model).encode()
    ).hexdigest()


def test_conflicting_initialization_options_fail_before_loading_configuration(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "patchwam", "train", "--config", "missing.yaml",
        "--resume", "checkpoint", "--initialize", "weights.safetensors",
    ])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
