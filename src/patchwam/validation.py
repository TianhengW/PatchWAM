"""Train, reload, and compare a short run using the selected real model."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path


def compare_weights(left, right, *, atol=0, rtol=0):
    import torch
    from safetensors import safe_open

    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    maximum, count = 0.0, 0
    with safe_open(left, framework="pt", device="cpu") as before, safe_open(right, framework="pt", device="cpu") as after:
        names = before.keys()
        if names != after.keys():
            raise ValueError("Checkpoint keys differ")
        for name in names:
            first, second = before.get_tensor(name), after.get_tensor(name)
            if first.shape != second.shape or first.dtype != second.dtype:
                raise ValueError("Checkpoint shape/dtype differs: " + name)
            if not torch.isfinite(first).all() or not torch.isfinite(second).all():
                raise ValueError("Nonfinite checkpoint tensor: " + name)
            if first.numel():
                difference = (first.float() - second.float()).abs().max().item()
                maximum = max(maximum, difference)
            if not torch.allclose(first, second, atol=atol, rtol=rtol):
                raise ValueError("Resumed weights differ: " + name + f" (max abs {difference})")
            count += 1
    return {"tensors": count, "max_abs_difference": maximum, "atol": atol, "rtol": rtol}


def validate_training(config, overrides=None, *, output, updates=2, num_processes=1,
                      allow_cpu=False, atol=0, rtol=0):
    import torch

    from .configuration import read_configuration

    if type(updates) is not int or updates < 2 or type(num_processes) is not int or num_processes < 1:
        raise ValueError("Validation needs at least two updates and one process")
    source = read_configuration(config, overrides)
    if not allow_cpu and (source.training.get("cpu", False) or not torch.cuda.is_available()):
        raise ValueError("Full-model validation requires CUDA; --allow-cpu is for local checks")
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("baseline", "resumed", "validation.json")):
        raise FileExistsError("Use a fresh validation output directory")
    precision = source.training.get("precision", "bf16")
    validation_overrides = list(overrides or []) + [
        f"training.max_updates={updates}", f"training.epochs={updates}",
        "training.batch_size=1", "training.accumulation=1", "training.workers=0",
        f"training.global_batch_size={num_processes}", "training.checkpoint_every=1", "training.log_every=1",
        "training.deterministic=true",
    ]
    prefix = [sys.executable, "-m", "accelerate.commands.launch"]
    if num_processes > 1:
        if source.training.get("cpu", False):
            prefix = [sys.executable, "-m", "torch.distributed.run",
                      "--rdzv_backend", "c10d", "--rdzv_endpoint", "127.0.0.1:0",
                      "--nproc_per_node", str(num_processes)]
        else:
            prefix += ["--multi_gpu", "--num_processes", str(num_processes),
                       "--num_machines", "1", "--mixed_precision", precision]
    else:
        prefix = [sys.executable]
    command = prefix + ["-m", "patchwam.cli", "train", "--config", str(Path(config).resolve())]
    report = {"status": "running", "config": str(Path(config).resolve()),
              "overrides": validation_overrides, "num_processes": num_processes,
              "scope": "selected model and dataset; reduced validation batch/update budget",
              "cuda": not source.training.get("cpu", False), "comparisons": {}}
    report["determinism"] = "deterministic algorithms; math SDPA; CUBLAS_WORKSPACE_CONFIG=:4096:8"
    report_path = directory / "validation.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        for phase in ("baseline", "resumed"):
            args = command.copy()
            if phase == "resumed":
                args += ["--resume", str(directory / "baseline" / "step_0000001")]
            args += validation_overrides + [f"training.output_dir={directory / phase}"]
            with (directory / (phase + ".log")).open("w") as output_log:
                subprocess.run(args, stdout=output_log, stderr=subprocess.STDOUT,
                               check=True, env=environment)
            final = directory / phase / f"step_{updates:07d}"
            cursor = json.loads((final / "cursor.json").read_text())
            if cursor["updates"] != updates or cursor["exhausted"]:
                raise ValueError("Validation did not reach its successful-update budget")
            metrics = [json.loads(line) for line in (directory / phase / "metrics.jsonl").read_text().splitlines()]
            if not metrics or any(not math.isfinite(value) for row in metrics for value in row.values() if isinstance(value, (int, float))):
                raise ValueError("Missing or nonfinite training metrics")
            report[phase] = {"checkpoint": str(final), "updates": cursor["updates"],
                             "run_signature": cursor["run_signature"], "metrics": metrics}
            save()
        left, right = Path(report["baseline"]["checkpoint"]), Path(report["resumed"]["checkpoint"])
        if report["baseline"]["run_signature"] != report["resumed"]["run_signature"]:
            raise ValueError("Resume contracts differ")
        files = sorted(path.name for path in left.glob("*.safetensors"))
        if not files:
            raise ValueError("No exported model weights")
        for name in files:
            report["comparisons"][name] = compare_weights(str(left / name), str(right / name), atol=atol, rtol=rtol)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        save()
        raise
    save()
    return report
