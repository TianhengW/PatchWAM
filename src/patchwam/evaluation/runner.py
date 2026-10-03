"""Fixed-episode evaluation with resumable result records."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def _write(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _summary(records, requested):
    completed = sum(item["status"] == "complete" for item in records)
    errors = len(records) - completed
    successes = sum(item["success"] for item in records if item["status"] == "complete")
    state = "invalid" if errors else "complete" if completed == requested else "incomplete"
    return {"status": state, "requested": requested, "completed": completed,
            "errors": errors, "successes": successes,
            "success_rate": successes / requested if state == "complete" else None}


def make_environment(spec, shape_meta, *, seed, episode_index):
    if spec["kind"] == "robotwin":
        from .robotwin import make_robotwin_environment
        return make_robotwin_environment(spec, shape_meta, seed=seed, episode_index=episode_index)
    from .simulators import make_environment as create
    return create(spec, shape_meta, seed=seed, episode_index=episode_index)


def run_episodes(policy, benchmark, settings, *, identity=None, resume=False, environment_factory=None):
    tasks = benchmark.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("benchmark.tasks must be a nonempty list of task mappings")
    count = _positive(settings["episodes_per_task"], "episodes_per_task")
    limit = _positive(settings["max_steps"], "max_steps")
    if type(settings.get("seed", 42)) is not int:
        raise ValueError("seed must be an integer")
    common = {key: value for key, value in benchmark.items() if key != "tasks"}
    for task in tasks:
        spec = {**common, **task}
        if "seed" in spec:
            raise ValueError("Use evaluation.seed or episode_seeds instead of benchmark/task seed")
        seeds = spec.get("episode_seeds")
        if seeds is not None and (not isinstance(seeds, list) or len(seeds) != count or any(type(seed) is not int for seed in seeds)):
            raise ValueError("episode_seeds must contain one integer per requested episode")
    directory = Path(settings["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    contract = {"benchmark": benchmark, "settings": {key: value for key, value in settings.items()
                if key != "output_dir"}, "identity": identity}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    records = {}
    if path.exists():
        if not resume:
            raise FileExistsError("Evaluation output exists; use --resume: " + str(path))
        previous = json.loads(path.read_text())
        if previous["signature"] != signature:
            raise ValueError("Evaluation resume requires unchanged weights and protocol")
        records = {(item["task_index"], item["episode_index"]): item for item in previous["episodes"]}
        if len(records) != len(previous["episodes"]):
            raise ValueError("Duplicate episode records")
        if any(not (0 <= task < len(tasks) and 0 <= episode < count) for task, episode in records):
            raise ValueError("Stored episode is outside the requested protocol")
    elif resume:
        raise FileNotFoundError("No evaluation result to resume: " + str(path))
    requested = count * len(tasks)

    def save():
        ordered = [records[key] for key in sorted(records)]
        report = {"signature": signature, "protocol": contract, "episodes": ordered,
                  "summary": _summary(ordered, requested)}
        _write(path, report)
        return report

    report = save()
    factory = environment_factory or make_environment
    for task_index, task in enumerate(tasks):
        spec = {**common, **task}
        spec["max_steps"] = min(_positive(spec.get("max_steps", limit), "task max_steps"), limit)
        seeds = spec.get("episode_seeds")
        if seeds is not None and (len(seeds) != count or any(type(seed) is not int for seed in seeds)):
            raise ValueError("episode_seeds must contain one integer per requested episode")
        for episode_index in range(count):
            key = task_index, episode_index
            if key in records and records[key]["status"] == "complete":
                continue
            seed = seeds[episode_index] if seeds is not None else settings.get("seed", 42) + episode_index
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            started, environment = time.monotonic(), None
            record = {"task_index": task_index, "episode_index": episode_index, "seed": seed,
                      "task": task, "status": "error", "success": False, "steps": 0}
            error = None
            try:
                environment = factory(spec, policy.processor.shape_meta, seed=seed, episode_index=episode_index)
                observation = environment.reset()
                policy.reset((task_index, episode_index, seed))
                for step in range(limit):
                    action = np.asarray(policy.act(observation), dtype=np.float32)
                    if action.ndim != 1 or not np.isfinite(action).all():
                        raise ValueError("Policy returned a nonfinite or non-vector action")
                    observation, reward, terminated, truncated, info = environment.step(action)
                    if not math.isfinite(float(reward)):
                        raise ValueError("Environment returned a nonfinite reward")
                    if "success" not in info:
                        raise KeyError("Environment must provide explicit success")
                    record.update(steps=step + 1, success=bool(info["success"]),
                                  terminated=bool(terminated), truncated=bool(truncated))
                    if record["success"] or terminated or truncated:
                        break
                record["status"] = "complete"
            except Exception as caught:  # noqa: BLE001
                error = caught
                record["error"] = type(caught).__name__ + ": " + str(caught)
            finally:
                if environment is not None:
                    try:
                        environment.close()
                    except Exception as caught:  # noqa: BLE001
                        error = caught
                        record.update(status="error", error="Environment close: " + str(caught))
                record["elapsed_seconds"] = time.monotonic() - started
                records[key] = record
                report = save()
            print(json.dumps(record), flush=True)
            if error is not None:
                raise RuntimeError("Evaluation failed; saved diagnostic record in " + str(path)) from error
    return report


def evaluate(path, overrides=None, *, resume=False):
    from accelerate import PartialState
    from safetensors import safe_open

    from ..checkpoints import load_policy_weights
    from ..configuration import construct, read_configuration, runtime_model_specification
    from ..data.scaling import read_statistics
    from .policy import OnlinePolicy

    configuration = read_configuration(path, overrides)
    evaluation = OmegaConf.to_container(configuration, resolve=True)
    specification = evaluation["policy"]
    training = read_configuration(specification["training_config"], specification.get("overrides"))
    runtime = PartialState(cpu=bool(training.training.get("cpu", False)))
    if runtime.num_processes != 1:
        raise ValueError("Evaluation runs in one process; launch independent task shards explicitly")
    seed = int(evaluation["evaluation"].get("seed", 42))
    torch.manual_seed(seed)
    weights = Path(specification["weights"]).expanduser().resolve()
    statistics = training.data.get("pretrained_norm_stats")
    if not statistics:
        raise ValueError("Evaluation requires saved training normalization statistics")
    if not weights.is_file() or not Path(statistics).expanduser().is_file():
        raise FileNotFoundError("Prepare native policy weights and training statistics before evaluation")
    model = construct(runtime_model_specification(training.model, runtime.device))
    load_policy_weights(model, weights)
    model.eval()
    processor = construct(training.data.processor).eval()
    processor.set_normalizer_from_stats(read_statistics(statistics))
    settings = evaluation["evaluation"]
    actor = OnlinePolicy(
        model, processor, video_size=training.data.video_size,
        concat_multi_camera=training.data.get("concat_multi_camera", "horizontal"),
        robotwin_camera_layout=training.data.get("robotwin_camera_layout", "compact_288x256"),
        horizon=settings.get("horizon", 16), steps=settings.get("steps", 10),
        replan_horizon=settings.get("replan_horizon", 16),
        guidance_scale=settings.get("guidance_scale", 1),
        history_slots=training.data.get("history_slots", 0),
        history_interval_s=training.data.get("history_interval_s", 1),
        vl_image_size=training.data.get("vl_image_size", 448),
        head_camera_key=training.data.get("head_camera_key"),
        separate_camera_views=training.data.get("separate_camera_views", False),
        prompt_template=training.data.get("prompt_template", "A video recorded from a robot's point of view executing the following instruction: {task}"),
        device=runtime.device, seed=seed,
    )
    with safe_open(weights, framework="pt", device="cpu") as archive:
        metadata = archive.metadata() or {}
    identity = {"weights": str(weights), "size": weights.stat().st_size,
                "mtime_ns": weights.stat().st_mtime_ns,
                "weight_variant": metadata.get("weight_variant", "live"),
                "normalization_sha256": hashlib.sha256(Path(statistics).expanduser().read_bytes()).hexdigest(),
                "model_assets": getattr(model, "asset_signature", None),
                "training_config": OmegaConf.to_container(training, resolve=True)}
    return run_episodes(actor, evaluation["benchmark"], settings, identity=identity, resume=resume)
