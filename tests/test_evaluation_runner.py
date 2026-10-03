import json
from types import SimpleNamespace

import numpy as np
import pytest

from patchwam.evaluation.runner import run_episodes


class Actor:
    processor = SimpleNamespace(shape_meta={})

    def __init__(self):
        self.resets = []

    def reset(self, episode):
        self.resets.append(episode)

    def act(self, observation):
        return np.zeros(7)


class Environment:
    def __init__(self, success=True, fail=False):
        self.success, self.fail, self.closed = success, fail, False

    def reset(self):
        return {}

    def step(self, action):
        if self.fail:
            raise ValueError("broken controller")
        return {}, 0, False, False, {"success": self.success}

    def close(self):
        self.closed = True


def settings(path):
    return {"output_dir": str(path), "episodes_per_task": 2, "max_steps": 2, "seed": 4}


def test_episode_reset_fixed_seeds_results_and_resume(tmp_path):
    actor, environments = Actor(), []

    def factory(spec, shape, *, seed, episode_index):
        env = Environment(success=episode_index == 0)
        environments.append(env)
        return env

    benchmark = {"kind": "libero", "tasks": [{"task_id": 0}, {"task_id": 1}]}
    report = run_episodes(actor, benchmark, settings(tmp_path), environment_factory=factory)
    assert report["summary"] == {"status": "complete", "requested": 4, "completed": 4,
                                 "errors": 0, "successes": 2, "success_rate": 0.5}
    assert actor.resets == [(0, 0, 4), (0, 1, 5), (1, 0, 4), (1, 1, 5)]
    assert all(env.closed for env in environments)
    run_episodes(actor, benchmark, settings(tmp_path), resume=True, environment_factory=factory)
    assert len(environments) == 4
    with pytest.raises(ValueError, match="unchanged"):
        run_episodes(actor, benchmark, {**settings(tmp_path), "seed": 8}, resume=True)


def test_failed_episode_invalidates_score_and_can_retry(tmp_path):
    benchmark = {"kind": "robotwin", "tasks": [{"task_name": "example"}]}
    broken = Environment(fail=True)
    with pytest.raises(RuntimeError, match="saved diagnostic"):
        run_episodes(Actor(), benchmark, settings(tmp_path), environment_factory=lambda *args, **kwargs: broken)
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["summary"]["requested"] == 2
    assert report["summary"]["status"] == "invalid"
    assert report["summary"]["success_rate"] is None
    assert broken.closed
    report = run_episodes(Actor(), benchmark, settings(tmp_path), resume=True,
                          environment_factory=lambda *args, **kwargs: Environment())
    assert report["summary"]["success_rate"] == 1


def test_nonfinite_actions_and_existing_outputs_fail(tmp_path):
    actor = Actor()
    actor.act = lambda observation: np.full(7, np.nan)
    benchmark = {"kind": "libero", "tasks": [{"task_id": 0}]}
    with pytest.raises(RuntimeError):
        run_episodes(actor, benchmark, settings(tmp_path), environment_factory=lambda *args, **kwargs: Environment())
    with pytest.raises(FileExistsError):
        run_episodes(actor, benchmark, settings(tmp_path))


def test_seed_manifest_matches_executed_and_recorded_seed(tmp_path):
    observed = []

    def factory(spec, shape, *, seed, episode_index):
        observed.append(seed)
        return Environment()

    benchmark = {"kind": "robotwin", "tasks": [{"task_name": "example", "episode_seeds": [90, 23]}]}
    report = run_episodes(Actor(), benchmark, settings(tmp_path), environment_factory=factory)
    assert observed == [90, 23] == [item["seed"] for item in report["episodes"]]
    with pytest.raises(ValueError, match="benchmark/task seed"):
        run_episodes(Actor(), {**benchmark, "seed": 3}, settings(tmp_path / "bad"))
