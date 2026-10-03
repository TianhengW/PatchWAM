from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
import yaml

from patchwam.data import SampleProcessor
from patchwam.evaluation.policy import OnlinePolicy
from patchwam.evaluation.robotwin import make_robotwin_environment
from patchwam.evaluation.runner import run_episodes


class RuntimeTaskFixture:
    def __init__(self):
        self.setups, self.actions, self.closed, self.observation_calls = [], [], 0, 0
        self.expert_valid, self.fail_observation = True, False
        self.success_after, self.step_lim = 2, 3

    def setup_demo(self, **arguments):
        self.setups.append(arguments)
        self.take_action_cnt, self.eval_success = 0, False
        self.vector = np.zeros(14, dtype=np.float32)
        self.info = {"info": {"object": "red block"}}
        self.cwd = Path.cwd()

    def get_obs(self):
        self.observation_calls += 1
        if self.fail_observation:
            raise RuntimeError("camera failed")
        return {
            "observation": {
                name: {"rgb": np.full((2, 4, 3), value, dtype=np.uint8)}
                for name, value in (("head_camera", 10), ("left_camera", 20), ("right_camera", 30))
            },
            "joint_action": {
                "vector": self.vector,
                "left_arm": self.vector[:6],
                "left_gripper": self.vector[6],
                "right_arm": self.vector[7:13],
                "right_gripper": self.vector[13],
            },
        }

    def set_instruction(self, instruction):
        self.instruction = instruction

    def take_action(self, action, *, action_type):
        self.actions.append((action.copy(), action_type))
        self.vector = action.copy()
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt >= self.success_after

    def check_success(self):
        return self.eval_success

    def play_once(self):
        self.vector.fill(99)
        self.plan_success = self.expert_valid
        self.eval_success = self.expert_valid
        return {"info": {"object": "blue block"}}

    def close_env(self, *, clear_cache):
        assert not clear_cache
        self.closed += 1


@pytest.fixture
def task_runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    for directory in (root / "envs", root / "task_config", root / "assets" / "robot"):
        directory.mkdir(parents=True)

    def write(path, value):
        path.write_text(yaml.safe_dump(value))

    write(
        root / "task_config" / "demo_clean.yml",
        {
            "embodiment": ["aloha"],
            "camera": {"head_camera_type": "D435"},
            "data_type": {"rgb": True, "qpos": True},
            "render_freq": 0,
        },
    )
    write(root / "task_config" / "_camera_config.yml", {"D435": {"h": 480, "w": 640}})
    write(root / "task_config" / "_embodiment_config.yml", {"aloha": {"file_path": "assets/robot"}})
    write(
        root / "assets" / "robot" / "config.yml",
        {"arm_joints_name": [[str(i) for i in range(6)]] * 2},
    )
    module = ModuleType("robotwin_runtime_fixture")
    module.Task = RuntimeTaskFixture
    module.clock = lambda task: task.take_action_cnt * 0.125
    calls = []

    def instructions(task_name, episodes, count):
        calls.append((task_name, episodes, count))
        return [
            {"seen": ["pick the blue block", "lift the blue block"], "unseen": ["grasp blue block"]}
        ]

    module.instructions = instructions
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    spec = {
        "root": str(root),
        "task_name": "test_task",
        "task_config": "demo_clean",
        "task_factory": module.__name__ + ":Task",
        "control_hz": 50,
        "instructions": ["pick the block", "lift the block"],
    }
    shape = {
        "images": [{"key": name} for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")],
        "state": [{"key": "default", "raw_shape": 14}],
        "action": [{"key": "default", "raw_shape": 14}],
    }
    return spec, shape, calls


def make(task_runtime, **overrides):
    spec, shape, _ = task_runtime
    return make_robotwin_environment(spec | overrides, shape, seed=123, episode_index=4)


def test_direct_task_setup_maps_raw_state_images_and_joint_action_without_clipping(task_runtime):
    spec, _, _ = task_runtime
    before = Path.cwd()
    env = make(task_runtime)
    first = env.reset()
    setup = env.task.setups[0]
    assert (
        setup["seed"] == 123
        and setup["now_ep_num"] == 4
        and setup["is_test"]
        and setup["eval_mode"]
    )
    assert setup["left_robot_file"] == str(Path(spec["root"]) / "assets" / "robot")
    assert setup["head_camera_h"] == 480 and setup["head_camera_w"] == 640
    assert first["timestamp"] == 0 and first["embodiment"] is None
    assert set(first["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert first["images"]["cam_left_wrist"].dtype == np.uint8
    np.testing.assert_array_equal(first["state"]["default"], np.zeros(14))
    action = np.arange(14, dtype=np.float32) - 5
    observation, reward, terminated, truncated, info = env.step(action)
    np.testing.assert_array_equal(env.task.actions[0][0], action)
    np.testing.assert_array_equal(observation["state"]["default"], action)
    assert env.task.actions[0][1] == "qpos"
    assert observation["timestamp"] == 0.02 and info["timestamp_clock"] == "policy_steps"
    assert info["runtime_embodiment"] == "aloha"
    assert reward == 0 and not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == 1 and terminated and not truncated and info["success"]
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(action)
    env.close()
    assert env.task.closed == 1 and Path.cwd() == before


def test_normalization_embodiment_is_explicit_and_separate_from_runtime_robot_name(task_runtime):
    env = make(task_runtime, embodiment="dataset_robot_identifier")
    observation = env.reset()
    assert observation["embodiment"] == "dataset_robot_identifier"
    _, _, _, _, info = env.step(np.zeros(14))
    assert info["runtime_embodiment"] == "aloha"
    env.close()


def test_runtime_time_limit_and_explicit_timestamp_getter(task_runtime):
    env = make(task_runtime, max_steps=1, timestamp_getter="robotwin_runtime_fixture:clock")
    env.reset()
    observation, reward, terminated, truncated, info = env.step(np.zeros(14))
    assert not terminated and truncated and reward == 0
    assert observation["timestamp"] == 0.125 and info["timestamp_clock"] == "runtime_getter"
    env.close()


def test_expert_validity_check_resets_exact_same_seed_before_policy_and_preserves_info(
    task_runtime,
):
    env = make(
        task_runtime,
        expert_check=True,
        instruction_provider="robotwin_runtime_fixture:instructions",
        instruction_variant="unseen",
    )
    observation = env.reset()
    assert len(env.task.setups) == 2 and [setup["seed"] for setup in env.task.setups] == [123, 123]
    assert observation["instruction"] == "grasp blue block"
    np.testing.assert_array_equal(observation["state"]["default"], np.zeros(14))
    assert task_runtime[2] == [("test_task", [{"object": "blue block"}], 1)]
    env.close()


def test_failed_expert_seed_is_reported_without_replacement(task_runtime):
    env = make(task_runtime, expert_check=True)
    env.task.expert_valid = False
    with pytest.raises(RuntimeError, match="requested seed 123"):
        env.reset()
    assert len(env.task.setups) == 1 and env.task.closed == 1 and not env.active


def test_instruction_and_seed_selection_are_deterministic_and_episode_seed_can_be_explicit(
    task_runtime,
):
    env = make(task_runtime, episode_seeds=[10, 11, 12, 13, 456])
    first = env.reset()
    second = env.reset()
    assert first["instruction"] == second["instruction"] and env.seed == 456
    env.close()


def test_instruction_file_has_seen_unseen_and_seed_specific_variants(task_runtime, tmp_path):
    path = tmp_path / "instructions.json"
    path.write_text('{"episode_seeds": {"123": {"seen": ["specific episode instruction"]}}}')
    env = make(task_runtime, instruction_file=str(path))
    assert env.reset()["instruction"] == "specific episode instruction"
    env.close()


@pytest.mark.parametrize("action", [np.zeros(13), np.zeros((1, 14)), np.full(14, np.nan)])
def test_invalid_actions_are_rejected_before_runtime_mutation(task_runtime, action):
    env = make(task_runtime)
    env.reset()
    with pytest.raises(ValueError, match="14D"):
        env.step(action)
    assert not env.task.actions
    env.close()


def test_reset_failure_cleans_runtime_and_restores_working_directory(task_runtime):
    env = make(task_runtime)
    before = Path.cwd()
    env.task.fail_observation = True
    with pytest.raises(RuntimeError, match="camera failed"):
        env.reset()
    assert not env.active and env.task.closed == 1 and Path.cwd() == before


def test_nonstandard_state_fields_require_explicit_mapping(task_runtime):
    spec, shape, _ = task_runtime
    shape = shape | {"state": [{"key": "left", "raw_shape": 6}, {"key": "right", "raw_shape": 6}]}
    with pytest.raises(ValueError, match="state_map"):
        make_robotwin_environment(spec, shape, seed=1, episode_index=0)
    env = make_robotwin_environment(
        spec | {"state_map": {"left": "joint_action.left_arm", "right": "joint_action.right_arm"}},
        shape,
        seed=1,
        episode_index=0,
    )
    assert set(env.reset()["state"]) == {"left", "right"}
    env.close()


def test_missing_instruction_or_wrong_embodiment_cannot_be_silently_substituted(task_runtime):
    spec, shape, _ = task_runtime
    env = make_robotwin_environment(
        {key: value for key, value in spec.items() if key != "instructions"},
        shape,
        seed=1,
        episode_index=0,
    )
    with pytest.raises(ValueError, match="instructions"):
        env.reset()
    config = Path(spec["root"]) / "assets" / "robot" / "config.yml"
    config.write_text(yaml.safe_dump({"arm_joints_name": [[str(i) for i in range(7)]] * 2}))
    with pytest.raises(ValueError, match="six arm"):
        make(task_runtime)


def test_different_action_mode_or_missing_clock_is_rejected(task_runtime):
    with pytest.raises(ValueError, match="qpos"):
        make(task_runtime, action_type="ee")
    with pytest.raises(KeyError, match="control_hz"):
        spec, shape, _ = task_runtime
        make_robotwin_environment(
            {key: value for key, value in spec.items() if key != "control_hz"},
            shape,
            seed=1,
            episode_index=0,
        )


@pytest.mark.parametrize("seeds", [[], [1], [1, 2, 3, 4, -5], [1, 2, 3, 4, True]])
def test_incomplete_or_invalid_explicit_seed_manifest_is_rejected(task_runtime, seeds):
    with pytest.raises(ValueError, match="episode_seeds"):
        make(task_runtime, episode_seeds=seeds)


def test_prepared_external_checkout_and_actual_task_api_are_required(task_runtime, monkeypatch):
    spec, shape, _ = task_runtime
    with pytest.raises(FileNotFoundError, match="prepared RoboTwin checkout"):
        make(task_runtime, root=str(Path(spec["root"]) / "missing"))
    module = ModuleType("robotwin_incompatible_fixture")
    module.Task = lambda: object()
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    with pytest.raises(TypeError, match="direct evaluation API"):
        make_robotwin_environment(
            spec | {"task_factory": module.__name__ + ":Task"}, shape, seed=1, episode_index=0
        )


def test_task_from_another_imported_checkout_is_rejected(task_runtime, tmp_path, monkeypatch):
    spec, shape, _ = task_runtime
    module = ModuleType("envs.test_task")
    module.__file__ = str(tmp_path / "different_checkout" / "envs" / "test_task.py")
    task_class = type("test_task", (RuntimeTaskFixture,), {"__module__": module.__name__})
    module.test_task = task_class
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="incompatible RoboTwin task tree"):
        make_robotwin_environment(
            {key: value for key, value in spec.items() if key != "task_factory"},
            shape,
            seed=1,
            episode_index=0,
        )


def test_online_policy_and_runner_use_decoded_joint_actions_and_reset_causal_history(
    task_runtime, tmp_path
):
    spec, shape, _ = task_runtime
    shape = shape | {
        "images": [field | {"shape": [3, 4, 4]} for field in shape["images"]],
        "state": [field | {"shape": 14} for field in shape["state"]],
        "action": [field | {"shape": 14} for field in shape["action"]],
    }
    processor = SampleProcessor(shape, norm_default_mode="z-score")
    processor.set_normalizer_from_stats(
        {
            "action": {
                "default": {"global_mean": (np.arange(14) + 10).tolist(), "global_std": [2] * 14}
            },
            "state": {"default": {"global_mean": [0] * 14, "global_std": [1] * 14}},
        }
    )

    class ConstantSampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.batches, self.draws = [], []

        def sample_actions(self, batch, *, horizon, steps, guidance_scale, generator):
            assert steps == 10 and guidance_scale == 1
            assert "action" not in batch and "future_tokens" not in batch
            self.batches.append(
                {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
            )
            self.draws.append(torch.rand((), generator=generator).item())
            return {"action": torch.full((1, horizon, 14), 0.5)}

    model = ConstantSampler()
    actor = OnlinePolicy(
        model,
        processor,
        video_size=(4, 12),
        concat_multi_camera="horizontal",
        head_camera_key="cam_high",
        vl_image_size=4,
        replan_horizon=1,
        history_slots=1,
        history_interval_s=0.02,
        history_tolerance_s=0.005,
    )
    environments = []

    def factory(parameters, metadata, *, seed, episode_index):
        environment = make_robotwin_environment(
            parameters, metadata, seed=seed, episode_index=episode_index
        )
        environments.append(environment)
        return environment

    report = run_episodes(
        actor,
        spec | {"kind": "robotwin", "episode_seeds": [901, 29], "tasks": [{}]},
        {
            "output_dir": str(tmp_path / "evaluation"),
            "episodes_per_task": 2,
            "max_steps": 3,
            "seed": 7,
        },
        environment_factory=factory,
    )
    assert report["summary"]["status"] == "complete" and report["summary"]["success_rate"] == 1
    assert (
        [episode["seed"] for episode in report["episodes"]]
        == [environment.seed for environment in environments]
        == [901, 29]
    )
    assert [episode["steps"] for episode in report["episodes"]] == [2, 2]
    for environment in environments:
        assert environment.task.closed == 1
        for action, mode in environment.task.actions:
            assert action.shape == (14,) and action.dtype == np.float32 and mode == "qpos"
            np.testing.assert_allclose(action, np.arange(14) + 11)
    assert len(model.batches) == 4
    assert [batch["history_valid"].item() for batch in model.batches] == [False, True, False, True]
    torch.testing.assert_close(model.batches[0]["proprio"], torch.zeros(1, 14))
    torch.testing.assert_close(model.batches[2]["proprio"], torch.zeros(1, 14))
    assert model.draws[0] == model.draws[2]
