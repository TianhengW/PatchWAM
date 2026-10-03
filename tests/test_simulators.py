"""Public simulator API fixtures; these tests do not render MuJoCo."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from patchwam.evaluation.simulators import _axis_angle, make_environment


class _UnexpectedStateObject:
    pass


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def _meta(action=12, state=16, cameras=("robot0_agentview_left",)):
    return {"action": [{"key": "default", "raw_shape": action, "shape": action}],
            "state": [{"key": "default", "raw_shape": state, "shape": state}],
            "images": [{"key": key, "shape": [3, 2, 3]} for key in cameras]}


def _pixels():
    return np.arange(18, dtype=np.uint8).reshape(2, 3, 3)


def _mobile_observation():
    return {"state.base_position": np.array([1, 2, 3]),
            "state.base_rotation": np.array([4, 5, 6, 7]),
            "state.end_effector_position_relative": np.array([8, 9, 10]),
            "state.end_effector_rotation_relative": np.array([11, 12, 13, 14]),
            "state.gripper_qpos": np.array([15, 16]),
            "video.robot0_agentview_left": _pixels(),
            "annotation.human.task_description": "Open the cabinet"}


@pytest.fixture
def mobile_api(monkeypatch):
    instances, calls = [], []

    class GymKitchen:
        def __init__(self):
            widths = {"base_motion": 4, "control_mode": 1, "end_effector_position": 3,
                      "end_effector_rotation": 3, "gripper_close": 1}
            self.action_space = {"action." + key: SimpleNamespace(shape=(width,))
                                 for key, width in widths.items()}
            self.closed, self.success, self.done, self.truncated = False, False, False, False
            self.actions, self.seeds = [], []

        def reset(self, *, seed):
            self.seeds.append(seed)
            return _mobile_observation(), {"success": False}

        def step(self, action):
            self.actions.append(action)
            # Fixture for the public Gym dict-to-controller contract.
            self.controller_action = np.concatenate([
                action["action.end_effector_position"], action["action.end_effector_rotation"],
                [1 if action["action.gripper_close"][0] >= 0.5 else -1],
                action["action.base_motion"],
                [1 if action["action.control_mode"][0] >= 0.5 else -1],
            ])
            return (_mobile_observation(), float(self.success), self.done, self.truncated,
                    {"success": self.success})

        def close(self):
            self.closed = True

    class NativeKitchen:
        action_spec = np.full(12, -1), np.full(12, 1)

        def __init__(self):
            self.closed, self.success, self.done = False, False, False
            self.actions, self.seeds = [], []

        def reset(self):
            self.seeds.append(self.rng.integers(0, 100000))
            return {"camera_image": _pixels(), "eef": [1, 2, 3]}

        def get_ep_meta(self):
            return {"lang": "Move the arm"}

        def step(self, action):
            self.actions.append(action.copy())
            return self.reset(), 0.0, self.done, {}

        def _check_success(self):
            return self.success

        def close(self):
            self.closed = True

    def gym_make(name, **kwargs):
        calls.append((name, kwargs))
        env = GymKitchen()
        instances.append(env)
        return env

    def create_env(name, **kwargs):
        calls.append((name, kwargs))
        env = NativeKitchen()
        instances.append(env)
        return env

    _module(monkeypatch, "robosuite")
    _module(monkeypatch, "robocasa")
    _module(monkeypatch, "robocasa.wrappers.gym_wrapper")
    _module(monkeypatch, "gymnasium", make=gym_make)
    _module(monkeypatch, "robocasa.utils.env_utils", create_env=create_env)
    return SimpleNamespace(instances=instances, calls=calls)


def _mobile_spec(**kwargs):
    return {"kind": "robocasa", "task": "PnPCounterToCab", "action_layout": "human300",
            "state_layout": "human300", **kwargs}


def test_mobile_dataset_actions_reach_correct_controller_parts(mobile_api):
    env = make_environment(_mobile_spec(), _meta(), seed=31, episode_index=7)
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(np.zeros(12))
    observation = env.reset()
    np.testing.assert_array_equal(observation["state"]["default"], np.arange(1, 17))
    np.testing.assert_array_equal(observation["images"]["robot0_agentview_left"], _pixels())
    assert observation["state"]["default"].dtype == np.float32
    assert observation["instruction"] == "Open the cabinet"
    vector = np.arange(12, dtype=np.float32) / 10
    observation, _, terminated, truncated, info = env.step(vector)
    runtime = mobile_api.instances[0]
    np.testing.assert_array_equal(runtime.actions[0]["action.base_motion"], vector[:4])
    np.testing.assert_array_equal(runtime.actions[0]["action.end_effector_position"], vector[5:8])
    np.testing.assert_array_equal(runtime.controller_action,
                                  np.r_[vector[5:11], 1, vector[:4], -1])
    assert observation["timestamp"] == 0.05
    assert not terminated and not truncated and info["success"] is False
    assert mobile_api.calls[0] == ("robocasa/PnPCounterToCab",
                                   {"split": "pretrain", "seed": 31, "control_freq": 20,
                                    "camera_widths": 256, "camera_heights": 256})
    env.reset()
    assert runtime.seeds == [31, 31]
    assert env.calls == 0
    env.close()
    env.close()
    assert runtime.closed


@pytest.mark.parametrize("closedness,mode,gripper,base_mode", [
    (0, 0, -1, -1), (0.499, 0.499, -1, -1), (0.5, 0.5, 1, 1), (1, 1, 1, 1),
])
def test_mobile_gripper_and_mode_threshold_are_owned_by_gym(
    mobile_api, closedness, mode, gripper, base_mode,
):
    env = make_environment(_mobile_spec(), _meta(), seed=0, episode_index=0)
    env.reset()
    vector = np.zeros(12)
    vector[4], vector[11] = mode, closedness
    env.step(vector)
    runtime = mobile_api.instances[0]
    assert runtime.actions[0]["action.gripper_close"][0] == pytest.approx(closedness)
    assert runtime.controller_action[6] == gripper
    assert runtime.controller_action[11] == base_mode


def test_mobile_modality_order_is_explicit(mobile_api, tmp_path):
    import json

    # A valid EEF-first dataset variant must differ from the Human300 layout.
    action = {"end_effector_position": [0, 3], "end_effector_rotation": [3, 6],
              "gripper_close": [6, 7], "base_motion": [7, 11], "control_mode": [11, 12]}
    state = {"gripper_qpos": [0, 2], "base_position": [2, 5], "base_rotation": [5, 9],
             "end_effector_position_relative": [9, 12],
             "end_effector_rotation_relative": [12, 16]}
    path = tmp_path / "modality.json"
    path.write_text(json.dumps({"action": {key: {"start": span[0], "end": span[1]}
                                           for key, span in action.items()},
                                "state": {key: {"start": span[0], "end": span[1]}
                                          for key, span in state.items()}}))
    env = make_environment(_mobile_spec(action_layout="modality", state_layout="modality",
                                        modality_json=str(path)), _meta(), seed=0, episode_index=0)
    observation = env.reset()
    np.testing.assert_array_equal(observation["state"]["default"], np.r_[15, 16, np.arange(1, 15)])
    action_vector = np.arange(12, dtype=np.float32)
    env.step(action_vector)
    np.testing.assert_array_equal(mobile_api.instances[0].actions[0]["action.base_motion"], [7, 8, 9, 10])


@pytest.mark.parametrize("change,match", [
    ({"action_layout": None}, "explicit action_layout"),
    ({"state_layout": None}, "explicit state_layout"),
    ({"api": "robosuite"}, "robosuite_native"),
    ({"env_kwargs": {"seed": 123}}, "episode seed"),
    ({"image_orientation": "guess"}, "image_orientation"),
])
def test_mobile_unknown_contracts_fail_before_creation(mobile_api, change, match):
    with pytest.raises(ValueError, match=match):
        make_environment(_mobile_spec(**change), _meta(), seed=0, episode_index=0)
    assert not mobile_api.instances


def test_native_kitchen_is_identity_with_explicit_state(mobile_api):
    spec = _mobile_spec(api="robosuite", action_layout="robosuite_native",
                        state_fields={"default": ["eef"]}, camera_map={"image": "camera_image"},
                        env_kwargs={"control_freq": 10})
    env = make_environment(spec, _meta(state=3, cameras=("image",)), seed=7, episode_index=9)
    observation = env.reset()
    np.testing.assert_array_equal(observation["images"]["image"], _pixels()[::-1])
    vector = np.arange(12, dtype=np.float32)
    observation, *_ = env.step(vector)
    np.testing.assert_array_equal(mobile_api.instances[0].actions[0], vector)
    assert observation["timestamp"] == 0.1
    env.reset()
    runtime = mobile_api.instances[0]
    assert runtime.seeds[0] == runtime.seeds[-1]
    assert mobile_api.calls[0][1]["camera_names"] == ["camera"]


def test_native_action_dimension_mismatch_closes_runtime(mobile_api):
    spec = _mobile_spec(api="robosuite", action_layout="robosuite_native",
                        state_fields={"default": ["eef"]})
    with pytest.raises(ValueError, match="Native simulator action dimension"):
        make_environment(spec, _meta(action=7, state=3), seed=0, episode_index=0)
    assert mobile_api.instances[0].closed


def test_control_frequency_drives_simulator_and_observation_clock(mobile_api, tabletop_api):
    for spec, meta, fixture in [
        (_mobile_spec(control_hz=10), _meta(), mobile_api),
        (_tabletop_spec(control_hz=10, settle_steps=0), _tabletop_meta(), tabletop_api),
    ]:
        env = make_environment(spec, meta, seed=0, episode_index=0)
        env.reset()
        observation, *_ = env.step(np.zeros(env.action_dim))
        assert observation["timestamp"] == 0.1
        call = fixture.calls[-1]
        kwargs = call[1] if isinstance(call, tuple) else call
        assert kwargs["control_freq"] == 10


@pytest.mark.parametrize("frequency", [0, -1, True, float("nan"), "20"])
def test_invalid_control_frequency_fails_before_creation(mobile_api, frequency):
    with pytest.raises(ValueError, match="finite positive"):
        make_environment(_mobile_spec(control_hz=frequency), _meta(), seed=0, episode_index=0)
    assert not mobile_api.instances


def test_conflicting_control_frequencies_fail_before_creation(tabletop_api):
    with pytest.raises(ValueError, match="must agree"):
        make_environment(_tabletop_spec(control_hz=10, env_kwargs={"control_freq": 20}),
                         _tabletop_meta(), seed=0, episode_index=0)
    assert not tabletop_api.instances


@pytest.mark.parametrize("success,done,truncated,expected", [
    (False, False, False, (False, False)), (True, False, False, (True, False)),
    (False, True, False, (False, True)), (False, False, True, (False, True)),
])
def test_success_is_distinct_from_horizon(mobile_api, success, done, truncated, expected):
    env = make_environment(_mobile_spec(), _meta(), seed=0, episode_index=0)
    env.reset()
    runtime = mobile_api.instances[0]
    runtime.success, runtime.done, runtime.truncated = success, done, truncated
    result = env.step(np.zeros(12))
    assert result[2:4] == expected
    assert result[4]["success"] is success
    if any(expected):
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(np.zeros(12))


@pytest.fixture
def tabletop_api(monkeypatch):
    instances, calls, conversions, task_calls = [], [], [], []

    def quaternion_conversion(quaternion):
        conversions.append(quaternion.copy())
        # Stand-in for the simulator API; preserve the supplied hemisphere.
        return np.array([0, 0, np.pi / 2 if quaternion[3] >= 0 else 3 * np.pi / 2])

    def raw():
        return {"agentview_image": _pixels(), "robot0_eye_in_hand_image": _pixels() + 30,
                "robot0_eef_pos": [1, 2, 3],
                "robot0_eef_quat": np.array([0, 0, 2 ** -0.5, 2 ** -0.5]),
                "robot0_gripper_qpos": [0.01, -0.01]}

    class Suite:
        n_tasks = 2

        def __init__(self, task_order_index):
            task_calls.append(("order", task_order_index))

        def get_task(self, index):
            task_calls.append(("task", index))
            return SimpleNamespace(problem_folder="suite", bddl_file=f"task_{index}.bddl",
                                   language=f"Do task {index}")

        def get_task_init_states(self, index):
            task_calls.append(("states", index))
            return np.array([[11, 12, 13], [21, 22, 23]])

    class RenderEnvironment:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            instances.append(self)
            self.env = SimpleNamespace(action_spec=(np.full(7, -1), np.full(7, 1)))
            self.seeds, self.initial_states, self.actions = [], [], []
            self.closed, self.success, self.done = False, False, False

        def seed(self, seed):
            self.seeds.append(seed)

        def reset(self):
            return raw()

        def set_init_state(self, state):
            self.initial_states.append(state.copy())
            state[0] = -123  # The adapter must retain an independent initial state.
            return raw()

        def step(self, action):
            self.actions.append(action.copy())
            return raw(), float(self.success), self.done, {}

        def check_success(self):
            return self.success

        def close(self):
            self.closed = True

    _module(monkeypatch, "libero.libero", get_libero_path=lambda kind: "/assets/" + kind)
    _module(monkeypatch, "libero.libero.benchmark", get_benchmark_dict=lambda: {"libero_spatial": Suite})
    _module(monkeypatch, "libero.libero.envs", OffScreenRenderEnv=RenderEnvironment)
    _module(monkeypatch, "robosuite.utils.transform_utils", quat2axisangle=quaternion_conversion)
    return SimpleNamespace(instances=instances, calls=calls, conversions=conversions,
                           task_calls=task_calls)


def _tabletop_spec(**kwargs):
    return {"kind": "libero", "suite": "libero_spatial", "task_id": 1, **kwargs}


def _tabletop_meta():
    return _meta(action=7, state=8, cameras=("image", "wrist_image"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_official_style_numpy_initial_state_loading_is_scoped(tabletop_api, tmp_path, monkeypatch, dtype):
    import pickle

    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    path = tmp_path / "task.pruned_init"
    states = np.array([[11, 12, 13], [21, 22, 23]], dtype=dtype)
    torch.save(states, path)
    original_load = torch.load
    original_globals = torch.serialization.get_safe_globals()
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        torch.load(path)
    suite = sys.modules["libero.libero.benchmark"].get_benchmark_dict()["libero_spatial"]

    def official_load(self, task_id):
        assert task_id == 1
        return torch.load(path)

    monkeypatch.setattr(suite, "get_task_init_states", official_load)
    env = make_environment(_tabletop_spec(settle_steps=0), _tabletop_meta(), seed=1, episode_index=1)
    env.reset()
    runtime = tabletop_api.instances[0]
    np.testing.assert_array_equal(runtime.initial_states[0], states[1])
    assert runtime.initial_states[0].dtype == dtype
    assert torch.load is original_load
    assert torch.serialization.get_safe_globals() == original_globals
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        torch.load(path)
    env.close()


def test_initial_state_loader_rejects_unknown_objects(tabletop_api, tmp_path, monkeypatch):
    import pickle

    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    path = tmp_path / "unknown.pruned_init"
    torch.save(_UnexpectedStateObject(), path)
    suite = sys.modules["libero.libero.benchmark"].get_benchmark_dict()["libero_spatial"]
    monkeypatch.setattr(suite, "get_task_init_states", lambda self, task_id: torch.load(path))
    original_globals = torch.serialization.get_safe_globals()
    with pytest.raises(pickle.UnpicklingError, match="_UnexpectedStateObject"):
        make_environment(_tabletop_spec(), _tabletop_meta(), seed=0, episode_index=0)
    assert torch.serialization.get_safe_globals() == original_globals
    assert not tabletop_api.instances


def test_initial_state_loader_rejects_unsafe_environment_override(tabletop_api, monkeypatch):
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(ValueError, match="weights-only loading"):
        make_environment(_tabletop_spec(), _tabletop_meta(), seed=0, episode_index=0)
    assert not tabletop_api.instances


def test_tabletop_task_initial_state_images_and_raw_actions(tabletop_api):
    env = make_environment(_tabletop_spec(), _tabletop_meta(), seed=19, episode_index=1)
    observation = env.reset()
    runtime = tabletop_api.instances[0]
    assert tabletop_api.task_calls == [("order", 0), ("task", 1), ("states", 1)]
    assert tabletop_api.calls[0]["bddl_file_name"] == "/assets/bddl_files/suite/task_1.bddl"
    assert tabletop_api.calls[0]["camera_widths"] == 256
    np.testing.assert_array_equal(runtime.initial_states[0], [21, 22, 23])
    assert runtime.seeds == [19]
    assert len(runtime.actions) == 5
    for action in runtime.actions:
        np.testing.assert_array_equal(action, [0, 0, 0, 0, 0, 0, -1])
    np.testing.assert_array_equal(observation["images"]["image"], _pixels()[::-1, ::-1])
    np.testing.assert_array_equal(observation["images"]["wrist_image"], (_pixels() + 30)[::-1, ::-1])
    np.testing.assert_allclose(observation["state"]["default"], [1, 2, 3, 0, 0, np.pi / 2, 0.01, -0.01])
    assert observation["instruction"] == "Do task 1"
    assert observation["timestamp"] == 0
    vector = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.25], dtype=np.float32)
    observation, _, terminated, truncated, info = env.step(vector)
    np.testing.assert_array_equal(runtime.actions[-1], vector)
    assert observation["timestamp"] == 0.05
    assert not terminated and not truncated and info["success"] is False
    env.reset()
    assert runtime.seeds == [19, 19]
    np.testing.assert_array_equal(runtime.initial_states[1], [21, 22, 23])
    env.close()
    assert runtime.closed


@pytest.mark.parametrize("gripper,expected", [(0, 1), (1, -1), (0.25, 0.5)])
def test_tabletop_explicit_open01_gripper_mapping(tabletop_api, gripper, expected):
    env = make_environment(_tabletop_spec(gripper_convention="open01", settle_steps=0),
                           _tabletop_meta(), seed=0, episode_index=0)
    env.reset()
    vector = np.r_[np.arange(6) / 10, gripper].astype(np.float32)
    env.step(vector)
    np.testing.assert_array_equal(tabletop_api.instances[0].actions[-1][:6], vector[:6])
    assert tabletop_api.instances[0].actions[-1][-1] == expected
    assert vector[-1] == gripper  # Caller-owned arrays are never modified.


def test_axis_angle_calls_simulator_with_original_hemisphere(tabletop_api):
    quaternion = np.array([0, 0, 2 ** -0.5, -2 ** -0.5], dtype=np.float32)
    original = quaternion.copy()
    angle = _axis_angle(quaternion)
    np.testing.assert_array_equal(tabletop_api.conversions[-1], quaternion)
    np.testing.assert_array_equal(quaternion, original)
    assert angle[-1] == pytest.approx(3 * np.pi / 2)
    with pytest.raises(ValueError, match="unit norm"):
        _axis_angle(np.zeros(4))


@pytest.mark.parametrize("change,match", [
    ({"initial_state_index": 2}, "never silently cycled"),
    ({"task_id": 2}, "outside"),
    ({"task_id": -1}, "nonnegative integer"),
    ({"suite": "unknown"}, "Unknown LIBERO suite"),
    ({"env_kwargs": {"controller": "JOINT_POSITION"}}, "OSC_POSE"),
    ({"image_orientation": "automatic"}, "image_orientation"),
])
def test_tabletop_contract_errors_do_not_start_simulation(tabletop_api, change, match):
    with pytest.raises(ValueError, match=match):
        make_environment(_tabletop_spec(**change), _tabletop_meta(), seed=0, episode_index=0)
    assert not tabletop_api.instances


@pytest.mark.parametrize("change,match", [
    ({"gripper_convention": "auto"}, "gripper_convention"),
    ({"settle_steps": -1}, "settle_steps"),
    ({"binarize_gripper": "false"}, "boolean"),
])
def test_tabletop_post_creation_errors_close_environment(tabletop_api, change, match):
    with pytest.raises(ValueError, match=match):
        make_environment(_tabletop_spec(**change), _tabletop_meta(), seed=0, episode_index=0)
    assert tabletop_api.instances[0].closed


def test_tabletop_settling_failure_is_not_a_valid_episode(tabletop_api):
    env = make_environment(_tabletop_spec(), _tabletop_meta(), seed=0, episode_index=0)
    tabletop_api.instances[0].success = True
    with pytest.raises(RuntimeError, match="during settling"):
        env.reset()
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(np.zeros(7))


@pytest.mark.parametrize("action", [np.zeros(6), np.full(7, np.nan), np.zeros((1, 7))])
def test_tabletop_rejects_invalid_decoded_actions(tabletop_api, action):
    env = make_environment(_tabletop_spec(settle_steps=0), _tabletop_meta(), seed=0, episode_index=0)
    env.reset()
    with pytest.raises(ValueError, match="finite 7D"):
        env.step(action)
    assert not tabletop_api.instances[0].actions


def test_explicit_camera_state_and_orientation_overrides(tabletop_api):
    spec = _tabletop_spec(settle_steps=0, image_orientation="identity", embodiment="Panda",
                          camera_map={"front": "agentview_image"},
                          state_fields={"position": "robot0_eef_pos"})
    shape = _tabletop_meta()
    shape["images"] = [{"key": "front", "shape": [3, 2, 3]}]
    shape["state"] = [{"key": "position", "raw_shape": 3, "shape": 3}]
    env = make_environment(spec, shape, seed=0, episode_index=0)
    observation = env.reset()
    np.testing.assert_array_equal(observation["images"]["front"], _pixels())
    np.testing.assert_array_equal(observation["state"]["position"], [1, 2, 3])
    assert observation["embodiment"] == "Panda"
    assert not tabletop_api.conversions
