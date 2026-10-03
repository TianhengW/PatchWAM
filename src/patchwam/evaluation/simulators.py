"""RoboCasa and LIBERO adapters for decoded physical actions."""

import importlib
import json
import math
import os
from pathlib import Path

import numpy as np

_MOBILE_ACTION = {
    "base_motion": (0, 4), "control_mode": (4, 5),
    "end_effector_position": (5, 8), "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}
_MOBILE_STATE = {
    "base_position": (0, 3), "base_rotation": (3, 7),
    "end_effector_position_relative": (7, 10),
    "end_effector_rotation_relative": (10, 14), "gripper_qpos": (14, 16),
}
_ACTION_WIDTHS = {key: end - start for key, (start, end) in _MOBILE_ACTION.items()}
_STATE_WIDTHS = {key: end - start for key, (start, end) in _MOBILE_STATE.items()}


def _width(field):
    shape = field.get("raw_shape", field.get("shape"))
    width = shape if type(shape) is int else math.prod(shape)
    if type(width) is not int or width < 1:
        raise ValueError("Raw field dimensions must be positive integers")
    return width


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def _frequency(spec, kwargs):
    frequency = spec.get("control_hz", kwargs.get("control_freq", 20))
    if (isinstance(frequency, bool) or not isinstance(frequency, (int, float))
            or not math.isfinite(frequency) or frequency <= 0):
        raise ValueError("control_hz/control_freq must be finite positive numbers")
    if "control_hz" in spec and "control_freq" in kwargs and kwargs["control_freq"] != frequency:
        raise ValueError("control_hz and env_kwargs.control_freq must agree")
    kwargs["control_freq"] = frequency


def _vector(value, width, name):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {width}D vector")
    return vector.copy()


def _axis_angle(value):
    """Preserve robosuite's XYZW quaternion hemisphere convention."""
    quaternion = _vector(value, 4, "Quaternion").astype(np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isclose(norm, 1, atol=1e-3, rtol=0):
        raise ValueError("Quaternion must have unit norm")
    utility = importlib.import_module("robosuite.utils.transform_utils")
    return _vector(utility.quat2axisangle(quaternion), 3, "Axis angle")


def _reference(value):
    module, separator, attribute = value.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("Custom readers must use module:function")
    return getattr(importlib.import_module(module), attribute)


def _initial_states(suite, task_id):
    """Allow only the NumPy types needed by official float32/float64 state files."""
    import torch
    from numpy.core.multiarray import _reconstruct

    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "").lower() in {"1", "y", "yes", "true"}:
        raise ValueError("LIBERO initial states require weights-only loading; remove TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    allowed = [np.ndarray, _reconstruct, np.dtype,
               type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        return suite.get_task_init_states(task_id)


def _segments(value, expected, label):
    """Validate named, nonoverlapping slices of one flattened dataset vector."""
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(label + " must name exactly: " + ", ".join(expected))
    segments = {}
    for key, item in value.items():
        if isinstance(item, dict):
            start, end = item["start"], item["end"]
        else:
            start, end = item
        if (type(start) is not int or type(end) is not int or start < 0
                or end - start != expected[key]):
            raise ValueError(label + " has an invalid slice for " + key)
        segments[key] = start, end
    cursor = 0
    for start, end in sorted(segments.values()):
        if start != cursor:
            raise ValueError(label + " slices must cover the vector without gaps or overlap")
        cursor = end
    return segments


def _one_state(shape_meta, width, components):
    fields = shape_meta.get("state", [])
    if len(fields) != 1 or _width(fields[0]) != width:
        raise ValueError(f"Built-in state layout requires one raw {width}D field; use state_fields")
    return {fields[0]["key"]: components}


class _Observations:
    def __init__(self, spec, shape_meta, camera_map, state_fields, orientation):
        self.spec, self.shape_meta = spec, shape_meta
        self.camera_map = camera_map | spec.get("camera_map", {})
        self.state_fields = spec.get("state_fields", state_fields)
        self.reader = _reference(spec["state_reader"]) if "state_reader" in spec else None
        self.orientation = spec.get("image_orientation", orientation)
        if self.orientation not in {"identity", "flip_vertical", "rotate_180"}:
            raise ValueError("image_orientation must be identity, flip_vertical, or rotate_180")
        if any(field["key"] not in self.camera_map for field in shape_meta.get("images", [])):
            raise ValueError("Every model camera needs an explicit camera_map entry")
        if self.reader is None and (not isinstance(self.state_fields, dict) or any(
            field["key"] not in self.state_fields for field in shape_meta.get("state", [])
        )):
            raise ValueError("Every model state field needs an explicit state_fields entry")

    @staticmethod
    def _state(raw, components):
        if isinstance(components, (str, dict)):
            components = [components]
        values = []
        for component in components:
            source = component if isinstance(component, str) else component["key"]
            value = np.asarray(raw[source], dtype=np.float32).reshape(-1)
            if isinstance(component, dict):
                transform = component.get("transform", "identity")
                if transform == "quat_xyzw_to_axis_angle":
                    value = _axis_angle(value)
                elif transform != "identity":
                    raise ValueError("Unknown state transform: " + transform)
                if "slice" in component:
                    start, end = component["slice"]
                    value = value[start:end]
            values.append(value)
        return np.concatenate(values)

    def convert(self, raw, *, instruction, timestamp):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Environment instructions must be nonempty strings")
        images = {}
        for field in self.shape_meta.get("images", []):
            key = field["key"]
            image = np.asarray(raw[self.camera_map[key]])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Camera observations must be HWC uint8 RGB")
            if self.orientation == "flip_vertical":
                image = image[::-1]
            elif self.orientation == "rotate_180":
                image = image[::-1, ::-1]
            images[key] = image.copy()
        custom = self.reader(raw, self.shape_meta) if self.reader else None
        state = {}
        for field in self.shape_meta.get("state", []):
            key = field["key"]
            value = custom[key] if custom is not None else self._state(raw, self.state_fields[key])
            state[key] = _vector(value, _width(field), "State " + key)
        observation = {"images": images, "state": state, "instruction": instruction,
                       "timestamp": timestamp}
        if "embodiment" in self.spec:
            observation["embodiment"] = self.spec["embodiment"]
        return observation


class _Simulation:
    def __init__(self, env, observations, spec, *, seed, action_dim, control_hz):
        self.env, self.observations, self.spec = env, observations, spec
        self.seed, self.action_dim, self.control_hz = seed, action_dim, float(control_hz)
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_freq must be finite and positive")
        self.calls, self.ready, self.closed = 0, False, False

    def _success(self, info):
        if "success" in info:
            return bool(info["success"])
        checker = getattr(self.env, "check_success", None)
        if not callable(checker):
            checker = getattr(self.env, "_check_success", None)
        if not callable(checker):
            raise TypeError("Simulator must expose an explicit success predicate")
        return bool(checker())

    def _instruction(self, raw):
        if "instruction" in self.spec:
            return self.spec["instruction"]
        if "instruction_key" in self.spec:
            return raw[self.spec["instruction_key"]]
        return self.instruction

    def _observe(self, raw):
        return self.observations.convert(raw, instruction=self._instruction(raw),
                                         timestamp=self.calls / self.control_hz)

    def step(self, action):
        if self.closed or not self.ready:
            raise RuntimeError("Reset the environment before stepping or after termination")
        vector = _vector(action, self.action_dim, "Decoded action")
        result = self.env.step(self._action(vector))
        if len(result) == 5:
            raw, reward, done, truncated, info = result
        else:
            raw, reward, done, info = result
            truncated = bool(info.get("TimeLimit.truncated", False))
        success = self._success(info)
        # Robosuite's done denotes a horizon; task success has its own predicate.
        terminated = success
        truncated = bool(truncated or (done and not success))
        self.calls += 1
        self.ready = not (terminated or truncated)
        info = {**info, "success": success, "seed": self.seed}
        return self._observe(raw), float(reward), terminated, truncated, info

    def _action(self, vector):
        return vector

    def close(self):
        if not self.closed:
            self.closed, self.ready = True, False
            self.env.close()


class _MobileKitchen(_Simulation):
    def __init__(self, env, observations, spec, *, seed, action_dim, control_hz, action_fields):
        super().__init__(env, observations, spec, seed=seed, action_dim=action_dim,
                         control_hz=control_hz)
        self.action_fields = action_fields
        self.api = spec.get("api", "gym")
        self.instruction = ""

    def reset(self):
        if self.closed:
            raise RuntimeError("Cannot reset a closed environment")
        self.ready, self.calls = False, 0
        if self.api == "gym":
            raw, _ = self.env.reset(seed=self.seed)
            self.instruction = raw.get("annotation.human.task_description", "")
        else:
            self.env.rng = np.random.default_rng(self.seed)
            raw = self.env.reset()
            self.instruction = self.env.get_ep_meta().get("lang", "")
        observation = self._observe(raw)
        self.ready = True
        return observation

    def _action(self, vector):
        if self.api == "robosuite":
            return vector
        # The official Gym adapter converts closedness/mode at 0.5 to +/-1.
        return {"action." + key: vector[start:end].copy()
                for key, (start, end) in self.action_fields.items()}


class _TabletopTasks(_Simulation):
    def __init__(self, env, observations, spec, *, seed, control_hz, initial_state, instruction):
        super().__init__(env, observations, spec, seed=seed, action_dim=7, control_hz=control_hz)
        self.initial_state, self.instruction = np.asarray(initial_state).copy(), instruction
        self.settle_steps = _integer(spec.get("settle_steps", 5), "settle_steps")
        self.settle_action = _vector(spec.get("settle_action", [0, 0, 0, 0, 0, 0, -1]),
                                     7, "Settling action")
        self.gripper_convention = spec.get("gripper_convention", "raw_pm1")
        if self.gripper_convention not in {"raw_pm1", "open01"}:
            raise ValueError("LIBERO gripper_convention must be raw_pm1 or open01")
        if type(spec.get("binarize_gripper", False)) is not bool:
            raise ValueError("binarize_gripper must be boolean")

    def _action(self, vector):
        if self.gripper_convention == "open01":
            vector[-1] = 1 - 2 * vector[-1]
        if self.spec.get("binarize_gripper", False):
            vector[-1] = np.sign(vector[-1])
        return vector

    def reset(self):
        if self.closed:
            raise RuntimeError("Cannot reset a closed environment")
        self.ready, self.calls = False, 0
        self.env.seed(self.seed)
        self.env.reset()
        raw = self.env.set_init_state(self.initial_state.copy())
        for _ in range(self.settle_steps):
            result = self.env.step(self.settle_action.copy())
            raw, _, done, info = result
            if done or self._success(info):
                raise RuntimeError("Requested initial state terminated or succeeded during settling")
        observation = self._observe(raw)
        self.ready = True
        return observation


def _robocasa(spec, shape_meta, seed):
    api, layout = spec.get("api", "gym"), spec.get("action_layout")
    modality = None
    if "modality_json" in spec:
        modality = json.loads(Path(spec["modality_json"]).expanduser().read_text())
    if api == "gym":
        if layout == "human300":
            action_fields = _MOBILE_ACTION
        elif layout == "modality" and modality is not None:
            action_fields = _segments(modality["action"], _ACTION_WIDTHS, "Modality action")
        elif layout == "fields" and "action_fields" in spec:
            action_fields = _segments(spec["action_fields"], _ACTION_WIDTHS, "Action fields")
        else:
            raise ValueError("Gym RoboCasa needs explicit action_layout: human300, modality, or fields")
        cameras = {field["key"]: "video." + field["key"]
                   for field in shape_meta.get("images", [])}
        orientation = "identity"
    elif api == "robosuite" and layout == "robosuite_native":
        action_fields = None
        cameras = {field["key"]: field["key"] + "_image"
                   for field in shape_meta.get("images", [])}
        orientation = "flip_vertical"
    else:
        raise ValueError("Native RoboCasa requires api: robosuite and action_layout: robosuite_native")
    state_fields = None
    if "state_fields" not in spec and "state_reader" not in spec:
        if api != "gym":
            raise ValueError("Native RoboCasa requires explicit state_fields or state_reader")
        if spec.get("state_layout") == "human300":
            state_segments = _MOBILE_STATE
        elif spec.get("state_layout") == "modality" and modality is not None:
            state_segments = _segments(modality["state"], _STATE_WIDTHS, "Modality state")
        else:
            raise ValueError("Gym RoboCasa needs explicit state_layout or state_fields")
        components = ["state." + key for key in sorted(state_segments,
                                                     key=lambda key: state_segments[key][0])]
        state_fields = _one_state(shape_meta, 16, components)
    observations = _Observations(spec, shape_meta, cameras, state_fields, orientation)
    action_dim = sum(_width(field) for field in shape_meta.get("action", []))
    if action_fields is not None and action_dim != 12:
        raise ValueError("This RoboCasa dataset action layout requires exactly twelve raw coordinates")
    task = spec.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("RoboCasa task must name a registered environment")
    kwargs = dict(spec.get("env_kwargs", {}))
    if any(key in kwargs for key in ("seed", "env_name", "split")):
        raise ValueError("Set task/split explicitly; episode seed comes from the evaluation runner")
    _frequency(spec, kwargs)
    kwargs.setdefault("camera_widths", 256)
    kwargs.setdefault("camera_heights", 256)
    importlib.import_module("robosuite")
    importlib.import_module("robocasa")
    if api == "gym":
        importlib.import_module("robocasa.wrappers.gym_wrapper")
        gym = importlib.import_module("gymnasium")
        env = gym.make("robocasa/" + task, split=spec.get("split", "pretrain"), seed=seed,
                       **kwargs)
    else:
        utility = importlib.import_module("robocasa.utils.env_utils")
        kwargs.setdefault("camera_names", list(dict.fromkeys(
            source.removesuffix("_image") for source in observations.camera_map.values())))
        env = utility.create_env(task, seed=seed, split=spec.get("split", "pretrain"), **kwargs)
    try:
        if api == "robosuite":
            lower, upper = env.action_spec
            if np.shape(lower) != (action_dim,) or np.shape(upper) != (action_dim,):
                raise ValueError("Native simulator action dimension differs from decoded model actions")
        else:
            for key, (start, end) in action_fields.items():
                if env.action_space["action." + key].shape != (end - start,):
                    raise ValueError("RoboCasa Gym action space differs from the declared dataset layout")
        return _MobileKitchen(env, observations, spec, seed=seed, action_dim=action_dim,
                              control_hz=kwargs["control_freq"], action_fields=action_fields)
    except Exception:
        env.close()
        raise


def _libero(spec, shape_meta, seed, episode_index):
    if sum(_width(field) for field in shape_meta.get("action", [])) != 7:
        raise ValueError("LIBERO OSC_POSE requires seven decoded raw action coordinates")
    if spec.get("action_layout", "osc_pose") != "osc_pose":
        raise ValueError("LIBERO supports action_layout: osc_pose (delta pose6 and raw gripper1)")
    state_fields = None
    if "state_fields" not in spec and "state_reader" not in spec:
        if spec.get("state_layout", "eef_axis_angle8") != "eef_axis_angle8":
            raise ValueError("Alternative LIBERO state layouts require explicit state_fields")
        state_fields = _one_state(shape_meta, 8, [
            "robot0_eef_pos", {"key": "robot0_eef_quat", "transform": "quat_xyzw_to_axis_angle"},
            "robot0_gripper_qpos",
        ])
    observations = _Observations(spec, shape_meta,
                                 {"image": "agentview_image", "wrist_image": "robot0_eye_in_hand_image"},
                                 state_fields, "rotate_180")
    benchmark = importlib.import_module("libero.libero.benchmark")
    suites = benchmark.get_benchmark_dict()
    suite_name = spec.get("suite")
    if suite_name not in suites:
        raise ValueError("Unknown LIBERO suite: " + str(suite_name))
    suite = suites[suite_name](task_order_index=_integer(spec.get("task_order_index", 0),
                                                       "task_order_index"))
    task_id = _integer(spec.get("task_id"), "task_id")
    if task_id >= suite.n_tasks:
        raise ValueError("LIBERO task_id is outside the selected suite")
    task = suite.get_task(task_id)
    initial_states = _initial_states(suite, task_id)
    state_index = _integer(spec.get("initial_state_index", episode_index), "initial_state_index")
    if state_index >= len(initial_states):
        raise ValueError("LIBERO initial_state_index is unavailable; states are never silently cycled")
    root = importlib.import_module("libero.libero")
    bddl = Path(root.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kwargs = dict(spec.get("env_kwargs", {}))
    if "bddl_file_name" in kwargs or "seed" in kwargs:
        raise ValueError("LIBERO task file and episode seed are selected explicitly")
    kwargs.setdefault("camera_heights", 256)
    kwargs.setdefault("camera_widths", 256)
    _frequency(spec, kwargs)
    if kwargs.get("controller", "OSC_POSE") != "OSC_POSE":
        raise ValueError("The LIBERO pose action contract requires controller: OSC_POSE")
    environments = importlib.import_module("libero.libero.envs")
    env = environments.OffScreenRenderEnv(bddl_file_name=str(bddl), **kwargs)
    try:
        lower, upper = env.env.action_spec
        if np.shape(lower) != (7,) or np.shape(upper) != (7,):
            raise ValueError("LIBERO simulator must expose a seven-coordinate OSC_POSE action space")
        return _TabletopTasks(env, observations, spec, seed=seed, control_hz=kwargs["control_freq"],
                              initial_state=initial_states[state_index], instruction=task.language)
    except Exception:
        env.close()
        raise


def make_environment(spec, shape_meta, *, seed, episode_index):
    """Build one fixed task/episode; optional simulator packages load lazily."""
    _integer(seed, "seed")
    _integer(episode_index, "episode_index")
    if spec.get("kind") == "robocasa":
        return _robocasa(spec, shape_meta, seed)
    if spec.get("kind") == "libero":
        return _libero(spec, shape_meta, seed, episode_index)
    raise ValueError("Simulator kind must be robocasa or libero")
