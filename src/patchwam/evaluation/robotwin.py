"""Direct RoboTwin task API for decoded Aloha joint-position actions.

The external runtime owns simulation, planning, assets and instruction templates.
This adapter targets ``setup_demo/get_obs/take_action`` tasks. Timestamps use an
explicit nominal policy-step clock, unless a runtime timestamp getter is supplied;
the variable-duration internal physics trajectory is a separate clock.
"""

import copy
import importlib
import json
import math
import os
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml


def _import_reference(reference):
    module, separator, attribute = reference.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("External API references must use module:attribute")
    return getattr(importlib.import_module(module), attribute)


def _read_mapping(path):
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def _resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _field_width(field):
    raw = field.get("raw_shape", field.get("shape"))
    return raw if isinstance(raw, int) else math.prod(raw)


def _lookup(value, path):
    for name in path.split("."):
        value = value[name]
    return value


@contextmanager
def _working_directory(root):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _runtime_arguments(root, spec):
    selected = str(spec["task_config"])
    config_path = (
        _resolve(root, selected)
        if Path(selected).suffix
        else root / "task_config" / (selected + ".yml")
    )
    arguments = _read_mapping(config_path)
    arguments.update(copy.deepcopy(spec.get("setup_overrides", {})))
    if any(name in arguments for name in ("seed", "now_ep_num", "is_test")):
        raise ValueError("Episode seed/index/test mode are supplied by the evaluation runner")
    arguments.update(
        task_name=spec["task_name"],
        task_config=config_path.stem,
        eval_mode=True,
        eval_video_save_dir=None,
        collect_data=False,
    )
    camera_path = _resolve(root, spec.get("camera_config_path", "task_config/_camera_config.yml"))
    cameras = _read_mapping(camera_path)
    head = cameras[arguments["camera"]["head_camera_type"]]
    arguments["head_camera_h"], arguments["head_camera_w"] = int(head["h"]), int(head["w"])
    embodiments = _read_mapping(
        _resolve(root, spec.get("embodiment_config_path", "task_config/_embodiment_config.yml"))
    )
    selected = arguments.get("embodiment")
    if not isinstance(selected, list) or len(selected) not in (1, 3):
        raise ValueError(
            "RoboTwin embodiment must contain one robot or left/right robots plus separation"
        )
    names = [selected[0], selected[0] if len(selected) == 1 else selected[1]]
    for side, name, joint_group in zip(("left", "right"), names, (0, 1)):
        location = embodiments[name].get("file_path")
        if not location:
            raise ValueError("The selected embodiment needs a local file_path")
        directory = _resolve(root, location)
        configuration = _read_mapping(directory / "config.yml")
        if len(configuration["arm_joints_name"][joint_group]) != 6:
            raise ValueError("This adapter requires six arm joints and one gripper per side")
        arguments[side + "_robot_file"] = str(directory)
        arguments[side + "_embodiment_config"] = configuration
    arguments["dual_arm_embodied"] = len(selected) == 1
    if len(selected) == 3:
        arguments["embodiment_dis"] = selected[2]
    data_type = arguments.get("data_type", {})
    if not data_type.get("rgb") or not data_type.get("qpos"):
        raise ValueError("RoboTwin task configuration must enable RGB and qpos observations")
    return arguments, "+".join(map(str, dict.fromkeys(names))), config_path


class JointPositionTask:
    """An explicitly configured external task; failed seeds are never substituted."""

    def __init__(self, root, task, spec, shape_meta, arguments, embodiment, *, seed, episode_index):
        self.root, self.task = root, task
        self.spec, self.shape_meta, self.arguments = spec, shape_meta, arguments
        self.seed, self.episode_index, self.embodiment = seed, episode_index, embodiment
        self.control_hz = float(spec["control_hz"])
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError(
                "control_hz must explicitly specify a positive recording/control cadence"
            )
        defaults = {
            "cam_high": "head_camera",
            "cam_left_wrist": "left_camera",
            "cam_right_wrist": "right_camera",
        }
        self.camera_map = defaults | spec.get("camera_map", {})
        self.state_map = spec.get("state_map")
        states = shape_meta.get("state", [])
        if self.state_map is None:
            if len(states) != 1 or _field_width(states[0]) != 14:
                raise ValueError("A nonstandard state layout needs an explicit state_map")
            self.state_map = {states[0]["key"]: "joint_action.vector"}
        if any(field["key"] not in self.camera_map for field in shape_meta.get("images", [])):
            raise ValueError("Every model camera needs an explicit RoboTwin camera mapping")
        if any(field["key"] not in self.state_map for field in states):
            raise ValueError("Every model state field needs an explicit source mapping")
        getter = spec.get("timestamp_getter")
        self.timestamp_getter = _import_reference(getter) if getter else None
        self.active, self.finished, self.calls = False, False, 0
        self.instruction = None

    def _runtime(self):
        return _working_directory(self.root)

    def _instructions(self, episode_info):
        if "instruction" in self.spec:
            options = [self.spec["instruction"]]
        elif "instruction_provider" in self.spec:
            provider = _import_reference(self.spec["instruction_provider"])
            previous_python, previous_numpy = random.getstate(), np.random.get_state()
            try:
                random.seed(self.seed)
                np.random.seed(self.seed)
                generated = provider(self.spec["task_name"], [episode_info], 1)
            finally:
                random.setstate(previous_python)
                np.random.set_state(previous_numpy)
            if not isinstance(generated, list) or len(generated) != 1:
                raise ValueError(
                    "Instruction provider must return one episode's instruction variants"
                )
            options = generated[0]
        elif "instruction_file" in self.spec:
            with _resolve(self.root, self.spec["instruction_file"]).open() as stream:
                options = json.load(stream)
        else:
            options = self.spec.get("instructions")
        if isinstance(options, dict):
            if "episode_seeds" in options:
                options = options["episode_seeds"][str(self.seed)]
            if isinstance(options, dict):
                options = options[self.spec.get("instruction_variant", "seen")]
        if (
            not isinstance(options, list)
            or not options
            or any(not isinstance(item, str) or not item.strip() for item in options)
        ):
            raise ValueError(
                "Supply explicit nonempty instructions, an instruction file, or an external provider"
            )
        return random.Random(self.seed).choice(options)

    def _observation(self):
        raw = self.task.get_obs()
        joint_state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
        if joint_state.shape != (14,) or not np.isfinite(joint_state).all():
            raise ValueError(
                "RoboTwin joint_action.vector must contain fourteen finite raw coordinates"
            )
        images, states = {}, {}
        for field in self.shape_meta.get("images", []):
            key = field["key"]
            image = np.asarray(raw["observation"][self.camera_map[key]]["rgb"])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("RoboTwin RGB observations must be uint8 HWC images")
            images[key] = image.copy()
        for field in self.shape_meta.get("state", []):
            value = np.asarray(
                _lookup(raw, self.state_map[field["key"]]), dtype=np.float32
            ).reshape(-1)
            if value.size != _field_width(field) or not np.isfinite(value).all():
                raise ValueError("RoboTwin state field does not match the model's raw shape")
            states[field["key"]] = value.copy()
        timestamp = (
            self.calls / self.control_hz
            if self.timestamp_getter is None
            else float(self.timestamp_getter(self.task))
        )
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Runtime timestamp must be finite nonnegative seconds")
        return {
            "images": images,
            "state": states,
            "instruction": self.instruction,
            "timestamp": timestamp,
            "embodiment": self.spec.get("embodiment"),
        }

    def reset(self):
        with self._runtime():
            self.close()
            self.calls, self.finished = 0, False
            self.active = True
            try:
                self.task.setup_demo(
                    now_ep_num=self.episode_index, seed=self.seed, is_test=True, **self.arguments
                )
                episode_info = getattr(self.task, "info", {}).get("info", {})
                if self.spec.get("expert_check", False):
                    episode = self.task.play_once()
                    if not self.task.plan_success or not self.task.check_success():
                        raise RuntimeError(
                            f"Expert validity check failed for requested seed {self.seed}"
                        )
                    episode_info = episode["info"]
                    self.close()
                    self.active = True
                    self.task.setup_demo(
                        now_ep_num=self.episode_index,
                        seed=self.seed,
                        is_test=True,
                        **self.arguments,
                    )
                self.instruction = self._instructions(episode_info)
                self.task.set_instruction(instruction=self.instruction)
                limit = getattr(self.task, "step_lim", None)
                if type(limit) is not int or limit < 1:
                    raise ValueError("The task runtime must expose a positive integer step_lim")
                requested = self.spec.get("max_steps", limit)
                if type(requested) is not int:
                    raise ValueError("max_steps must be an integer")
                self.limit = min(limit, requested)
                if self.limit < 1:
                    raise ValueError("max_steps must be positive")
                return self._observation()
            except Exception:
                self.close()
                raise

    def step(self, action):
        if not self.active or self.finished:
            raise RuntimeError(
                "Reset the task before stepping, including after episode termination"
            )
        vector = np.asarray(action, dtype=np.float32)
        if vector.shape != (14,) or not np.isfinite(vector).all():
            raise ValueError("Decoded RoboTwin actions must be finite 14D joint-position vectors")
        with self._runtime():
            self.task.take_action(vector.copy(), action_type="qpos")
            self.calls += 1
            success = bool(getattr(self.task, "eval_success", False) or self.task.check_success())
            truncated = self.calls >= self.limit and not success
            self.finished = success or truncated
            observation = self._observation()
        info = {
            "success": success,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "action_type": "qpos",
            "timestamp_clock": "policy_steps"
            if self.timestamp_getter is None
            else "runtime_getter",
            "control_hz": self.control_hz,
            "take_action_count": getattr(self.task, "take_action_cnt", self.calls),
            "runtime_embodiment": self.embodiment,
        }
        return observation, float(success), success, truncated, info

    def close(self):
        if self.active:
            self.active = False
            with self._runtime():
                if callable(getattr(self.task, "close_env", None)):
                    self.task.close_env(clear_cache=False)
                else:
                    self.task.close()


def make_robotwin_environment(spec, shape_meta, *, seed, episode_index):
    """Build an external direct-task runtime; never download assets or replace seeds."""
    root = Path(spec["root"]).expanduser().resolve()
    if not (root / "envs").is_dir():
        raise FileNotFoundError("root must point to a prepared RoboTwin checkout containing envs/")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec["task_name"]):
        raise ValueError("task_name must name one external RoboTwin task module")
    if (
        spec.get("action_type", "qpos") != "qpos"
        or sum(_field_width(field) for field in shape_meta.get("action", [])) != 14
    ):
        raise ValueError("This adapter supports only decoded 14D qpos actions")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("Episode index must be a nonnegative integer")
    selected_seed = spec.get("seed", seed)
    if "episode_seeds" in spec:
        seeds = spec["episode_seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(value) is not int or value < 0 for value in seeds)
            or episode_index >= len(seeds)
        ):
            raise ValueError(
                "episode_seeds must provide a nonnegative integer for this episode index"
            )
        selected_seed = seeds[episode_index]
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("Episode seed must be a nonnegative integer")
    arguments, embodiment, _ = _runtime_arguments(root, spec)
    sys.path.insert(0, str(root))
    reference = spec.get("task_factory", f"envs.{spec['task_name']}:{spec['task_name']}")
    with _working_directory(root):
        factory = _import_reference(reference)
        if "task_factory" not in spec:
            module = sys.modules[factory.__module__]
            location = Path(module.__file__).resolve()
            if root not in location.parents:
                raise RuntimeError(
                    "An incompatible RoboTwin task tree is already imported; use a fresh process"
                )
        task = factory()
    required = ("setup_demo", "get_obs", "take_action", "check_success", "set_instruction")
    if spec.get("expert_check", False):
        required += ("play_once",)
    if any(not callable(getattr(task, name, None)) for name in required):
        raise TypeError("External task does not implement the RoboTwin direct evaluation API")
    if not any(callable(getattr(task, name, None)) for name in ("close_env", "close")):
        raise TypeError("External task must expose close_env or close")
    return JointPositionTask(
        root,
        task,
        spec,
        shape_meta,
        arguments,
        embodiment,
        seed=selected_seed,
        episode_index=episode_index,
    )
