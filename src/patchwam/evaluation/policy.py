"""Online observations, causal history, and decoded action chunks."""

import math
from collections import deque

import torch
from torch.nn import functional as F

from patchwam.data.cameras import compose_cameras, prepare_rgb, resize_clip
from patchwam.data.history import CausalObservationBuffer


class OnlinePolicy:
    """Reset per episode; call act once after each environment observation."""

    def __init__(
        self, model, processor, *, video_size, concat_multi_camera,
        robotwin_camera_layout="compact_288x256", horizon=16, steps=10,
        replan_horizon=16, guidance_scale=1, history_slots=0, history_interval_s=1.0,
        vl_image_size=448, head_camera_key=None, device=None, seed=42,
        separate_camera_views=False, history_tolerance_s=0.05,
        prompt_template="A video recorded from a robot's point of view executing the following instruction: {task}",
    ):
        if any(type(value) is not int or value < 1 for value in (horizon, steps, replan_horizon, vl_image_size)):
            raise ValueError("Action/image horizons and solver steps must be positive integers")
        if replan_horizon > horizon:
            raise ValueError("Replanning cannot consume more actions than the predicted horizon")
        if type(history_slots) is not int or history_slots < 0:
            raise ValueError("History slots must be a nonnegative integer")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Guidance scale must be finite and nonnegative")
        if len(video_size) != 2 or any(type(value) is not int or value < 1 for value in video_size):
            raise ValueError("Video size must contain positive integer height and width")
        if processor.codec is None:
            raise ValueError("The data processor must have training normalization statistics")
        if not processor.shape_meta["images"]:
            raise ValueError("At least one camera is required")
        self.model, self.processor = model, processor
        processor.eval()
        self.video_size, self.concat_multi_camera = tuple(video_size), concat_multi_camera
        self.robotwin_camera_layout = robotwin_camera_layout
        self.horizon, self.steps, self.replan_horizon = horizon, steps, replan_horizon
        self.guidance_scale, self.seed = float(guidance_scale), int(seed)
        self.vl_image_size, self.separate_camera_views = vl_image_size, bool(separate_camera_views)
        self.head_camera_key = head_camera_key or processor.shape_meta["images"][0]["key"]
        if self.head_camera_key not in {item["key"] for item in processor.shape_meta["images"]}:
            raise ValueError("The head camera is absent from shape metadata")
        self.prompt_template = prompt_template
        parameters = getattr(model, "parameters", None)
        parameter = next(iter(parameters()), None) if callable(parameters) else None
        self.device = torch.device(device) if device is not None else (parameter.device if parameter is not None else torch.device("cpu"))
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("OnlinePolicy supports CPU and CUDA sampling")
        if parameter is not None and parameter.device != self.device:
            model.to(self.device)
        if callable(getattr(model, "eval", None)):
            model.eval()
        self.history = CausalObservationBuffer(history_slots, history_interval_s, tolerance_s=history_tolerance_s)
        self._episode_id = None
        self._actions = deque()
        self._timestamp = self._action_timestamp = None
        self._raw_observation = self._batch = None
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def reset(self, episode_id):
        if episode_id is None:
            raise ValueError("Reset requires an episode identifier")
        self._episode_id = episode_id
        self._actions.clear()
        self.history.reset()
        self.generator.manual_seed(self.seed)
        self._timestamp = self._action_timestamp = None
        self._raw_observation = self._batch = None

    @staticmethod
    def _camera(value):
        image = torch.as_tensor(value).detach().cpu()
        if image.ndim != 3:
            raise ValueError("Camera observations must be HWC uint8 or CHW float RGB")
        if image.dtype == torch.uint8:
            if image.shape[-1] != 3:
                raise ValueError("Uint8 camera observations must be HWC RGB")
            image = image.permute(2, 0, 1)
        elif not image.is_floating_point() or image.shape[0] != 3:
            raise ValueError("Floating camera observations must be CHW RGB")
        if not torch.isfinite(image).all() or (image.is_floating_point() and (image.min() < 0 or image.max() > 1)):
            raise ValueError("Float RGB observations must be finite and in [0,1]")
        return image.contiguous().clone()

    def _canonical_observation(self, observation):
        timestamp = float(observation["timestamp"])
        instruction = observation["instruction"]
        if not math.isfinite(timestamp) or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Observations require a finite timestamp and nonempty instruction")
        images = {meta["key"]: self._camera(observation["images"][meta["key"]]) for meta in self.processor.shape_meta["images"]}
        state = {}
        for meta in self.processor.shape_meta["state"]:
            value = torch.as_tensor(observation["state"][meta["key"]], dtype=torch.float32).detach().cpu()
            width = meta.get("raw_shape", meta["shape"])
            if value.shape != (width,) or not torch.isfinite(value).all():
                raise ValueError(f"State {meta['key']} must be finite with raw width {width}")
            state[meta["key"]] = value.clone()
        embodiment = observation.get("embodiment")
        return {"timestamp": timestamp, "instruction": instruction, "images": images,
                "state": state, "embodiment": None if embodiment is None else str(embodiment)}

    def _prepare(self, observation):
        embodiment = self._normalization_embodiment(observation)
        fields = {key: value.unsqueeze(0) for key, value in observation["state"].items()}
        transformed = self.processor.transform_fields({
            "state": fields,
            "action": {meta["key"]: torch.zeros(1, meta.get("raw_shape", meta["shape"]))
                       for meta in self.processor.shape_meta["action"]},
        })
        fields = self.processor.codec.encode({"state": transformed["state"]}, embodiment)["state"]
        proprio = torch.cat([fields[meta["key"]] for meta in self.processor.shape_meta["state"]], -1)
        if proprio.shape[-1] > self.processor.proprio_output_dim:
            raise ValueError("State width exceeds the trained proprioception interface")
        proprio = F.pad(proprio, (0, self.processor.proprio_output_dim - proprio.shape[-1]))
        cameras = [resize_clip(prepare_rgb(observation["images"][meta["key"]].unsqueeze(0)), meta["shape"][-2:]) for meta in self.processor.shape_meta["images"]]
        video = compose_cameras(cameras, self.concat_multi_camera, self.video_size, self.robotwin_camera_layout)
        batch = {"video": video.unsqueeze(0), "proprio": proprio,
                 "prompt": [self.prompt_template.format(task=observation["instruction"])]}
        if self.separate_camera_views:
            if len({tuple(camera.shape) for camera in cameras}) != 1:
                raise ValueError("Separate camera views require equal image shapes")
            batch["camera_video"] = torch.stack([(camera * 2 - 1).permute(1, 0, 2, 3) for camera in cameras]).unsqueeze(0)
        head = observation["images"][self.head_camera_key]
        current = prepare_rgb(head.unsqueeze(0))
        batch["vl_current"] = resize_clip(current, (self.vl_image_size,) * 2) * 2 - 1
        past = self.history.history(observation["timestamp"])
        valid = past["history_valid"]
        if past["history_video"].shape[0]:
            raw_history = prepare_rgb(past["history_video"])
            head_meta = next(meta for meta in self.processor.shape_meta["images"] if meta["key"] == self.head_camera_key)
            history = resize_clip(raw_history, head_meta["shape"][-2:]) * 2 - 1
            vl_history = resize_clip(raw_history, (self.vl_image_size,) * 2) * 2 - 1
        else:
            history = current.new_empty((0, 3, *cameras[0].shape[-2:]))
            vl_history = current.new_empty((0, 3, self.vl_image_size, self.vl_image_size))
        batch["history_video"] = history.masked_fill(~valid[:, None, None, None], 0).unsqueeze(0)
        batch["vl_history"] = vl_history.masked_fill(~valid[:, None, None, None], 0).unsqueeze(0)
        batch["history_valid"] = batch["vl_history_valid"] = valid.unsqueeze(0)
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _normalization_embodiment(self, observation):
        return observation["embodiment"] if self.processor.codec.statistics.get("type") == "per_embodiment" else None

    def observe(self, observation):
        if self._episode_id is None:
            raise RuntimeError("Reset the policy with an episode identifier before observing")
        current = self._canonical_observation(observation)
        if self._timestamp is not None and current["timestamp"] < self._timestamp:
            raise ValueError("Observation timestamps must increase within an episode")
        if self._raw_observation is not None:
            previous = self._raw_observation
            if current["embodiment"] != previous["embodiment"]:
                raise ValueError("Embodiment changes require an episode reset")
            if current["timestamp"] == self._timestamp:
                unchanged = all(torch.equal(current[field][key], value) for field in ("images", "state") for key, value in previous[field].items())
                if not unchanged:
                    raise ValueError("Repeated observation timestamps must carry the same images and state")
                if current["instruction"] == previous["instruction"]:
                    return
            if current["instruction"] != previous["instruction"]:
                self._actions.clear()
        if current["timestamp"] != self._timestamp:
            if self.history.frames and current["images"][self.head_camera_key].dtype != self.history.frames[-1][1].dtype:
                raise ValueError("Head-camera dtype changes require an episode reset")
            self.history.append(current["images"][self.head_camera_key], current["timestamp"], episode_id=self._episode_id)
        self._timestamp, self._raw_observation = current["timestamp"], current
        self._batch = None

    @torch.no_grad()
    def act(self, observation):
        self.observe(observation)
        if self._timestamp == self._action_timestamp:
            raise ValueError("Call act once per environment observation")
        if not self._actions:
            batch = self._prepare(self._raw_observation)
            self._batch = batch
            prediction = self.model.sample_actions(
                batch, horizon=self.horizon, steps=self.steps, guidance_scale=self.guidance_scale,
                generator=self.generator,
            )
            normalized = prediction["action"].detach().float().cpu()
            if normalized.shape != (1, self.horizon, self.processor.action_output_dim) or not torch.isfinite(normalized).all():
                raise ValueError("Sampled actions must be finite and match the trained chunk interface")
            decoded = self.processor.decode_actions(normalized, batch["proprio"].detach().float().cpu().unsqueeze(1), embodiment=self._normalization_embodiment(self._raw_observation))
            for meta in self.processor.shape_meta["action"]:
                if decoded[meta["key"]].shape != (1, self.horizon, meta.get("raw_shape", meta["shape"])):
                    raise ValueError(f"Decoded {meta['key']} actions must match the raw feature width")
            actions = torch.cat([decoded[meta["key"]] for meta in self.processor.shape_meta["action"]], -1)[0]
            if not torch.isfinite(actions).all():
                raise ValueError("Decoded actions must be finite")
            self._actions.extend(action.numpy().copy() for action in actions[:self.replan_horizon])
        self._action_timestamp = self._timestamp
        return self._actions.popleft()
