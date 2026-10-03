"""Robot feature assembly with explicit temporal and dimension validity.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Yuyang "Alice.L"
Source contract attribution: LICENSE-MIT and PROVENANCE.json.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F

from .cameras import prepare_rgb, resize_clip
from .scaling import RobotFeatureCodec


class SampleProcessor:
    def __init__(self, shape_meta, action_output_dim=None, proprio_output_dim=None,
                 norm_default_mode="q01/q99", norm_exception_mode=None,
                 use_stepwise_action_norm=False, relative_joint_keys=None,
                 delta_action_dim_mask=None):
        self.shape_meta = deepcopy(shape_meta)
        self.action_output_dim = action_output_dim or sum(item["shape"] for item in shape_meta["action"])
        self.proprio_output_dim = proprio_output_dim or sum(item["shape"] for item in shape_meta["state"])
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self.stepwise = use_stepwise_action_norm
        self.relative_joint_keys = relative_joint_keys or []
        self.delta_action_dim_mask = delta_action_dim_mask or {}
        for key, mask in self.delta_action_dim_mask.items():
            meta = next((item for item in shape_meta["action"] if item["key"] == key), None)
            if meta is None or len(mask) != meta["shape"]:
                raise ValueError(f"Invalid delta action dimension mask for {key}")
        self.codec = None
        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def set_normalizer_from_stats(self, statistics):
        self.codec = RobotFeatureCodec(self.shape_meta, statistics, self.norm_default_mode, self.norm_exception_mode, self.stepwise)
        return self

    def transform_fields(self, fields):
        result = {field: {key: value.clone() for key, value in values.items()} for field, values in fields.items()}
        for key in self.relative_joint_keys:
            result["action"][key] -= result["state"][key][..., :1, :]
        for field, values in result.items():
            for item in self.shape_meta[field]:
                if values[item["key"]].shape[-1] != item["shape"]:
                    raise ValueError(f"{field}.{item['key']} has unexpected feature width")
        return result

    def _assemble(self, fields, field, target, existing_mask=None):
        value = torch.cat([fields[field][meta["key"]] for meta in self.shape_meta[field]], -1)
        width = value.shape[-1]
        if width > target:
            raise ValueError(f"{field} width {width} exceeds requested output width {target}")
        mask = torch.arange(target, device=value.device) >= width
        if existing_mask is not None:
            supplied = torch.as_tensor(existing_mask, dtype=torch.bool, device=value.device)
            if supplied.ndim != 1 or supplied.numel() > width:
                raise ValueError(f"Invalid {field} dimension mask")
            mask[:supplied.numel()] |= supplied
        return F.pad(value, (0, target - width)).masked_fill(mask, 0), mask

    def preprocess(self, sample):
        fields = {field: {key: value.clone() for key, value in sample[field].items()} for field in ("action", "state")}
        for key, mask in self.delta_action_dim_mask.items():
            action_pad = sample["action_is_pad"].bool().unsqueeze(-1)
            fields["action"][key] = fields["action"][key].masked_fill(action_pad & torch.as_tensor(mask).bool(), 0)
        fields = self.transform_fields(fields)
        if self.codec is None:
            raise RuntimeError("Set feature statistics before preprocessing")
        fields = self.codec.encode(fields, sample.get("embodiment"))
        action, action_mask = self._assemble(fields, "action", self.action_output_dim, sample.get("action_dim_is_pad"))
        proprio, proprio_mask = self._assemble(fields, "state", self.proprio_output_dim, sample.get("state_dim_is_pad"))
        images = []
        for item in self.shape_meta["images"]:
            clip = prepare_rgb(sample["images"][item["key"]])
            images.append(resize_clip(clip, item["shape"][-2:]))
        return {
            "images": images, "action": action, "proprio": proprio,
            "action_dim_is_pad": action_mask, "proprio_dim_is_pad": proprio_mask,
            "action_is_pad": sample["action_is_pad"], "proprio_is_pad": sample["state_is_pad"],
            "image_is_pad": sample["image_is_pad"], "instruction": sample["task"],
        }

    def decode_actions(self, actions, proprio, embodiment=None):
        """Decode a T/D or B/T/D action tensor without removing horizon steps."""
        if self.codec is None:
            raise RuntimeError("Set feature statistics before decoding actions")
        fields = {}
        for field, value in (("action", actions), ("state", proprio)):
            fields[field] = {}
            offset = 0
            for item in self.shape_meta[field]:
                fields[field][item["key"]] = value[..., offset:offset + item["shape"]]
                offset += item["shape"]
        fields = self.codec.decode(fields, embodiment)
        for key in self.relative_joint_keys:
            fields["action"][key] += fields["state"][key][..., :1, :]
        return fields["action"]
