"""Clip augmentation preserving the attributed data-processing recipe.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Yuyang "Alice.L"
See LICENSE-MIT and PROVENANCE.json in this directory.
"""

from __future__ import annotations

import torch
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as vision


class ClipAugment:
    """Use one probability draw per sample and shared temporal parameters per view."""

    def __init__(self, p=1.0, augment_types=("corrupt_only", "color_only", "both"),
                 color_jitter=None, gamma=None, exposure=None, gaussian_noise=None,
                 random_resized_crop=None, rotate=None):
        if not 0 <= p <= 1:
            raise ValueError("Augmentation probability must be between 0 and 1")
        self.p = p
        self.types = tuple(augment_types) or ("both",)
        if not set(self.types) <= {"corrupt_only", "color_only", "both"}:
            raise ValueError("Unknown augmentation group")
        self.color = color_jitter or {}
        self.gamma, self.exposure = gamma or {}, exposure or {}
        self.noise, self.crop, self.rotate = gaussian_noise or {}, random_resized_crop or {}, rotate or {}

    @staticmethod
    def _uniform(low, high, device):
        return float(torch.empty((), device=device).uniform_(low, high))

    def _view(self, frames):
        if frames.ndim != 4 or not frames.is_floating_point() or frames.min() < -1e-6 or frames.max() > 1 + 1e-6:
            raise ValueError("ClipAugment expects float TCHW frames in [0, 1]")
        group = self.types[int(torch.randint(len(self.types), ()))]
        if group in {"color_only", "both"}:
            operators = []
            for key, function in (("brightness", vision.adjust_brightness), ("contrast", vision.adjust_contrast), ("saturation", vision.adjust_saturation), ("hue", vision.adjust_hue)):
                amount = float(self.color.get(key, 0))
                if amount > 0:
                    low, high = (-min(amount, 0.5), min(amount, 0.5)) if key == "hue" else (max(0, 1 - amount), 1 + amount)
                    operators.append((function, self._uniform(low, high, frames.device)))
            for index in torch.randperm(len(operators)).tolist():
                function, factor = operators[index]
                frames = function(frames, factor)
            if self.gamma:
                frames = vision.adjust_gamma(frames.clamp(0, 1), self._uniform(*self.gamma.get("range", [0.9, 1.1]), frames.device))
            if self.exposure:
                ev = self._uniform(*self.exposure.get("ev_range", self.exposure.get("range", [-0.1, 0.1])), frames.device)
                frames = frames * 2 ** ev
        if group in {"corrupt_only", "both"} and self.noise:
            noise = torch.randn((1, *frames.shape[1:]), device=frames.device, dtype=frames.dtype)
            frames = frames + noise * float(self.noise.get("std", 0.01))
        if self.crop:
            height, width = frames.shape[-2:]
            scale, ratio = self.crop.get("scale", [0.95, 1]), self.crop.get("ratio", "preserve")
            if ratio == "preserve":
                fraction = self._uniform(*scale, frames.device)
                crop_h, crop_w = max(1, min(height, round(height * fraction))), max(1, min(width, round(width * fraction)))
                top, left = int(torch.randint(height - crop_h + 1, (), device=frames.device)), int(torch.randint(width - crop_w + 1, (), device=frames.device))
            else:
                top, left, crop_h, crop_w = RandomResizedCrop.get_params(frames[0], tuple(scale), tuple(ratio))
            frames = vision.resized_crop(frames, top, left, crop_h, crop_w, [height, width], vision.InterpolationMode.BILINEAR, antialias=True)
        if self.rotate:
            degrees = self.rotate.get("degrees", 5.0)
            bounds = degrees if isinstance(degrees, (list, tuple)) else [-float(degrees), float(degrees)]
            fill = self.rotate.get("fill", "mean")
            frames = vision.rotate(frames, self._uniform(*bounds, frames.device), vision.InterpolationMode.BILINEAR, fill=float(frames.mean()) if fill == "mean" else fill)
        return frames.clamp(0, 1)

    def __call__(self, views):
        if float(torch.rand(())) >= self.p:
            return views
        return [self._view(view) for view in views]
