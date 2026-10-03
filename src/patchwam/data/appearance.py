"""Appearance augmentation adapted from the locked C2R-DR4 data recipe.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Yuyang "Alice.L"
See LICENSE-MIT and PROVENANCE.json for source hashes and modification notes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as neural
from torchvision.transforms import functional as vision


class AppearanceRandomizer:
    """Geometry-preserving appearance changes, shared across frames per camera.

    One sample-level probability draw; independent camera parameters.
    """

    def __init__(self, p=0.8, photometric=None, style=None, fourier=None, background=None):
        if not 0 <= p <= 1:
            raise ValueError("Appearance randomization probability must be in [0, 1]")
        if (background or {}).get("enabled", False):
            raise NotImplementedError("Mask-guided background replacement is not supported by AppearanceRandomizer")
        self.p = float(p)
        self.photometric = self._config({
            "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1,
            "gamma": (0.7, 1.4), "exposure": (-0.5, 0.5), "color_temperature": 0.15,
            "gaussian_noise_std": 0.03, "blur_sigma": (0.0, 1.5), "blur_prob": 0.3,
        }, photometric)
        self.style = self._config({
            "enabled": False, "p": 0.5, "target_mean": (0.2, 0.8),
            "target_std": (0.05, 0.4), "strength": (0.4, 1.0), "eps": 1e-5,
        }, style)
        self.fourier = self._config({
            "enabled": False, "p": 0.5, "amp_jitter": (0.5, 1.5),
            "envelope_strength": (0.0, 0.6), "envelope_grid": 8, "eps": 1e-8,
        }, fourier)
        if not 0 <= self.style["p"] <= 1 or not 0 <= self.fourier["p"] <= 1:
            raise ValueError("Style and Fourier probabilities must be in [0, 1]")
        if int(self.fourier["envelope_grid"]) < 1:
            raise ValueError("Fourier envelope grid must be positive")

    @staticmethod
    def _config(defaults, overrides):
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError("Appearance stage configuration must be a mapping")
        extra = set(overrides or {}) - set(defaults)
        if extra:
            raise TypeError(f"Unknown appearance configuration keys: {sorted(extra)}")
        return {**defaults, **(overrides or {})}

    @staticmethod
    def _uniform(low, high, device):
        return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())

    def _lighting(self, frames):
        cfg, device = self.photometric, frames.device
        operations = []
        for key, function in (("brightness", vision.adjust_brightness), ("contrast", vision.adjust_contrast),
                              ("saturation", vision.adjust_saturation), ("hue", vision.adjust_hue)):
            amount = float(cfg[key])
            if amount > 0:
                bounds = (-min(amount, 0.5), min(amount, 0.5)) if key == "hue" else (max(0, 1 - amount), 1 + amount)
                operations.append((function, self._uniform(*bounds, device)))
        if operations:
            for position in torch.randperm(len(operations)).tolist():
                function, factor = operations[position]
                frames = function(frames, factor)
        if cfg["gamma"]:
            frames = vision.adjust_gamma(frames.clamp(0, 1), self._uniform(*cfg["gamma"], device))
        if cfg["exposure"]:
            frames = frames * 2 ** self._uniform(*cfg["exposure"], device)
        temperature = float(cfg["color_temperature"])
        if temperature > 0:
            gains = torch.tensor([1 + self._uniform(-temperature, temperature, device),
                                  1 + self._uniform(-0.5 * temperature, 0.5 * temperature, device),
                                  1 + self._uniform(-temperature, temperature, device)],
                                 device=device, dtype=frames.dtype).view(1, 3, 1, 1)
            frames = frames * gains
        frames = frames.clamp(0, 1)
        if cfg["blur_sigma"] and self._uniform(0, 1, device) < float(cfg["blur_prob"]):
            sigma = self._uniform(max(1e-3, cfg["blur_sigma"][0]), cfg["blur_sigma"][1], device)
            if sigma > 1e-2:
                kernel = int(2 * round(3 * sigma) + 1)
                frames = vision.gaussian_blur(frames, [kernel, kernel], [sigma, sigma])
        if float(cfg["gaussian_noise_std"]) > 0:
            noise = torch.randn((1, *frames.shape[1:]), device=device, dtype=frames.dtype)
            frames = frames + noise * float(cfg["gaussian_noise_std"])
        return frames.clamp(0, 1)

    def _channel_statistics(self, frames):
        cfg, device = self.style, frames.device
        if not cfg["enabled"] or self._uniform(0, 1, device) >= float(cfg["p"]):
            return frames
        channels = frames.shape[1]
        flattened = frames.permute(1, 0, 2, 3).reshape(channels, -1)
        mean = flattened.mean(1).view(1, channels, 1, 1)
        std = flattened.std(1).view(1, channels, 1, 1).clamp_min(float(cfg["eps"]))
        target_mean = torch.empty(channels, device=device, dtype=frames.dtype).uniform_(*cfg["target_mean"]).view(1, channels, 1, 1)
        target_std = torch.empty(channels, device=device, dtype=frames.dtype).uniform_(*cfg["target_std"]).view(1, channels, 1, 1)
        strength = self._uniform(*cfg["strength"], device)
        effective_mean = mean + strength * (target_mean - mean)
        effective_std = std + strength * (target_std - std)
        return ((frames - mean) / std * effective_std + effective_mean).clamp(0, 1)

    def _spectrum(self, frames):
        cfg, device = self.fourier, frames.device
        if not cfg["enabled"] or self._uniform(0, 1, device) >= float(cfg["p"]):
            return frames
        spectrum = torch.fft.fft2(frames.float(), dim=(-2, -1))
        amplitude, phase = spectrum.abs(), spectrum.angle()
        grid = int(cfg["envelope_grid"])
        jitter = torch.empty(1, 1, grid, grid, device=device, dtype=torch.float32).uniform_(*cfg["amp_jitter"])
        jitter = neural.interpolate(jitter, size=frames.shape[-2:], mode="bilinear", align_corners=False)
        strength = self._uniform(*cfg["envelope_strength"], device)
        envelope = torch.empty(1, 1, grid, grid, device=device, dtype=torch.float32).uniform_(0.5, 1.5)
        envelope = neural.interpolate(envelope, size=frames.shape[-2:], mode="bilinear", align_corners=False)
        modulation = jitter * (1 - strength) + envelope * strength
        modified = torch.polar(amplitude * modulation.clamp_min(0), phase)
        return torch.fft.ifft2(modified, dim=(-2, -1)).real.to(frames.dtype).clamp(0, 1)

    def _view(self, frames):
        if frames.ndim != 4 or frames.shape[1] != 3 or not frames.is_floating_point():
            raise ValueError("AppearanceRandomizer expects float TCHW RGB frames")
        if not torch.isfinite(frames).all() or frames.min() < -1e-6 or frames.max() > 1 + 1e-6:
            raise ValueError("AppearanceRandomizer expects finite frames in [0, 1]")
        return self._spectrum(self._channel_statistics(self._lighting(frames)))

    def __call__(self, views):
        """Accept a camera list, or RGB tensors in CHW, TCHW, or camera/TCHW form."""
        if isinstance(views, torch.Tensor) and (views.ndim not in {3, 4, 5} or not views.is_floating_point()):
            raise ValueError("AppearanceRandomizer expects float CHW/TCHW/camera-TCHW tensors")
        if float(torch.rand(())) >= self.p:
            return views
        if isinstance(views, torch.Tensor):
            if views.ndim == 3:
                return self._view(views.unsqueeze(0)).squeeze(0)
            if views.ndim == 4:
                return self._view(views)
            return torch.stack([self._view(view) for view in views])
        return [self._view(view) for view in views]
