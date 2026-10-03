# SPDX-License-Identifier: Apache-2.0
"""Continuous rectified-flow training and the descending Euler schedule."""

import math

import torch
from torch import Tensor


class ShiftedFlow:
    """Paper schedule and noise-level weight, in sigma units [0, 1]."""

    def __init__(self, shift: float = 5.0, *, quadrature_points: int = 16384):
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("shift must be finite and positive")
        if quadrature_points < 2:
            raise ValueError("quadrature_points must be at least two")
        self.shift = float(shift)
        # Midpoint integration computes E_tau[w_raw(phi(tau))], without RNG.
        midpoints = (torch.arange(quadrature_points, dtype=torch.float64) + 0.5) / quadrature_points
        self.weight_normalizer = self._raw_weight(self.warp(midpoints)).mean().item()

    def warp(self, uniform_time: Tensor) -> Tensor:
        return self.shift * uniform_time / (1 + (self.shift - 1) * uniform_time)

    @staticmethod
    def _raw_weight(sigma: Tensor) -> Tensor:
        return (torch.exp(-2 * (sigma - 0.5).square()) - math.exp(-0.5)).clamp_min(0)

    def weight(self, sigma: Tensor) -> Tensor:
        return self._raw_weight(sigma.float()) / self.weight_normalizer

    def sample(self, batch: int, *, device=None, generator=None) -> Tensor:
        return self.warp(torch.rand(batch, device=device, generator=generator))

    def schedule(self, steps: int, *, device=None) -> Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        return self.warp(torch.linspace(1, 0, steps + 1, device=device))

    @staticmethod
    def perturb(clean: Tensor, noise: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        if noise.shape != clean.shape or sigma.shape != (clean.shape[0],):
            raise ValueError("Noise or sigma shape does not match clean samples")
        amount = sigma.to(clean).reshape(clean.shape[0], *([1] * (clean.ndim - 1)))
        return torch.lerp(clean, noise, amount), noise - clean


def masked_sample_mse(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
    """Average independently per sample; an entirely padded sample has loss zero."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    if valid is None:
        return (prediction.float() - target.float()).square().flatten(1).mean(1)
    valid = torch.broadcast_to(valid.bool(), prediction.shape)
    difference = torch.where(valid, prediction.float() - target.float(), 0)
    count = valid.flatten(1).sum(1).clamp_min(1)
    return difference.square().flatten(1).sum(1) / count
