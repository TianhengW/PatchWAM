"""Continuous rectified-flow training and the descending Euler schedule."""

import math

import torch
from torch import Tensor


class ShiftedFlow:
    """Shifted flow schedule and weights in sigma units [0,1]."""

    def __init__(self, shift: float = 5.0, *, quadrature_points: int = 16384, normalization: str = "continuous"):
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("shift must be finite and positive")
        if quadrature_points < 2:
            raise ValueError("quadrature_points must be at least two")
        self.shift = float(shift)
        if normalization not in ("continuous", "endpoint_1000"):
            raise ValueError("normalization must be continuous or endpoint_1000")
        self.normalization = normalization
        # Deterministic weight normalization over the sampling distribution.
        midpoints = (torch.arange(quadrature_points, dtype=torch.float64) + 0.5) / quadrature_points
        points = torch.arange(1, 1001, dtype=torch.float64) / 1000 if normalization == "endpoint_1000" else midpoints
        self.weight_normalizer = self._raw_weight(self.warp(points)).mean().item()

    def warp(self, uniform_time: Tensor) -> Tensor:
        return self.shift * uniform_time / (1 + (self.shift - 1) * uniform_time)

    @staticmethod
    def _raw_weight(sigma: Tensor) -> Tensor:
        return (torch.exp(-2 * (sigma - 0.5).square()) - math.exp(-0.5)).clamp_min(0)

    def weight(self, sigma: Tensor) -> Tensor:
        denominator = self.weight_normalizer + (1e-10 if self.normalization == "endpoint_1000" else 0)
        return self._raw_weight(sigma.float()) / denominator

    def sample(self, batch: int, *, device=None, generator=None) -> Tensor:
        return self.warp(torch.rand(batch, device=device, generator=generator))

    def dual_timesteps(
        self, batch: int, future_length: int, horizon: int, *, mask_ratio: float = 0.25,
        structured: bool = False, structure_ratio: float = 0.5, edge_ratio: float = 0.5,
        device=None, generator=None, first=None, second=None, mask=None, branches=None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Self-Flow Eq. 4: mixed token times and a cleaner teacher time.

        Structured branches mix random masks, modality planes, and clean edges.
        """
        if not 0 <= mask_ratio <= 0.5 or batch < 1 or min(future_length, horizon) < 1:
            raise ValueError("Dual timesteps require positive sizes and mask_ratio in [0, 0.5]")
        first = self.sample(batch, device=device, generator=generator) if first is None else first.to(device)
        second = self.sample(batch, device=device, generator=generator) if second is None else second.to(device)
        for value in (first, second):
            if value.shape != (batch,) or not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError("Both dual timesteps must be finite [B] values in [0,1]")
        length = future_length + horizon
        if not 0 <= structure_ratio <= 1 or not 0 <= edge_ratio <= 1:
            raise ValueError("Structured and edge ratios must be in [0,1]")
        if mask is None:
            mask = torch.rand((batch, length), device=device, generator=generator) < mask_ratio
        if mask.shape != (batch, length):
            raise ValueError("Dual-timestep mask must be [B, future_length + horizon]")
        token_times = torch.where(mask.to(device=device, dtype=torch.bool), second[:, None], first[:, None])
        if structured:
            if branches is None:
                draw = torch.rand(batch, device=device, generator=generator)
                thresholds = torch.tensor((1 - structure_ratio, 1 - structure_ratio * edge_ratio,
                                           1 - structure_ratio * edge_ratio / 2), device=device)
                branches = (draw[:, None] >= thresholds).sum(1)
            if branches.shape != (batch,) or ((branches < 0) | (branches > 3)).any():
                raise ValueError("Structured branches must be [B] values 0=random, 1=plane, 2=inverse, 3=forward")
            branches = branches.to(device=device)
            plane = torch.cat((first[:, None].expand(-1, future_length), second[:, None].expand(-1, horizon)), 1)
            token_times = torch.where((branches != 0)[:, None], plane, token_times)
            token_times[:, :future_length] = torch.where((branches == 2)[:, None], 0, token_times[:, :future_length])
            token_times[:, future_length:] = torch.where((branches == 3)[:, None], 0, token_times[:, future_length:])
            return first, token_times, token_times.min(1).values
        return first, token_times, torch.minimum(first, second)

    def schedule(self, steps: int, *, device=None) -> Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        return self.warp(torch.linspace(1, 0, steps + 1, device=device))

    @staticmethod
    def perturb(clean: Tensor, noise: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        if noise.shape != clean.shape or sigma.shape not in ((clean.shape[0],), clean.shape[:2]):
            raise ValueError("Noise or sigma shape does not match clean samples")
        amount = sigma.to(clean).reshape(*sigma.shape, *([1] * (clean.ndim - sigma.ndim)))
        return torch.lerp(clean, noise, amount), noise - clean


def weighted_masked_sample_mse(prediction: Tensor, target: Tensor, weight: Tensor, valid: Tensor | None = None) -> Tensor:
    """Weighted token MSE, averaged over each sample's valid coordinates."""
    if prediction.shape != target.shape or weight.shape != prediction.shape[:2]:
        raise ValueError("Expected matching [B,N,C] predictions and [B,N] weights")
    valid = torch.ones_like(prediction, dtype=torch.bool) if valid is None else torch.broadcast_to(valid.bool(), prediction.shape)
    difference = torch.where(valid, prediction.float() - target.float(), 0)
    count = valid.flatten(1).sum(1).clamp_min(1)
    return (difference.square() * weight.float()[..., None]).flatten(1).sum(1) / count


def masked_sample_mse(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
    """Per-sample MSE; fully padded samples have zero loss."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    if valid is None:
        return (prediction.float() - target.float()).square().flatten(1).mean(1)
    valid = torch.broadcast_to(valid.bool(), prediction.shape)
    difference = torch.where(valid, prediction.float() - target.float(), 0)
    count = valid.flatten(1).sum(1).clamp_min(1)
    return difference.square().flatten(1).sum(1) / count
