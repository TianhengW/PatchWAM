# SPDX-License-Identifier: Apache-2.0
"""Joint latent flow objective and action-chunk sampling."""

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn

from .codec import RepeatedActionCodec
from .flow import ShiftedFlow, masked_sample_mse
from .geometry import joint_visibility, sequence_coordinates
from .tiny import SmallJointTransformer


class PatchFlowPolicy(nn.Module):
    """A backbone-independent implementation of the published core method.

    ``forward`` consumes normalized actions and encoded visual/text tokens:
    reference_tokens [B,R,128], future_tokens [B,S,128], text_tokens [B,L,C],
    action [B,H,A], and optional proprio [B,P] or [B,H,P]. Explicit four-axis
    reference_ids / future_ids preserve raster and multiview geometry; the
    sequence-coordinate fallback exists for synthetic examples only.

    Padding flags use the dataset convention: True means invalid. A dimension
    flag may be [A], [B,A], or [B,H,A]; a step flag must be [B,H].
    """

    def __init__(
        self, denoiser: nn.Module, *, action_dim: int, text_dim: int,
        proprio_dim: int | None = None, token_dim: int = 128, action_scale: float = 1,
        shift: float = 5, video_weight: float = 0.5, action_weight: float = 1,
        isolate_actions: bool = False,
    ):
        super().__init__()
        if video_weight < 0 or action_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        self.denoiser = denoiser
        self.codec = RepeatedActionCodec(action_dim, token_dim, action_scale)
        self.flow = ShiftedFlow(shift)
        self.state_projection = None if proprio_dim is None else nn.Linear(proprio_dim, text_dim)
        self.text_dim = text_dim
        self.video_weight, self.action_weight = video_weight, action_weight
        self.isolate_actions = isolate_actions

    def _context(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor | None]:
        context = batch["text_tokens"]
        if context.ndim != 3 or context.shape[-1] != self.text_dim:
            raise ValueError(f"text_tokens must be [B,L,{self.text_dim}]")
        valid = batch.get("text_valid")
        if valid is not None and valid.shape != context.shape[:2]:
            raise ValueError("text_valid shape must match text_tokens")
        if valid is not None:
            valid = valid.to(device=context.device, dtype=torch.bool)
            # A masked key containing NaN still contaminates SDPA's dot product.
            context = torch.where(valid[..., None], context, 0)
        if self.state_projection is not None:
            state = batch.get("proprio")
            if state is None:
                raise ValueError("The configured state projection requires proprio")
            if state.ndim == 3:
                state = state[:, 0]
            if state.ndim != 2 or state.shape[0] != context.shape[0]:
                raise ValueError("proprio must be [B,P] or [B,H,P]")
            projected = self.state_projection(state.to(context)).unsqueeze(1)
            context = torch.cat((context, projected), 1)
            if valid is not None:
                state_valid = torch.ones(context.shape[0], 1, device=valid.device, dtype=torch.bool)
                valid = torch.cat((valid.bool(), state_valid), 1)
        return context, valid

    def _padding(self, batch: Mapping[str, Any], actions: Tensor) -> tuple[Tensor, Tensor]:
        dimensions = batch.get("action_dim_is_pad")
        if dimensions is None:
            valid_dimensions = torch.ones_like(actions, dtype=torch.bool)
        else:
            dimensions = dimensions.to(device=actions.device, dtype=torch.bool)
            if dimensions.ndim == 2:
                dimensions = dimensions[:, None, :]
            try:
                valid_dimensions = ~torch.broadcast_to(dimensions.bool(), actions.shape)
            except RuntimeError as exc:
                raise ValueError("action_dim_is_pad cannot broadcast to [B,H,A]") from exc
        steps = batch.get("action_is_pad")
        if steps is None:
            valid_steps = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        else:
            if steps.shape != actions.shape[:2]:
                raise ValueError("action_is_pad must be [B,H]")
            valid_steps = ~steps.to(device=actions.device, dtype=torch.bool)
        return valid_dimensions & valid_steps[..., None], valid_steps

    def _velocity(self, batch: Mapping[str, Any], noisy_future: Tensor, noisy_action: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        reference = batch["reference_tokens"]
        context, text_valid = self._context(batch)
        b, future_length = noisy_future.shape[:2]
        if reference.ndim != 3 or reference.shape[0] != b or reference.shape[-1] != self.codec.token_dim:
            raise ValueError("reference_tokens must be [B,R,token_dim]")
        if context.shape[0] != b:
            raise ValueError("Text and visual batch sizes differ")
        device = reference.device
        # Raster IDs must be supplied when working with encoded real images.
        ref_ids = batch.get("reference_ids")
        if ref_ids is None:
            ref_ids = sequence_coordinates(b, reference.shape[1], group=10, device=device)
        future_ids = batch.get("future_ids")
        if future_ids is None:
            future_ids = sequence_coordinates(b, future_length, group=0, device=device)
        action_ids = sequence_coordinates(b, noisy_action.shape[1], group=20, device=device)
        text_ids = sequence_coordinates(b, context.shape[1], device=device)
        visibility = joint_visibility(
            context.shape[1], reference.shape[1], future_length, noisy_action.shape[1],
            text_valid=text_valid, isolate_actions=self.isolate_actions, device=device,
        )
        prediction = self.denoiser(
            reference, torch.cat((noisy_future, noisy_action), 1), sigma, context,
            reference_ids=ref_ids, noisy_ids=torch.cat((future_ids, action_ids), 1),
            context_ids=text_ids, visibility=visibility,
        )
        if prediction.shape != (b, future_length + noisy_action.shape[1], self.codec.token_dim):
            raise ValueError("Denoiser returned an incompatible velocity shape")
        return prediction[:, :future_length], prediction[:, future_length:]

    def forward(
        self, batch: Mapping[str, Any], *, generator=None, sigma: Tensor | None = None,
        future_noise: Tensor | None = None, action_noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        future, actions = batch["future_tokens"], batch["action"]
        if future.ndim != 3 or future.shape[-1] != self.codec.token_dim:
            raise ValueError("future_tokens must be [B,S,token_dim]")
        if actions.ndim != 3 or actions.shape[0] != future.shape[0]:
            raise ValueError("action must be [B,H,A] with matching batch size")
        valid_dimensions, valid_steps = self._padding(batch, actions)
        clean_actions = torch.where(valid_dimensions, actions, 0)
        action_tokens = self.codec.encode(clean_actions).to(future)
        if sigma is None:
            sigma = self.flow.sample(future.shape[0], device=future.device, generator=generator)
        if sigma.shape != (future.shape[0],) or not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
            raise ValueError("sigma must be finite [B] values in [0,1]")
        sigma = sigma.to(device=future.device, dtype=torch.float32)
        if future_noise is None:
            future_noise = torch.randn(future.shape, device=future.device, dtype=future.dtype, generator=generator)
        if action_noise is None:
            action_noise = torch.randn(action_tokens.shape, device=future.device, dtype=future.dtype, generator=generator)
        x, velocity_x = self.flow.perturb(future, future_noise, sigma)
        u, velocity_u = self.flow.perturb(action_tokens, action_noise, sigma)
        prediction_x, prediction_u = self._velocity(batch, x, u, sigma)
        valid_token_coordinates = (self.codec.coordinate_validity(valid_dimensions) & valid_steps[..., None]).to(future.device)
        per_video = masked_sample_mse(prediction_x, velocity_x)
        per_action = masked_sample_mse(prediction_u, velocity_u, valid_token_coordinates)
        weight = self.flow.weight(sigma)
        loss_video, loss_action = (weight * per_video).mean(), (weight * per_action).mean()
        return {
            "loss": self.video_weight * loss_video + self.action_weight * loss_action,
            "loss_video": loss_video, "loss_action": loss_action,
            "sigma_mean": sigma.float().mean().detach(),
        }

    def loss(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        return self(batch, **kwargs)

    @torch.no_grad()
    def sample_actions(
        self, batch: Mapping[str, Any], *, horizon: int = 16, future_length: int | None = None,
        steps: int = 20, generator=None, initial_future: Tensor | None = None, initial_action: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Generate a joint latent; return normalized controls for the data boundary."""
        reference = batch["reference_tokens"]
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        future_length = reference.shape[1] if future_length is None else future_length
        if future_length <= 0:
            raise ValueError("future_length must be positive")
        b, _, width = reference.shape
        x = initial_future
        if x is None:
            x = torch.randn((b, future_length, width), device=reference.device, dtype=reference.dtype, generator=generator)
        u = initial_action
        if u is None:
            u = torch.randn((b, horizon, width), device=reference.device, dtype=reference.dtype, generator=generator)
        if x.shape != (b, future_length, width) or u.shape != (b, horizon, width):
            raise ValueError("Initial latent shapes differ from the requested output")
        schedule = self.flow.schedule(steps, device=reference.device)
        for start, end in pairwise(schedule):
            vx, vu = self._velocity(batch, x, u, start.expand(b))
            x, u = x + (end - start) * vx, u + (end - start) * vu
        return {"action": self.codec.decode(u), "future_tokens": x}


def make_tiny_policy(
    *, action_dim: int = 14, proprio_dim: int | None = 14, text_dim: int = 32,
    token_dim: int = 128, width: int = 64, heads: int = 4, depth: int = 2, **policy_options,
) -> PatchFlowPolicy:
    denoiser = SmallJointTransformer(token_dim=token_dim, text_dim=text_dim, width=width, heads=heads, depth=depth)
    return PatchFlowPolicy(
        denoiser, action_dim=action_dim, token_dim=token_dim, text_dim=text_dim, proprio_dim=proprio_dim, **policy_options,
    )
