# SPDX-License-Identifier: Apache-2.0
"""Token layouts and visibility for the joint world/action prediction."""

import torch
from torch import Tensor


def raster_coordinates(batch: int, height: int, width: int, *, group: float, device=None) -> Tensor:
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij",
    )
    coordinates = torch.zeros(height * width, 4, device=device, dtype=torch.float32)
    coordinates[:, 0] = group
    coordinates[:, 1] = rows.flatten()
    coordinates[:, 2] = columns.flatten()
    return coordinates.unsqueeze(0).expand(batch, -1, -1)


def sequence_coordinates(batch: int, length: int, *, group: float = 0, device=None) -> Tensor:
    coordinates = torch.zeros(batch, length, 4, device=device, dtype=torch.float32)
    coordinates[..., 0] = group
    coordinates[..., 3] = torch.arange(length, device=device)
    return coordinates


def joint_visibility(
    text_length: int, reference_length: int, future_length: int, horizon: int,
    *, text_valid: Tensor | None = None, isolate_actions: bool = False, device=None,
) -> Tensor:
    """Return SDPA visibility: True means query may attend to key.

    Layout: [text (+ state), reference | future, action]. A clean prefix is
    closed under attention. Both generated groups see the complete prefix.
    """
    lengths = (text_length, reference_length, future_length, horizon)
    if min(lengths) < 0 or text_length + reference_length == 0:
        raise ValueError("Nonnegative lengths and a nonempty prefix are required")
    prefix = text_length + reference_length
    total = prefix + future_length + horizon
    allowed = torch.ones(total, total, device=device, dtype=torch.bool)
    allowed[:prefix, prefix:] = False
    if isolate_actions:
        action_start = prefix + future_length
        allowed[prefix:action_start, action_start:] = False
        allowed[action_start:, prefix:action_start] = False
    if text_valid is None:
        return allowed[None, None]
    if text_valid.ndim != 2 or text_valid.shape[1] != text_length:
        raise ValueError("text_valid must have shape [batch, text_length]")
    allowed = allowed[None, None].expand(text_valid.shape[0], 1, total, total).clone()
    allowed[..., :text_length] &= text_valid[:, None, None].to(device=allowed.device, dtype=torch.bool)
    return allowed


def flatten_image_latents(latents: Tensor) -> Tensor:
    """Official FLUX.2 AE output is already spatially packed to 128 channels."""
    if latents.ndim != 4:
        raise ValueError("Expected [batch, channels, height, width]")
    return latents.flatten(2).transpose(1, 2)


def restore_image_latents(tokens: Tensor, height: int, width: int) -> Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError("Token count does not match the requested raster")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
