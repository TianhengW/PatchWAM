"""Camera composition and temporally consistent RGB preparation.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Yuyang "Alice.L"
Source contract attribution: LICENSE-MIT and PROVENANCE.json.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_clip(clip: torch.Tensor, size, mode="bilinear") -> torch.Tensor:
    """Resize a TCHW RGB clip in [0, 1]."""
    return F.interpolate(clip, size=tuple(size), mode=mode, align_corners=False, antialias=True)


def prepare_rgb(value) -> torch.Tensor:
    rgb = torch.as_tensor(value)
    if rgb.ndim != 4:
        raise ValueError(f"Expected TCHW or THWC camera frames, received {tuple(rgb.shape)}")
    if rgb.shape[1] != 3 and rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    if rgb.shape[1] != 3:
        raise ValueError("Camera frames must have three RGB channels")
    return rgb.float() / 255 if rgb.dtype == torch.uint8 else rgb.float()


def compose_cameras(clips, layout="horizontal", size=(224, 448), robotwin_layout="compact_288x256"):
    """Return CTHW video in [-1, 1], preserving camera order from shape metadata."""
    clips = [prepare_rgb(clip) for clip in clips]
    if not clips:
        raise ValueError("At least one camera is required")
    if layout == "robotwin":
        if len(clips) != 3:
            raise ValueError("RoboTwin composition requires head, left wrist, right wrist cameras")
        if robotwin_layout in {"compact", "compact_288x256", "288x256"}:
            top, wrist = (192, 256), (96, 128)
        elif robotwin_layout in {"legacy", "legacy_384x320", "384x320"}:
            top, wrist = (256, 320), (128, 160)
        else:
            raise ValueError(f"Unknown RoboTwin camera layout: {robotwin_layout}")
        result = torch.cat((resize_clip(clips[0], top), torch.cat([resize_clip(clip, wrist) for clip in clips[1:]], -1)), -2)
    elif len(clips) == 1:
        result = clips[0]
    elif layout in {"horizontal", "vertical"}:
        result = torch.cat(clips, -1 if layout == "horizontal" else -2)
    else:
        raise ValueError(f"Unsupported camera layout: {layout!r}")
    height, width = result.shape[-2:]
    scale = max(size[0] / height, size[1] / width)
    result = resize_clip(result, (int(height * scale + 0.5), int(width * scale + 0.5)), mode="bicubic")
    offset_h, offset_w = round((result.shape[-2] - size[0]) / 2), round((result.shape[-1] - size[1]) / 2)
    result = result[..., offset_h:offset_h + size[0], offset_w:offset_w + size[1]]
    return (result * 2 - 1).permute(1, 0, 2, 3).contiguous()
