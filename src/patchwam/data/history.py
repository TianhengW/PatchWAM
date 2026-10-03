# SPDX-License-Identifier: Apache-2.0
"""Causal frame selection and episode-local online observation storage."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PastFrameSelector:
    """Slots are ordered from one interval ago to ``slots`` intervals ago.

    Frame indices refer to the original episode clock, never a filtered sample
    index. World-model dropout is independent of the undropped VL history.
    """

    slots: int = 20
    interval_s: float = 1.0
    jitter_s: float = 0.4
    whole_dropout: float = 0.2
    slot_dropout: float = 0.2
    truncate_recent: bool = True
    vl_jitter_frames: int = 1

    def __post_init__(self):
        if type(self.slots) is not int or self.slots < 0:
            raise ValueError("History slots must be a nonnegative integer")
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError("History interval must be finite and positive")
        if not math.isfinite(self.jitter_s) or not 0 <= self.jitter_s < self.interval_s:
            raise ValueError("History jitter must be nonnegative and smaller than the interval")
        if any(not 0 <= value <= 1 for value in (self.whole_dropout, self.slot_dropout)):
            raise ValueError("History dropout probabilities must be in [0,1]")
        if type(self.vl_jitter_frames) is not int or self.vl_jitter_frames < 0:
            raise ValueError("VL history jitter must be a nonnegative frame count")

    def select(self, current_row: int, fps: float, *, training: bool = False, vision_language: bool = False):
        if current_row < 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("A nonnegative episode row and positive FPS are required")
        centers = current_row - np.arange(1, self.slots + 1) * self.interval_s * fps
        if training and self.slots:
            if vision_language:
                centers = centers + np.random.randint(-self.vl_jitter_frames, self.vl_jitter_frames + 1, self.slots)
            else:
                centers = centers + np.random.uniform(-self.jitter_s, self.jitter_s, self.slots) * fps
        # The current frame and every future frame are excluded even at low FPS.
        rows = np.rint(centers).astype(np.int64)
        valid = (rows >= 0) & (rows < current_row)
        rows = np.maximum(0, np.minimum(rows, max(0, current_row - 1)))
        if training and not vision_language and self.slots:
            if np.random.random() < self.whole_dropout:
                valid[:] = False
            else:
                valid &= np.random.random(self.slots) >= self.slot_dropout
                if self.truncate_recent:
                    keep = int(np.random.randint(self.slots + 1))
                    valid[np.flatnonzero(valid)[keep:]] = False
        return rows, torch.from_numpy(valid.copy())


class CausalObservationBuffer:
    """Store head-camera RGB frames within one explicitly identified episode.

    ``append`` automatically clears storage when the episode identifier changes.
    Timestamps are seconds on a monotonic episode clock. ``history`` chooses
    observations nearest each past slot and returns zero-filled invalid slots.
    Calling ``reset`` is required when reusing an episode identifier.
    """

    def __init__(self, slots: int = 20, interval_s: float = 1.0, *, tolerance_s: float = 0.05):
        self.selector = PastFrameSelector(slots=slots, interval_s=interval_s, jitter_s=0)
        if not math.isfinite(tolerance_s) or tolerance_s < 0:
            raise ValueError("History tolerance must be finite and nonnegative")
        self.tolerance_s = tolerance_s
        self.reset()

    def reset(self):
        self.episode_id = None
        self.start_time = None
        self.frames = deque()

    def append(self, frame: torch.Tensor, timestamp: float, *, episode_id):
        if episode_id is None or not math.isfinite(timestamp):
            raise ValueError("An episode identifier and finite timestamp are required")
        if frame.ndim != 3 or frame.shape[0] != 3 or not torch.isfinite(frame).all():
            raise ValueError("Observation frames must be finite CHW RGB tensors")
        if self.episode_id != episode_id:
            self.reset()
            self.episode_id, self.start_time = episode_id, float(timestamp)
        if self.frames and timestamp <= self.frames[-1][0]:
            raise ValueError("Observation timestamps must increase within an episode")
        if self.frames and frame.shape != self.frames[-1][1].shape:
            raise ValueError("Observation shape changed within an episode")
        self.frames.append((float(timestamp), frame.detach().clone()))
        cutoff = timestamp - self.selector.slots * self.selector.interval_s - self.tolerance_s
        while len(self.frames) > 1 and self.frames[1][0] < cutoff:
            self.frames.popleft()

    def history(self, current_time: float):
        if not self.frames or not math.isfinite(current_time):
            raise ValueError("Append a current observation before requesting history")
        if current_time < self.frames[-1][0]:
            raise ValueError("History cannot be requested before the latest observation")
        prototype = self.frames[-1][1]
        result = prototype.new_zeros((self.selector.slots, *prototype.shape))
        valid = torch.zeros(self.selector.slots, dtype=torch.bool, device=prototype.device)
        candidates = [(time, frame) for time, frame in self.frames if time < current_time]
        for slot in range(self.selector.slots):
            target = current_time - (slot + 1) * self.selector.interval_s
            if target < self.start_time or not candidates:
                continue
            timestamp, frame = min(candidates, key=lambda item: abs(item[0] - target))
            if abs(timestamp - target) <= self.tolerance_s:
                result[slot] = frame
                valid[slot] = True
        return {"history_video": result, "history_valid": valid}
