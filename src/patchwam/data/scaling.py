"""Feature scaling adapted from the attributed data processing implementation.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Yuyang "Alice.L"
See LICENSE-MIT and PROVENANCE.json in this directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def read_statistics(path: str | Path) -> dict[str, Any]:
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            try:
                return torch.tensor(value, dtype=torch.float32)
            except (ValueError, TypeError):
                return [convert(item) for item in value]
        return value

    return convert(json.loads(Path(path).read_text(encoding="utf-8")))


def write_statistics(stats: dict[str, Any], path: str | Path) -> None:
    def convert(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    Path(path).write_text(json.dumps(convert(stats), indent=2), encoding="utf-8")


class FeatureScaler:
    """Invertible affine map with the original clipping and constant-field rules."""

    def __init__(self, statistics: dict[str, Any], mode: str = "q01/q99"):
        self.statistics = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in statistics.items()}
        if mode == "z-score":
            self.scale = (self.statistics["std"] + 1e-8).reciprocal()
            self.offset = -self.statistics["mean"] * self.scale
        else:
            if mode in {"min/max", "q01/q99"}:
                low_key, high_key = mode.split("/")
                low, high = self.statistics[low_key], self.statistics[high_key]
            else:
                low_value, high_value = map(float, mode.split("/"))
                template = next(iter(self.statistics.values()))
                low, high = torch.full_like(template, low_value), torch.full_like(template, high_value)
            width = high - low
            constant = width < 1e-4
            self.scale = 2 / torch.where(constant, torch.full_like(width, 2), width)
            self.offset = torch.where(constant, -low, -1 - low * self.scale)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return (value * self.scale.to(value) + self.offset.to(value)).clamp(-5, 5)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.offset.to(value)) / self.scale.to(value)


class RobotFeatureCodec:
    """Per-field and optional per-embodiment statistics for actions and states."""

    def __init__(self, shape_meta, statistics, mode="q01/q99", exceptions=None, stepwise=False):
        self.shape_meta = shape_meta
        self.statistics = statistics
        self.tables = {}
        groups = statistics["embodiments"] if statistics.get("type") == "per_embodiment" else {"default": statistics}
        for embodiment, group in groups.items():
            self.tables[embodiment] = {}
            for field in ("action", "state"):
                for meta in shape_meta[field]:
                    key = meta["key"]
                    prefix = "stepwise_" if field == "action" and stepwise else "global_"
                    stats = group[field][key]
                    selected = {name.removeprefix(prefix): item for name, item in stats.items() if name.startswith(prefix)}
                    if not selected:
                        selected = stats
                    selected_mode = (exceptions or {}).get(field, {}).get(key, mode)
                    self.tables[embodiment][(field, key)] = FeatureScaler(selected, selected_mode)

    def _table(self, embodiment):
        if embodiment is None and len(self.tables) == 1:
            return next(iter(self.tables.values()))
        if embodiment not in self.tables:
            raise KeyError(f"Missing feature statistics for embodiment {embodiment!r}")
        return self.tables[embodiment]

    def encode(self, fields, embodiment=None):
        table = self._table(embodiment)
        return {field: {key: table[(field, key)].encode(value) for key, value in values.items()} for field, values in fields.items()}

    def decode(self, fields, embodiment=None):
        table = self._table(embodiment)
        return {field: {key: table[(field, key)].decode(value) for key, value in values.items()} for field, values in fields.items()}
