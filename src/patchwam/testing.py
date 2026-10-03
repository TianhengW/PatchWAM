# SPDX-License-Identifier: Apache-2.0
"""Small deterministic tensor examples for local verification without model downloads."""

import hashlib
import json

import torch
from torch.utils.data import Dataset


class TensorExamples(Dataset):
    def __init__(self, length=16, action_dim=14, proprio_dim=14, text_dim=32, horizon=16):
        self.length = length
        self.action_dim, self.proprio_dim = action_dim, proprio_dim
        self.text_dim, self.horizon = text_dim, horizon

    def __len__(self):
        return self.length

    def fingerprint(self):
        return hashlib.sha256(json.dumps(vars(self), sort_keys=True).encode()).hexdigest()

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(1000 + index)
        def sample(*shape):
            return torch.randn(*shape, generator=generator)
        return {
            "reference_tokens": sample(4, 128), "future_tokens": sample(4, 128),
            "text_tokens": sample(3, self.text_dim),
            "action": sample(self.horizon, self.action_dim),
            "proprio": sample(self.horizon, self.proprio_dim),
            "action_is_pad": torch.zeros(self.horizon, dtype=torch.bool),
            "action_dim_is_pad": torch.zeros(self.action_dim, dtype=torch.bool),
        }
