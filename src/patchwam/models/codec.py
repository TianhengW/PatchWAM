"""Parameter-free action coordinates in the visual token space."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RepeatedActionCodec:
    """Repeat normalized actions into tokens; decode by group averaging.

    The data processor handles normalization; the codec has no learned weights.
    """

    action_dim: int
    token_dim: int = 128
    scale: float = 1.0

    def __post_init__(self):
        if not 0 < self.action_dim <= self.token_dim:
            raise ValueError("action_dim must be positive and no greater than token_dim")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")

    @property
    def copies(self) -> int:
        return self.token_dim // self.action_dim

    @property
    def occupied(self) -> int:
        return self.action_dim * self.copies

    def encode(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} action coordinates")
        expanded = actions.unsqueeze(-1).expand(*actions.shape, self.copies)
        result = actions.new_zeros(*actions.shape[:-1], self.token_dim)
        result[..., :self.occupied] = expanded.flatten(-2) * self.scale
        return result

    def decode(self, tokens: Tensor) -> Tensor:
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected {self.token_dim} token coordinates")
        groups = tokens[..., :self.occupied].unflatten(-1, (self.action_dim, self.copies))
        return groups.mean(-1) / self.scale

    def coordinate_validity(self, valid_dimensions: Tensor) -> Tensor:
        """Expand valid dimensions; keep fill coordinates supervised."""
        if valid_dimensions.shape[-1] != self.action_dim:
            raise ValueError("Dimension mask does not match action_dim")
        result = torch.ones(
            *valid_dimensions.shape[:-1], self.token_dim,
            device=valid_dimensions.device, dtype=torch.bool,
        )
        result[..., :self.occupied] = valid_dimensions.bool().unsqueeze(-1).expand(
            *valid_dimensions.shape, self.copies,
        ).flatten(-2)
        return result
