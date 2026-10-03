"""Randomly initialized transformer for CPU checks, not benchmark scores."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _four_axis_rotation(query: Tensor, key: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    per_axis = query.shape[-1] // 4
    frequencies = torch.exp(
        -math.log(2000) * torch.arange(0, per_axis, 2, device=query.device).float() / per_axis,
    )
    phase = (coordinates.float().unsqueeze(-1) * frequencies).flatten(-2).unsqueeze(1)
    cosine, sine = phase.cos(), phase.sin()

    def rotate(value):
        pairs = value.float().unflatten(-1, (-1, 2))
        a, b = pairs.unbind(-1)
        return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2).to(value)

    return rotate(query), rotate(key)


class ConditionalTokenBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, width * 3)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, states: Tensor, time: Tensor, coordinates: Tensor, visible: Tensor) -> Tensor:
        modulation = self.modulation(F.silu(time))
        offset, scale, gate = (modulation.unsqueeze(1) if time.ndim == 2 else modulation).chunk(3, -1)
        modulated = self.norm(states) * (1 + scale) + offset
        qkv = self.qkv(modulated).unflatten(-1, (3, self.heads, -1))
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = _four_axis_rotation(query, key, coordinates)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=visible)
        attended = attended.transpose(1, 2).flatten(-2)
        states = states + torch.tanh(gate) * self.mix(attended)
        return states + self.ff(self.ff_norm(states))


class SmallJointTransformer(nn.Module):
    def __init__(self, *, token_dim: int = 128, text_dim: int = 32, width: int = 64, heads: int = 4, depth: int = 2):
        super().__init__()
        if width % (heads * 8) != 0 or depth < 1:
            raise ValueError("width / heads must be divisible by eight, and depth positive")
        self.token_dim, self.text_dim = token_dim, text_dim
        self.representation_dim, self.representation_layers = width, depth
        self.token_input = nn.Linear(token_dim, width, bias=False)
        self.text_input = nn.Linear(text_dim, width, bias=False)
        self.time_input = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(ConditionalTokenBlock(width, heads) for _ in range(depth))
        self.output_norm = nn.LayerNorm(width)
        self.token_output = nn.Linear(width, token_dim, bias=False)

    def forward(
        self, reference: Tensor, noisy: Tensor, sigma: Tensor, context: Tensor,
        *, reference_ids: Tensor, noisy_ids: Tensor, context_ids: Tensor, visibility: Tensor,
        token_sigma: Tensor | None = None, representation_layer: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if representation_layer is not None and not 1 <= representation_layer <= self.representation_layers:
            raise ValueError("representation_layer must be a 1-based transformer layer")
        text_length, reference_length = context.shape[1], reference.shape[1]
        states = torch.cat((self.text_input(context), self.token_input(torch.cat((reference, noisy), 1))), 1)
        coordinates = torch.cat((context_ids, reference_ids, noisy_ids), 1)
        frequencies = torch.exp(-math.log(10000) * torch.arange(32, device=sigma.device).float() / 32)
        if token_sigma is not None:
            if token_sigma.shape != noisy.shape[:2]:
                raise ValueError("token_sigma must match the noisy token shape [B,N]")
            sigma = torch.cat((sigma[:, None].expand(-1, text_length + reference_length), token_sigma), 1)
        angles = sigma.float()[..., None] * 1000 * frequencies
        time = self.time_input(torch.cat((angles.cos(), angles.sin()), -1).to(states))
        features = None
        for index, block in enumerate(self.blocks, 1):
            states = block(states, time, coordinates, visibility)
            if index == representation_layer:
                features = states[:, text_length + reference_length:]
        velocity = self.token_output(self.output_norm(states[:, text_length + reference_length:]))
        return velocity if representation_layer is None else (velocity, features)
