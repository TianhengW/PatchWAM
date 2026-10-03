# SPDX-License-Identifier: Apache-2.0
"""Configuration loading, without process-global resolvers or hidden environment files."""

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


def read_configuration(path: str | Path, overrides: list[str] | None = None):
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def construct(specification):
    return instantiate(specification, _convert_="all")


def runtime_model_specification(specification, device):
    """Bind eager CUDA asset loading to the process device, preserving saved config."""
    import torch

    bound = OmegaConf.create(OmegaConf.to_container(specification, resolve=True))
    if "device" not in bound:
        return bound
    requested = torch.device(bound.device)
    runtime = torch.device(device)
    if requested.type == "cuda":
        if runtime.type != "cuda":
            raise ValueError("The model requests CUDA but the training runtime uses " + str(runtime))
        if requested.index is not None and requested != runtime:
            raise ValueError(
                f"Model device {requested} differs from process device {runtime}; use device=cuda"
            )
        bound.device = str(runtime)
    return bound
