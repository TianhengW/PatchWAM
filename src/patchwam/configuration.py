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
