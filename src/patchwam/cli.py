# SPDX-License-Identifier: Apache-2.0
"""Explicit command-line entry points for training and local smoke verification."""

import argparse
import json
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="patchwam")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train from a YAML configuration")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", help="A complete PatchWAM accelerator state directory")
    train.add_argument("--initialize", help="Explicitly map full single-stream research weights")
    train.add_argument("overrides", nargs="*", help="Dotted key=value settings")
    smoke = commands.add_parser("smoke", help="Run a small CPU optimization with a tiny backbone")
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()

    from .engine import OptimizationRun, RunSettings
    if args.command == "train":
        from .configuration import construct, read_configuration
        from omegaconf import OmegaConf
        config = read_configuration(args.config, args.overrides)
        import torch
        torch.manual_seed(int(config.training.get("seed", 42)))
        model = construct(config.model)
        if args.initialize and args.resume:
            parser.error("Choose either weight initialization or training-state resume")
        if args.initialize:
            from .checkpoints import import_research_weights
            print(json.dumps(import_research_weights(model, args.initialize)))
        dataset = construct(config.data)
        settings = RunSettings(**OmegaConf.to_container(config.training, resolve=True))
        settings.model_config_hash = hashlib.sha256(OmegaConf.to_yaml(config.model).encode()).hexdigest()
        run = OptimizationRun(model, dataset, settings)
        if args.resume:
            run.restore(args.resume)
        if run.accelerator.is_main_process:
            Path(settings.output_dir, "configuration.yaml").write_text(OmegaConf.to_yaml(config))
    else:
        import torch
        from .models import make_tiny_policy
        from .testing import TensorExamples
        torch.manual_seed(42)
        model = make_tiny_policy()
        settings = RunSettings(
            output_dir=args.output, epochs=args.updates,
            max_updates=args.updates, batch_size=2, accumulation=1,
            precision="no", cpu=True, workers=0, checkpoint_every=args.updates, log_every=1,
        )
        run = OptimizationRun(model, TensorExamples(), settings)
    checkpoint = run.train()
    if run.accelerator.is_main_process:
        print(json.dumps({"updates": run.updates, "checkpoint": checkpoint}))


if __name__ == "__main__":
    main()
