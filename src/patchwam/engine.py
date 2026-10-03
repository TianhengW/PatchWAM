# SPDX-License-Identifier: Apache-2.0
"""Distributed optimization and restartable state for the patch flow objective."""

import json
import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Sampler


@dataclass
class RunSettings:
    output_dir: str = "runs/patchwam"
    epochs: int = 5
    max_updates: int | None = None
    batch_size: int = 4
    accumulation: int = 2
    global_batch_size: int | None = None
    model_config_hash: str | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_updates: int | None = None
    warmup_fraction: float = 0.05
    minimum_lr_fraction: float = 0.01
    precision: str = "bf16"
    cpu: bool = False
    workers: int = 8
    gradient_clip: float = 1.0
    checkpoint_every: int = 5000
    log_every: int = 10
    seed: int = 42
    drop_last: bool = True

    def validate(self):
        for key in ("epochs", "batch_size", "accumulation", "checkpoint_every", "log_every"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.max_updates is not None and self.max_updates < 1:
            raise ValueError("max_updates must be positive")
        if self.learning_rate <= 0 or self.workers < 0 or (self.warmup_updates is not None and self.warmup_updates < 0):
            raise ValueError("Invalid optimizer or loader settings")
        if len(self.betas) != 2 or any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("AdamW betas must be in [0,1)")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        if not 0 <= self.minimum_lr_fraction <= 1:
            raise ValueError("minimum_lr_fraction must be in [0,1]")


class EpochPermutation(Sampler):
    """The epoch number determines shuffling, so restarting does not change sample order."""

    def __init__(self, dataset, seed: int):
        self.length, self.seed, self.epoch = len(dataset), seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.length, generator=generator).tolist())


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


class OptimizationRun:
    def __init__(self, model, dataset, settings: RunSettings):
        settings.validate()
        self.settings = settings
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        self.accelerator = Accelerator(
            mixed_precision=settings.precision,
            gradient_accumulation_steps=settings.accumulation,
            cpu=settings.cpu,
        )
        effective_batch = settings.batch_size * settings.accumulation * self.accelerator.num_processes
        if settings.global_batch_size is not None and effective_batch != settings.global_batch_size:
            raise ValueError(
                f"Global batch differs from recipe: {effective_batch} vs {settings.global_batch_size}",
            )
        self.directory = Path(settings.output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("The model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay,
            betas=tuple(settings.betas),
        )
        self.sampler = EpochPermutation(dataset, settings.seed)
        self.loader_generator = torch.Generator()
        loader = DataLoader(
            dataset, batch_size=settings.batch_size, sampler=self.sampler,
            num_workers=settings.workers, drop_last=settings.drop_last,
            worker_init_fn=seed_worker, persistent_workers=settings.workers > 0,
            pin_memory=torch.cuda.is_available(),
            generator=self.loader_generator,
        )
        self.model, self.optimizer, self.loader = self.accelerator.prepare(model, optimizer, loader)
        from .checkpoints import register_compact_policy_state
        register_compact_policy_state(self.accelerator)
        if not len(self.loader):
            raise ValueError("The dataset is too small for this batch size and drop_last setting")
        epoch_budget = math.ceil(len(self.loader) / settings.accumulation) * settings.epochs
        self.total_updates = min(settings.max_updates, epoch_budget) if settings.max_updates else epoch_budget
        warmup = settings.warmup_updates
        if warmup is None:
            warmup = int(self.total_updates * settings.warmup_fraction)
        warmup = min(warmup, self.total_updates - 1)
        fingerprint = getattr(dataset, "fingerprint", None)
        fingerprint = fingerprint() if callable(fingerprint) else fingerprint
        self.dataset_signature = {
            "type": type(dataset).__module__ + "." + type(dataset).__qualname__,
            "length": len(dataset), "fingerprint": fingerprint,
        }
        continuation = asdict(settings)
        for key in ("output_dir", "checkpoint_every", "log_every"):
            continuation.pop(key)
        self.run_signature = hashlib.sha256(json.dumps({
            "settings": continuation, "dataset": self.dataset_signature,
            "model": self._model_signature(model),
        }, sort_keys=True).encode()).hexdigest()

        def rate(update):
            if warmup and update < warmup:
                start = 1 / warmup
                return start + (1 - start) * update / warmup
            progress = (update - warmup) / max(
                1, self.total_updates - warmup,
            )
            cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return settings.minimum_lr_fraction + (1 - settings.minimum_lr_fraction) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, rate)
        self.accelerator.register_for_checkpointing(self.scheduler)
        self.updates = self.start_epoch = self.start_batch = 0
        self.last_checkpoint = None
        if self.accelerator.is_main_process:
            _write_json(self.directory / "settings.json", asdict(settings))

    def restore(self, directory: str | Path):
        directory = Path(directory)
        metadata = json.loads((directory / "cursor.json").read_text())
        if metadata["world_size"] != self.accelerator.num_processes:
            raise ValueError("Exact continuation requires the same world size")
        if metadata["accumulation"] != self.settings.accumulation:
            raise ValueError("Exact continuation requires the same gradient accumulation")
        if metadata["total_updates"] != self.total_updates:
            raise ValueError("Exact continuation requires the same scheduler duration")
        if metadata.get("run_signature") != self.run_signature:
            raise ValueError("Continuation requires unchanged batch, optimizer, seed, and dataset settings")
        self.accelerator.load_state(str(directory))
        self.updates = metadata["updates"]
        self.start_epoch, self.start_batch = metadata["epoch"], metadata["next_batch"]
        self.last_checkpoint = str(directory)

    @staticmethod
    def _model_signature(model):
        policy = getattr(model, "policy", model)
        contract = {
            "type": type(model).__module__ + "." + type(model).__qualname__,
            "parameters": [(name, list(value.shape), str(value.dtype), value.requires_grad)
                           for name, value in model.named_parameters()],
        }
        if hasattr(policy, "codec"):
            contract["codec"] = asdict(policy.codec)
            contract["flow"] = {"shift": policy.flow.shift, "weight_normalizer": policy.flow.weight_normalizer}
            contract["objective"] = [policy.video_weight, policy.action_weight, policy.isolate_actions]
        return contract

    def checkpoint(self, epoch: int, next_batch: int):
        final = self.directory / f"step_{self.updates:07d}"
        pending = self.directory / f".step_{self.updates:07d}.pending"
        conflict = torch.tensor(
            int(self.accelerator.is_main_process and (final.exists() or pending.exists())),
            device=self.accelerator.device,
        )
        if self.accelerator.reduce(conflict, reduction="sum").item():
            raise FileExistsError(f"Checkpoint destination already exists: {final}")
        if self.accelerator.is_main_process:
            pending.mkdir()
        self._synchronize()
        self.accelerator.save_state(str(pending), safe_serialization=True)
        self._synchronize()
        if self.accelerator.is_main_process:
            _write_json(pending / "cursor.json", {
                "updates": self.updates, "epoch": epoch, "next_batch": next_batch,
                "world_size": self.accelerator.num_processes,
                "accumulation": self.settings.accumulation,
                "total_updates": self.total_updates,
                "run_signature": self.run_signature,
                "dataset_signature": self.dataset_signature,
            })
            os.replace(pending, final)
        self._synchronize()
        self.last_checkpoint = str(final)
        return final

    def _synchronize(self):
        if self.accelerator.device.type == "cpu" and torch.distributed.is_initialized():
            # Device IDs force an MPS barrier on macOS in Accelerate 1.12 / Torch 2.7.
            torch.distributed.barrier()
        else:
            self.accelerator.wait_for_everyone()

    def _log(self, values):
        if self.accelerator.is_main_process:
            record = {"update": self.updates, **values}
            with (self.directory / "metrics.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    def train(self):
        self.model.train()
        # Keep asset encoders frozen even when the trainable policy enters train mode.
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "freeze_asset_encoders"):
            unwrapped.freeze_asset_encoders()
        self.optimizer.zero_grad(set_to_none=True)
        if self.updates >= self.total_updates:
            return self.last_checkpoint
        for epoch in range(self.start_epoch, self.settings.epochs):
            self.loader_generator.manual_seed(self.settings.seed + epoch)
            self.sampler.set_epoch(epoch)
            if hasattr(self.loader, "set_epoch"):
                self.loader.set_epoch(epoch)
            skip = self.start_batch if epoch == self.start_epoch else 0
            epoch_loader = self.accelerator.skip_first_batches(self.loader, skip) if skip else self.loader
            for batch_number, batch in enumerate(epoch_loader, start=skip):
                with self.accelerator.accumulate(self.model):
                    result = self.model(batch)
                    loss = result["loss"]
                    invalid = (~torch.isfinite(loss.detach())).to(torch.int32)
                    if self.accelerator.reduce(invalid, reduction="sum").item():
                        raise FloatingPointError("Non-finite loss on at least one rank")
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        if self.settings.gradient_clip:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.settings.gradient_clip,
                            )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if not self.accelerator.sync_gradients or self.accelerator.optimizer_step_was_skipped:
                        continue
                    self.scheduler.step()
                    self.updates += 1
                    if self.updates == 1 or self.updates % self.settings.log_every == 0:
                        values = {}
                        for name, value in result.items():
                            if torch.is_tensor(value) and value.numel() == 1:
                                values[name] = self.accelerator.reduce(
                                    value.detach().float(), reduction="mean",
                                ).item()
                        values["learning_rate"] = self.scheduler.get_last_lr()[0]
                        self._log(values)
                    next_epoch, next_batch = epoch, batch_number + 1
                    if next_batch == len(self.loader):
                        next_epoch, next_batch = epoch + 1, 0
                    finished = self.updates >= self.total_updates
                    if finished or self.updates % self.settings.checkpoint_every == 0:
                        self.checkpoint(next_epoch, next_batch)
                    if finished:
                        return self.last_checkpoint
        # A skipped fp16 optimizer step can exhaust the epoch budget early.
        final = self.directory / f"step_{self.updates:07d}"
        if not final.exists():
            self.checkpoint(self.settings.epochs, 0)
        return self.last_checkpoint
