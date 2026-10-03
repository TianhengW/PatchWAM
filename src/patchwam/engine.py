# SPDX-License-Identifier: Apache-2.0
"""Distributed optimization and restartable state for the patch flow objective."""

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
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
    ema_decay: float | None = None
    ema_warmup_updates: int = 0

    def validate(self):
        for key in ("epochs", "batch_size", "accumulation", "checkpoint_every", "log_every"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.max_updates is not None and self.max_updates < 1:
            raise ValueError("max_updates must be positive")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or self.workers < 0
            or (self.warmup_updates is not None and self.warmup_updates < 0)
        ):
            raise ValueError("Invalid optimizer or loader settings")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.weight_decay, self.gradient_clip)
        ):
            raise ValueError("Weight decay and gradient clipping must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("AdamW betas must be in [0,1)")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        if not 0 <= self.minimum_lr_fraction <= 1:
            raise ValueError("minimum_lr_fraction must be in [0,1]")
        if self.ema_decay is not None and (
            not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1
        ):
            raise ValueError("EMA decay must be finite and in [0,1)")
        if type(self.ema_warmup_updates) is not int or self.ema_warmup_updates < 0:
            raise ValueError("EMA warmup updates must be a nonnegative integer")
        if self.ema_decay is None and self.ema_warmup_updates:
            raise ValueError("EMA warmup requires enabled EMA")


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
        if self.accelerator.distributed_type not in (
            DistributedType.NO,
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
        ):
            raise ValueError("This engine supports single-process execution and CPU/GPU DDP")
        effective_batch = (
            settings.batch_size * settings.accumulation * self.accelerator.num_processes
        )
        if settings.global_batch_size is not None and effective_batch != settings.global_batch_size:
            raise ValueError(
                f"Global batch differs from recipe: {effective_batch} vs {settings.global_batch_size}",
            )
        self.directory = Path(settings.output_dir)
        self._on_main(
            "creating the output directory",
            lambda: self.directory.mkdir(parents=True, exist_ok=True),
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("The model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=tuple(settings.betas),
        )
        self.sampler = EpochPermutation(dataset, settings.seed)
        self.loader_generator = torch.Generator()
        loader = DataLoader(
            dataset,
            batch_size=settings.batch_size,
            sampler=self.sampler,
            num_workers=settings.workers,
            drop_last=settings.drop_last,
            worker_init_fn=seed_worker,
            persistent_workers=settings.workers > 0,
            pin_memory=torch.cuda.is_available(),
            generator=self.loader_generator,
        )
        self.model, self.optimizer, self.loader = self.accelerator.prepare(model, optimizer, loader)
        from .checkpoints import register_compact_policy_state

        register_compact_policy_state(self.accelerator)
        if not len(self.loader):
            raise ValueError("The dataset is too small for this batch size and drop_last setting")
        epoch_budget = math.ceil(len(self.loader) / settings.accumulation) * settings.epochs
        self.total_updates = (
            min(settings.max_updates, epoch_budget) if settings.max_updates else epoch_budget
        )
        warmup = settings.warmup_updates
        if warmup is None:
            warmup = int(self.total_updates * settings.warmup_fraction)
        warmup = min(warmup, self.total_updates - 1)
        fingerprint = getattr(dataset, "fingerprint", None)
        fingerprint = fingerprint() if callable(fingerprint) else fingerprint
        self.dataset_signature = {
            "type": type(dataset).__module__ + "." + type(dataset).__qualname__,
            "length": len(dataset),
            "fingerprint": fingerprint,
        }
        continuation = asdict(settings)
        for key in ("output_dir", "checkpoint_every", "log_every"):
            continuation.pop(key)
        self.run_signature = hashlib.sha256(
            json.dumps(
                {
                    "settings": continuation,
                    "dataset": self.dataset_signature,
                    "model": self._model_signature(model),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        def rate(update):
            if warmup and update < warmup:
                start = 1 / warmup
                return start + (1 - start) * update / warmup
            progress = (update - warmup) / max(
                1,
                self.total_updates - warmup,
            )
            cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return settings.minimum_lr_fraction + (1 - settings.minimum_lr_fraction) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, rate)
        self.accelerator.register_for_checkpointing(self.scheduler)
        self.ema = None
        unwrapped = self.accelerator.unwrap_model(self.model)
        if settings.ema_decay is not None:
            from .averaging import PolicyWeightAverage

            self.ema = PolicyWeightAverage(
                unwrapped,
                decay=settings.ema_decay,
                warmup_updates=settings.ema_warmup_updates,
            )
            self.accelerator.register_for_checkpointing(self.ema)
        policy = getattr(unwrapped, "policy", unwrapped)
        if hasattr(policy, "set_optimizer_updates"):
            policy.set_optimizer_updates(0)
        if getattr(policy, "self_flow_variant", 0):
            if self.ema is None:
                raise ValueError("Self-Flow training requires EMA")
            prefix = "policy.denoiser." if policy is not unwrapped else "denoiser."
            policy.attach_self_flow_teacher(
                lambda batch, **kwargs: self.ema.call(
                    unwrapped,
                    batch,
                    parameter_prefixes=(prefix,),
                    submodule="policy" if policy is not unwrapped else None,
                    **kwargs,
                )
            )
        self.updates = self.start_epoch = self.start_batch = 0
        self.last_checkpoint = None

        def write_settings():
            path = self.directory / "settings.json"
            if not path.exists():
                _write_json(path, asdict(settings))

        self._on_main("writing run settings", write_settings)

    def _propagate_failure(self, error, operation):
        failed = torch.tensor(int(error is not None), device=self.accelerator.device)
        if self.accelerator.reduce(failed, reduction="sum").item():
            if error is not None:
                raise error
            raise RuntimeError(f"Another process failed while {operation}")

    def _on_main(self, operation, action):
        error = None
        if self.accelerator.is_main_process:
            try:
                action()
            except Exception as caught:  # noqa: BLE001 - propagate failures before another rank waits
                error = caught
        self._propagate_failure(error, operation)

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
            raise ValueError(
                "Continuation requires unchanged batch, optimizer, seed, dataset, model, and assets"
            )
        cursor = [metadata["updates"], metadata["epoch"], metadata["next_batch"]]
        if any(type(value) is not int or value < 0 for value in cursor):
            raise ValueError("Invalid continuation cursor")
        if cursor[0] > self.total_updates or cursor[1] > self.settings.epochs:
            raise ValueError("Continuation cursor exceeds the training budget")
        if cursor[2] >= len(self.loader) or (cursor[1] == self.settings.epochs and cursor[2]):
            raise ValueError("Continuation cursor exceeds the epoch")
        if cursor[2] % self.settings.accumulation:
            raise ValueError("Continuation must start at an accumulation boundary")
        self.accelerator.load_state(str(directory))
        if self.ema is not None and self.ema.updates != metadata["updates"]:
            raise ValueError("EMA update count differs from the training continuation cursor")
        self.updates = metadata["updates"]
        unwrapped = self.accelerator.unwrap_model(self.model)
        policy = getattr(unwrapped, "policy", unwrapped)
        if hasattr(policy, "set_optimizer_updates"):
            policy.set_optimizer_updates(self.updates)
        self.start_epoch, self.start_batch = metadata["epoch"], metadata["next_batch"]
        self.last_checkpoint = str(directory)

    @staticmethod
    def _model_signature(model):
        policy = getattr(model, "policy", model)
        contract = {
            "type": type(model).__module__ + "." + type(model).__qualname__,
            "parameters": [
                (name, list(value.shape), str(value.dtype), value.requires_grad)
                for name, value in model.named_parameters()
            ],
            "assets": getattr(model, "asset_signature", None),
        }
        if hasattr(policy, "codec"):
            contract["codec"] = asdict(policy.codec)
            contract["flow"] = {
                "shift": policy.flow.shift,
                "weight_normalizer": policy.flow.weight_normalizer,
            }
            contract["objective"] = [
                policy.video_weight,
                policy.action_weight,
                policy.isolate_actions,
            ]
            contract["policy_features"] = {
                name: value
                for name, value in vars(policy).items()
                if name == "condition_dropout" or name.startswith("self_flow_")
            }
        feature_contract = getattr(model, "checkpoint_contract", None)
        if callable(feature_contract):
            contract["conditioning"] = feature_contract()
        return contract

    def checkpoint(self, epoch: int, next_batch: int, *, exhausted=False):
        name = f"step_{self.updates:07d}" + ("_exhausted" if exhausted else "")
        final = self.directory / name
        pending = self.directory / f".{name}.pending"
        conflict = torch.tensor(
            int(self.accelerator.is_main_process and (final.exists() or pending.exists())),
            device=self.accelerator.device,
        )
        if self.accelerator.reduce(conflict, reduction="sum").item():
            raise FileExistsError(f"Checkpoint destination already exists: {final}")
        self._on_main("creating the checkpoint directory", pending.mkdir)
        error = None
        try:
            self.accelerator.save_state(str(pending), safe_serialization=True)
        except Exception as caught:  # noqa: BLE001 - every rank must observe a failed state save
            error = caught
        self._propagate_failure(error, "saving checkpoint state")

        def publish():
            if self.ema is not None:
                from .checkpoints import save_policy_average

                save_policy_average(
                    self.ema,
                    self.accelerator.unwrap_model(self.model),
                    pending / "ema_policy.safetensors",
                )
            _write_json(
                pending / "cursor.json",
                {
                    "updates": self.updates,
                    "epoch": epoch,
                    "next_batch": next_batch,
                    "world_size": self.accelerator.num_processes,
                    "accumulation": self.settings.accumulation,
                    "total_updates": self.total_updates,
                    "run_signature": self.run_signature,
                    "dataset_signature": self.dataset_signature,
                    "exhausted": exhausted,
                },
            )
            os.replace(pending, final)

        self._on_main("publishing the checkpoint", publish)
        self.last_checkpoint = str(final)
        return final

    def _log(self, values):
        def write():
            record = {"update": self.updates, **values}
            with (self.directory / "metrics.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        self._on_main("writing training metrics", write)

    def train(self):
        self.model.train()
        # Keep asset encoders frozen even when the trainable policy enters train mode.
        unwrapped = self.accelerator.unwrap_model(self.model)
        policy = getattr(unwrapped, "policy", unwrapped)
        if hasattr(unwrapped, "freeze_asset_encoders"):
            unwrapped.freeze_asset_encoders()
        self.optimizer.zero_grad(set_to_none=True)
        if self.updates >= self.total_updates or self.start_epoch >= self.settings.epochs:
            return self.last_checkpoint
        for epoch in range(self.start_epoch, self.settings.epochs):
            self.loader_generator.manual_seed(self.settings.seed + epoch)
            self.sampler.set_epoch(epoch)
            if hasattr(self.loader, "set_epoch"):
                self.loader.set_epoch(epoch)
            skip = self.start_batch if epoch == self.start_epoch else 0
            epoch_loader = (
                self.accelerator.skip_first_batches(self.loader, skip) if skip else self.loader
            )
            group_metrics = {}
            group_count = 0
            for batch_number, batch in enumerate(epoch_loader, start=skip):
                with self.accelerator.accumulate(self.model):
                    result = self.model(batch)
                    loss = result["loss"]
                    invalid = (~torch.isfinite(loss.detach())).to(torch.int32)
                    if self.accelerator.reduce(invalid, reduction="sum").item():
                        raise FloatingPointError("Non-finite loss on at least one rank")
                    group_start = (
                        batch_number // self.settings.accumulation * self.settings.accumulation
                    )
                    group_size = min(self.settings.accumulation, len(self.loader) - group_start)
                    # Accelerate divides by the configured accumulation even for a short final group.
                    self.accelerator.backward(loss * (self.settings.accumulation / group_size))
                    for name, value in result.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            detached = value.detach().float()
                            group_metrics[name] = group_metrics.get(name, 0) + detached
                    group_count += 1
                    if self.accelerator.sync_gradients:
                        norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.settings.gradient_clip or float("inf"),
                        )
                        if self.accelerator.scaler is None:
                            invalid_gradient = (~torch.isfinite(norm)).to(torch.int32)
                            if self.accelerator.reduce(invalid_gradient, reduction="sum").item():
                                self.optimizer.zero_grad(set_to_none=True)
                                raise FloatingPointError("Non-finite gradient on at least one rank")
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if not self.accelerator.sync_gradients:
                        continue
                    values = {name: value / group_count for name, value in group_metrics.items()}
                    group_metrics, group_count = {}, 0
                    if self.accelerator.optimizer_step_was_skipped:
                        continue
                    self.scheduler.step()
                    self.updates += 1
                    if self.ema is not None:
                        # Gradient finiteness was checked before this successful optimizer step.
                        self.ema.update(
                            self.accelerator.unwrap_model(self.model), check_finite=False
                        )
                    if hasattr(policy, "set_optimizer_updates"):
                        policy.set_optimizer_updates(self.updates)
                    if self.updates == 1 or self.updates % self.settings.log_every == 0:
                        values = {
                            name: self.accelerator.reduce(value, reduction="mean").item()
                            for name, value in values.items()
                        }
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
        self.checkpoint(self.settings.epochs, 0, exhausted=True)
        return self.last_checkpoint
