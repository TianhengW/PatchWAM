# Training and checkpoints

The `train` entry point trains the full model selected by its YAML configuration.
First complete [installation](getting_started.md), [model assets](models.md), and
[dataset/normalization setup](data.md). Choose the matched
[training recipe](training_recipes.md), then run commands below from the repository
root. Synthetic examples use the tiny model; benchmark recipes use the configured
FLUX.2 backbone.

- [Configuration overrides](#configuration-overrides)
- [Single-node GPU launch](#single-node-gpu-launch)
- [Multi-node GPU launch](#multi-node-gpu-launch)
- [Full-model training validation](#full-model-training-validation)
- [Checkpoints and continuation](#checkpoints-and-continuation)

## Configuration overrides

The CLI reads a YAML file and accepts dotted `key=value` overrides. For example:

```bash
patchwam train --config configs/smoke.yaml \
  training.output_dir=runs/custom-smoke \
  training.max_updates=4 \
  training.epochs=2 \
  training.log_every=1
```

Model, data, and training settings can all be overridden. Changing the camera
layout, label transforms, normalization, or sample selection changes the experiment.

## Single-node GPU launch

For an eight-GPU node, this RobotWin example preserves global batch 256:

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 \
  -m patchwam.cli train --config configs/robotwin.yaml \
  training.batch_size=4 \
  training.accumulation=8 \
  training.output_dir=runs/robotwin-8gpu
```

The effective batch is:

```text
global_batch = world_size × training.batch_size × training.accumulation
```

The engine checks this against `training.global_batch_size` when that value is
configured. The original RobotWin recipe uses 32 processes × batch 4 × accumulation 2.
Preserving global batch on fewer processes is useful for setup, but does not establish
numerical equivalence with the original distributed run.

## Multi-node GPU launch

Prepare an Accelerate DDP configuration for your cluster. For example, 32 total
processes across four nodes with eight GPUs each:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 32
num_machines: 4
machine_rank: 0
main_process_ip: 10.0.0.1
main_process_port: 29500
same_network: true
rdzv_backend: static
use_cpu: false
```

Save it as `/path/to/ddp.yaml`, replace the main-node address, and run the following
on **every node**, setting `NODE_RANK` to 0, 1, 2, or 3:

```bash
export NODE_RANK=0
accelerate launch --config_file /path/to/ddp.yaml \
  --machine_rank "$NODE_RANK" \
  -m patchwam.cli train --config configs/robotwin.yaml \
  training.output_dir=/shared/runs/robotwin
```

Use shared output storage and consistent asset/dataset paths across processes.
The eager model loader binds `device: cuda` to each process's local GPU. Explicit
CUDA indices must match that process's device.

For the default RoboCasa or C2R configuration, set the launcher to 16 total
processes and choose the corresponding configuration. With two eight-GPU nodes,
set `num_machines: 2` and use `NODE_RANK=0` or `NODE_RANK=1` on the respective node:

```bash
accelerate launch --config_file /path/to/ddp-16.yaml \
  --machine_rank "$NODE_RANK" \
  -m patchwam.cli train --config configs/robotwin_c2r.yaml \
  training.output_dir=/shared/runs/robotwin-c2r
```

Single-process and CPU/GPU DDP execution are supported. Sharded model/optimizer
launch modes require additional implementation and are rejected by the current engine.
Multi-node full-size GPU execution still requires validation on the target cluster.

## Full-model training validation

Validate the actual configured model and dataset on an allocated GPU before a
long training run. Use a fresh output directory:

```bash
patchwam validate --config configs/robotwin.yaml \
  --output runs/validate-robotwin --updates 2
```

This keeps the model, preprocessing, and training objective. It uses batch 1,
accumulation 1, workers 0, a short epoch/update budget, and per-update checkpoints.
The ordinary recipe is not overwritten. The reduced update budget also changes
the validation learning-rate schedule; this check is separate from a paper training run.

The validator runs an uninterrupted baseline and a separate process resumed
from update 1. It checks finite training metrics, successful-update cursors, matching
run signatures, and equality of all exported live/EMA safetensors. It enables
deterministic algorithms and math SDPA and defaults to exact equality; unsupported
deterministic operations fail explicitly. Use `--atol` / `--rtol` only with
recorded reasons.

For a single-node multi-GPU check:

```bash
patchwam validate --config configs/robotwin.yaml \
  --output runs/validate-robotwin-ddp --updates 2 --num-processes 8
```

Invoke the validator directly inside the allocation; it launches its own workers.
`validation.json`, `baseline.log`, and `resumed.log` record the result. The full
4B model and optimizer still require their ordinary parameter-state memory;
a small batch does not reduce parameter/optimizer storage.

The local CPU equivalent checks this orchestration without loading real weights:

```bash
patchwam validate --config configs/smoke.yaml \
  --output runs/validate-cpu --allow-cpu
```

Single-process and two-process CPU training/resume checks pass. Real 4B GPU
execution and simulator success rates remain separate verification stages.
See [closed-loop evaluation](evaluation.md) for benchmark execution.

## Checkpoints and continuation

### Run outputs

The output directory records:

```text
run_directory/
├── configuration.yaml       # Resolved configuration for YAML launches
├── settings.json            # Optimization settings
├── metrics.jsonl            # Logged optimizer-update metrics
└── step_0005000/
    ├── policy_0.safetensors # Trainable policy for the frozen-encoder wrapper
    ├── optimizer.bin
    ├── custom_checkpoint_0.pkl
    ├── random_states_0.pkl
    └── cursor.json
```

The scheduler is registered as custom checkpoint state. Additional ranks have
their own RNG-state files; mixed-precision scaler files may also be present.
The tiny model uses `model.safetensors` instead of `policy_0.safetensors`.
Frozen encoder weights are kept in the original model-asset locations.
With trained VL adapters, the native policy checkpoint also includes their weights.
EMA-enabled runs additionally save `ema_policy.safetensors` and registered averaging
state; select the EMA file explicitly when loading inference weights.

Checkpoints are written to a temporary pending directory and published after
state saving finishes. A complete checkpoint has `cursor.json`. Existing checkpoint
destinations are protected against overwrite. If skipped fp16 steps exhaust the
epoch budget, an `_exhausted` checkpoint records the actual final cursor and update count.

### Resume complete training state

Keep the same launcher and experiment configuration, then add `--resume`:

```bash
accelerate launch --config_file /path/to/ddp.yaml \
  --machine_rank "$NODE_RANK" \
  -m patchwam.cli train --config configs/robotwin.yaml \
  --resume /shared/runs/robotwin/step_0005000 \
  training.output_dir=/shared/runs/robotwin
```

Resume restores model, optimizer, scheduler, RNG states, the epoch/batch cursor,
and EMA averaging state when enabled.
It requires the same world size, accumulation, scheduler duration, model/asset
contract, dataset fingerprint, and relevant training settings. To start a changed
experiment, initialize weights into a fresh run instead.
The saved `cursor.json` carries the run signature, dataset signature, update count,
world size, and continuation cursor used to check that contract before loading.

Exact CPU continuation is verified with `training.workers=0`. Prefetched asynchronous
worker state is not saved, so mid-epoch continuation with multiple workers can
produce different augmentation draws.

### Initialize policy weights

`--initialize` loads complete native policy weights into a new optimization run:

```bash
accelerate launch --config_file /path/to/ddp.yaml \
  --machine_rank "$NODE_RANK" \
  -m patchwam.cli train --config configs/robotwin.yaml \
  --initialize /path/to/policy_0.safetensors \
  training.output_dir=/shared/runs/robotwin-initialized
```

The loader checks every parameter name, shape, and floating-point value before
changing the policy. It initializes the trainable transformer and state projection;
the frozen encoder assets must still be supplied separately. Optimizer, scheduler,
RNG, and EMA averaging states start fresh. `--initialize` and `--resume` are mutually exclusive.
Historical checkpoint formats require a separate validated conversion to the native schema.
