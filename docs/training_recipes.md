# Training recipes

These development configurations describe the intended training contracts.
Full-size GPU execution and closed-loop reproduction remain unverified.
See [training and checkpoints](training.md) for launch, validation, and resume
commands; run those commands from the repository root.

Additional matched controls, VLM/history recipes, and three source-checked Self-Flow
variants are described in [`conditioning_and_training.md`](conditioning_and_training.md).
CFG and EMA are explicit training/inference options; they are not automatically
enabled in the ordinary benchmark configurations.

| Configuration | Action / state width | Composed image, H×W | Normalization | Per-process batch × accumulation | Intended global batch |
| --- | --- | --- | --- | --- | --- |
| [RobotWin](../configs/robotwin.yaml) | 14 / 14 | 288×256: high view above both wrist views | Z-score | 4 × 2 | 256 with 32 processes |
| [RobotWin C2R](../configs/robotwin_c2r.yaml) | 14 / 14 | 288×256: same geometry, appearance randomization | Z-score | 16 × 1 | 256 with 16 processes |
| [RoboCasa](../configs/robocasa.yaml) | 12 / 16 | 288×256: left agent view above right agent and wrist views | Z-score | 16 × 1 | 256 with 16 processes |
| [LIBERO](../configs/libero.yaml) | 7 / 8 | 224×448: front and wrist views side by side | Min/max | 8 × 1 | 8 × world size |
| [CPU smoke](../configs/smoke.yaml) | 14 / 14 | Synthetic encoded tokens | Synthetic tensors | 2 × 1 | 2 for one process |

Benchmark configurations use 16 future actions and endpoint image pairs. RobotWin uses
50Hz data; RoboCasa and LIBERO use 20Hz data. The 128D fixed codec consumes the
configured raw action dimension. Changing action dimensions, normalization,
camera geometry, or sampling filters defines a separate experiment.
Verify the selected dataset's `meta/info.json` and action semantics against the
recipe. The supplied RoboCasa configuration describes a specific 12D-action/16D-state
interface; other releases may use different transforms or layouts.

The common benchmark optimizer is AdamW with betas `(0.9, 0.95)`, weight decay `0.01`,
learning rate `0.0001`, 5% linear warmup, and a cosine schedule with a 1% rate
floor. The denoising objective uses a shared shifted sigma with shift 5 and
image/action weights 0.5/1. Benchmark configurations enable gradient checkpointing
and clip gradients at `1.0`.

RobotWin and RoboCasa run for five configured epochs; LIBERO uses 50. C2R uses
100 epochs with a cap of 150,000 optimizer updates. The actual update budget is
bounded by both the epoch budget and `training.max_updates`.
The 1% held-out split in RobotWin/RoboCasa configurations reserves data; the current
training CLI does not run a validation loop automatically.

RobotWin uses its non-idle frame ranges. The C2R recipe additionally selects
the first 50 episodes of each 550-episode block and applies photometric, style,
and Fourier appearance augmentation with probability 0.8. Background replacement
is disabled. It runs up to 150000 updates. Its `inference` section records the
intended evaluation protocol: 10 denoising steps and a 16-action replanning
horizon. The training CLI does not execute evaluation or apply these inference
settings. A direct API caller must pass `steps=10` to `sample_actions` and implement
the controller's replanning schedule; `patchwam evaluate` uses its separate
[evaluation configuration](evaluation.md). Other recipes keep their separately configured
sample sets and augmentation parameters.

Dataset roots, normalization statistics, non-idle filters, official model source,
and pretrained model assets are supplied through explicit configuration paths;
see [data setup](data.md) and [model assets](models.md).
The engine verifies the configured global batch before optimization.
Eager model loading binds `device: cuda` to each launched process's local CUDA
device. Explicit CUDA indices must match the selected process device.

Weight-only initialization accepts complete native policy safetensors parameters.
Every parameter name and shape must match before loading. Full training-state
continuation uses the optimizer, scheduler, RNG states, and data cursor saved by
the optimization engine. Frozen encoder assets are prepared separately.

Data processing attribution and source records are in [NOTICE](../NOTICE),
the [retained data-processing MIT license](../licenses/ImageWAM-MIT.txt), and
the [provenance manifest](../src/patchwam/data/PROVENANCE.json).
