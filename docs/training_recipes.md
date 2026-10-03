# Training recipes

These development configurations describe the intended training contracts.
Full-size GPU execution and closed-loop reproduction remain unverified.

| Configuration | Action / state dimensions | Camera layout | Normalization | Global batch |
| --- | --- | --- | --- | --- |
| `configs/robotwin.yaml` | 14 / 14 | High view above both wrist views, 288x256 | Z-score | 256: 32 processes x batch 4 x accumulation 2 |
| `configs/robotwin_c2r.yaml` | 14 / 14 | High view above both wrist views, 288x256 | Z-score | 256: 16 processes x batch 16 |
| `configs/robocasa.yaml` | 12 / 16 | Left agent view above right agent and wrist views, 288x256 | Z-score | 256: 16 processes x batch 16 |
| `configs/libero.yaml` | 7 / 8 | Front and wrist views side by side, 224x448 | Min/max | Launcher-defined |

All configurations use 16 future actions and endpoint image pairs. RobotWin uses
50Hz data; RoboCasa and LIBERO use 20Hz data. The 128D fixed codec consumes the
configured raw action dimension. Changing action dimensions, normalization,
camera geometry, or sampling filters defines a separate experiment.

The common optimizer is AdamW with betas `(0.9, 0.95)`, weight decay `0.01`,
learning rate `0.0001`, 5% linear warmup, and a cosine schedule with a 1% rate
floor. The denoising objective uses a shared shifted sigma with shift 5 and
image/action weights 0.5/1.

RobotWin uses its non-idle frame ranges. The C2R recipe additionally selects
the first 50 episodes of each 550-episode block and applies photometric, style,
and Fourier appearance augmentation with probability 0.8. Background replacement
is disabled. It runs up to 150000 updates. Its `inference` section records the
intended evaluation protocol: 10 denoising steps and a 16-action replanning
horizon. The training CLI does not execute evaluation or apply these inference
settings; an evaluation caller must pass `steps=10` to `sample_actions` and
implement the controller's replanning schedule. Other recipes keep their separately configured
sample sets and augmentation parameters.

Dataset roots, normalization statistics, non-idle filters, official model source,
and pretrained model assets are supplied through explicit configuration paths.
The engine verifies the configured global batch before optimization.
Eager model loading binds `device: cuda` to each launched process's local CUDA
device. Explicit CUDA indices must match the selected process device.

Weight-only initialization accepts complete native policy safetensors parameters.
Every parameter name and shape must match before loading. Full training-state
continuation uses the optimizer, scheduler, RNG states, and data cursor saved by
the optimization engine. Frozen encoder assets are prepared separately.

Data processing attribution and source records are in `NOTICE`,
the retained data-processing MIT license, and `src/patchwam/data/PROVENANCE.json`.
