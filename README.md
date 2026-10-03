# PatchWAM

Code for **An Action Is Worth One Patch: Unified World–Action Modeling with PatchWAM**.

PatchWAM places visual predictions and robot actions in a shared patch-token space.
Each action vector becomes one fixed, parameter-free token. A single denoising
transformer jointly predicts future visual tokens and an action chunk, conditioned
on the current observation, language instruction, and robot state.

This repository provides the core model, a FLUX.2 adapter, LeRobot v2 data loading,
and distributed training configurations for RoboTwin/RobotWin, RoboCasa, and LIBERO.
It is a development implementation: CPU integration checks have passed; full-size
GPU training and closed-loop reproduction of the paper's results remain pending.
See the [implementation and verification record](docs/migration.md) for current scope.

## Contents

- [Method overview](#method-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Model assets](#model-assets)
- [Dataset preparation](#dataset-preparation)
- [Training configurations](#training-configurations)
- [Training](#training)
- [Checkpoints and continuation](#checkpoints-and-continuation)
- [Action sampling](#action-sampling)
- [Verification and troubleshooting](#verification-and-troubleshooting)
- [Repository layout](#repository-layout)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Method overview

### Fixed action patches

For an action vector with `A` coordinates, the default codec repeats each coordinate
`floor(128 / A)` times and fills the remaining token coordinates with zeros. An
`H`-step action chunk therefore becomes `H` tokens of width 128. Decoding averages
the repeated coordinates and reverses the configured action scale.

For example, a 14-dimensional action occupies 126 token coordinates, with two
zero-filled coordinates. A 16-step chunk becomes a `[16, 128]` token sequence.
The codec has no learned parameters. Time padding and action-dimension padding
are tracked separately and expanded into the token-level loss mask.

### Joint flow matching

The policy uses one noise level for both future-image and action tokens. A sampled
uniform time `u` is shifted to `sigma = s*u / (1 + (s-1)*u)`, where the benchmark
configurations use `s = 5`. Training interpolates clean tokens with Gaussian noise
and predicts the velocity `noise - clean`.

The default objective combines weighted visual and action losses:

```text
loss = 0.5 * loss_video + 1.0 * loss_action
```

Language, proprioception, and current-image tokens form the conditioning prefix.
The prefix cannot attend to noisy future or action tokens. Generated tokens can
attend to the prefix and to each other. The FLUX.2 adapter implements this attention
contract while retaining the official transformer's parameter structure.

The autoencoder and text encoder remain frozen. The denoising transformer and
proprioception projection are trained. Action sampling follows a descending
Euler flow schedule and returns normalized actions together with future latent tokens.

## Installation

### Environment

Use Python **3.11** for the verified setup. The package supports Python 3.10–3.12
and pins PyTorch 2.7.1, torchvision 0.22.1, and NumPy 1.26.4. CPU execution is
available for the tiny model and local checks; benchmark configurations request
CUDA with bfloat16 precision.

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/TianhengW/PatchWAM.git
cd PatchWAM

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
```

To train with the FLUX.2 backbone, also install the text-encoding dependencies:

```bash
uv pip install -e '.[flux]'
```

With standard Python tooling, the equivalent setup is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,flux]'
```

Prepare the official model source separately as described below. Keep PatchWAM's
pinned runtime when using that source; installing its full dependency set can
select different Torch versions.

## Quick start

These commands need no pretrained weights or robotics datasets. They use a small,
randomly initialized transformer and deterministic synthetic examples.
Use a fresh output directory for each new run.

```bash
patchwam smoke --output runs/quickstart --updates 2
```

The YAML entry point exercises the same configuration and training path used by
the benchmark recipes:

```bash
patchwam train --config configs/smoke.yaml \
  training.output_dir=runs/quickstart-config
```

The final JSON output reports the completed update count and checkpoint directory.
The synthetic smoke run checks optimization and state saving; it does not measure
robot performance.

Run the available local checks with:

```bash
python -m pytest -q
```

Tests requiring the official FLUX.2 source are skipped until that source is on
`PYTHONPATH`. A CUDA-only device-integration check is skipped on CPU hosts. See
[verification](#verification-and-troubleshooting) for the complete CPU test setup.

## Model assets

The benchmark configurations use `FluxAssetPolicy.from_local_assets` with the
`klein-base-4b` variant. Prepare these local assets:

| Environment variable | Required asset |
| --- | --- |
| `FLUX2_SRC` | Official FLUX.2 repository root, containing `src/flux2/` |
| `PATCHWAM_TRANSFORMER` | FLUX.2 Klein Base 4B transformer safetensors file |
| `PATCHWAM_AUTOENCODER` | FLUX.2 autoencoder safetensors file |
| `PATCHWAM_TEXT_ENCODER` | Local Qwen3-4B Hugging Face model directory, including weights and tokenizer files |

The adapter is checked against official source revision
`50fe5162777813d869182b139e83b10743caef15`. From the PatchWAM directory:

```bash
git clone https://github.com/black-forest-labs/flux2.git ../flux2
git -C ../flux2 checkout 50fe5162777813d869182b139e83b10743caef15
export FLUX2_SRC="$(cd ../flux2 && pwd)"

export PATCHWAM_TRANSFORMER=/path/to/flux-2-klein-base-4b.safetensors
export PATCHWAM_AUTOENCODER=/path/to/ae.safetensors
export PATCHWAM_TEXT_ENCODER=/path/to/Qwen3-4B
```

The loader uses local files and strict parameter loading. The text encoder must
provide the expected hidden-state layers and feature width for the selected
backbone. Frozen encoders are reconstructed from these assets when restoring a
training run; keep the same files and source revision available.

Paths may also be recorded in a local environment file:

```bash
cp .env.example .env
# Edit .env to point to your actual assets and dataset files, then export it:
set -a
source .env
set +a
```

The CLI reads exported environment variables; it does not automatically load `.env`.
Additional adapter details are in [model setup](docs/models.md).

## Dataset preparation

### LeRobot v2 layout

Each entry in `data.dataset_dirs` must be a dataset root with its own `meta/info.json`.
A typical root contains:

```text
dataset_root/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        │   └── episode_000000.mp4
        ├── observation.images.cam_left_wrist/
        │   └── episode_000000.mp4
        └── observation.images.cam_right_wrist/
            └── episode_000000.mp4
```

Camera names vary by recipe. Images may also be embedded in parquet records. Custom
data/video path templates come from `meta/info.json`. LeRobot v3 merged storage
requires a separate reader and is currently unsupported.

The default feature columns are `action`, `observation.state`, and
`observation.images.<camera_name>`. Configure `data.shape_meta` to match the actual
fields, camera order, dimensions, and resize shapes. A field's `lerobot_key` can
override the default column name.

Set the root and normalization-statistics paths:

```bash
export PATCHWAM_DATA_ROOT=/path/to/lerobot/dataset_root
export PATCHWAM_NORM_STATS=/path/to/statistics.json
# Required by the RobotWin and C2R configurations:
export PATCHWAM_NONIDLE_FILTER=/path/to/nonidle_ranges.json
```

For several roots, override the list explicitly:

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 \
  -m patchwam.cli train --config configs/robocasa.yaml \
  'data.dataset_dirs=[/data/task_a,/data/task_b]' \
  training.accumulation=2 \
  training.output_dir=runs/robocasa-multi-root
```

All roots in one dataset must share FPS. The reader validates timestamps against
that FPS and matches decoded frames with a default tolerance of `1e-4` seconds.
`data.lerobot_tolerance_s` changes the tolerance explicitly. The reader does not
resample datasets to a different control frequency.

### Windows, normalization, and filters

The supplied recipes use `num_frames: 17`, `endpoint_frames_only: true`, and an
action/video frequency ratio of 1. A sample contains current/future images at
`t` and `t+16`, and 16 actions at `t ... t+15`. Episode-end windows repeat the
last available observation and mark padded action steps separately.

Use normalization statistics from the exact dataset and action contract being
trained. RobotWin and the supplied RoboCasa configuration use z-score scaling;
LIBERO uses min/max scaling. The data processor also supports quantile scaling,
stepwise action statistics, and per-embodiment tables when configured explicitly.
Preserve the same transforms when decoding actions at inference.

Setting `data.pretrained_norm_stats=null` computes statistics from the selected
training split. To keep those statistics for later inference or validation,
export them explicitly:

```python
from patchwam.configuration import construct, read_configuration
from patchwam.data.scaling import write_statistics

config = read_configuration(
    "configs/robotwin.yaml", ["data.pretrained_norm_stats=null"]
)
dataset = construct(config.data)
write_statistics(dataset.statistics, "statistics.json")
```

This assumes the other environment paths are configured. For reproduction, use
the original statistics rather than silently fitting a new normalization contract.

RobotWin configurations accept episode-indexed non-idle ranges. Ranges select
episode rows and use an exclusive end index; windows are assembled from this
filtered row sequence and may cross between retained intervals. Missing episode
entries keep all rows for that episode, so check that the filter belongs to the supplied roots.
For multiple roots with separate filters, `data.nonidle_filter_path` can be a mapping
from each resolved absolute dataset-root path to its corresponding filter file.
The C2R configuration additionally keeps the first 50 episodes of each 550-episode
block and applies temporally consistent appearance augmentation with probability 0.8.

### Optional text caches

The FLUX.2 policy can consume cached Qwen3 features. Set these data options:

```yaml
qwen_text_cache_dir: /path/to/text_cache
qwen_text_cache_format: qwen3_flux2
qwen_context_len: 128
```

Cache files use the SHA256 hash of the final formatted prompt:
`<prompt_hash>.qwen3_flux2_len128.pt`. Each payload contains `text_hidden_states`
with shape `[128, text_feature_width]` and `text_attention_mask` with shape `[128]`.
The reader supplies native `text_tokens` and `text_valid` to the policy. Cached
features must match the encoder, prompt template, and context length used in training.
Incompatible cache formats are rejected by the FLUX.2 policy.

Dataset contracts and fingerprint limitations are documented in [data setup](docs/data.md).

## Training configurations

| Configuration | Action / state width | Composed image, H×W | Normalization | Per-process batch × accumulation | Intended global batch |
| --- | --- | --- | --- | --- | --- |
| [RobotWin](configs/robotwin.yaml) | 14 / 14 | 288×256: high view above both wrist views | Z-score | 4 × 2 | 256 with 32 processes |
| [RobotWin C2R](configs/robotwin_c2r.yaml) | 14 / 14 | 288×256: same geometry, appearance randomization | Z-score | 16 × 1 | 256 with 16 processes |
| [RoboCasa](configs/robocasa.yaml) | 12 / 16 | 288×256: left agent view above right agent and wrist views | Z-score | 16 × 1 | 256 with 16 processes |
| [LIBERO](configs/libero.yaml) | 7 / 8 | 224×448: front and wrist views side by side | Min/max | 8 × 1 | 8 × world size |
| [CPU smoke](configs/smoke.yaml) | 14 / 14 | Synthetic encoded tokens | Synthetic tensors | 2 × 1 | 2 for one process |

The intended dataset frequencies are 50Hz for RobotWin and 20Hz for RoboCasa/LIBERO.
Verify `meta/info.json` and action semantics against the selected recipe. The supplied
RoboCasa configuration describes a specific 12D-action/16D-state data interface;
other releases may use different transforms or layouts.

Benchmark optimizer defaults are AdamW, learning rate `1e-4`, betas `(0.9, 0.95)`,
weight decay `0.01`, gradient clipping at `1.0`, 5% linear warmup, and cosine decay
to a 1% learning-rate floor. All benchmark models use shift 5, image/action loss
weights 0.5/1, and gradient checkpointing.

RobotWin and RoboCasa run for five configured epochs; LIBERO uses 50. C2R uses
100 epochs with a cap of 150,000 optimizer updates. The actual update budget is
bounded by both the epoch budget and `training.max_updates`.
The 1% held-out split in RobotWin/RoboCasa configurations reserves data; the current
training CLI does not run a validation loop automatically.

For recipe-specific filtering, augmentation, and evaluation-protocol metadata,
see [training recipes](docs/training_recipes.md).

## Training

### Configuration overrides

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

### Single-node GPU launch

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

### Multi-node GPU launch

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

Resume restores model, optimizer, scheduler, RNG states, and the epoch/batch cursor.
It requires the same world size, accumulation, scheduler duration, model/asset
contract, dataset fingerprint, and relevant training settings. To start a changed
experiment, initialize weights into a fresh run instead.

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
the frozen encoder assets must still be supplied separately. Optimizer and scheduler
states start fresh. `--initialize` and `--resume` are mutually exclusive.
Historical checkpoint formats require a separate validated conversion to the native schema.

## Action sampling

The policy API provides `sample_actions`; the training CLI currently has no
benchmark evaluation command. This executable CPU example demonstrates the API
with a random tiny policy:

```python
import torch
from torch.utils.data import DataLoader

from patchwam.models import make_tiny_policy
from patchwam.testing import TensorExamples

torch.manual_seed(42)
policy = make_tiny_policy().eval()
batch = next(iter(DataLoader(TensorExamples(length=2), batch_size=2)))
prediction = policy.sample_actions(batch, horizon=16, steps=3)

print(prediction["action"].shape)         # torch.Size([2, 16, 14])
print(prediction["future_tokens"].shape)  # torch.Size([2, 4, 128])
```

With the real assets, dataset, and a native trained checkpoint prepared, the following
example samples a RobotWin observation and decodes its action chunk. Use the resolved
`configuration.yaml` from that training run to preserve its model and data settings.
Disable training video augmentation when inspecting the observation:

```python
from torch.utils.data import DataLoader

from patchwam.checkpoints import load_policy_weights
from patchwam.configuration import construct, read_configuration

config = read_configuration(
    "/path/to/robotwin-run/configuration.yaml", ["data.video_augmentation=null"]
)
dataset = construct(config.data)
policy = construct(config.model)
load_policy_weights(policy, "/path/to/policy_0.safetensors")
policy.eval()

batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
prediction = policy.sample_actions(batch, horizon=16, steps=20)
embodiment = batch["embodiment"][0] if "embodiment" in batch else None
actions = dataset.processor.decode_actions(
    prediction["action"].float().cpu(),
    batch["proprio"][:, :1].float().cpu(),
    embodiment=embodiment,
)
print(actions["default"].shape)  # torch.Size([1, 16, 14])
```

The decoder returns a dictionary keyed by the action fields in `shape_meta`.
This example inspects a dataset observation; it does not run a closed-loop benchmark.

The FLUX.2 wrapper accepts a prepared inference batch containing:

| Field | Shape / meaning |
| --- | --- |
| `video` | `[B, 3, 1, H, W]` or `[B, 3, H, W]`, current composed RGB image in `[-1, 1]` |
| `prompt` | One formatted instruction per sample |
| `proprio` | `[B, P]` or `[B, H, P]`, state transformed and normalized with the training processor |
| `text_tokens`, `text_valid` | Optional cached language features and valid-token mask, replacing prompt encoding |

Load the trained native policy weights, put the model in evaluation mode, and call
`sample_actions(batch, horizon=16, steps=...)`. Returned actions are normalized.
Use the same data processor and statistics to decode them before applying the
benchmark controller's coordinate, gripper, and replanning conventions.
`future_tokens` are latent tokens; generating RGB predictions additionally requires
restoring their raster layout and applying the autoencoder decoder.

The C2R configuration's `inference` section records a 10-step solver and a 16-action
replanning horizon. A caller must explicitly pass `steps=10` and implement that
controller schedule; the training CLI does not consume the evaluation settings.
Closed-loop adapters, history/VL reasoning paths, expert variants, and other
backbone implementations remain pending.

## Verification and troubleshooting

### Complete CPU checks

After preparing the pinned official source and installing `.[dev,flux]`:

```bash
PYTHONPATH="$FLUX2_SRC/src${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q
python -m ruff check src tests
python -m build
```

Coverage includes action codec inversion, visibility masks, masked losses, official
FLUX.2 micro-layer execution, checkpointed gradients, miniature local asset loading,
LeRobot parquet/MP4 samples, normalization, augmentation, native weight validation,
and complete-state continuation. Two-process CPU checks also verified exact resumed
weights, partial accumulation gradients, and collective checkpoint failure handling.

The [CPU verification workflow](.github/workflows/cpu-checks.yml) installs the optional
dependencies, checks out the pinned model source, runs tests and a two-update smoke,
and builds wheel/source distributions. These checks establish implementation coverage;
GPU throughput and closed-loop benchmark results require separate experiments.

### Common setup issues

| Symptom | Check |
| --- | --- |
| An environment interpolation cannot be resolved | Export the variables listed above; source `.env` explicitly if using it. |
| `meta/info.json` or a feature column is missing | Supply individual LeRobot v2 roots and match `shape_meta`/`lerobot_key` to the actual parquet schema. |
| Global batch differs from the recipe | Recalculate world size × per-process batch × accumulation. |
| Video timestamps cannot be decoded | Check dataset FPS, timestamp cadence, and video synchronization; inspect the encoding before changing tolerance. |
| Text cache or encoder width is incompatible | Use the matching Qwen3 encoder, cache format, context length, and prompt template. |
| Another FLUX.2 source is already imported | Use the pinned source in a fresh Python process. |
| Continuation settings or fingerprint differ | Restore the original configuration/assets/data, or initialize weights into a fresh experiment. |
| A checkpoint destination already exists | Use `--resume` for continuation or a new output directory for a new run. |

## Repository layout

```text
configs/                     Benchmark recipes and CPU smoke configuration
docs/                        Data, model, training, and verification notes
licenses/                    Retained third-party license texts
src/patchwam/
├── cli.py                   Training and smoke entry points
├── configuration.py         YAML construction and process-device binding
├── engine.py                Distributed optimization, metrics, and continuation
├── checkpoints.py           Native weights and compact policy state
├── models/
│   ├── codec.py             Fixed action-patch representation
│   ├── flow.py              Shifted flow training and sampling schedule
│   ├── geometry.py          Token coordinates and attention visibility
│   ├── policy.py            Joint objective and action sampling
│   ├── flux.py              Official model adapter and local frozen encoders
│   └── tiny.py              Small transformer for CPU checks
└── data/
    ├── episodes.py          LeRobot v2 windows, filters, caches, and fingerprints
    ├── processing.py        Action/state transforms and feature assembly
    ├── scaling.py           Feature normalization and inverse transforms
    ├── cameras.py           RGB preparation and multi-camera composition
    ├── augmentation.py      Temporally consistent clip augmentation
    ├── appearance.py        C2R appearance randomization
    └── PROVENANCE.json      Data-processing attribution and modification records
tests/                       Unit and CPU integration checks
```

## Acknowledgements

The data processing implementation is adapted from
[ImageWAM](https://github.com/yuyangalin/ImageWAM), including dataset preprocessing,
action/state normalization, multi-camera composition, and image/appearance augmentation.
We thank the ImageWAM authors for these data-processing components.
The source commit and per-file adaptation records are documented in
[the data provenance manifest](src/patchwam/data/PROVENANCE.json), and the original
[MIT license and copyright notice](licenses/ImageWAM-MIT.txt) are retained.

We also thank the authors of FLUX.2, Qwen, LeRobot, RoboCasa, RoboTwin, and LIBERO for
their models, dataset formats, and benchmark environments.

## License

New PatchWAM code is released under [Apache-2.0](LICENSE). Adapted data-processing
components retain the permissions and notices listed in [NOTICE](NOTICE) and their
provenance records. Model weights, external source packages, datasets, and benchmark
environments retain their respective licenses.
