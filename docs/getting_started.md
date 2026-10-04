# Getting started

[Overview](../README.md) · [Documentation](index.md) · [Model assets](models.md) · [Data](data.md)

Run commands from the repository root.

## Environment

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

To train with the FLUX.2 backbone, install the text-encoding dependencies:

```bash
uv pip install -e '.[flux]'
```

Vision-language configurations additionally use `.[vlm]`, with Transformers 4.57.1
for the official Qwen3-VL modules:

```bash
uv pip install -e '.[flux,vlm]'
```

With standard Python tooling, the equivalent setup is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,flux]'
```

Prepare the [official model source and local assets](models.md) separately before
a benchmark run. Keep PatchWAM's pinned runtime when using that source; installing
its full dependency set can select different Torch versions.

## CPU quick start

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
[development and verification](development.md) for the complete CPU test setup.

## Next steps

Prepare [model assets](models.md) and [datasets and statistics](data.md), select a
[training recipe](training_recipes.md), and follow the [training and resume guide](training.md).
The [evaluation guide](evaluation.md) covers action decoding, episode history
reset, and simulator setup for RoboCasa, RobotWin/RoboTwin, and LIBERO.
