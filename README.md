# PatchWAM

Code for **An Action Is Worth One Patch: Unified World–Action Modeling with PatchWAM**.

PatchWAM represents robot action sequences as fixed, parameter-free patch tokens. Future
image patches and action patches share one denoising backbone and one flow-matching time.

This repository is being rebuilt as a standalone implementation. The current development
version contains a new model package, a new optimization engine, and local data processing
interfaces. CPU verification does not establish compatibility with historical checkpoints
or reproduce the paper's closed-loop benchmark results. Track the remaining work in
[the migration record](docs/migration.md).

## Install

Use Python 3.11. Install the package in an isolated environment:

```bash
uv venv --python 3.11
uv pip install -e '.[dev]'
```

For the FLUX.2 backbone, install `.[flux]` and prepare the upstream model source and weights
as described in [model setup](docs/models.md).

## Verify and train

```bash
python -m pytest
patchwam smoke --output runs/smoke --updates 2
patchwam train --config configs/smoke.yaml training.output_dir=runs/config-smoke
```

The smoke commands use a tiny random backbone and synthetic tensors. Benchmark training
requires real LeRobot datasets, normalization statistics, and pretrained model assets.
Benchmark configurations use explicit environment variables for paths. See
[dataset setup](docs/data.md).

```bash
accelerate launch --config_file /path/to/distributed.yaml -m patchwam.cli train \
  --config configs/robotwin.yaml training.output_dir=runs/robotwin
```

`--resume` accepts a complete state directory written by this implementation. A directory
contains model weights, optimizer and scheduler states, RNG states, and a completed cursor
record. Historical training checkpoints require an explicit mapping and parity validation.

`configs/robotwin.yaml` preserves global batch 256 with 32 processes, batch 4, and
accumulation 2. `configs/robotwin_c2r.yaml` and the legacy RoboCasa recipe use 16
processes with batch 16. The engine checks the configured global batch before training.

Use `--initialize /path/to/research.pt` for the strict full single-stream weight mapping.
This initializes model weights; it does not resume the historical optimizer or sampler.

## Layout

```text
src/patchwam/models/   patch representation, joint flow objective, backbone adapters
src/patchwam/data/     LeRobot readers, normalization, camera packing, filtering
src/patchwam/engine.py distributed optimization and checkpoint state
configs/              portable experiment configurations
tests/                CPU verification
licenses/             retained third-party license texts
```

## Acknowledgements

The data processing implementation is adapted from
[ImageWAM](https://github.com/yuyangalin/ImageWAM), including dataset preprocessing,
action/state normalization, and multi-camera composition. We thank the ImageWAM authors.
The source commit and per-file adaptation records are documented in
[the data provenance manifest](src/patchwam/data/PROVENANCE.json), and the original
[MIT license and copyright notice](licenses/ImageWAM-MIT.txt) are retained.

We also thank the authors of FLUX.2, LeRobot, RoboCasa, RoboTwin, and LIBERO for their
models, dataset formats, and benchmark environments.

## License

New PatchWAM code is released under [Apache-2.0](LICENSE). Adapted data processing
components retain the permissions and notices listed in [NOTICE](NOTICE) and their
provenance records. Model weights and external datasets retain their respective licenses.
