# PatchWAM

Code for **[An Action Is Worth One Patch: Unified World–Action Modeling with PatchWAM](https://arxiv.org/abs/2609.25961)**.

PatchWAM represents each robot action as a fixed, parameter-free patch token.
A single denoising transformer jointly predicts future visual tokens and actions,
conditioned on the current observation, language instruction, and robot state.

The repository includes a FLUX.2 adapter, LeRobot v2 data processing, and training
and closed-loop evaluation entry points for RoboCasa, RobotWin/RoboTwin, and LIBERO.
VLM, causal history, CFG, EMA, and Self-Flow are configurable options.
CPU integration checks have passed; full-size GPU training and simulator results
still require verification. See the [verification record](docs/migration.md).

## Installation

Use Python **3.11** for the verified setup:

```bash
git clone https://github.com/TianhengW/PatchWAM.git
cd PatchWAM
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,flux,vlm]'
```

See [installation and environment setup](docs/getting_started.md) for dependency
versions, optional extras, and alternative commands. Pretrained weights and
simulator packages are prepared separately.

## Quick start

Run a two-update CPU check with synthetic data; no model weights or robot datasets
are needed. Use a fresh output directory for each run:

```bash
patchwam smoke --output runs/quickstart --updates 2
```

For benchmark experiments, prepare the [model assets](docs/models.md) and
[datasets](docs/data.md), then follow the [training guide](docs/training.md) and
[closed-loop evaluation guide](docs/evaluation.md).

## Documentation

The [documentation index](docs/index.md) lists the setup and experiment workflow.

| Topic | Guide |
| --- | --- |
| Installation and CPU quick start | [Getting started](docs/getting_started.md) |
| Pretrained weights and official model source | [Model assets](docs/models.md) |
| Datasets, normalization, filtering, and text caches | [Data preparation](docs/data.md) |
| RoboCasa, RobotWin, C2R, and LIBERO configurations | [Training recipes](docs/training_recipes.md) |
| GPU training, validation, and checkpoint resume | [Training](docs/training.md) |
| Action decoding, history reset, and benchmark evaluation | [Closed-loop evaluation](docs/evaluation.md) |
| VLM, history, CFG, EMA, and Self-Flow | [Conditioning and training options](docs/conditioning_and_training.md) |
| Policy APIs and action sampling | [Inference](docs/inference.md) |
| Action patches and joint flow matching | [Architecture](docs/architecture.md) |
| Tests, troubleshooting, and repository layout | [Development](docs/development.md) |

## Citation

If you use PatchWAM in your research, please cite our paper:
[An Action Is Worth One Patch: Unified World-Action Modeling with PatchWAM](https://arxiv.org/abs/2609.25961).

```bibtex
@misc{wang2026patchwam,
  title={An Action Is Worth One Patch: Unified World-Action Modeling with {PatchWAM}},
  author={Tianheng Wang and Zhou Xie and Heng Jia and Jianhua Xu and Tong Zhang and Kaicheng Yu},
  year={2026},
  eprint={2609.25961},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.25961}
}
```

## Acknowledgements

The data processing implementation is adapted from
[ImageWAM](https://github.com/yuyangalin/ImageWAM). We thank its authors for the
preprocessing, normalization, camera composition, and augmentation components.
See the [data provenance manifest](src/patchwam/data/PROVENANCE.json) and retained
[MIT license and copyright notice](licenses/ImageWAM-MIT.txt).

We also thank the authors of FLUX.2, Qwen, LeRobot, RoboCasa, RoboTwin, and LIBERO
for their models, dataset formats, and benchmark environments.

## License

New PatchWAM code is released under [Apache-2.0](LICENSE). Adapted data-processing
components retain the permissions and notices in [NOTICE](NOTICE). External model
weights, source packages, datasets, and environments retain their own licenses.
