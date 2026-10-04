# Development and verification

## Verification and troubleshooting

### Complete CPU checks

Run from the repository root after [installation](getting_started.md), preparing
the [pinned official model source](models.md), and installing `.[dev,flux]`:

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

The [CPU verification workflow](../.github/workflows/cpu-checks.yml) installs the optional
dependencies, checks out the pinned model source, runs tests and a two-update smoke,
and builds wheel/source distributions. These checks establish implementation coverage;
GPU throughput and closed-loop benchmark results require separate experiments.
The [implementation and verification record](migration.md) tracks the current
coverage. Use [full-model training validation](training.md#full-model-training-validation)
and [closed-loop evaluation](evaluation.md) for the corresponding real-model checks.

### Common setup issues

| Symptom | Check |
| --- | --- |
| An environment interpolation cannot be resolved | Export the variables in the [model](models.md) and [data](data.md) setup guides; source `.env` explicitly if using it. |
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
├── cli.py                   Training, smoke, validation, and evaluation entry points
├── configuration.py         YAML construction and process-device binding
├── engine.py                Distributed optimization, metrics, and continuation
├── checkpoints.py           Native weights and compact policy state
├── validation.py            Selected-model training and resume checks
├── evaluation/              Online actions and benchmark adapters
├── averaging.py             Successful-update EMA and detached teacher execution
├── models/
│   ├── codec.py             Fixed action-patch representation
│   ├── flow.py              Shifted flow training and sampling schedule
│   ├── geometry.py          Token coordinates and attention visibility
│   ├── policy.py            Joint objective and action sampling
│   ├── flux.py              Official model adapter and local frozen encoders
│   ├── vision_language.py   Causal Qwen3-VL features, language LoRA, and subtask loss
│   └── tiny.py              Small transformer for CPU checks
└── data/
    ├── episodes.py          LeRobot v2 windows, filters, caches, and fingerprints
    ├── history.py           Independent causal slot selection and episode-local buffer
    ├── processing.py        Action/state transforms and feature assembly
    ├── scaling.py           Feature normalization and inverse transforms
    ├── cameras.py           RGB preparation and multi-camera composition
    ├── augmentation.py      Temporally consistent clip augmentation
    ├── appearance.py        C2R appearance randomization
    └── PROVENANCE.json      Data-processing attribution and modification records
tests/                       Unit and CPU integration checks
```
