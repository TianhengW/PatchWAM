# Model assets

The FLUX.2 adapter uses the upstream transformer parameter modules with an independently
implemented joint attention path. Clean conditioning tokens cannot attend to the noisy
future or action tokens. The upstream default attention path is not interchangeable
with that contract.

Install the optional text-encoding dependencies:

```bash
uv pip install -e '.[flux]'
```

Prepare the official [FLUX.2 source](https://github.com/black-forest-labs/flux2) and
local model files before starting a benchmark run. The adapter's CPU layer-contract tests
use revision `50fe5162777813d869182b139e83b10743caef15`; use that revision for the current
development version. Model preparation never silently substitutes a different model.

```bash
export FLUX2_SRC=/path/to/official/flux2
export PATCHWAM_TRANSFORMER=/path/to/flux-2-klein-base-4b.safetensors
export PATCHWAM_AUTOENCODER=/path/to/ae.safetensors
export PATCHWAM_TEXT_ENCODER=/path/to/Qwen3-4B
export PATCHWAM_DATA_ROOT=/path/to/local/lerobot/dataset
export PATCHWAM_NORM_STATS=/path/to/normalization/statistics.json
export PATCHWAM_NONIDLE_FILTER=/path/to/nonidle_ranges.json
```

The model factory is `patchwam.models.FluxAssetPolicy.from_local_assets`. Only local
assets are loaded. The autoencoder and text encoder remain frozen and in evaluation
mode; the transformer and proprioception projection are trainable.

Weight-only initialization uses complete native policy safetensors files. Every parameter
name and tensor shape must match before any model weights are changed. Frozen encoder
assets are prepared separately. Continuing training requires the complete state directory
written by the optimization engine.
