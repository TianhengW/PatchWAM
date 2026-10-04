# Model assets

[Overview](../README.md) · [Installation](getting_started.md) · [Data](data.md) · [Training](training.md)

## Official source and local files

The benchmark configurations use `FluxAssetPolicy.from_local_assets` with the
`klein-base-4b` variant. Prepare these local assets:

| Environment variable | Required asset |
| --- | --- |
| `FLUX2_SRC` | Official FLUX.2 repository root, containing `src/flux2/` |
| `PATCHWAM_TRANSFORMER` | FLUX.2 Klein Base 4B transformer safetensors file |
| `PATCHWAM_AUTOENCODER` | FLUX.2 autoencoder safetensors file |
| `PATCHWAM_TEXT_ENCODER` | Local Qwen3-4B Hugging Face model directory, including weights and tokenizer files |
| `PATCHWAM_VLM` | Optional local Qwen3-VL-4B-Instruct directory for VL configurations |

Install the [optional text-encoding dependencies](getting_started.md#environment)
and prepare the official [FLUX.2 source](https://github.com/black-forest-labs/flux2).
The adapter's CPU layer-contract tests use revision
`50fe5162777813d869182b139e83b10743caef15`; use that revision for the current
development version. From the PatchWAM directory:

```bash
git clone https://github.com/black-forest-labs/flux2.git ../flux2
git -C ../flux2 checkout 50fe5162777813d869182b139e83b10743caef15
export FLUX2_SRC="$(cd ../flux2 && pwd)"

export PATCHWAM_TRANSFORMER=/path/to/flux-2-klein-base-4b.safetensors
export PATCHWAM_AUTOENCODER=/path/to/ae.safetensors
export PATCHWAM_TEXT_ENCODER=/path/to/Qwen3-4B
```

Use the source checkout through `FLUX2_SRC` with PatchWAM's pinned runtime.
Installing the external repository's full dependency set would select a different
Torch runtime. Model preparation never silently substitutes a different model.

The loader uses local files and strict parameter loading. The text encoder must
provide the expected hidden-state layers and feature width for the selected
backbone. Frozen encoders are reconstructed from these assets when restoring a
training run; keep the same files and source revision available.

## Environment file

Paths may also be recorded in a local environment file:

```bash
cp .env.example .env
# Edit .env to point to your actual assets and dataset files, then export it:
set -a
source .env
set +a
```

The CLI reads exported environment variables; it does not automatically load `.env`.
The [data guide](data.md) lists dataset, normalization, and filter paths.

## Adapter and checkpoint contract

The FLUX.2 adapter uses the upstream transformer's parameter modules with an
independently implemented joint attention path. Clean conditioning tokens cannot
attend to noisy future or action tokens. The upstream default attention path is
not interchangeable with that contract.

The model factory is `patchwam.models.FluxAssetPolicy.from_local_assets`. The
autoencoder and base text encoder remain frozen and in evaluation mode; the
transformer and proprioception projection are trainable. Vision-language recipes
can additionally train language LoRA adapters.

Weight-only initialization uses complete native policy safetensors files. Every
parameter name and tensor shape must match before any model weights are changed.
Frozen encoder assets are prepared separately. Continuing training requires the
complete state directory written by the optimization engine. See
[checkpoints and continuation](training.md#checkpoints-and-continuation).

Vision-language conditioning, pooled causal histories, CFG, EMA, and the three
Self-Flow training variants are described in
[conditioning and training options](conditioning_and_training.md). Their CPU
checks do not establish compatibility with full-size historical checkpoints or
benchmark scores.
