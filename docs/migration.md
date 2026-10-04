# Migration record

This file distinguishes prepared code from verified reproduction results.

| Work item | Current state |
| --- | --- |
| Source snapshots | Archived outside this repository with commit and file hashes |
| New `TianhengW/PatchWAM` repository | Created independently; now public |
| Independent model and optimization code | New implementation; CPU verification passed |
| Data processing provenance and licensing | Complete per-file manifest, MIT notices, normalization/augmentation oracle checks |
| H800 / Digua training sources | 10 read-only snapshots, 5952 text files; all SHA256 verified; legacy recipes ported |
| Native policy initialization | Strict safetensors parameter-name and shape validation |
| VLM and causal histories | Independent Qwen3-VL, pooled histories, language LoRA and subtask path; CPU checks passed, real checkpoint parity unverified |
| CFG and EMA | Explicit conditioning dropout/guidance and successful-update averaging; CPU checks passed |
| Self-Flow | Three independent source-checked experiment ports; CPU checks passed, real-weight/GPU parity unverified |
| Closed-loop adapters | Independent RoboCasa, RobotWin, and LIBERO adapters; CPU interface/action/history tests passed, real simulator scores unverified |
| Full-model validation workflow | Short selected-model training and complete-state resume command; single/two-process CPU checks passed, 4B GPU run unverified |
| Real-data GPU forward and backward | Unverified |
| Paired fixed-seed closed-loop parity | Unverified |
| Alternate backbones and expert variants | Pending independent implementation |
| Public release | Public development version; alternate implementations and GPU reproduction still pending |

Source review records are archived in the local migration workspace. Retained data
processing keeps its provenance and required license notices.

The latest corrected AAP337/NeMo RoboCasa and RobotWin runs are distinct from the paper's
FLUX.2 action-patch implementation. Their preprocessing contracts must be audited and
recorded separately before selecting an initialization or comparing benchmark scores.

The new optimization engine preserves model/optimizer/scheduler/RNG state and deterministic
epoch sample permutations. Mid-epoch continuation with asynchronous data workers may produce
different augmentation draws because prefetched worker state is not checkpointed; use
`workers=0` for exact CPU resume comparisons. Single-process and two-process CPU resume have
dedicated verification. Shared-filesystem multi-node GPU checkpointing and full-size GPU
throughput still need validation on the target cluster. The supported launch modes are a
single process, CPU distributed data parallel, and GPU distributed data parallel; sharded
optimizer/model launch modes are rejected until their state-saving contracts are implemented.


## Verification in this development version

CPU tests cover patch packing, attention visibility, official FLUX.2 micro-layer execution,
native policy weight validation and rejection, complete-state resume, scheduler curves, realistic
LeRobot v2 parquet/MP4 samples, normalization, camera composition, and data augmentation.
A two-process CPU gloo optimization completed two updates and saved a complete checkpoint.
An additional two-process run verified exact interrupted/resumed weights after four
updates, partial-group gradients, single-rank nonfinite-gradient rejection, and collective
checkpoint directory/save/publish failure handling.
Wheel and source distribution builds include Apache-2.0, MIT notices, and data provenance.
Regression coverage also checks partial accumulation groups, accumulation-group metric
averages, finite gradients, process-local eager CUDA asset loading, and checkpoint I/O
failure propagation. CPU CI installs the optional text-encoding dependencies and runs the
official micro-layer tests against the pinned source revision rather than skipping them.
The local asset factory also has a full CPU test using miniature transformer/autoencoder
weights and a 28-layer text encoder, including raw-image training and action sampling.
The feature audit passes 131 CPU tests, with one CUDA-only test awaiting a GPU.
It includes an actual miniature Qwen3-VL language model combined with three camera
views, causal history, Self-Flow pseudo-label warmup, LoRA, and EMA; interrupted
training resumes with bitwise-identical live, adapter, and averaged weights.
Each Self-Flow variant also completed a two-update CLI smoke run. A separate
two-process CPU run verified exact EMA/adapter resume over four updates.
Resume signatures include frozen asset and source identities. Large model weights,
videos, and text-cache files use path/size/modification-time metadata; small model config
and source files also have content hashes. Complete large payloads are not hashed.

Current numerical and interface differences requiring GPU parity review:

- Ordinary recipes retain continuous midpoint flow-weight normalization. The matched
  control and Self-Flow recipes select the historical 1000-point endpoint grid;
  its values match the archived source oracle exactly on 4,097 sampled timesteps.
- Action-dimension padding is explicitly expanded into patch coordinates; historical
  versions varied in whether codec fill coordinates were excluded.
- The raw asset wrapper accepts separate current/future views and masked pooled history.
  The VL encoder accepts unaugmented current/past observations; its original full-size
  history/LoRA checkpoint and fixed training budget still require matching.
- LeRobot v3 and heterogeneous canonical-80 data contracts are not supported by this reader.
- Shared-filesystem multi-node GPU checkpointing and real benchmark evaluation remain unverified.

Optional model and training features are described in
[`conditioning_and_training.md`](conditioning_and_training.md). Self-Flow configurations
were checked against archived H800 source recipes, including the 25% withheld-label
variant with a 10,000-update label warmup and eight-step teacher solver. Their ports
preserve explicit protocol settings rather than enabling the features in every baseline.
The main shared-noise configuration retains its previous defaults.

[`evaluation.md`](evaluation.md) documents the new benchmark runners, action/layout
contracts, per-episode history reset, evaluation resume, and selected-model
training validation. Two-process CPU validation in separate baseline/resume
launches reached the same weights exactly. Real 4B CUDA and simulator scores
remain unverified; interface tests do not establish those results.
The final CPU regression suite passes 217 tests, with one CUDA-only test skipped.
