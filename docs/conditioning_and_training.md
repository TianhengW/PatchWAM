# Vision-language conditioning, history, CFG, EMA, and Self-Flow

These features have separate configurations. Enabling every feature at once changes
the experiment. The ordinary RobotWin/LIBERO recipes retain the shared-noise core;
the paper evaluates without classifier-free guidance. Real-data GPU and closed-loop
parity of these independent ports still requires verification.

## Configuration map

| Configuration | Purpose | Global batch |
| --- | --- | --- |
| `configs/robotwin_matched.yaml` | Shared-noise control, one of every 20 window starts, 10 epochs | 64: 8 processes × 4 × 2 |
| `configs/robotwin_self_flow_v1.yaml` | Dual timesteps and EMA representation teacher | 64: 8 × 4 × 2 |
| `configs/robotwin_self_flow_v2.yaml` | Variant 1 plus modality-structured timestep masks | 64: 8 × 4 × 2 |
| `configs/robotwin_self_flow_v3.yaml` | Variant 2 plus withheld action labels and teacher pseudo-labels | 64: 8 × 4 × 2 |
| `configs/robotwin_vlm.yaml` | Optional frozen current-image vision-language conditioning | 256: 32 × 4 × 2 |
| `configs/robodojo_vlm_history.yaml` | Separate cameras, causal histories, VL LoRA, and subtask supervision | 256: 32 × 8 × 1 |

Use immutable data roots, matching normalization/filter files, and the same source
and model assets for a comparison. These files specify implementations and intended
training contracts; they are not evidence of reproduced success rates.

## Qwen3-VL and two history paths

Install `.[flux,vlm]` and supply a local `Qwen3-VL-4B-Instruct` directory:

```bash
uv pip install -e '.[flux,vlm]'
export PATCHWAM_VLM=/path/to/Qwen3-VL-4B-Instruct
```

The pinned Transformers 4.57.1 provides the official Qwen3-VL modules. The adapter
uses local assets and does not download models or execute remote model code. See
[Qwen's model documentation](https://github.com/QwenLM/Qwen3-VL) for asset preparation.

The RoboDojo configuration maintains two distinct causal histories:

- **World-model prefix:** 20 past head-camera slots at one-second intervals.
  Frozen autoencoder features are average-pooled to 4×4 tokens per slot, with
  positional group `-k`. Training uses up to 0.4 seconds of timing jitter,
  whole-history dropout 0.2, slot dropout 0.2, and random retention of the nearest
  remaining slots. Unavailable or dropped slots are masked as attention keys.
- **VL input:** unaugmented current and past head-camera images resized to
  448×448. Past visual features and their DeepStack features are pooled to 4×4
  before the language model. Training jitters each past slot by at most one frame
  and keeps all available slots. Future targets never enter this path.

The world model encodes three separate 256×256 camera views. Current view groups
are 10/11/12 and future view groups are 0/1/2. Adding historical prefix tokens does
not increase the number of future tokens that sampling generates.

VL conditioning reads instruction-token states from layers 9/18/27 and adds a
last-prompt summary from layers 18/27/36. Frozen VL vision/base weights are kept
separate from trained language LoRA weights. The RoboDojo recipe uses rank 64,
alpha 128, and dropout 0.05 on language attention and feed-forward projections.

Training also requires a nonempty subtask sentence in each row's `subtask` parquet
column. Change `data.subtask_column` for another schema. The adapter appends these
labels after the conditioning prompt and applies causal cross-entropy with weight
0.1 and label smoothing 0.1. Conditioning states cannot attend to those labels;
inference ignores subtask labels. Missing required annotations fail explicitly.
Trainable VL adapters cannot be bypassed with cached text features.

Set `PATCHWAM_MAX_UPDATES` to the original RoboDojo run's fixed optimizer-update
budget before loading its configuration. The original 20-slot VL checkpoint and
complete training recipe have not yet been matched to this port; no budget is
silently substituted. The configured epoch count is an upper bound, and the engine
still stops at the smaller of the epoch and update budgets.

For online history, `patchwam.data.history.CausalObservationBuffer` stores head-camera
frames with their episode timestamps. Append observations at the recording cadence,
use a new episode identifier or call `reset()` at each episode, and retrieve the
past slots with `history(current_time)`. Resize and prepare separate VL inputs with
the same training contract. The buffer never retains observations across episodes.
This helper does not implement a benchmark controller or evaluation runner.

## Classifier-free guidance

`model.condition_dropout` controls training-time dropout of language conditioning.
The visual prefix and proprioception remain available to both branches. Default 0
keeps the original conditional training path. For an explicitly guided experiment:

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 \
  -m patchwam.cli train --config configs/robotwin_matched.yaml \
  model.condition_dropout=0.1 \
  training.output_dir=runs/robotwin-guidance
```

At inference, `sample_actions(..., guidance_scale=g)` uses
`v_uncond + g * (v_cond - v_uncond)`. `action_guidance_scale` can override the action
branch's scale. Scale 1 runs a single conditional forward and disables guidance;
scale 0 selects the branch without language. Non-unit scales require training with
conditioning dropout. Use scale 1 and 10 solver steps for the paper's evaluations.

## EMA training and inference weights

`training.ema_decay=null` disables averaging. A value such as 0.999 maintains FP32
shadows of trainable policy weights and optional VL adapters. Frozen encoders are
not replicated. The average updates once per successful optimizer update, including
gradient accumulation; skipped fp16 or failed updates do not advance it.

`training.ema_warmup_updates` optionally copies live weights for the first N
successful updates, then applies the fixed decay. It is 0 in the Self-Flow recipes
and is distinct from pseudo-label warmup.

EMA state and update count are saved with optimizer/scheduler/RNG state. An EMA run
also exports `ema_policy.safetensors` inside each completed checkpoint. Load that
file with `load_policy_weights` or `--initialize` to select averaged inference or
initialization weights explicitly. `--resume` restores the complete live training
state and its average. Record whether benchmark results use live or EMA weights.

A trainable VL checkpoint includes both `policy.*` tensors and trained
`text_encoder.*` adapters. The frozen base model and autoencoder must still be
supplied. The native loader rejects incomplete or incompatible files.

FP32 averaging adds about 16 GB for four billion averaged parameters, plus adapter
weights and ordinary training memory. The current engine supports DDP; sharded
EMA/optimizer storage is not implemented.

## Three Self-Flow variants

The independent implementation follows the dual-timestep and representation-learning
mechanism of [Self-Flow](https://arxiv.org/abs/2603.06507), with the three PatchWAM
experiment settings checked against the archived H800 configurations.

All three recipes retain one of every 20 window starts, train for 10 epochs with
global batch 64, and save every 2,500 updates. They use the historical 1,000-point
endpoint flow-weight normalization (`model.flow_normalization=endpoint_1000`).
The matched shared-noise control uses that same normalization.

- **Variant 1:** two independently sampled shifted noise levels, token-mask ratio
  0.25, and a teacher conditioned at the cleaner noise level. A trainable projection
  aligns generated-token representations from approximately 30% of the student
  depth with approximately 70% of the EMA teacher depth. Representation weight is
  0.8; EMA decay is 0.9999.
- **Variant 2:** half of samples use the random token mask. The other half uses
  modality blocks: one quarter has separate video/action times, one eighth fixes
  video at zero noise, and one eighth fixes actions at zero noise. Clean blocks are
  excluded from flow supervision. Other Variant 1 settings remain the same.
- **Variant 3:** withhold action labels for 25% of samples. Before 10,000 successful
  updates, those actions are excluded from action and action-representation
  supervision. Afterwards, the detached EMA teacher generates labels with an
  eight-step inverse-dynamics solver that clamps the true future visual tokens.
  Pseudo-label action loss weight is 0.5; EMA decay is 0.999.

For example, launch Variant 1 on one eight-GPU node:

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 \
  -m patchwam.cli train --config configs/robotwin_self_flow_v1.yaml \
  training.output_dir=runs/self-flow-v1
```

Self-Flow training requires the registered EMA teacher; omitting EMA fails at setup.
The teacher uses averaged denoiser parameters with detached live conditioning. It
does not become an optimizer module or re-encode historical observations. Resume
restores its average and the successful-update cursor used by pseudo-label warmup.

The representation metric uses `1 - cosine_similarity`; a historical negative-cosine
metric differs by an additive constant while giving the same gradients. Real-weight
numerical parity, multi-node GPU behavior, and benchmark scores still need paired
validation. Adding these features alone does not establish paper reproduction.
