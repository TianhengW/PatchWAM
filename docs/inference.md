# Action sampling and inference

The policy API provides `sample_actions`; [closed-loop evaluation](evaluation.md)
uses `patchwam evaluate` for benchmark execution. All commands and examples below
run from the repository root after [installation](getting_started.md).

## Synthetic CPU example

This example demonstrates the API with a random tiny policy:

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

The random policy and synthetic observations check the sampling interface; they
do not measure robot performance.

## Real observations and action decoding

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
prediction = policy.sample_actions(batch, horizon=16, steps=10)
embodiment = batch["embodiment"][0] if "embodiment" in batch else None
actions = dataset.processor.decode_actions(
    prediction["action"].float().cpu(),
    batch["proprio"][:, :1].float().cpu(),
    embodiment=embodiment,
)
print(actions["default"].shape)  # torch.Size([1, 16, 14])
```

The decoder returns a dictionary keyed by the action fields in `shape_meta`,
undoes training normalization, and restores relative actions using the state at
the start of the predicted chunk. Preserve the same statistics, field order,
padding widths, and embodiment identifier used by training. Do not fit new
statistics for inference or move the relative anchor as queued actions execute.
This example inspects a dataset observation; it does not run a closed-loop benchmark.

## Prepared inference batch

The FLUX.2 wrapper accepts a prepared inference batch containing:

| Field | Shape / meaning |
| --- | --- |
| `video` | `[B, 3, 1, H, W]` or `[B, 3, H, W]`, current composed RGB image in `[-1, 1]` |
| `prompt` | One formatted instruction per sample |
| `proprio` | `[B, P]` or `[B, H, P]`, state transformed and normalized with the training processor |
| `text_tokens`, `text_valid` | Optional cached language features and valid-token mask, replacing prompt encoding |

Configurations with separate views, world-model history, or Qwen3-VL additionally
use the following fields. Supply only the paths required by the trained recipe:

| Field | Shape / meaning |
| --- | --- |
| `camera_video` | `[B, V, 3, 1, H, W]`, separate current camera views in `[-1, 1]`; used instead of composed `video` |
| `history_video` | `[B, K, 3, H, W]`, past head-camera RGB observations in `[-1, 1]` for the world-model prefix |
| `history_valid` | Boolean `[B, K]`, marking available world-model history slots |
| `vl_current` | `[B, 3, 448, 448]`, unaugmented current head-camera RGB observation in `[-1, 1]` for the default VL recipe |
| `vl_history` | `[B, K, 3, 448, 448]`, unaugmented past head-camera RGB observations in `[-1, 1]` for the default VL recipe |
| `vl_history_valid` | Boolean `[B, K]`, marking available VL history slots |

History slots are nearest first at the trained temporal spacing. Missing slots are
zero-filled and masked; historical visual tokens remain conditioning context and
do not enlarge the predicted future raster. Qwen3-VL requires image-conditioned
inputs rather than a text-only cache. Trained language adapters cannot be bypassed
with cached text features. Future images and subtask labels are not inference
inputs. The [conditioning guide](conditioning_and_training.md#qwen3-vl-and-two-history-paths)
records the two history contracts, feature pooling, and trained language adapters.

Load the trained native policy weights, put the model in evaluation mode, and call
`sample_actions(batch, horizon=16, steps=...)`. Returned actions are normalized.
Use the same data processor and statistics to decode them before applying the
benchmark controller's coordinate, gripper, and replanning conventions.
`future_tokens` are latent tokens; generating RGB predictions additionally requires
restoring their raster layout and applying the autoencoder decoder.

## Guidance and inference weights

The paper protocol uses `steps=10` and `guidance_scale=1`. Scale 1 uses a single
conditional forward. For a separately trained guidance experiment,
`sample_actions(batch, horizon=16, steps=10, guidance_scale=g)` combines
`v_uncond + g * (v_cond - v_uncond)`; `action_guidance_scale` can set a different
scale for actions. Non-unit scales require language-conditioning dropout during
training. Visual observations and proprioception remain in both branches. See
[CFG training](conditioning_and_training.md#classifier-free-guidance).

Select live or averaged inference weights explicitly. The real-data example loads
the live `policy_0.safetensors`; a completed EMA checkpoint also contains
`ema_policy.safetensors`:

```python
load_policy_weights(policy, "/path/to/step_0005000/ema_policy.safetensors")
policy.eval()
```

This replaces the weight-loading line in the real-data example. Both selections
still require the same frozen model assets and processor statistics. Record the
selection with evaluation results. Weight-only loading does not resume training;
complete-state restoration is described in [training and checkpoints](training.md).
See [EMA settings](conditioning_and_training.md#ema-training-and-inference-weights)
for successful-update averaging, warmup, exports, and memory cost.

## Online actions, history, and replanning

The C2R configuration's `inference` section records a 10-step solver and a 16-action
replanning horizon. A caller must explicitly pass `steps=10` and implement that
controller schedule; the training CLI does not consume the evaluation settings.
VL and causal-history input preparation are available in the optional configurations.
Closed-loop adapters apply the explicit evaluation settings. Expert variants and
other backbone implementations remain pending.

`patchwam.evaluation.policy.OnlinePolicy` accepts raw observations containing
`images`, `state`, a nonempty `instruction`, and a `timestamp` in seconds. Camera
and state keys follow the training processor's `shape_meta`. Images are HWC uint8
or CHW floating RGB in `[0, 1]`; state values have their raw, unnormalized widths.
Supply `embodiment` when the trained statistics are per-embodiment.

Call `reset(episode_id)` before the first observation of every episode, then call
`act(observation)` once per environment observation. The policy prepares and
normalizes inputs, decodes a predicted chunk, and returns one physical action
vector at a time. It records every observation in causal history even while
serving queued actions. Reset clears history, queued actions, the observation
clock, and the sampling generator; an instruction change clears queued actions.

For manual history preparation, `CausalObservationBuffer.append` stores CHW frames
at their recorded timestamps and `history(current_time)` returns the past slots
and validity mask. Use a new episode ID or call `reset()` between episodes, and
resize world-model and VL inputs according to the trained recipe. This buffer
alone does not execute a controller. The [evaluation guide](evaluation.md) covers
the complete adapters, clocks, episode seeds, controller conventions, and result
records. Their interface checks do not establish real full-model simulator scores.
