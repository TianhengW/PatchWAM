# Closed-loop evaluation and training validation

`patchwam evaluate` connects a native PatchWAM checkpoint to RoboCasa, RobotWin /
RoboTwin, or LIBERO. `patchwam validate` checks training and complete-state resume
with the selected model and dataset. Simulator packages, their assets, model
weights, and training normalization files are external dependencies.

## Training and resume

The existing `train` entry point trains the full configured FLUX.2 model. For
example, the RobotWin recipe uses 32 GPUs and global batch 256; preserve that
contract across machines when reproducing the original run. See
[`training_recipes.md`](training_recipes.md) and the README's multi-node launch.

Before a long run, validate the real model on an allocated GPU:

```bash
patchwam validate --config configs/robotwin.yaml \
  --output runs/validate-robotwin --updates 2
```

This keeps the model, preprocessing, and training objective. It uses batch 1,
accumulation 1, workers 0, a short update budget, and per-update checkpoints.
The ordinary recipe is not overwritten. The validator runs an uninterrupted
baseline and a separate process resumed from update 1, compares all exported
live/EMA safetensors, and checks finite training metrics and update cursors.
It enables deterministic algorithms and math SDPA and defaults to exact equality;
unsupported deterministic operations fail explicitly. Use `--atol` / `--rtol`
only with recorded reasons.
The reduced update budget also changes the validation learning-rate schedule.

For a single-node multi-GPU check:

```bash
patchwam validate --config configs/robotwin.yaml \
  --output runs/validate-robotwin-ddp --updates 2 --num-processes 8
```

Run the validator directly inside the allocation; it starts its own workers.
`validation.json`, `baseline.log`, and `resumed.log` record the result. Full 4B
parameters and optimizer state still require their ordinary memory; a small
batch does not make parameter/optimizer storage small.

The local CPU equivalent checks this orchestration without loading real weights:

```bash
patchwam validate --config configs/smoke.yaml \
  --output runs/validate-cpu --allow-cpu
```

Resume a real training run with the original complete checkpoint and recipe:

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 32 --mixed_precision bf16 \
  -m patchwam.cli train --config configs/robotwin.yaml \
  --resume /shared/runs/robotwin/step_0005000 \
  training.output_dir=/shared/runs/robotwin
```

Use the actual machine/process layout of the original run. World size, data,
model assets, and training contract must match. `--initialize` loads weights
for a new run; it does not resume optimizer, scheduler, RNG, or EMA state.

## Shared online policy contract

The evaluator loads `policy.training_config`, its `data.processor`, the saved
normalization statistics, and `policy.weights`. Supply the resolved training
configuration for a trained checkpoint rather than changing its input contract.
It does not fit fresh statistics during evaluation.

`OnlinePolicy` normalizes live state with the training codec, prepares camera
inputs in the configured order/layout, samples an action chunk, and decodes it
with the same statistics and relative-joint anchor used at prediction time.
The simulator adapter maps those physical actions to its controller contract.

Every environment observation advances the history buffer, including steps
served from an existing chunk. Each episode resets history, the queued actions,
the observation clock, and the sampling generator. History contains current/past
observations only; no future targets or subtask labels enter online inference.
The sampling generator restarts from the configured policy seed per episode.

The evaluation configuration explicitly sets solver steps, predicted horizon,
replanning horizon, and guidance. The paper protocol uses 10 steps and guidance 1.
RoboDojo predicts 16 actions and replans after 8; preserve that setting when using
its VLM/history configuration.

## Simulator setup

First complete the README's model-asset and normalization environment setup.
The templates reference training recipes whose environment variables must
resolve even though evaluation does not construct the dataset. Alternatively,
append `policy.training_config=/path/to/run/configuration.yaml` to each command
to use that run's saved, resolved asset/data settings.

Install each benchmark and its simulator/assets using the version associated
with the training dataset. Keep its Python, MuJoCo/SAPIEN, controller, camera,
and task configuration versions in the experiment record. PatchWAM does not
install simulator dependencies into the model environment automatically.

Primary simulator references:

- [RoboCasa environment/action utilities](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/env_utils.py)
- [RoboCasa Gym wrapper](https://github.com/robocasa/robocasa/blob/main/robocasa/wrappers/gym_wrapper.py)
- [LIBERO environment and initialization example](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/benchmark_scripts/render_single_task.py)
- [RoboTwin source](https://github.com/RoboTwin-Platform/RoboTwin)

### LIBERO

```bash
export PATCHWAM_POLICY=/path/to/native/policy_0.safetensors
patchwam evaluate --config configs/evaluation/libero.yaml
```

The template evaluates task 0 of `libero_spatial`, one episode, as a smoke run.
It uses the task's provided initial state at `episode_index`, five settling
steps, and explicit 180-degree image orientation. Initial-state indices outside
the task's available list fail rather than wrapping to a different state.
NumPy float32/float64 initial-state arrays work with the pinned Torch loader;
custom serialized object types are rejected.

`gripper_convention=raw_pm1` passes native -1/open and +1/close commands. Set
`open01` only for datasets whose decoded gripper stores 1/open and 0/close;
the adapter then maps it with `1 - 2*g`. `binarize_gripper` is a separate explicit
option. Confirm both conventions against the exact dataset release.

For selected tasks and a larger episode count:

```bash
patchwam evaluate --config configs/evaluation/libero.yaml \
  'benchmark.tasks=[{task_id:0},{task_id:1}]' \
  evaluation.episodes_per_task=50 \
  evaluation.output_dir=runs/evaluation/libero-selected \
  evaluation.protocol=selected-50
```

This evaluates the requested subset; it is not a complete four-suite score.
Select every required suite/task and its prescribed horizon for a formal result.

### RoboCasa

The template targets the Human300 Gym interface and explicitly selects its
dataset layout: base motion 4, control mode 1, EEF position 3, EEF rotation 3,
gripper closedness 1. Its state has base position/quaternion, relative EEF
position/quaternion, and two gripper joints. The Gym adapter maps actions by
name; it does not treat the dataset vector as the native robosuite ordering.

```bash
export PATCHWAM_POLICY=/path/to/native/policy_0.safetensors
export PATCHWAM_ROBOCASA_TASK=PnPCounterToCab
patchwam evaluate --config configs/evaluation/robocasa.yaml
```

Provide the task's original `env_kwargs` and split directly in YAML or through
dotted CLI overrides. Other layouts can supply
`modality_json` or explicit `action_fields` / `state_fields`. Native robosuite
execution requires `api=robosuite`, `action_layout=robosuite_native`, explicit
state fields, and the original robot/controller configuration. Layouts and
field widths are checked; unsupported mappings fail before a controller step.

### RobotWin / RoboTwin

The adapter targets the direct task API: `setup_demo`, `get_obs`, `take_action`,
`check_success`, and `close_env`. Provide the external RoboTwin checkout,
task configuration, embodiments/assets, and instruction. It executes decoded
14D joint-position commands in left-arm/gripper then right-arm/gripper order.

```bash
export PATCHWAM_POLICY=/path/to/native/policy_0.safetensors
export PATCHWAM_ROBOTWIN_ROOT=/path/to/RoboTwin
export PATCHWAM_ROBOTWIN_TASK=adjust_bottle
export PATCHWAM_ROBOTWIN_TASK_CONFIG=/path/to/task_config.yml
export PATCHWAM_INSTRUCTION='Adjust the bottle.'
patchwam evaluate --config configs/evaluation/robotwin.yaml
```

Instructions can be supplied per episode or through an explicit external
provider. Use the original seen/unseen instruction protocol and valid seed
manifest for formal comparisons. `expert_check` checks a supplied seed; it
does not silently search for another seed. Camera/state mappings can be overridden.

`control_hz=50` timestamps policy observations with a policy-step clock. The
direct task may advance a variable number of physics steps internally. Match
the recording clock before reporting history-conditioned temporal parity;
use an explicit timestamp provider when the runtime exposes that clock.

## Results, errors, and evaluation resume

Each template is a one-episode smoke protocol. Add the original task list,
episode seeds, horizons, splits, instructions, and episode count for a formal
benchmark. `benchmark.tasks` is a list of task mappings; each mapping overrides
the shared benchmark fields. `episode_seeds`, when supplied, must have one
entry per requested episode. Set the default seed in `evaluation.seed`; a separate
benchmark/task `seed` is rejected to keep recorded and executed seeds identical.

The evaluator saves `results.json` after every attempt. It records requested
episodes, completed episodes, errors, successes, seeds, task settings, checkpoint
identity, normalization hash, and live/EMA weight selection. A success rate is
reported only when every requested episode completes without execution errors.
Simulator/controller failures keep a diagnostic record and stop the run.

```bash
patchwam evaluate --config configs/evaluation/libero.yaml --resume
```

Resume skips completed episodes and retries failed attempts under the same
weights and protocol. Changed checkpoint identity or normalization/configuration
is rejected. Use a new output directory for a changed experiment.

The runner uses one process. Shard task lists explicitly into distinct output
directories when using multiple GPUs, and aggregate only matching protocols.
CPU interface/decoder tests and training-resume checks are development evidence;
full-size GPU and real simulator success rates remain separate validation stages.
