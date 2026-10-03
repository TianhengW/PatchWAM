# Server source audit

Audit date: 2026-10-04. The source archives are stored outside this repository,
under the migration workspace's `sources/<source-id>/` directories. Each archive
has a `SNAPSHOT_MANIFEST.json` with its original location, Git state, and every
copied file's SHA256. The public repository contains no server checkout, dataset,
checkpoint, environment, credentials, or log archive.

These are source references for the rewrite. Downloading an archive does not
establish runtime parity, a completed integration, or reproduction of a score.

## Version matrix

| Source ID | Version | Role | Archive manifest SHA256 |
| --- | --- | --- | --- |
| `h800-paper-actionpatch` | `action-as-patch`, `e99fa8cab47dff84f33005f577663c690ea0dc7a`; 120 changed/untracked entries | Current H800 FLUX.2 research workspace; LIBERO/LIBERO-plus training and evaluation references | `b8350c704096d2a0bb924eb07a7ce6204efa5f31c449bd0774fbcb1cbb6f6be4` |
| `digua-paper-actionpatch-current` | `latent`, `db3f2132ad2562702208d43a949d4de60a1108d5`; 97 changed/untracked entries | Current Digua workspace; changed after the historical C2R run | `91697c05aa7b3d13c0ae92e5cf778217aef99f87de33e06fe401a54038f692d7` |
| `local-c2r-paper-locked` | Clean `main`, `18a2e96285d0911f8ecc2aeb8a7c5bb60aeb8285` | Canonical historical C2R-DR4 reconstruction, resolved config, original launch manifest, and source lock | `cf9fd17534cf24160b4030dec42ca7d27327a09a5169c244da2a572c58eab076` |
| `digua-robotwin-c2r-paper-run` | No Git metadata; config and dataset statistics only | Original C2R-DR4 run metadata, separate from the later workspace | `bfec2ca4e16b2300dcf83153033e9672c7710a92faa7da30b9cea9b125ec2c7a` |
| `local-robocasa-human300-release` | No Git metadata; released config and inference source | Legacy Human300 12D policy release associated with step 559660 | `eabd4a694c4c0d36eb9797c57bb837181c31261e70319bab675aa9016d11abd4` |
| `h800-robocasa-flux2-legacy` | No Git metadata; source archive | H800 legacy RoboCasa FLUX.2 training workspace; current files are not an immutable training-time checkout | `9c82cf7868067aaac1612d8c3032abd9d27ded33ba89c47cc00294f648396972` |
| `h800-robocasa-command-q0199` | No Git metadata; source archive | Later RoboCasa command-gripper and quantile-normalization fixes | `615d080b5be7647cc3305ebbb339dc5e9e6c524a5795ad4f0478fd07048505ad` |
| `h800-robocasa-native-framework` | No Git metadata; source archive | NeMo runtime used by the later RoboCasa recipe | `c04cfd65068cae50a5fdbcb3af81982144721e5c78c5c3e8dda68004609afd74` |
| `h800-robocasa-adapter-dependencies` | No Git metadata; source archive | Shared RoboCasa sampler and video-augmentation dependencies | `c70a94adc784afca766aeb4cb88ee80cd09f60dfc434689d3f8cb49c087b6bca` |
| `digua-robotwin-c2r-fixed` | No Git metadata; source archive | Later NeMo/AP337 C2R numeric-contract and inactive-dimension fixes | `62791771b5575eee0fcd96e03dbac17732afa22d03e8435668a690d6eb0368e2` |

The dirty workspaces are locked by their file manifests in addition to HEAD.
The clean C2R archive includes `repro/SOURCE_LOCK.sha256` and
`repro/REPRODUCIBILITY.md`. Its canonical model/factory and DR4 files were
reconstructed from backups preceding subsequent linear-codec/DR5 changes.
The selected text-only snapshot omits external simulator vendors and large
assets; it is not a complete simulator installation.

## FLUX.2 recipe contracts

The following contracts are the selected historical integration references.
The current NeMo formal runs are listed separately below.

### RoboTwin/RobotWin C2R-DR4

- Archive/config: `local-c2r-paper-locked/repro/configs/c2r_dr4_resolved.yaml`.
  Config SHA256: `5e474227d4a234960f362b13ef8760168785ff933d585820eaa6f654daf304b9`.
- AdamW: learning rate `1e-4`, weight decay `1e-2`, betas `(0.9, 0.95)`;
  cosine schedule with legacy 5% warmup and a 1% learning-rate floor
  (`1e-6`); seed 42; bf16; gradient clip 1.0.
- Actual launcher: 16 ranks, batch 16 per rank, accumulation 1, global batch 256;
  100 epochs / 150000-update cap; checkpoint every 5000 updates. An obsolete
  four-node comment in the task YAML is superseded by the resolved config and
  original launcher.
- Data: 50Hz, raw 14D action and 14D state, horizon 16, 17-frame windows with
  two endpoint image frames. No chunk-relative transform; global z-score
  normalization. Keep the original normalization statistics.
- Cameras: high view above the left/right wrist views; high resized to
  192x256 and wrists to 96x128, final 288x256. Initial per-view resize is 240x320.
- Selection: dense starts inside the non-idle ranges plus
  `periodic_prefix(period=550, keep_first=50)` episode filtering. Both filters
  are part of the recipe. Time padding is excluded from action loss; the
  14 raw dimensions are valid dimensions.
- DR4: application probability 0.8; photometric, AdaIN-style, and Fourier
  amplitude perturbations; background replacement disabled.
- Historical formal checkpoint: step 140000; clean 91.56%, random 66.72%,
  average 79.14%; 50 tasks x two conditions x 100 episodes = 10000 episodes.
  This result belongs to the historical recipe and checkpoint.

### RoboCasa Human300 legacy release

- Archive/config: `local-robocasa-human300-release/training_config/resolved_config.yaml`.
  Config SHA256: `a5d6f1f2512154259f3dfb04519692eeb537f02a277aa6112b0dedfed233becd`.
- Released configuration: AdamW learning rate `1e-4`, weight decay `1e-2`,
  cosine schedule with a 1% learning-rate floor, legacy 5% warmup, seed 42,
  bf16, gradient clip 1.0. The
  associated legacy trainer uses betas `(0.9, 0.95)`; the release archive does
  not independently lock the complete training-time runtime.
- 16 ranks x batch 16 x accumulation 1 = global batch 256; five epochs,
  559660 updates; checkpoint interval 5000. The archived `resume` field is a
  later state-resume path and must not become a fresh-run initialization.
- Data: Human300 only, 301 roots; action 12D, proprioception 16D, 20Hz,
  horizon 16 / 17-frame window, two endpoint frames. Keep raw command action
  semantics, z-score statistics, and no chunk-relative action transform.
- Cameras: agentview-left above agentview-right and eye-in-hand, compact
  288x256 layout; per-view input 256x256. This is the legacy layout, not the
  later AP337 wrist-above-agents layout.
- Dense global stride 1, no non-idle filter; train/validation split 0.01.
  Preserve time-padding masks and the 12D action / 16D state dimensions.
- Release checkpoint: step 559660, weight SHA256
  `7877dd2546d7e2fe489aab79f84476266724b7392d48fe2f2861e9752cd4dccb`.
  The current audit did not download or recompute the weight hash; this is
  archive metadata.

### LIBERO and LIBERO-plus

- Archive: `h800-paper-actionpatch`; task configs:
  `configs/task/libero_flux2_klein_4b_actionpatch.yaml` and
  `configs/task/libero_plus_flux2_klein_4b_actionpatch.yaml`.
- Respective task-config SHA256:
  `4611d3801d7fa060138bb88d1066ff1212f30ebf15d8cd42d5792dec743eb7d1` and
  `01822f131af9c02b1856b954ced8c6bd8439dcf36e73e43fcdb3042cd3b861c6`.
- AdamW learning rate `1e-4`, weight decay `1e-2`, betas `(0.9, 0.95)`;
  cosine with legacy 5% warmup and a 1% learning-rate floor, bf16.
  Task defaults are batch 8 per rank,
  accumulation 1, 50 epochs, checkpoint interval 2000; launcher overrides
  determine the actual batch/world size. No immutable resolved launch
  configuration was identified here, so these defaults do not certify a
  particular paper score or checkpoint.
- Standard LIBERO selects the spatial, object, goal, and long-horizon
  no-noops datasets; LIBERO-plus selects its own LeRobot release. Keep their
  separate episode sets and evaluation protocols.
- Data: 20Hz, action 7D / proprioception 8D, horizon 16 / 17-frame window,
  endpoint frames. Front/wrist views resize to 224x224 and concatenate
  horizontally into 224x448. Global sample stride 1, no non-idle filter.
- Global min/max normalization; the first six action dimensions are delta
  coordinates and the seventh is the gripper. The delta mask zeros padded
  delta labels; padded steps remain excluded by the action loss mask.

All three legacy families use the parameter-free 128D repeated-action codec,
scale 1, a shared image/action sigma, shift 5, image/action loss weights 0.5/1,
and FLUX.2-Klein base 4B plus the FLUX.2 autoencoder and Qwen3-4B language
encoder. Asset versions, positional axes, prefix attention visibility, and
state conditioning require paired checks when moving to the rewritten model.

## Later fixes: separate migration targets

The NeMo/AP337 sources are not integrated by placing the legacy FLUX.2 code
and configurations in one repository.

- Later RoboCasa: unified80 action/state masks, observed future EEF pose
  relative to the chunk-start frame, command gripper from the original
  control label, q01/q99 normalization with an explicit statistics hash,
  20Hz/H16, wrist above the agent views, final 320x224 image, Qwen3-VL
  conditioning. `rc20_command.py`, `rc20_stats.py`, `rc_codec.py`,
  `normalization_command_q0199.json`, and the run's `formal.yaml` record the
  changed contract. It differs from legacy Human300 in labels, dimensions,
  normalization, image geometry, and encoder.
- Later C2R: raw14 scattered into unified80 slots
  `[0,1,2,3,4,5,16,29,30,31,32,33,34,45]`; joint actions are chunk-start
  relative and grippers remain absolute. q01/q99 statistics are checked
  against the raw cache; inactive normalized dimensions remain zero and
  are masked in training and sampling. 50Hz/H16 and compact288x256 geometry
  are retained. `robotwin_dataset.py`, `stats_contract.py`, `raw_stats.py`,
  `train_entry.py`, and the patched NeMo action/sampling modules record this
  separate contract.

These fixes must have their own named configurations and validation evidence.
Historical C2R/legacy Human300 results cannot be assigned to them.

## Legacy checkpoint conversion contract

Legacy policies store a payload containing `mot`, `proprio_encoder`, `step`,
and sometimes optimizer state. Encoders and the autoencoder are external
assets and are not supplied by this payload.

| Legacy weight | Rewritten `FluxAssetPolicy` destination |
| --- | --- |
| `mot["mixtures.video.transformer." + key]` | `policy.denoiser.transformer.` + key |
| `proprio_encoder[key]` | `policy.state_projection.` + key |

For a bare `PatchFlowPolicy`, remove the initial `policy.` from the destination.
The mapping is a specification, not evidence that a converted checkpoint was
loaded or evaluated. A converter must check tensor shapes, action/state
dimensions, all missing/unexpected keys, codec options, assets, and numerical
output parity. Legacy `dit` payloads, action experts, LoRA payloads, and NeMo
checkpoints need distinct explicit converters.

A legacy weights file is a warm start. Its `step` does not recover the new
optimizer, scheduler, rank RNG states, or data cursor. Only the new complete
training-state format can provide continuation under a matching data/optimizer
contract.

## Attribution and validation status

The legacy source uses MIT with its original copyright notice. Retained
data-processing code must keep that notice and its provenance. NeMo source
uses Apache-2.0 and retains its own copyright headers. The new Apache-2.0
license applies to the new implementation; it does not replace retained
third-party notices.

Archive transfer and local file-hash checks are complete. Training-recipe,
checkpoint-output, simulator-controller, and closed-loop score parity remain
separate validation gates. Current formal jobs were not edited, restarted,
stopped, or reclassified by this audit.
