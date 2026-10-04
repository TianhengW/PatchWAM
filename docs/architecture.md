# Architecture and action patches

PatchWAM represents robot actions and visual predictions in a shared patch-token
space. A single denoising transformer predicts future visual tokens and an action
chunk from the current observation, instruction, and robot state.

## Fixed action patches

For an action vector with `A` coordinates, the default codec repeats each coordinate
`floor(128 / A)` times and fills the remaining token coordinates with zeros. An
`H`-step action chunk therefore becomes `H` tokens of width 128. Decoding averages
the repeated coordinates and reverses the configured action scale.

For example, a 14-dimensional action occupies 126 token coordinates, with two
zero-filled coordinates. A 16-step chunk becomes a `[16, 128]` token sequence.
The codec has no learned parameters. Time padding and action-dimension padding
are tracked separately and expanded into the token-level loss mask.

## Joint flow matching

The policy uses one noise level for both future-image and action tokens. A sampled
uniform time `u` is shifted to `sigma = s*u / (1 + (s-1)*u)`, where the benchmark
configurations use `s = 5`. Training interpolates clean tokens with Gaussian noise
and predicts the velocity `noise - clean`.

The default objective combines weighted visual and action losses:

```text
loss = 0.5 * loss_video + 1.0 * loss_action
```

Language, proprioception, and current-image tokens form the conditioning prefix.
The prefix cannot attend to noisy future or action tokens. Generated tokens can
attend to the prefix and to each other. The FLUX.2 adapter implements this attention
contract while retaining the official transformer's parameter structure.

The autoencoder and base text-encoder weights remain frozen. The denoising transformer,
proprioception projection, and optional RoboDojo VL LoRA adapters are trained.
Action sampling follows a descending
Euler flow schedule and returns normalized actions together with future latent tokens.

See [model assets](models.md) for the official source revision and frozen encoders,
[action sampling](inference.md) for decoding and controller inputs, and
[conditioning and training options](conditioning_and_training.md) for optional
vision-language history, guidance, EMA, and Self-Flow variants.
