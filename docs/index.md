# PatchWAM documentation

[Repository overview](../README.md) · [Paper](https://arxiv.org/abs/2609.25961)

Commands in these guides run from the repository root unless stated otherwise.

## Start here

1. [Install PatchWAM and run the CPU smoke check](getting_started.md).
2. Prepare [model assets](models.md) and [LeRobot v2 datasets and statistics](data.md).
3. Select a [benchmark recipe](training_recipes.md), then follow
   [GPU launch, training validation, and checkpoint resume](training.md).
4. Run [RoboCasa, RobotWin/RoboTwin, or LIBERO evaluation](evaluation.md) with the
   checkpoint's original action, camera, normalization, and controller contract.

CPU checks establish implementation behavior. Full-size GPU training and real
simulator results require separate verification; see the [verification record](migration.md).

## Guides

| Guide | Contents |
| --- | --- |
| [Getting started](getting_started.md) | Python setup, optional dependencies, and synthetic checks |
| [Model assets](models.md) | Pinned FLUX.2 source, transformer, autoencoder, and Qwen assets |
| [Data preparation](data.md) | LeRobot v2 layout, field mapping, statistics, filters, and text caches |
| [Training recipes](training_recipes.md) | Benchmark configurations, dimensions, data selection, and optimizer settings |
| [Training](training.md) | Overrides, single/multi-node GPU launch, validation, outputs, and resume |
| [Closed-loop evaluation](evaluation.md) | Simulator setup, action decoding, history reset, seeds, results, and evaluation resume |
| [Conditioning and training options](conditioning_and_training.md) | VLM, causal histories, CFG, EMA, and the three Self-Flow variants |
| [Inference](inference.md) | Policy APIs, synthetic and real-observation sampling, and physical action decoding |
| [Architecture](architecture.md) | Fixed action patches, flow matching, attention, and loss masks |
| [Development](development.md) | CPU checks, setup troubleshooting, and repository layout |
| [Verification record](migration.md) | Provenance, completed checks, and pending reproduction work |

Citation, acknowledgements, and licensing remain in the [README](../README.md#citation).
