# Data setup

Use local LeRobot v2 datasets with episode tables under `data/`, camera videos under
`videos/`, and metadata under `meta/`. Configure dataset roots explicitly; this repository
does not download or include research data.

Preserve each benchmark's action semantics, action horizon, normalization statistics,
non-idle window filter, and camera arrangement when reproducing an existing run. A changed
data contract is a separate experiment, even when the optimizer settings are identical.

The default paper recipes use 16 future actions and endpoint image pairs. RoboTwin uses
14 action/state dimensions and a compact three-camera image. LIBERO uses 7 action dimensions,
8 state dimensions, and a horizontal two-camera image. RoboCasa interfaces and normalization
depend on the dataset version; use the statistics belonging to that specific release.

See `src/patchwam/data/PROVENANCE.json` for the adapted data-processing functions and their
source licenses. The reader must raise on unsupported formats or missing action fields;
it must not silently turn a changed dataset into a successful reproduction claim.

Video timestamps must match the declared FPS and requested frame times. The default
matching tolerance is `1e-4` seconds; change `lerobot_tolerance_s` explicitly when the
dataset's encoding requires a different tolerance.
Native cached text features require `qwen_text_cache_format=qwen3_flux2`; incompatible
cache formats are rejected by the FLUX.2 policy.

The dataset fingerprint includes metadata, normalization, selected rows, and file
size/modification-time records for episode tables, external camera videos, and active
text caches. It does not hash full payloads or inspect external image paths embedded
inside parquet records. Keep those data assets immutable during a run.
