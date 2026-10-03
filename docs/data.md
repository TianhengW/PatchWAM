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
