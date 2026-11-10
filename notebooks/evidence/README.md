# Portable evidence

`curves.json.gz` contains verified per-seed curves for weak-signal M=64,
endpoints for all 384 unique runs, environment preflight facts and saved-history
audit summaries. It is an explicit derived export, not a replacement for raw
policy snapshots and trajectories.

`manifest.json` records source study identities, individual run-manifest hashes,
training-source hashes and SHA-256 hashes of the exported bundle and plot assets.
The loader verifies these before the notebook displays results. No authentication,
external download, local absolute path, or raw training cache is needed.

Maintainers can regenerate it with `python scripts/export_walkthrough.py` when
the original completed studies are present under `runs/`. A new experiment must
use a new run folder and must not overwrite the bundled historical evidence.
