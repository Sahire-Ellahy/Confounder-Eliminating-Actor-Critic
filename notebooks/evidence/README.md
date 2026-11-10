# Portable evidence

`curves.json.gz` contains 132 verified per-seed curves for weak-signal M=4,16,64,
endpoints for 192 retained runs from 384 verified unique source runs, environment preflight facts and saved-history
audit summaries. Per-curve configurations allow the same matched-condition audit
used by the raw-cache report. These are nuisance-diversity controls, not a
physical undertow/false-beacon strength sweep. It is an explicit derived export, not a replacement for raw
policy snapshots and trajectories.

`manifest.json` records source study identities, individual run-manifest hashes,
training-source hashes and SHA-256 hashes of the exported bundle and plot assets.
The loader verifies these before the notebook displays results. No authentication,
external download, local absolute path, or raw training cache is needed.

Maintainers can regenerate it with `python scripts/export_walkthrough.py` when
the original completed studies are present under `runs/`. A new experiment must
use a new run folder and must not overwrite the bundled historical evidence.

The 192 retired custom-control runs are excluded from display. Their source run
hashes remain in the manifest. `source_bundle.tar.gz` pins the exact historical
training source. The loader checks that archive and the current diagnostic
kernels separately; replacing the runner does not change historical evidence.
No historical OIR-surrogate curve is relabeled as IDAC.
