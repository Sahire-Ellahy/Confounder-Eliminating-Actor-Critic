# Portable evidence

The maintained export contains 336 physical-strength seed curves and endpoints:
two environments, both stationary modes, four diversion probabilities, three
seeds, seven methods, and 2,048 episodes per cell. The main controls are CEAC,
offline Graves-Lai, full-support Bayesian IDS, misspecified-prior Bayesian IDS,
IDAC, UCB Q-learning and optimistic POMD. Each has 48 runs and appears in the
main figures for both environments. The Bayesian priors are `(0.5, 0.5)` and
`(1, 0)`; the latter excludes the truth in mode 1, with mode 0 as a negative control.

The compressed curve bundle includes physical exposure counters, terminal
success, evaluator regret and information traces, per-seed strength statistics,
history audits and matched-domain preflight facts. The manifest records run,
study, source archive, figure and report PDF hashes. No authentication, download, local
absolute path, raw cache, or legacy package is needed to execute the notebook.

The [complete PDF report](report.pdf) is copied from the verified study without
modification. Its checksum is included in the portable evidence manifest.

Regenerate this maintained export after rendering the study with `--pdf`:

    python scripts/export_walkthrough.py --study runs/pathology_strength_20261110

The exporter first verifies the complete study and every recorded report
artifact. The notebook verifies the source archive and live diagnostic kernels
separately before displaying results. The original historical studies remain
under their local run folders and Git history; none of their surrogate curves
has been renamed as IDAC or a published optimism algorithm.
