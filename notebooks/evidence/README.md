# Portable evidence

The maintained export contains 336 physical-strength seed curves and endpoints:
two environments, both stationary modes, four diversion probabilities, three
seeds, seven methods, and 2,048 episodes per cell. The main controls are CEAC,
offline Graves-Lai, full-support Bayesian IDS, IDAC, UCB Q-learning and optimistic
POMD; zero-support IDS is a separately labeled sensitivity arm.

The compressed curve bundle includes physical exposure counters, terminal
success, evaluator regret and information traces, per-seed strength statistics,
history audits and matched-domain preflight facts. The manifest records run,
study, source archive and figure hashes. No authentication, download, local
absolute path, raw cache, or legacy package is needed to execute the notebook.

Regenerate this maintained export with:

    python scripts/export_walkthrough.py --study runs/pathology_strength_20261110

The exporter first verifies the complete study and every recorded report
artifact. The notebook verifies the source archive and live diagnostic kernels
separately before displaying results. The original historical studies remain
under their local run folders and Git history; none of their surrogate curves
has been renamed as IDAC or a published optimism algorithm.
