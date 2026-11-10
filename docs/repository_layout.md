# Repository layout and cleanup

The maintained CEAC runtime remains in `ceac/`; its training-source files and
completed run paths are unchanged. Presentation code lives separately in
`ceac_walkthrough/`, with one notebook under `notebooks/`.

- `ceac/`: environments, finite models, learners, runner, immutable evidence, reports.
- `ceac_walkthrough/`: explanatory figures and portable evidence loading.
- `notebooks/`: executed environment and results walkthrough.
- `notebooks/evidence/`: compact, hash-checked exports from completed studies.
- `scripts/`: evidence export, notebook verification and historical audit tools.
- `tests/`: algorithm and presentation-interface checks.
- `audits/`: current mathematical ledgers; historical audit outputs under `archive/`.
- `runs/`: local raw training records and reports, excluded from Git.
- `legacy/inforatio/`: local archive of the unrelated 2022 package, old notebooks,
  examples, launch scripts and dependencies, excluded from this CEAC publication.

No raw training run was deleted or relocated. The old root `setup.py` moved with
its original package so installing CEAC no longer pulls Ray/Torch/old Gym.
The previous root README is retained in the local legacy archive.
