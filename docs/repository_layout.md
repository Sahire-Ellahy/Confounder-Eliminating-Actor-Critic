# Repository layout and cleanup

The maintained CEAC runtime lives in `ceac/`. Presentation code lives separately
in `ceac_walkthrough/`, with one notebook under `notebooks/`. Completed raw
studies keep their original paths and archived runtime; current source changes
do not relabel or overwrite their results.

- `ceac/`: environments, finite models, learners, runner, immutable evidence, reports.
- `ceac_walkthrough/`: explanatory figures and portable evidence loading.
- `notebooks/`: executed environment and results walkthrough.
- `notebooks/evidence/`: compact, hash-checked exports from completed studies.
- `scripts/`: evidence export, notebook verification and historical audit tools.
- `tests/`: algorithm and presentation-interface checks, with a pinned upstream
  IDAC fixture under `tests/fixtures/` for source-parity checks.
- `audits/`: mathematical ledgers and a compact comparator-validation record;
  historical audit outputs remain under the local `archive/`.
- `docs/`: repository guide, validation status and comparator source/adaptation ledger.
- `runs/`: local raw training records and reports, excluded from Git.
- `legacy/inforatio/`: local archive of the unrelated 2022 package, old notebooks,
  examples, launch scripts and dependencies, excluded from this CEAC installation.
  A small attributed source fixture supports IDAC parity tests independently of
  the local archive.

No raw training run was deleted or relocated. The old root `setup.py` moved with
its original package so installing CEAC no longer pulls Ray/Torch/old Gym.
The previous root README is retained in the local legacy archive.

The comparator update removes retired learner registrations and two superseded
aggregate plot assets. The current figures use explicit environment/mode names.
Raw run caches, local virtual environments and generated build metadata stay
excluded from Git; they are not needed to execute the published notebook.
