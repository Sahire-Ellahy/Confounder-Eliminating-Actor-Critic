# Publication validation

The branch was checked in a newly created, repository-local Python 3.11 virtual
environment installed with `python -m pip install -e '.[notebook]'`.

- `pip check`: no broken requirements.
- Core algorithm and reporting suite: 58 tests passed.
- Walkthrough geometry and evidence suite: 5 tests passed.
- Notebook: all 10 code cells executed, with seven rendered figures and no errors.
- Default notebook execution launched no training.
- Export: 108 complete seed traces and 384 unique source-run endpoints; all
  source studies and raw run hashes were verified before export.
- The maintained training-source hashes match the completed studies. Cleanup
  changes packaging, documentation and presentation, not the learning runtime.

`notebooks/execution.json` records the executed notebook digest, figure digests
and exact package versions. `notebooks/evidence/manifest.json` records source
provenance and export hashes. The GitHub workflow repeats tests and headless
notebook execution from the published tree.

The saved data establishes the exact-model experimental behavior. It is not an
independent learned-critic ablation, a hidden-change-point experiment, or a
confidence interval based on only three seeds. These limits remain explicit in
the notebook and mathematical ledgers.
