# Publication validation

The maintained branch uses the repository-local Python 3.11 environment,
installed with `python -m pip install -e '.[notebook]'`. The current comparator
revision passed:

- Core algorithm and reporting suite: 65 tests.
- Walkthrough geometry and evidence suite: five tests.
- Notebook: all ten code cells executed, with eight rendered figures and no errors.
- Default notebook execution launched no training.
- Comparator integration: 48 completed cells covering two environments, two
  stationary modes, two seeds and six methods, with 128 episodes per cell.
- Portable historical export: 132 complete seed traces and 192 retained endpoints
  from 384 verified source runs. The 192 retired custom-control runs are excluded
  from display and retain their original provenance.

The primitive IDAC update is compared numerically against the unmodified pinned
upstream class. UCB-Q and optimistic POMD have published-equation checks;
additional tests verify information isolation, update ordering and episodic
padding. See [comparator provenance](comparator_provenance.md) for the source
references and declared adapters, and
[the integration audit](../audits/comparator_validation.json) for run hashes and
update counters.

`notebooks/execution.json` records the executed notebook digest, figure digests
and exact package versions. `notebooks/evidence/manifest.json` records export
hashes and historical source provenance. Its archived runtime is verified
separately from the current diagnostic kernels: the new comparator runtime is
not represented as the source of old results. New studies archive their source
at launch. The GitHub workflow repeats tests and headless notebook execution
from the published tree, without requiring raw local caches or the legacy package.

The integration runs validate execution, not convergence. In these short runs,
POMD's published bonuses leave all optimistic costs zero and the policy uniform.
The historical figures vary nuisance alphabet size, not physical strength.
The requested physical-strength sweep is not yet implemented or launched.
Neither study is a learned-critic causal ablation or a hidden-change-point
experiment; seed bands are descriptive standard deviations, not confidence
intervals. These limits remain explicit in the notebook and mathematical ledgers.
