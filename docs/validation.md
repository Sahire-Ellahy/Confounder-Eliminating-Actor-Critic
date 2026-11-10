# Publication validation

The maintained branch uses the repository-local Python 3.11 environment.
The current physical-strength/reporting revision passed:

- Core algorithm and reporting suite: 75 tests.
- Walkthrough geometry and evidence suite: five tests.
- Notebook: eleven code cells and fourteen rendered figures, with no training.
- Physical-strength sweep: 336 completed matched cells, 2,048 episodes each,
  totaling 688,128 environment episodes. All seven recorded methods are in
  the portable notebook export.
- Scientific controls: fixed useful-route values/KL, optimal values, reward
  range and state/action domain; simulator/enumerator parity; closed-form
  DeepSea nuisance value; and actual beacon dwell/resumption.
- Reporting checks: identical seed identities and budgets, no duplicate or
  missing strength cells, fixed non-strength configuration/runtime, and paired
  subtraction before computing variation. All seven methods, including both
  Bayesian priors, appear in the primary figures for both environments.
- Prior audit: 48 full-support and 48 restricted-prior runs, matched at every
  environment/mode/strength/seed. Method labels are checked against actual
  recorded priors; the notebook verifies posterior support and exact linear
  exclusion regret, including the mode-0 negative control.

The earlier 48-cell comparator integration audit remains a distinct historical
validation. IDAC is checked numerically against the unmodified pinned upstream
class; UCB-Q and optimistic POMD have published-equation checks. Additional
tests verify information isolation, update ordering and episodic padding.
See [comparator provenance](comparator_provenance.md) and
[the physical-strength protocol](pathology_sweep.md).

The notebook execution record stores the executed notebook digest, figure
digests and package versions. Its evidence manifest stores source identities
and export hashes. The runtime archive and current diagnostic kernels are
verified separately. New studies archive their training source at launch.
Reports and the notebook share the same rendering functions; the GitHub
workflow repeats tests and notebook execution without local raw caches.

The new physical-strength sweep is distinct from the archived nuisance-alphabet
experiments. No historical surrogate curve is relabeled. The report explicitly
separates POMD bonus saturation, finite-budget value gaps, nuisance exposure,
and deliberately excluded Bayesian prior support. Full-support Bayesian IDS
remains a separate control. These are not learned-critic causal ablations or
hidden-change-point experiments; seed bands are descriptive standard deviations,
not confidence intervals or asymptotic impossibility claims.
