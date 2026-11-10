For an executed visual introduction and a fresh-clone installation, start with
[the repository guide](../README.md) and [the notebook](../notebooks/Undertow_FalseBeacon.ipynb).
The default notebook uses the portable evidence export and needs no raw run cache.

# Maintained finite-model CEAC experiments

This package implements the corrected finite-model single-loop lane for Shifting
DeepSea and Shifting-Shore FrozenLake. It is separate from the legacy
`inforatio` package and does not import the released agents or runners.

The v90 manuscript and historical release remain reference artifacts under
`IDS/doc/ceac_v90_accessibility_release`. The initial audit is in
`code/audits/archive/ceac_v90_fidelity_20261110.md (local historical archive)`; mathematical specialization choices
and remaining theorem-level limits are in
`code/audits/ceac_v90_corrections_20261110.md`.

## Install and launch from one place

Run these commands from `Research/2027/IDS/code` with Python 3.10 or newer:

```bash
python3 -m venv .venv-ceac
source .venv-ceac/bin/activate
python -m pip install -r requirements-ceac-report.txt
python -m unittest discover -s tests/ceac -v
python -m ceac.run --output runs/my_ceac_check --episodes 256
python -m ceac.report --input runs/my_ceac_check
```

The learner dependencies are NumPy and SciPy (`requirements-ceac.txt`).
The report requirements additionally install Matplotlib. Gymnasium is optional; the benchmark
kernels retain their released compatibility implementation. The archived
`legacy/inforatio/setup.py` is separate from CEAC; its Ray/Torch dependencies are
unrelated to this lane.

A new output directory is required. `--resume` verifies identical source,
configuration, runtime versions, seed identity and prior, and skips only fully
completed, hash-verified cells. An incomplete cell is deliberately not silently
reused. Preserve it and use a new output folder. Never point a new experiment at
a historical release-results directory.

The deployment comparison uses the same launch interface:

```bash
python -m ceac.run --output runs/ceac_deployment_comparison_20261110 --environments deepsea frozenlake --methods ceac ceac_actor oracle confidence policy_ucb oir_idac coverage ids ids_rare ids_zero --modes 0 1 --seeds 0 1 2 --episodes 256
python -m ceac.report --input runs/ceac_deployment_comparison_20261110
```

The completed run with that name is already preserved. Choose a **new output
name** for a new experiment; matching completed runs may be verified with
`--resume`. The earlier `runs/ceac_v90_validation_20261110` used actor-only CEAC
and remains an archived result with its original report, not a current launch
target. The current renderer rejects relabeling that old study as certified
CEAC.

Add `--pdf` to the report command when system `pandoc` and `pdflatex` are
installed. The Markdown source uses `$...$` / `$$...$$`; preview needs a math
extension. A compiled PDF requires no Markdown math support. Renderer and figure
sources are hashed separately from the runtime, so plot edits do not invalidate
experiment caches.

`--contexts 0 1 2` checks all visible contexts. Each context is a separate
stationary run with its own evidence state; this runner does not inject hidden
temporal shifts or pool different contexts' evidence. `ids_rare` assigns mode 1
prior probability 0.001; `ids_zero` assigns it zero. The latter is an explicit
support-exclusion treatment, with the asymmetric support reported in each run.

## Architecture and audit checkpoints

| Module | Responsibility | Checkpoint |
|---|---|---|
| `environments/` | Benchmark reset/step dynamics | Exact return/KL parity with released dynamics |
| `model.py` | Public finite candidate family; exact policy evaluation and response | Common support, finite horizon, common-policy occupancy-ratio and information-sufficiency checks |
| `contracts.py` | Immutable policy, transition and episode records | Frozen policy digest and explicit behavior label |
| `design.py` | Bounded Graves–Lai and target/calibration LPs | Primal/dual, row, global column and phase-I infeasibility certificates |
| `learning.py` | On-policy updates, likelihood confidence, replay and cumulative debt | No off-policy actor updates; zero-price identity; no discarded quota; single credit |
| `deployment.py` | Reward-only response and certificate | Optimal for every plausible model; revoked when the model set changes |
| `agents.py`, `ids.py` | CEAC, exact IDS and offline oracle control | Full policy responses; only the oracle receives the realized model |
| `comparators.py` | Confidence, policy-UCB, OIR/IDAC and coverage | Explicit model access; shared enriched bank for the trajectory-only controls |
| `run.py` | Experiment construction, seeds and execution | Method-order-independent RNGs; evaluator-only truth |
| `evidence.py` | Streaming evidence and cache identity | Source/config/version identity; immutable completed cells; artifact hashes |
| `report.py` | Render verified saved evidence | Counts, hashes and study/run identity checked before rendering |

The runner is the only component that owns the realized environment. Agents
receive `FiniteModel`, which independently enumerates *all declared candidate
models*. A hypothetical candidate copy carrying label 0 or 1 is model knowledge,
not access to which candidate generated the experiment. Privileged simulator
`info` dictionaries never cross the learner interface.

Each cell contains `manifest.json`, `status.json`, `metrics.jsonl`,
`trajectories.jsonl`, `state_index.npz`, `policies/<digest>.npz`,
`audit_events.json`, and `result.json`. Frozen actor behavior policies are saved
even when they were never admitted to the diagnostic bank. The original audit
utility remains a read-only diagnostic of the **historical** release.

## Interpretation

This is an explicitly labeled **exact-model specialization** of v90. Model
uncertainty is handled through likelihood evidence; candidate-model returns and
KLs are evaluated exactly rather than estimated with an unproved replay radius.
Exact separators may add a policy before the actor discovers it. The default
`ceac` also uses the reward-only certified deployment response; `ceac_actor`
retains actor-only execution as an ablation. The deployment correction closes a
missing execution interface without claiming actor SGD has converged. Actor updates
remain fresh, on-policy updates, and the report separates actor convergence
from diagnostic-design correctness. No coefficient-one finite-sample or
nonstationary theorem is asserted by passing the numerical tests.


## Undertow DeepSea and False Beacon Frozen Lake

Public benchmark settings live in `ceac/benchmarks.py`, separate from the learner.
The default `legacy` setting preserves the earlier experiment dynamics. See
`audits/ceac_undertow_design_20261110.md` for the equations, exact preflight checks,
allocation-bound calibration and interpretation limits.

From `IDS/code`, the complete exploratory slate is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m ceac.run --output runs/ceac_undertow_20261110 \
  --benchmarks strong_signal_m4 weak_signal_m4 weak_signal_m16 weak_signal_m64 \
  --methods ceac oracle confidence policy_ucb oir_idac coverage ids \
  --modes 0 1 --seeds 0 1 2 --episodes 512 --workers 2 --report-pdf
```

Use a fresh output name for a new execution. This is 336 cells / 172,032 episodes,
using CPU finite-model planning. The short `ceac_undertow_preflight_20261110`
study is an integration test, not a performance result. Refined settings default
to mass bound and normalization 128; `--mass-bound` makes this choice explicit.

Outputs remain under the study folder, grouped by benchmark and environment.
`status.json` counts completed cells, `results.json` indexes their endpoints,
`environment_audit.json` records mathematical preflight evidence, and the final
`report.pdf` includes event traces and nuisance-size sensitivity. Current progress
can be read without importing the learning package:

```bash
cat runs/ceac_undertow_20261110/status.json
```

This slate fixes the task mode within each run. No hidden temporal shift or
Bayesian-support exclusion is introduced. A larger nuisance alphabet expands
observable states, while physical geometry and the number of seed policies are
held fixed across the weak-signal arms.


## Prior-support comparison without rerunning controls

The approved display names are **Undertow DeepSea** and **False Beacon Frozen
Lake**. Their stable cache identifiers are `deepsea` and `frozenlake`.

The support extension uses the same training runtime with the existing
`ids_rare` (mode-1 prior 0.001) and `ids_zero` (mode-1 prior 0) arms:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m ceac.run --output runs/ceac_prior_support_20261110 \
  --benchmarks strong_signal_m4 weak_signal_m64 \
  --methods ids_rare ids_zero --modes 0 1 --seeds 0 1 2 \
  --episodes 512 --workers 1

python -m ceac.report --input runs/ceac_prior_support_20261110 \
  --reference runs/ceac_undertow_20261110 \
  --reference-methods ceac oracle confidence policy_ucb oir_idac coverage ids --band sd --pdf
```

Both studies have completed. The report command is independent of training and
can be rerun against the saved studies. Use a fresh output name to launch another
execution. This adds 48 runs; the consolidated report reads 168 compatible
controls from the reference study, giving 216 verified runs across nine methods. It fails closed
on source/runtime, environment, learner-setting, budget or seed incompatibility.
`report_audit.json` records provenance for both studies; `prior_support_audit.json`
checks the per-episode absorbing posterior and exact linear-regret prediction.
`report_status.json` distinguishes training completion from report completion.

This is a deliberately excluded-prior-support treatment. It does not imply
failure of correctly specified full-support IDS. See the mathematical benchmark
ledger for the assumptions and the matched mode-0 negative control.


## Seed-shaded traces and evidence attribution

The report defaults to `--band sd`: each time point shows the seed mean plus or
minus one **sample** standard deviation (denominator n-1). `--band seed_range`
selects the observed min/max instead. Both are descriptive, not confidence
intervals; the current slate has only three seeds per curve. One-seed curves have
no variability band. Expected-return smoothing uses a causal 16-episode mean
within each seed before aggregation. The episode index is not a replicate.

The first two figures summarize weak-signal M=64 for both environments: policy
value, cumulative pseudo-regret and observed log-likelihood evidence. Mode 0
compares all seven standard methods; mode 1 emphasizes CEAC, offline oracle and
IDS prior support. Detailed plots retain all benchmark/method comparisons.

`trace_statistics.csv.gz` exports each curve's seed count, mean, sample SD,
min/max, band edges and smoothing width. Values are not clipped when a plot's
nonnegative domain clips the lower band. `history_audit.json` independently
reconstructs the cumulative likelihood mixture and confidence sets from saved
transitions, and checks all reported policy credits against physical episodes.
Both files are hashed by `report_audit.json`. Raw run artifacts are unchanged.

These curves validate cumulative history accounting and certified finite-model
deployment. They **do not isolate learned-critic improvement**: in the full
336-run slate, 34 of 48 CEAC runs have zero actor/evidence-critic updates, and no
learned actor proposal is admitted in any of the 48. Diagnostic design and
response certification use exact declared-model evaluation. Retaining trajectory
strata and likelihood sufficient statistics is distinct from resampling replay
into learned critic updates. An active learned-estimator lane and matched
ablations would be needed for that causal claim.


## Aggregate nuisance sensitivity and CEAC labels

The base report now opens with `figures/aggregate_nuisance_context0_mode0.pdf`
and its mode-1 counterpart. Each combines both environments and shows total
regret, extra regret relative to M=4, and nuisance visits. Only the matched
weak-signal M=4,16,64 settings share this axis. M counts nuisance observation
symbols; it is not physical current strength or useful task-signal amplitude.

`nuisance_sensitivity.json` preserves the per-seed endpoints and descriptive
summaries. Differences are taken within each seed before calculating the SD.
The renderer rejects missing/duplicate seed pairs or any change to another run
configuration field. Modes and environments are never pooled into extra seeds.

The plotted label is **CEAC (exact-model single loop)**. This is the maintained
single episode-loop specialization with exact candidate-model evaluation and
policy responses. Certified deployment selects a reward-only response whose
gap is within tolerance for every currently plausible model, after outstanding
statistical/design debt is served. It uses no true-mode label. The **offline
Graves–Lai oracle** receives that label before training, fixes its normalized
allocation once, and executes its debt or the true-model optimal policy.
Certification does not imply that the learned actor has converged.

The completed normal-mode sensitivity supports increasing OIR/IDAC cost and
stable CEAC cost. Full-support IDS is also stable. Other comparators have mostly
unchanged costs as M increases; the report does not relabel those flat curves as
increasing failures. The excluded-prior IDS treatment remains a separate axis.
