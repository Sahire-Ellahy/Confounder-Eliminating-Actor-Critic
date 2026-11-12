# Physical pathology sweep and standard overlays

**Interpretation corrected:** rho is a physical diversion parameter, not a
validated learning-hardness score. This sweep did not establish a transition
from a learnable mild case to failure at stronger pathology. See the
[replacement design](hardness_redesign.md) before planning another full sweep.


The maintained sweep varies a physical diversion probability rho at fixed
geometry, horizon, nuisance alphabet (four symbols), useful diagnostic quality,
reward range and route payoff gap. It uses rho = 0.05, 0.25, 0.50 and 0.75.
Strictly interior probabilities preserve common positive outcome support and
the same reachable state/action domain at every strength.

## Dynamics and scientific controls

For the optional nuisance action, both candidate modes have the same mixture

$$P_{m,\rho}=(1-\rho)P_{\mathrm{pass}}+\rho P_{\mathrm{divert}}.$$

In Undertow DeepSea, a nuisance action always consumes one depth step and its
existing nuisance fee. On the pass branch it advances rightward; on the
diversion branch it loses that progress. An all-nuisance policy has exact value

$$J_{\rho}^{\mathrm{sonar}}=\tfrac12(1-\rho)^8-8(0.005),$$

and zero task KL. A joint categorical draw samples the independent nuisance
symbol and physical event; the exact kernel enumerator uses this same joint
law. Direct, challenge and probe policies retain their original value/KL.

In False Beacon Frozen Lake, the first reed visit draws a symbol and a
mode-independent activation bit. Activation causes one forced dwell on the
next physical step. The observed reed-state code records no activation,
pending dwell, or completed dwell; the action mask exposes the forced step.
The state is Markov, one simulator step advances elapsed time by one, and
seed routes resume the intended leg after the dwell. A late activation need
not finish its dwell before truncation; activations and actual lost steps
are logged separately. Inner, outer, shore and deep-survey routes bypass
this event and retain their value/KL.

Existing benchmark defaults retain their previous dynamics. The new
pathology_rho identifiers prevent overwriting historical studies.
Preflight checks state/domain hashes, action masks, non-strength parameters,
optimal values, reward range, and useful-policy value/KL before launching.
Tests additionally compare the DeepSea noise-route value with its closed
form and execute the actual beacon wait/resumption.

This is a fixed-mode experiment, not a hidden change-point study. Increasing
harm is not guaranteed to increase every algorithm's regret: a learner may
avoid it more quickly. In particular, full-support Bayesian IDS may remain
robust. No desired ranking is encoded in the renderer.

## Registered comparison

The initial sweep uses two environments, both stationary modes, three seeds,
four strengths, 2,048 episodes and seven methods: CEAC, offline Graves–Lai,
full-support Bayesian IDS, misspecified-prior Bayesian IDS, source-backed
IDAC, UCB-Hoeffding Q-learning and optimistic POMD. Both Bayesian treatments
are primary plotted controls and are included in the default runner slate.
This is 336 cells and 688,128 environment episodes. Each Bayesian treatment
has 48 runs: two environments × two modes × four strengths × three seeds.
There is no outcome-dependent hyperparameter search.

| Bayesian treatment | Method identifier | Prior on modes (0, 1) | Interpretation |
|---|---|---|---|
| Full support | `ids` | (0.5, 0.5) | True model has positive prior mass in both modes |
| Misspecified prior | `ids_zero` | (1, 0) | Excludes the truth in mode 1; mode 0 is a negative control |

The algorithm, likelihood family, budget and seed identities are otherwise
matched. Exact zero mass is preserved by Bayesian updating; a small positive
prior is a distinct treatment (`ids_rare`) and is not part of this slate.
Physical strength and prior support are separate experimental factors.

IDAC is a frequentist actor-critic for a cost/occupancy-entropy ratio. It is
not a frequentist replacement for Bayesian IDS's posterior information
objective. The primitive controls learn from transitions; CEAC and Bayesian
IDS additionally know the declared candidate models. CEAC's exact-model
deployment does not establish convergence of its learned proposal actor.
The zero-prior arm is a separate information-assumption failure, not a
physical-strength effect. See [source provenance](comparator_provenance.md).

## One report and notebook interface

From the repository root:

    python -m ceac.run --output runs/my_pathology_strength \
      --benchmarks pathology_rho005 pathology_rho025 pathology_rho050 pathology_rho075 \
      --environments deepsea frozenlake \
      --methods ceac oracle ids idac ucb_q pomd ids_zero \
      --contexts 0 --modes 0 1 --seeds 0 1 2 --episodes 2048 --workers 4
    python -m ceac.report --input runs/my_pathology_strength --pdf
    python scripts/export_walkthrough.py --study runs/my_pathology_strength
    python scripts/verify_notebook.py --write

The report automatically creates, for each environment/mode:

- A headline **strength-on-x** view: total regret (symlog), paired extra
  regret above the mildest dose (linear, signed), and late expected value gap.
  The two environments share one figure, with intervention-specific labels.
- An annotated CEAC-advantage matrix: comparator regret minus CEAC regret
  at every dose. Negative entries retain CEAC losses. Bold entries indicate
  lower CEAC regret in every recorded paired seed, not a significance test.
- A generated comparison table separating total regret saved, the increase
  in that advantage over mild dose, and late value-gap reduction. Counts of
  lower/tied/higher CEAC regret accompany the paired mean and sample SD.

- All seven methods in the main overlays at each strength, including both
  Bayesian prior treatments, with consistent colors and a labeled symlog
  regret axis. Full-support IDS is blue; misspecified-prior IDS is fuchsia.
- STEERING-style training traces: one method per panel, strength-colored
  curves, and shared symlog regret axes (linear within ±0.01), including signed
  added regret. Markers sit above all lines at staggered recorded episodes;
  data values are never offset. Zero has a visible margin inside the axes.
- Paired added-regret traces relative to rho=0.05; subtraction happens within
  each seed before computing variation, and negative values are retained.
- A strongest-dose endpoint comparison with one row per method, numeric means
  and sample SD. Exact ties and near-equal controls remain individually readable.
- Strength diagnostics: cumulative regret, late expected value gap, terminal
  success, nuisance visits, lost progress/dwell, and useful task information.
- Dedicated Bayesian prior-support overlays supplement the main figures;
  the misspecified-prior arm also has its own strength-colored method panel.

The file ceac_comparisons.json retains each matched method-minus-CEAC
contrast and its summary. The extra advantage is

$$[R_A(\rho)-R_C(\rho)]-[R_A(\rho_{min})-R_C(\rho_{min})],$$

computed within each seed before aggregation. This distinguishes an existing
baseline advantage from added robustness as dose increases. Matching rejects
duplicate endpoints, missing seed/dose/method pairs and unequal budgets.
The independent unit remains the training seed, not episodes or prompts.

The file pathology_statistics.json retains per-seed endpoints, paired added
regret and descriptive diagnostic flags. The compressed trace_statistics.csv.gz
retains exported training-time statistics. Sample-SD bands are descriptive,
not confidence intervals; method panels require the same seed identities and
episode budgets. Missing doses, changed runtime sources, changed non-strength
configuration, or inconsistent trace/endpoint values fail validation.

The diagnostic notebook imports the drawing functions from ceac/pathology_report.py.
The compact main notebook audits whether this sweep establishes learning hardness.
Both read a hash-verified portable export. It never launches training.
All modes are present; its MODE setting switches the main displayed mode.

POMD bonus saturation, nuisance exposure and a finite-budget value gap are
reported separately. A baseline plateau does not demonstrate that increasing
pathology caused it, and a finite experiment is not an asymptotic impossibility
result. Figures show only measured results; no custom policy-bank comparator
is restored or renamed.

## Observed outcome of the first slate

All 336 cells completed. In shifted mode (1), the paired increase in
cumulative pseudo-regret from rho=0.05 to rho=0.75 was:

| Control | Undertow DeepSea | False Beacon Frozen Lake |
|---|---:|---:|
| UCB Q-learning | 48.74 ± 44.20 | 2.12 ± 3.67 |
| Optimistic POMD | 40.14 ± 0.00 | 0.062 ± 0.000 |
| IDAC | 35.94 ± 13.52 | -0.054 ± 0.171 |
| Full-support Bayesian IDS | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Misspecified-prior Bayesian IDS | 0.00 ± 0.00 | 0.00 ± 0.00 |

These are means ± one sample SD over three paired seeds, not confidence
intervals. POMD's zero seed variation in evaluator regret comes from its
unchanged uniform policy; sampled returns can still vary. Zero added regret
for misspecified-prior IDS means its 102.4 cumulative regret is unchanged
across physical strengths; it does not mean the policy has zero regret.

The primitive controls already have large value gaps at mild strength.
The new DeepSea effect adds to that baseline problem. The additional
False Beacon effect is small; the experiment does not establish substantial
strength-dependent deterioration there. UCB-Q's DeepSea mode-0 paired
change is negative and variable, further ruling out a blanket monotonic
failure claim. Full-support Bayesian IDS and CEAC reach the optimal
last-window value in both environments; the excluded-support IDS arm
incurs exactly 102.4 regret in mode 1 at every strength.

The plots therefore distinguish finite-budget poor learning, extra
nuisance cost, and prior-support failure. They do not establish that
evidence-directed exploration or a learned CEAC critic is the unique cause
of recovery, given the model-information differences between methods.

## Reading the CEAC comparison

At the strongest dose in mode 1, CEAC has roughly 1.37 regret in DeepSea and
1.55 in Frozen Lake, versus about 1,128–1,133 and 1,155–1,173 for the three
primitive optimism/IDAC controls. The much smaller incremental-cost panel
is essential: most of that separation already exists at the mild dose.
Full-support IDS has about 0.088 and 0.064 regret, respectively, so this is
not a claim that CEAC dominates all evidence-directed algorithms. Both
attain zero last-window expected value gap. The offline oracle remains an
explicit privileged reference, not a finite-time regret lower bound.

The current False Beacon dose is activation probability, **not breadth**.
Physical geometry, nuisance alphabet and the one-step dwell length remain
fixed. A breadth claim requires a separately identified experiment; these
results must not be relabeled to imply one. Likewise, excluded-prior IDS's
mode-1 failure is due to prior support, not stronger physical diversion.
