# Confounder-Eliminating Actor-Critic

The maintained [notebook](notebooks/Undertow_FalseBeacon.ipynb) asks whether **the same algorithm learns an easy problem and deteriorates as environmental pathology increases**. Its STEERING-style main figures follow **one algorithm over time**, with increasingly dark curves for stronger pathology and seed-SD shading. The story moves from irrelevant coverage and unresolved optimism to a broken Bayesian task–cue prior, then reconstructs CEAC’s actual reward/evidence updates.

Primary CEAC is the manuscript’s **reward critic, price-conditioned evidence critic, and evidence-priced policy-labeled replay**, implemented in `ceac_pseudocode`. The selector serves statistical debt, then design debt, then the actor. The replay master uses observed returns and likelihood ratios. Exact model responses supply global certificates; they do not replace the learned actor’s action selection. No ablation is in the primary comparison.

The verified studies contain 144 physical-burden cells and 90 fixed-prior-shift cells, with three held-out seeds per condition. These are **two-step junction variants**, separate from the historical grid environments. IDAC, UCB Q-learning, optimistic POMD, both Bayesian priors, CEAC and the offline Graves–Lai oracle have explicit information-access contracts.

- [Curated mechanism report](notebooks/evidence/mechanisms/report.md) and [PDF](notebooks/evidence/mechanisms/report.pdf): when each method fails and where evidence enters the actor.
- [Physical burden: methods as Undertow actions or beacon breadth grow](notebooks/evidence/hardness_burden/assessment.md).
- [Frozen Bayesian prior: both tasks, all doses and posterior diagnostics](notebooks/evidence/hardness_prior/assessment.md).
- [Actor-failure diagnosis](audits/ceac_actor_failure_20261112/assessment.md): dense evidence, early sampling noise, softmax saturation, and conservative replay prices.
- [Actor-stability assessment](notebooks/evidence/actor_stability/assessment.md) and [protocol](docs/actor_stability_protocol.md): implemented schedules, interior-simplex constraints and guarded natural/mirror updates. Calibration preserves the original endpoint-first selection; a separately registered comparison tests the natural update's learning-cost tradeoff.
- [Implementation and reproduction](docs/hardness_implementation.md), [pseudocode fidelity audit](audits/ceac_pseudocode_fidelity_20261111.md).

The original three-seed physical-burden study shows substantial deterioration of IDAC and optimistic POMD while CEAC’s final losses stay small on those seeds. Fresh five-seed optimizer confirmation exposes additional original-CEAC failures at Undertow B=64; the guarded natural update reduces those large actor errors, with a probability-floor cost in other cases. The original comparison figures retain their recorded optimizer and seed set. UCB’s increased learning cost is distinct from persistent final loss. The frozen correlated prior fails at full reversal; product-support IDS remains successful. **CEAC has a held-out task-0 actor-learning exception**, preserved in the means, bands and [saved-observation audit](audits/ceac_primary_actor_replay_20261111.json). The study does not support universal robustness or isolate each component’s causal effect.

The earlier rho sweep did not establish the intended easy-to-hard transition. Its [diagnostic notebook](notebooks/Undertow_FalseBeacon_Diagnostics.ipynb), [historical PDF](notebooks/evidence/report.pdf) and [audit](audits/hardness_review_20261111/assessment.md) remain available. Those historical CEAC-labeled runs include the earlier exact-deployment shortcut and must not be substituted for the corrected primary implementation.

## Install and run the notebook

From this repository's root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m ipykernel install --sys-prefix --name ceac-undertow --display-name "Python (CEAC)"
python -m jupyter lab notebooks/Undertow_FalseBeacon.ipynb
```

Choose **Python (CEAC)**. In VS Code, select this repository's `.venv/bin/python`.
The notebook dependencies include `ipykernel`; installing inside a different
Python environment does not repair the selected kernel. Windows activation is
`.venv\Scripts\Activate.ps1` in PowerShell.

The notebook runs offline after installation and launches no training. Its hash-verified exports contain the 234 original comparison cells plus 56 optimizer-calibration and 260 fresh-seed confirmation cells; raw trajectories remain local under ignored `runs/`. The diagnostic notebook retains the earlier 336 physical-strength cells. Portable exports include per-seed endpoints, uncertainty summaries, mechanism diagnostics and PDF figures.

For headless verification with the active interpreter:

```bash
python scripts/verify_notebook.py
python -m unittest discover -s tests/pseudocode -v
python -m unittest discover -s tests/stability -v
python -m unittest discover -s tests/hardness -v
python -m unittest discover -s tests/ceac -v
python -m unittest discover -s tests/walkthrough -v
```

## Historical grid comparisons

- **Undertow:** increasing nuisance diversion probability makes a noisy move
  more likely to lose rightward progress.
- **False Beacon:** increasing activation probability makes the reed detour
  more likely to require a forced physical dwell step.
- Preflight fixes the state/action domain, useful-route values, optimal values,
  reward range and diagnostic signal across strengths.
- The main overlays compare greedy UCB Q-learning, optimistic softmax POMD,
  frequentist IDAC, both full-support and misspecified-prior Bayesian IDS, CEAC
  and the offline oracle. All seven methods appear in both environments.
- Separate plots isolate paired additional regret and show value gaps,
  successful arrivals, nuisance exposure, wasted steps and useful information.
- Bayesian priors are explicit: full support uses `(0.5, 0.5)`; the restricted
  prior uses `(1, 0)`. The latter is misspecified when mode 1 is true; mode 0
  is its negative control. Both are in the main plots and default run slate.

The historical curve labeled CEAC includes an added exact-model deployment response. It is preserved for reproducibility, not used as primary CEAC evidence. The corrected primary results use `ceac_pseudocode`. In both studies, CEAC and IDS receive declared candidate laws, while primitive controls learn from trajectories; the oracle additionally receives the realized model.
The report diagnoses finite-budget plateaus; it does not assume that every
alternative deteriorates monotonically with nuisance strength.

See the [physical-strength protocol](docs/pathology_sweep.md),
[implementation ledger](audits/ceac_v90_corrections_20261110.md) and
[historical benchmark ledger](audits/ceac_undertow_design_20261110.md).

## Source-backed comparators

`idac` ports the original tabular softmax/two-critic update from the supplied
`inforatio-main.zip` and Sahire's pinned repository. `ucb_q` implements Jin et al.
(2018), Algorithm 1, at primitive state/action level. `pomd` implements optimistic softmax mirror descent from Shani et al. (2020),
Algorithm 2. None uses the old policy bank. See the [source and adaptation ledger](docs/comparator_provenance.md) for
exact equations, parity checks, reward normalization and reset semantics.
The earlier 48-cell [integration audit](audits/comparator_validation.json)
checks all three replacements. The diagnostic appendix contains matched
strength-sweep results; the historical audit records their scientific interpretation.

## Prepared physical-breadth experiment

A separate False Beacon experiment adds **0, 1, 2, 4 or 8 traversable chamber
cells** behind one uninformative entrance. Preflight verifies unchanged useful
routes, task signals, horizon, optimal values and oracle costs. The reachable
domain deliberately grows; this can also change published optimism bonuses.

The [registered plan](experiments/false_beacon_breadth.json) uses five seeds,
both modes, all seven methods and 8,192 episodes: **350 cells**. The full
sweep has not been launched. Its B=0 baseline still needs the revised easy-case
calibration before it can test learning hardness. The 28-cell, 32-episode integration smoke passed;
its outputs are validation, not evidence of a performance trend. See the
[protocol and launch commands](docs/false_beacon_breadth.md) and
[validation ledger](audits/beacon_breadth_validation_20261111.json).

Print the plan without starting training:

    .venv/bin/python scripts/run_beacon_breadth.py --stage plan

The standard report automatically adds breadth-response figures, chamber
exposure diagnostics and paired CEAC comparisons for completed breadth studies.

## Render the new studies and notebook

The standard renderer verifies the completed condition grids, reconstructs all saved CEAC actor updates (cached against input/source hashes), exports the mechanism report and PDF, and executes the notebook without training. It also incorporates an existing complete actor-stability export. Pass `--stability runs/ceac_actor_stability_20261112` to verify and rebuild that export from its raw cells:

```bash
python scripts/render_hardness.py \
  --burden runs/ceac_primary_burden_20261111 \
  --prior runs/ceac_primary_prior_20261111 --execute
```

For new training, use the calibration-gated commands in the [implementation guide](docs/hardness_implementation.md). The registered configuration is [experiments/ceac_primary.json](experiments/ceac_primary.json). Never compare uncalibrated hard cases to an already failing baseline.

## Run the historical grid benchmarks

```bash
python -m ceac.run --output runs/my_smoke --benchmarks weak_signal_m4 \
  --environments deepsea frozenlake --methods ceac oracle ids ids_zero idac ucb_q pomd \
  --modes 0 1 --seeds 0 --episodes 32 --workers 1
python -m ceac.report --input runs/my_smoke
```

This short run checks integration, not convergence. Always choose a new output
folder. The [sweep guide](docs/pathology_sweep.md) gives the full launch command.
The standard report automatically emits all method overlays, strength curves,
paired additional-cost plots, strength-on-x endpoint comparisons, signed
CEAC-advantage figures/tables and failure diagnostics. Per-seed method
contrasts and their summaries are exported in `ceac_comparisons.json`.
The diagnostic notebook uses the historical renderer. Reproduce the original
scientific assessment without training:

    python scripts/review_pathology_design.py --pdf

To regenerate the historical evidence from a completed study:

    python -m ceac.report --input runs/pathology_strength_20261110 --pdf
    python scripts/export_walkthrough.py --study runs/pathology_strength_20261110

Core dependencies are NumPy and SciPy; the report extra adds Matplotlib.
PDF export additionally requires system pandoc and pdflatex. Historical raw
studies remain local and keep their original source identities.

## Layout

| Path | Responsibility |
|---|---|
| `ceac/` | Problems, learners, runner, immutable evidence and report rendering |
| `ceac_pseudocode/` | Primary two-critic CEAC, replay master, calibration and studies |
| `ceac_hardness/` | Junction variants, finite model compiler, IDS, primitive controls and reports |
| `ceac_stability/` | Separately versioned full-CEAC optimizer settings, frozen calibration and fresh-seed assessment |
| `ceac_walkthrough/` | Explanatory maps, scientific assessment and portable evidence loader |
| `notebooks/` | Compact main walkthrough, detailed diagnostic appendix and verified evidence |
| `scripts/` | Staged breadth launcher, evidence export and notebook verification |
| `experiments/` | Registered configurations and explicitly labeled design proposals |
| `tests/` | Algorithm, statistical reporting and walkthrough checks |
| `audits/` | Mathematical scope and deviation ledgers |
| `runs/` | Local raw studies, excluded from Git |
| `legacy/upstream/` | Original standalone DeepSea script, retained unchanged |

The original `inforatio` package remains in a local archive; the new IDAC port
has a small pinned source fixture for numerical parity tests. Historical study
paths and archived runtimes are preserved. Details:
[repository layout](docs/repository_layout.md) and
[publication validation](docs/validation.md).
