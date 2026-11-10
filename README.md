# Confounder-Eliminating Actor-Critic

Can an RL agent purchase the specific trajectory evidence needed to rule out
models that change the optimal policy, instead of paying for irrelevant novelty?

This branch contains **Undertow DeepSea** and **False Beacon Frozen Lake**, a
maintained finite-model CEAC implementation, source-backed primitive controls, and an
executed visual walkthrough.

**Start with [the notebook](notebooks/Undertow_FalseBeacon.ipynb).** Its saved
outputs render on GitHub. It explains the route geometry, the confounding
mechanism, nuisance versus useful evidence, and what the completed runs show.

The [complete PDF report](notebooks/evidence/report.pdf) includes the corrected
seven-method comparisons, both Bayesian priors, diagnostics and audit details.

![Undertow route pathology](notebooks/assets/undertow.png)
![False Beacon route pathology](notebooks/assets/false_beacon.png)

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

The notebook runs offline after installation and launches no training. Its
portable evidence contains **336 physical-strength runs**: two environments,
two stationary modes, four diversion probabilities, three seeds and seven
methods, with 2,048 episodes per cell. It includes the source-backed optimism
and IDAC controls alongside both full-support and misspecified-prior Bayesian
IDS, CEAC and the oracle (48 runs per method).

For headless verification with the active interpreter:

```bash
python scripts/verify_notebook.py
python -m unittest discover -s tests/ceac -v
python -m unittest discover -s tests/walkthrough -v
```

## What the comparisons measure

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

The plotted CEAC is a single episode-loop, exact-model specialization. CEAC
and Bayesian IDS know the declared candidate laws; the primitive controls
learn from observed transitions. The offline oracle also knows the realized
mode. Exact-model deployment and learned actor convergence remain distinct.
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
checks all three replacements. The notebook now contains fresh matched
strength-sweep results, not renamed or pooled historical surrogate curves.

## Run new experiments

```bash
python -m ceac.run --output runs/my_smoke --benchmarks weak_signal_m4 \
  --environments deepsea frozenlake --methods ceac oracle ids ids_zero idac ucb_q pomd \
  --modes 0 1 --seeds 0 --episodes 32 --workers 1
python -m ceac.report --input runs/my_smoke
```

This short run checks integration, not convergence. Always choose a new output
folder. The [sweep guide](docs/pathology_sweep.md) gives the full launch command.
The standard report automatically emits all method overlays, strength curves,
paired additional-cost plots and failure diagnostics. The notebook imports
the same renderer and displays the selected audited study.

To regenerate its evidence and executed outputs from a completed study:

    python -m ceac.report --input runs/pathology_strength_20261110 --pdf
    python scripts/export_walkthrough.py --study runs/pathology_strength_20261110
    python scripts/verify_notebook.py --write

Core dependencies are NumPy and SciPy; the report extra adds Matplotlib.
PDF export additionally requires system pandoc and pdflatex. Historical raw
studies remain local and keep their original source identities.

## Layout

| Path | Responsibility |
|---|---|
| `ceac/` | Problems, learners, runner, immutable evidence and report rendering |
| `ceac_walkthrough/` | Explanatory maps and portable evidence loader |
| `notebooks/` | One executed notebook, compact evidence and figure assets |
| `scripts/` | Evidence export and notebook verification |
| `tests/` | Algorithm, statistical reporting and walkthrough checks |
| `audits/` | Mathematical scope and deviation ledgers |
| `runs/` | Local raw studies, excluded from Git |
| `legacy/upstream/` | Original standalone DeepSea script, retained unchanged |

The original `inforatio` package remains in a local archive; the new IDAC port
has a small pinned source fixture for numerical parity tests. Historical study
paths and archived runtimes are preserved. Details:
[repository layout](docs/repository_layout.md) and
[publication validation](docs/validation.md).
