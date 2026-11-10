# Confounder-Eliminating Actor-Critic

Can an RL agent purchase the specific trajectory evidence needed to rule out
models that change the optimal policy, instead of paying for irrelevant novelty?

This branch contains **Undertow DeepSea** and **False Beacon Frozen Lake**, a
maintained finite-model CEAC implementation, audited comparators, and an
executed visual walkthrough.

**Start with [the notebook](notebooks/Undertow_FalseBeacon.ipynb).** Its saved
outputs render on GitHub. It explains the route geometry, the confounding
mechanism, nuisance versus useful evidence, and what the completed runs show.

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

The notebook runs offline after installation and launches no training. A compact,
hash-verified evidence export is included; raw local training caches are not
required. It combines 336 base runs and 48 additional prior-support runs without
duplicating reused controls. There are three seeds per plotted condition.

For headless verification with the active interpreter:

```bash
python scripts/verify_notebook.py
python -m unittest discover -s tests/ceac -v
python -m unittest discover -s tests/walkthrough -v
```

## What these benchmarks establish

- **Undertow:** nuisance moves consume the navigation horizon without progress.
  Their symbols are diverse but have zero likelihood information about the task.
- **False Beacon:** reeds and shallow surveys can attract coverage while the
  inner route hides a profitable outer alternative. A deep survey reveals the mode.
- Increasing the nuisance alphabet worsens the finite-policy OIR/IDAC adaptation,
  clearly in DeepSea and modestly in Frozen Lake. CEAC and full-support IDS remain
  stable; not every comparator deteriorates with nuisance diversity.
- Zero-prior IDS is an explicitly misspecified support treatment. It does not
  refute full-support IDS or show that CEAC discovers undeclared models.

The plotted CEAC is a **single episode-loop, exact-model specialization**. The
realized mode is unknown to CEAC; candidate laws, evaluation and policy separation
are available. Certified deployment requires a reward-optimal response for every
currently plausible model. The offline Graves–Lai oracle additionally knows the
realized mode from the outset. These runs do not isolate learned-critic causality:
34/48 CEAC runs made zero actor/critic updates and no learned actor proposal was
admitted. See the [implementation ledger](audits/ceac_v90_corrections_20261110.md)
and [benchmark ledger](audits/ceac_undertow_design_20261110.md).

## Run new experiments

```bash
python -m ceac.run --output runs/my_smoke --benchmarks weak_signal_m4 \
  --environments deepsea frozenlake --methods ceac oracle confidence policy_ucb oir_idac coverage ids \
  --modes 0 1 --seeds 0 --episodes 32 --workers 1
python -m ceac.report --input runs/my_smoke
```

This short run checks integration, not convergence. Always choose a new output
folder. See [ceac/README.md](ceac/README.md) for the complete multi-seed slate,
cache identity, scientific assumptions and report controls. Core dependencies are
NumPy and SciPy; `pip install -e '.[report]'` adds plotting without Jupyter.
PDF report export additionally requires system `pandoc` and `pdflatex`.

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

The unrelated old `inforatio` files were moved to a local archive and excluded
from this publication. The existing training runtime and raw study paths were
preserved. Details: [repository layout](docs/repository_layout.md).
