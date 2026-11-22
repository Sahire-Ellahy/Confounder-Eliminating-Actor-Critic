#!/usr/bin/env python3
"""Build, execute, and SHA-256-audit notebooks/Pathological_Search_Suite.ipynb from portable evidence."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from ceac_pathologies.report import load_export as load_pathology_export
from ceac_ranked.study import load_gi_ranked_export


def build_notebook() -> nbf.NotebookNode:
    cells = []

    def md(cid: str, text: str):
        c = nbf.v4.new_markdown_cell(text.strip() + "\n")
        c.id = cid
        cells.append(c)

    def code(cid: str, text: str):
        src = text.strip() + "\n"
        compile(src, f"<{cid}>", "exec")
        c = nbf.v4.new_code_cell(src)
        c.id = cid
        cells.append(c)

    md(
        "pathologies-intro",
        """# Pathological Search Suite: Sequential Locks, Signalling Activation & Pairwise Interaction Screens

This notebook loads, verifies, and renders all three benchmark families extending Confounder-Eliminating Actor-Critic (CEAC) beyond two-step junctions:

1. **Sequential Combination Lock & Bidirectional ("Diabolical") Combination Lock (`ceac_pathologies`, `locks`)** — Sweeps gated lock horizon $L \\in \\{1, 2, 3, 4\\}$ ($H = L + 1$) with anti-shaped step costs, a 3-lane diffusing dead-end sink, and an opposing $0.50$ decoy chain while holding route returns and diagnostic KL strictly invariant across depth.
2. **Epistatic Signalling / Protein Activation under Pleiotropic Confounding (`ceac_pathologies`, `epistasis`)** — Tests discovery of a $1$-in-$128$ rare synthetic-rescue responder amidst $k \\in \\{0, 1, 2, 3\\}$ non-causal passenger-omics channels ($M = 128 \\cdot 2^k \\in \\{128, 256, 512, 1024\\}$) and $4^k - 1 \\in \\{0, 3, 15, 63\\}$ zero-reward off-pathway pleiotropic states.
3. **Pairwise Interaction Screen on the Published Horlbeck 2018 K562 Matrix + Greedy Price-Ranked CEAC (`ceac_gi` & `ceac_ranked`)** — Evaluates sequential pairwise discovery on a verified 10-variable / 45-pair human K562 interaction score panel (`FDPS–PPP2R1A`, $\\text{GI} = -5.24$) under $k \\in \\{0, 1, 2, 3\\}$ non-causal single-variable main-effect confounders, comparing certified Section-3 `ceac` against the **greedy maximally-disagreeing + dynamic evidence-price ranked replay** extension (`ceac_ranked`) and Russo–Van Roy optimal-action-MI IDS (`ids_action`).

All figures are rendered live from compact, SHA-256-verified portable exports under `notebooks/evidence/`; raw `/runs` trajectories are never required and no training is launched when executing the notebook.""",
    )

    code(
        "pathologies-setup",
        """from pathlib import Path
import sys

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "ceac_pathologies").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IPython import get_ipython
_ip = get_ipython()
if _ip is not None:
    _ip.run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import numpy as np

from ceac_pathologies.report import (
    KINDS,
    LABELS,
    draw_architecture,
    draw_epistasis_discovery_traces,
    draw_epistasis_overlaid_performance,
    draw_learning,
    draw_mechanisms,
    draw_response,
    endpoints,
    load_export,
    paired_ceac_comparisons,
)
from ceac_ranked.study import (
    RANKED_LABELS,
    draw_gi_discovery_traces,
    draw_gi_overlaid_performance,
    draw_gi_ranked_learning,
    draw_gi_ranked_mechanisms,
    draw_gi_ranked_response,
    load_gi_ranked_export,
)

locks = load_export(ROOT / "notebooks/evidence/pathologies_locks")
epistasis = load_export(ROOT / "notebooks/evidence/pathologies_epistasis")
gi_ranked = load_gi_ranked_export(ROOT / "notebooks/evidence/pathologies_gi_ranked")

print(
    f"Verified portable exports: {len(locks['runs'])} lock cells, "
    f"{len(epistasis['runs'])} signalling-activation cells, and "
    f"{len(gi_ranked['sweep_rows'])} GI sweep + {len(gi_ranked['calibration_rows'])} GI calibration cells. "
    "No training launched."
)""",
    )

    md(
        "locks-section",
        """## 1. Sequential Combination Lock & Bidirectional ("Diabolical") Lock (`locks`)

Both lock families share the calibrated $L = 1$ easy baseline (`seeds 1001, 1002`, `tasks 0, 1`, `late_gap <= 0.025`). Increasing $L \\in \\{1, 2, 3, 4\\}$ adds gated key steps with step tension $-0.004$ (rebated at the terminal gate so route returns remain identically $0.50$, $0.10/0.90$, and $0.48$) while wrong keys drop irreversibly into a 3-lane zero-cost dead sink. In `bidirectional_lock`, an opposing left chain unlocks deterministic $0.50$ under key $0$ while the right target chain requires key $1$.""",
    )

    code(
        "locks-architecture",
        """fig = draw_architecture("locks")
plt.show()""",
    )

    code(
        "locks-dose-response",
        """fig = draw_response(locks)
plt.show()""",
    )

    code(
        "locks-mechanisms",
        """fig = draw_mechanisms(locks)
plt.show()""",
    )

    code(
        "locks-learning-fans",
        """for metric in ("cumulative_regret", "window_gap"):
    fig = draw_learning(locks, "bidirectional_lock", task=1, metric=metric)
    plt.show()""",
    )

    code(
        "locks-summary-table",
        """lock_rows = [r for r in endpoints(locks) if r["task"] == 1 and r["dose"] in (1, 4)]
print(f"{'Environment':32s} | {'Algorithm':34s} | {'L=1 Regret':>11s} | {'L=4 Regret':>11s} | {'L=4 Gap':>8s}")
print("-" * 108)
for kind in ("bidirectional_lock", "combination_lock"):
    by_m = { (r["method"], r["dose"]): r for r in lock_rows if r["kind"] == kind }
    for m in ("idac", "pomd", "ucb_q", "ids_decoy", "ids_product", "ceac", "ceac_natural", "oracle"):
        r1, r4 = by_m[(m, 1)], by_m[(m, 4)]
        print(f"{KINDS[kind]:32s} | {LABELS[m]:34s} | {r1['regret_mean']:11,.1f} | {r4['regret_mean']:11,.1f} | {r4['late_gap_mean']:8.4f}")""",
    )

    md(
        "epistasis-section",
        """## 2. Signalling Pathway / Protein Activation under Off-Pathway Pleiotropy (`epistasis`)

Models discovering a rare $1$-in-$128$ synthetic-rescue signalling subtype (`K1 ∧ K2` synergy pays $0.90$ vs. $0.10$ in $127$ non-responders; baseline care pays $0.50$; phospho-probe pays $0.48$) as $k \\in \\{0, 1, 2, 3\\}$ non-causal passenger modules ($\\eta \\in \\{0, 1\\}^k$, $M = 128 \\cdot 2^k$) and $4^k - 1$ zero-reward off-pathway pleiotropic arms are added.

- **Whole-model MI (`ids_product`) vs. Optimal-action MI (`ids_action`)**: Whole-model mutual information $I((\\theta, \\eta); Z^\\pi)$ spends early episodes profiling decision-irrelevant passenger-omics channels, whereas Russo–Van Roy optimal-action MI $I(A^\\star; Z^\\pi)$ and CEAC's coordinate-free row dominance assign **zero** value to passenger channels.
- **Model-free coverage / bonus collapse**: `IDAC`, `Optimistic POMD`, and `UCB-Q` are drawn into the $4^k - 1$ off-pathway states as $k$ increases.""",
    )

    code(
        "epistasis-architecture",
        """fig = draw_architecture("epistasis")
plt.show()""",
    )

    code(
        "epistasis-dose-response",
        """fig = draw_response(epistasis)
plt.show()""",
    )

    code(
        "epistasis-mechanisms",
        """fig = draw_mechanisms(epistasis)
plt.show()""",
    )

    code(
        "epistasis-overlaid-performance",
        """fig = draw_epistasis_overlaid_performance(epistasis, task=1)
plt.show()""",
    )

    code(
        "epistasis-discovery-traces",
        """fig = draw_epistasis_discovery_traces(epistasis, dose=3, task=1)
plt.show()""",
    )

    code(
        "epistasis-learning-fans",
        """for metric in ("window_gap", "cumulative_regret"):
    fig = draw_learning(epistasis, "epistatic_signalling", task=1, metric=metric)
    plt.show()""",
    )

    code(
        "epistasis-summary-table",
        """epi_rows = { (r["method"], r["dose"]): r for r in endpoints(epistasis) if r["task"] == 1 }
print(f"{'Algorithm':38s} | {'k=0 Regret':>10s} | {'k=3 Regret':>10s} | {'k=3 Gap':>8s} | {'Omics%':>7s} | {'OffPath%':>8s}")
print("-" * 96)
for m in ("idac", "pomd", "ucb_q", "ids_canonical", "ids_product", "ids_action", "ceac", "ceac_natural", "oracle"):
    r0, r3 = epi_rows[(m, 0)], epi_rows[(m, 3)]
    print(
        f"{LABELS[m]:38s} | {r0['regret_mean']:10,.1f} | {r3['regret_mean']:10,.1f} | "
        f"{r3['late_gap_mean']:8.4f} | {100*r3['passenger_omics_fraction']:6.2f}% | {100*r3['pleiotropic_fraction']:7.1f}%"
    )""",
    )

    md(
        "gi-section",
        """## 3. Pairwise Interaction Screen (Horlbeck 2018 K562 Matrix) & Greedy Price-Ranked CEAC (`gi_screen` + `ceac_ranked`)

Grounded in the verified Horlbeck et al. (2018, *Cell*) K562 pairwise interaction score matrix (`data/external/horlbeck2018/GI_map_treeview.zip`, SHA-256 verified offline), `select_panel(10)` selects a 10-variable panel (`45` pairs) containing one strong interaction (`FDPS–PPP2R1A`, $\\text{GI} = -5.2413$) with all other 44 pairs $> -2.5$, permuted per seed to eliminate index tie-break artifacts.

### Why Greedy Disagreement + Dynamic Evidence-Price Buffer Ranking (`ceac_ranked`) Matters
1. **Eliminating uniform $\\binom{G}{2}$ seed-bank floors**: Certified Section-3 `ceac` pre-registers `control` + all `45` pair arms (`46` columns $\\times 513$ replay-floor episodes $\\approx 23,600$ startup episodes on `gi_arrayed`). `GreedyPricedCEAC` (`ceac_ranked`) starts with `bank_size = 1` (`control`) and acquires candidate columns lazily via coordinate-free confounder-projected disagreement $\\min_{j' \\in C'} I^m(\\pi; j')$, certifying `Task 1` with an active bank of **~2 columns** instead of `46`.
2. **Dynamic evidence-price replay ranking & KKT price attenuation**: `PriceRankedReplay` re-scores stored strata in $O(|\\mathcal{B}_t|)$ whenever $\\beta_t$ shifts, promoting informative strata with zero new environment steps, while attenuating effective actor exploration prices as $\\rho_t = \\min(1, \\text{scale}(t)/(1 + t_{\\text{actor}}))$ so `Task 0` (null panel) converges cleanly to `control` instead of perpetually probing low-reward pair arms.""",
    )

    code(
        "gi-architecture",
        """fig = draw_architecture("gi_screen")
plt.show()""",
    )

    code(
        "gi-dose-response",
        """fig = draw_gi_ranked_response(gi_ranked["summary_rows"])
plt.show()""",
    )

    code(
        "gi-mechanisms",
        """fig = draw_gi_ranked_mechanisms(gi_ranked["summary_rows"])
plt.show()""",
    )

    code(
        "gi-overlaid-performance",
        """fig = draw_gi_overlaid_performance(gi_ranked, dose=3)
plt.show()""",
    )

    code(
        "gi-discovery-traces",
        """fig = draw_gi_discovery_traces(gi_ranked, dose=3)
plt.show()""",
    )

    code(
        "gi-learning-fans",
        """for kind in ("gi_arrayed", "gi_multiplexed"):
    for metric in ("window_gap", "cumulative_regret"):
        fig = draw_gi_ranked_learning(gi_ranked, kind, metric=metric)
        plt.show()""",
    )

    code(
        "gi-summary-tables",
        """from collections import defaultdict

cal_by_key = defaultdict(list)
for r in gi_ranked["calibration_rows"]:
    cal_by_key[(r["kind"], r["task"], r["method"])].append(r)

print("=== Easy-Case Calibration Diagnostic (dose k=0, seeds 1001 & 1002, 65,536 episodes) ===")
print(f"{'Screen Variant':38s} | {'Task':>4s} | {'Method':46s} | {'Bank':>5s} | {'Gap':>7s} | {'Regret':>9s}")
print("-" * 122)
for (kind, task, m), grp in sorted(cal_by_key.items()):
    bank_s = f"{np.mean([x['bank_policies'] for x in grp]):5.1f}" if grp[0].get("bank_policies") is not None else "   — "
    print(
        f"{KINDS[kind][:38]:38s} | {task:4d} | {RANKED_LABELS[m][:46]:46s} | {bank_s} | "
        f"{np.mean([x['late_gap'] for x in grp]):7.4f} | {np.mean([x['cumulative_regret'] for x in grp]):9,.1f}"
    )

print("\\n=== Held-Out Sweep Summary (Task 1, k=0 -> k=3, seeds 0, 1, 2) ===")
gi_map = { (r["kind"], r["method"], r["dose"]): r for r in gi_ranked["summary_rows"] if r["task"] == 1 }
print(f"{'Screen Variant':38s} | {'Method':46s} | {'k=0 Reg':>9s} | {'k=3 Reg':>9s} | {'k=3 Gap':>7s} | {'Omics%':>6s}")
print("-" * 128)
for kind in ("gi_arrayed", "gi_multiplexed"):
    for m in ("idac", "pomd", "ucb_q", "ids_additive", "ids_product", "ids_empirical", "ids_action", "ceac", "ceac_natural", "ceac_ranked", "ceac_ranked_natural", "oracle"):
        r0, r3 = gi_map[(kind, m, 0)], gi_map[(kind, m, 3)]
        print(
            f"{KINDS[kind][:38]:38s} | {RANKED_LABELS[m][:46]:46s} | {r0['regret_mean']:9,.1f} | "
            f"{r3['regret_mean']:9,.1f} | {r3['late_gap_mean']:7.4f} | {r3['profiling_pct']:5.2f}%"
        )""",
    )

    md(
        "runners-section",
        """## 4. Experimental Runners & Reproducible CLI Entrypoints

All studies and portable evidence bundles can be reproduced from the repository root or invoked programmatically via the entrypoints below (disabled by default during notebook verification so rendering remains instantaneous and deterministic):""",
    )

    code(
        "runners-entrypoints",
        """RUNNER_COMMANDS = {
    "unit_tests": ".venv/bin/python -m unittest discover -s tests/pathologies -v",
    "locks_and_epistasis_suite": ".venv/bin/python scripts/run_pathologies_suite.py --tag repro_v1 --workers 10 --protocols locks epistasis",
    "gi_ranked_study": ".venv/bin/python -m ceac_ranked.study --tag repro_ranked_v1 --workers 10",
    "rebuild_this_notebook": ".venv/bin/python scripts/render_pathologies_notebook.py --execute",
}
for name, cmd in RUNNER_COMMANDS.items():
    print(f"[{name}]\\n  {cmd}")""",
    )

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "pathologies_suite_schema": 1,
        },
    )
    return nb


def execute_and_audit(nb_path: Path):
    import base64
    from contextlib import redirect_stdout
    import io
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Verify all three portable evidence exports before executing notebook
    locks_dir = ROOT / "notebooks/evidence/pathologies_locks"
    epi_dir = ROOT / "notebooks/evidence/pathologies_epistasis"
    gi_dir = ROOT / "notebooks/evidence/pathologies_gi_ranked"
    load_pathology_export(locks_dir)
    load_pathology_export(epi_dir)
    load_gi_ranked_export(gi_dir)

    notebook = nbf.read(nb_path, as_version=4)
    nbf.validate(notebook)

    exec_ns = {"__name__": "__main__"}
    exec_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        exec_count += 1
        cell.execution_count = exec_count
        cell.outputs = []
        pending_figs = []

        orig_show = plt.show

        def _capture_show(*_a, **_kw):
            for num in list(plt.get_fignums()):
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
                pending_figs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                plt.close(fig)

        plt.show = _capture_show
        stdout_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf):
                exec(compile(cell.source, f"<{cell.id}>", "exec"), exec_ns)
        finally:
            plt.show = orig_show
            plt.close("all")

        text_out = stdout_buf.getvalue()
        if text_out:
            cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=text_out))
        for png_b64 in pending_figs:
            cell.outputs.append(
                nbf.v4.new_output(
                    "display_data",
                    data={"image/png": png_b64, "text/plain": "<Figure>"},
                    metadata={},
                )
            )

    nbf.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    images = sum("image/png" in out.get("data", {}) for c in code_cells for out in c.outputs)
    if images < 12:
        raise AssertionError(f"Expected >= 12 rendered PNG figures, got {images}")

    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbf.write(notebook, nb_path)

    audit = dict(
        schema=1,
        status="complete",
        notebook=str(nb_path.relative_to(ROOT)),
        notebook_sha256=hashlib.sha256(nb_path.read_bytes()).hexdigest(),
        code_cells=len(code_cells),
        rendered_figures=images,
        training_launched=False,
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib", "nbclient", "ipykernel")
        },
        evidence_manifests={
            name: hashlib.sha256((ROOT / "notebooks/evidence" / name / "manifest.json").read_bytes()).hexdigest()
            for name in ("pathologies_locks", "pathologies_epistasis", "pathologies_gi_ranked")
        },
    )
    (ROOT / "notebooks/pathologies_execution.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Executed and verified {nb_path.name}: {len(code_cells)} code cells, {images} rendered figures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute notebook in-place and write SHA-256 audit")
    args = parser.parse_args()
    nb_path = ROOT / "notebooks/Pathological_Search_Suite.ipynb"
    nb = build_notebook()
    nbf.write(nb, nb_path)
    if args.execute:
        execute_and_audit(nb_path)


if __name__ == "__main__":
    main()
