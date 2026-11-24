#!/usr/bin/env python3
"""Build and execute notebooks/Undertow_FalseBeacon_Diagnostics.ipynb organized strictly by environment:
Part I: All Undertow DeepSea Variants (8x8 Grid -> Controlled Nuisance-Breadth Junction -> Multi-Step Path-Dependent Undertow)
Part II: All False Beacon Frozen Lake Variants (Shore Grid -> Controlled Chamber & Cue-Reversal Junctions -> Multi-Step Path-Dependent False Beacon)
Part III: Complete Integrity Assertions, Paired Advantage Tables & Reproducible CLI Entrypoints
"""
from __future__ import annotations
import base64
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf


def build_diagnostics_notebook() -> nbf.NotebookNode:
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
        "diag-overview",
        """# Undertow DeepSea & False Beacon Frozen Lake — Complete Environment-Organized Suite

This notebook organizes every benchmark variant, held-out Reward & Evidence Critic TD / Bellman diagnostic, and multi-step path-dependent stress test **strictly by environment family** so each domain can be read cohesively from its base navigation problem through to multi-step path-dependent confounding:

- **Color & Layout Convention (Shared Across All Figures)**:
  - **CEAC variants (shades of blue, boldface)**: Canonical CEAC (`ceac`, `ceac_natural`) in **Cobalt Blue** (`#0047AB` / `#1D4ED8`); Improved Greedy Disagreement + Price-Ranked Replay CEAC (`ceac_ranked`, `ceac_ranked_natural`) in **Periwinkle** (`#6C7CFA` / `#96A2FF`).
  - **Bayesian IDS variants (shades of red)**: Restricted/null priors in **Dark Wine Red** (`#7F1D1D`), full-model MI in **Scarlet Red** (`#DC2626`), policy-tree optimal-action MI $I(\\Pi^\\star; Z)$ in **Coral Red** (`#FB7185`), and on-path quotient MI $I(C_{\\text{on-path}}^\\star; Z)$ in **Ruby Red** (`#B91C1C`).
  - **IDAC (`idac`)**: **Emerald Green** (`#15803D`).
  - **Optimism-based RL (`pomd`, `ucb_q`)**: **Hot Pink** (`#FF1493`) and **Vivid Fuchsia-Magenta** (`#C026D3`).
  - All figures render as clean `1×1` or `1×2` single-environment time-trace overlays or dose curves (never massive small-multiple grids).

1. **Part I — Undertow DeepSea Family (`deepsea` $8\\times 8$ Grid $\\to$ `undertow_junction` $\\to$ `undertow_pathdep` $H=3$)**
2. **Part II — False Beacon Frozen Lake Family (`frozenlake` Shore Grid $\\to$ `false_beacon_junction` $\\to$ `false_beacon_pathdep` $H=3$)**
3. **Part III — Complete Data-Integrity Audits, Paired Tables & Reproducible CLI Entrypoints**""",
    )

    code(
        "diag-setup",
        """from pathlib import Path
import sys, importlib.util

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "pyproject.toml").is_file() and (p / "ceac/model.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook inside the cloned CEAC repository.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from IPython import get_ipython
_ip = get_ipython()
if _ip is not None:
    _ip.run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from ceac.model import FiniteModel
from ceac.pathology_report import (
    METHODS,
    ceac_advantage_statistics,
    ceac_comparison_section,
    draw_ceac_advantage,
    draw_single_env_grid_traces,
    draw_single_env_rho_scaling,
    strength_groups,
)
from ceac_walkthrough.evidence import verified_bundle
from ceac_walkthrough.visuals import draw_undertow, draw_false_beacon, draw_signal_laws, draw_policy_facts
from ceac_hardness.report import load_export as load_hardness_export
from ceac_walkthrough.mechanisms import load_mechanisms
from ceac_walkthrough.failure_report import (
    draw_critic_mechanism,
    draw_false_beacon_prior_reversal,
    draw_single_junction_burden,
)
from ceac_critic_diagnostics import (
    draw_suite_critic_dose_scaling,
    draw_suite_critic_traces,
    load_suite_critic_audit,
)
from ceac_critic_diagnostics.reconstruct_earlier import BURDEN_METHODS, GRID_METHODS, PRIOR_METHODS
from ceac_pathdep_grids.study import (
    AUDIT_METHODS as PATHDEP_AUDIT_METHODS,
    GRID_PATHDEP_SHORT,
    draw_single_grid_pathdep_scaling,
    draw_single_grid_pathdep_traces,
    load_grid_pathdep_export,
)

bundle, provenance = verified_bundle(ROOT / "notebooks")
groups = strength_groups(bundle["curves"])
MODE = 1

burden = load_hardness_export(ROOT / "notebooks/evidence/hardness_burden")
prior = load_hardness_export(ROOT / "notebooks/evidence/hardness_prior")
mech_audit = load_mechanisms(ROOT / "notebooks/evidence/mechanisms")

eb_dir = ROOT / "notebooks/evidence/critic_bellman_earlier"
grid_critic = load_suite_critic_audit(eb_dir, "grid_critic_audit.json.gz")
burden_critic = load_suite_critic_audit(eb_dir, "junction_burden_critic_audit.json.gz")
prior_critic = load_suite_critic_audit(eb_dir, "junction_prior_critic_audit.json.gz")

pd_dir = ROOT / "notebooks/evidence/pathdep_undertow_falsebeacon"
pd_exp = load_grid_pathdep_export(pd_dir)
pd_crit = load_suite_critic_audit(pd_dir, "critic_bellman_audit.json.gz")

print(
    f"Verified exports loaded (0 training runs launched):\\n"
    f"  • Grid strength suite: {provenance['curve_runs']} traces ({len(grid_critic['cells'])} critic audits)\\n"
    f"  • Controlled junctions: {len(burden['runs'])} burden + {len(prior['runs'])} prior cells ({len(burden_critic['cells']) + len(prior_critic['cells'])} critic audits)\\n"
    f"  • Multi-step Path-Dependent Undertow & False Beacon: {len(pd_exp['rows'])} cells ({len(pd_crit['cells'])} critic audits)"
)""",
    )

    # =========================================================================
    # PART I: ALL UNDERTOW DEEPSEA VARIANTS
    # =========================================================================
    md(
        "part1-undertow-header",
        """---
## Part I — Undertow DeepSea Family (All DeepSea Variants)

### 1.1 Undertow DeepSea ($8 \\times 8$ Grid): Geometry, Overlaid Performance & Critic Bellman Diagnostics

In the $8 \\times 8$ Undertow DeepSea navigation domain, the horizon is a tight rightward-progress budget: a diagnostic probe at `(2, 2)` resolves the latent task mode (`P(signal)=0.15` vs. `0.85`) while preserving progress toward `(8, 8)`, whereas nuisance sonar/TV actions emit high-entropy mode-invariant symbols and lose rightward progress with physical diversion probability $\\rho \\in \\{0.05, 0.25, 0.50, 0.75\\}$.""",
    )

    code(
        "part1-undertow-map",
        """fig = draw_undertow()
plt.show()""",
    )

    code(
        "part1-undertow-grid-perf",
        """fig = draw_single_env_grid_traces(groups, "deepsea", mode=MODE, rho=0.75)
plt.show()
fig = draw_single_env_rho_scaling(bundle["pathology_statistics"], "deepsea", mode=MODE)
plt.show()""",
    )

    code(
        "part1-undertow-grid-critics",
        """fig = draw_suite_critic_traces(
    grid_critic["cells"],
    [("deepsea", "Undertow DeepSea (8×8)")],
    target_dose=0.75,
    suptitle="Undertow DeepSea (8×8 Grid, ρ = 0.75, Mode 1) · Overlaid Held-Out Reward & Evidence Critic Error",
    method_order=GRID_METHODS,
)
plt.show()""",
    )

    md(
        "part1-undertow-junction-md",
        """### 1.2 Controlled Undertow Nuisance-Breadth Variant (`undertow_junction`, $B \\in \\{0, 4, 16, 64\\}$)

Isolates action-breadth growth at fixed two-step horizon by adding $B \\in \\{0, 4, 16, 64\\}$ zero-reward dead-end undertow actions alongside `safe` ($0.50$), `risky` ($0.10 / 0.90$), and `probe` ($0.48$). Occupancy-entropy (`idac`) and optimistic bonus methods (`pomd`, `ucb_q`) deteriorate sharply as $B$ grows, while `ceac` (cobalt) assigns zero likelihood-ratio price to dead-end actions and converges cleanly in both Held-Out Reward and Evidence Critic Bellman RMSE.""",
    )

    code(
        "part1-undertow-junction-plots",
        """fig = draw_single_junction_burden(burden, "undertow_junction")
plt.show()
fig = draw_critic_mechanism(mech_audit)
plt.show()
fig = draw_suite_critic_traces(
    burden_critic["cells"],
    [("undertow_junction", "Undertow Junction (B = 64)")],
    target_dose=64.0,
    suptitle="Controlled Undertow Junction (B = 64, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error",
    method_order=BURDEN_METHODS,
)
plt.show()""",
    )

    md(
        "part1-undertow-pathdep-md",
        """### 1.3 Multi-Step Path-Dependent Undertow DeepSea (`undertow_pathdep`, $H = 3$) — Breaking Optimal-Action IDS

In Sections 1.1 and 1.2 above, well-specified Bayesian IDS (`ids` / `ids_product`) did **not** fail because a single binary task variable $\\theta \\in \\{0, 1\\}$ determined the optimal route without state-contingent path dependence.

`undertow_pathdep` ($H = 3$, `32,768` episodes) introduces **genuine multi-step path dependence** across confounding modules $k \\in \\{0, 1, 2, 3\\}$:
- **Gated Optimal Action**: Step-0 `dive_thermocline` pays $-0.015$ depth cost ($0$ immediate KL) to enter `deep_trench`, unlocking $R = 8$ deep hydrothermal vents ($0.93$ vs. $0.10$). Unstaged plunges are swept away by the undertow ($0.24$).
- **Why Both Optimal-Action IDS Variants Break as $k$ Grows**:
  1. **Policy-Tree Optimal-Action IDS (`ids_action_tree`, $I(\\Pi^\\star; Z)$, coral red)**: $(3^k - 1)$ off-path undertow eddies carry local trim-rescue rules governed by $\\xi \\in \\{0, 1\\}^k$ (total return $\\le 0.26 < 0.50$, never optimal to enter), splitting full MDP optimal policy trees $\\Pi^\\star(\\theta, \\xi)$ by $2^k$ and trapping Russo–Van Roy IDS on surface current-vane sensors.
  2. **Quotiented On-Path Optimal-Action IDS (`ids_action_onpath`, $I(C_{\\text{on-path}}^\\star; Z)$, ruby red)**: Even after quotienting out all off-path states, $6k$ shallow acoustic/chemical echo plumes ($0.49999$) carry positive scalar MI about marginal shelf seeps ($0.5004$) while providing **zero KL** on deep hydrothermal vents, locking quotiented IDS into linear regret across $k = 1 \\to 3$.
- **Why Periwinkle `ceac_ranked` Succeeds (`~37` Regret Across All $k$)**: Tracks vector-valued pairwise log-likelihood ratios across undefeated on-path alternatives and propagates dual evidence prices $\\beta_t$ across the $-0.015$ thermocline gate via $Q_I(s, a; \\beta_t)$, driving Held-Out Evidence Critic RMSE smoothly to `0.0000`.""",
    )

    code(
        "part1-undertow-pathdep-plots",
        """fig = draw_single_grid_pathdep_scaling(pd_exp["rows"], "undertow_pathdep")
plt.show()
fig = draw_single_grid_pathdep_traces(pd_exp["rows"], "undertow_pathdep", target_dose=3)
plt.show()
fig = draw_suite_critic_traces(
    pd_crit["cells"],
    [("undertow_pathdep", GRID_PATHDEP_SHORT["undertow_pathdep"])],
    target_dose=3,
    suptitle="Path-Dependent Undertow DeepSea (k = 3, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error",
    method_order=PATHDEP_AUDIT_METHODS,
    ranked_ceac_present=True,
)
plt.show()""",
    )

    # =========================================================================
    # PART II: ALL FALSE BEACON FROZEN LAKE VARIANTS
    # =========================================================================
    md(
        "part2-falsebeacon-header",
        """---
## Part II — False Beacon Frozen Lake Family (All Frozen Lake Variants)

### 2.1 False Beacon Frozen Lake (Shore Grid): Geometry, Overlaid Performance & Critic Bellman Diagnostics

On the Frozen Lake shore grid, an `inner-bridge` route has mode-invariant payoff while the `outer-causeway` route becomes optimal in shifted Mode 1. Detouring to the `deep-survey` buoy yields genuine mode KL (`P(bit=1)=0.15` vs. `0.85`), whereas the noisy `reed-scan` ("False Beacon") emits $M$ uniform mode-invariant symbols (`Task KL = 0`) and adds a forced dwell step with probability $\\rho \\in \\{0.05, 0.25, 0.50, 0.75\\}$.""",
    )

    code(
        "part2-falsebeacon-maps",
        """fig = draw_false_beacon()
plt.show()
fig = draw_signal_laws()
plt.show()""",
    )

    code(
        "part2-falsebeacon-grid-perf",
        """fig = draw_single_env_grid_traces(groups, "frozenlake", mode=MODE, rho=0.75)
plt.show()
fig = draw_single_env_rho_scaling(bundle["pathology_statistics"], "frozenlake", mode=MODE)
plt.show()""",
    )

    code(
        "part2-falsebeacon-grid-critics",
        """fig = draw_suite_critic_traces(
    grid_critic["cells"],
    [("frozenlake", "False Beacon Shore (FrozenLake)")],
    target_dose=0.75,
    suptitle="False Beacon Frozen Lake (ρ = 0.75, Mode 1) · Overlaid Held-Out Reward & Evidence Critic Error",
    method_order=GRID_METHODS,
)
plt.show()""",
    )

    md(
        "part2-falsebeacon-junction-md",
        """### 2.2 Controlled False Beacon Chamber-Breadth & Cue-Reversal Variants (`false_beacon_junction`)

Evaluates two controlled interventions on the False Beacon two-step kernel:
1. **Chamber Breadth ($B \\in \\{0, 4, 16, 64\\}$)**: One nuisance action transitions uniformly into a zero-reward reed chamber of breadth $B$, inflating state-occupancy entropy (`idac`) and transition-bonus sums (`pomd`, `ucb_q`).
2. **Prior Cue Reversal ($d \\in [0, 1]$)**: Reverses the correlation between a mandatory pre-decision cue and the latent task under a frozen joint prior (`ids_correlated`, dark wine red), causing more observations to strengthen the wrong posterior belief while product-class `ceac` (cobalt) and `ids_product` (scarlet) survive.""",
    )

    code(
        "part2-falsebeacon-junction-plots",
        """fig = draw_single_junction_burden(burden, "false_beacon_junction")
plt.show()
fig = draw_false_beacon_prior_reversal(prior)
plt.show()
fig = draw_suite_critic_traces(
    burden_critic["cells"],
    [("false_beacon_junction", "False Beacon Junction (B = 64)")],
    target_dose=64.0,
    suptitle="Controlled False Beacon Junction (B = 64, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error",
    method_order=BURDEN_METHODS,
)
plt.show()
fig = draw_suite_critic_traces(
    prior_critic["cells"],
    [("false_beacon_junction", "False Beacon Cue-Reversal (d = 1.0)")],
    target_dose=1.0,
    suptitle="Controlled False Beacon Cue-Reversal (d = 1.0, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error",
    method_order=PRIOR_METHODS,
)
plt.show()""",
    )

    md(
        "part2-falsebeacon-pathdep-md",
        """### 2.3 Multi-Step Path-Dependent False Beacon Frozen Lake (`false_beacon_pathdep`, $H = 3$) — Breaking Optimal-Action IDS

`false_beacon_pathdep` ($H = 3$, `32,768` episodes) introduces **state-contingent icebreaker staging** across confounding complexity $k \\in \\{0, 1, 2, 3\\}$:
- **Gated Optimal Action**: Step-0 `stage_icebreaker` pays $-0.015$ staging cost ($0$ immediate KL) to reach `outer_soundings`, unlocking $R = 8$ outer causeway channels ($0.95$ vs. $0.09$), while unreinforced crossings hit thin shore ice ($0.23$).
- **Separation Against Optimal-Action IDS**: Adding $6k$ shoreline false-beacon optical glints ($0.49999$, informative only about marginal $0.5004$ coves with zero KL on outer causeways) plus $(3^k - 1)$ reed slush corridors with $\\xi$-dependent steering rescues breaks `ids_product`, Russo–Van Roy `ids_action_tree` ($I(\\Pi^\\star; Z)$), **and** quotiented `ids_action_onpath` ($I(C_{\\text{on-path}}^\\star; Z)$) as $k$ scales from $0 \\to 3$, whereas periwinkle `ceac_ranked` remains flat near `~40` regret with `0.0000` Held-Out Evidence Critic RMSE.""",
    )

    code(
        "part2-falsebeacon-pathdep-plots",
        """fig = draw_single_grid_pathdep_scaling(pd_exp["rows"], "false_beacon_pathdep")
plt.show()
fig = draw_single_grid_pathdep_traces(pd_exp["rows"], "false_beacon_pathdep", target_dose=3)
plt.show()
fig = draw_suite_critic_traces(
    pd_crit["cells"],
    [("false_beacon_pathdep", GRID_PATHDEP_SHORT["false_beacon_pathdep"])],
    target_dose=3,
    suptitle="Path-Dependent False Beacon Shore (k = 3, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error",
    method_order=PATHDEP_AUDIT_METHODS,
    ranked_ceac_present=True,
)
plt.show()""",
    )

    # =========================================================================
    # PART III: INTEGRITY AUDITS, SUMMARY TABLES & REPRODUCIBLE CLI
    # =========================================================================
    md(
        "part3-summary-header",
        """---
## Part III — Consolidated Integrity Verification, Paired Comparisons & CLI Entrypoints""",
    )

    code(
        "part3-integrity-and-tables",
        """# 1. Finite-model confounding & probe KL verification
models = {kind: FiniteModel(kind, benchmark="weak_signal_m4") for kind in ("deepsea", "frozenlake")}
for kind, model in models.items():
    policies = {p.label: p for p in model.seed_policies()}
    old, noise, probe = (("exploit", "sonar-sweep", "probe") if kind == "deepsea"
                         else ("inner-bridge", "reed-scan", "deep-survey"))
    values, information = model.evaluate(policies[old])
    np.testing.assert_allclose(information, 0, atol=1e-12)
    np.testing.assert_allclose(values[0], values[1], atol=1e-12)
    assert model.optimal_returns[1] - values[1] > 0.049
    np.testing.assert_allclose(model.evaluate(policies[noise])[1], 0, atol=1e-12)
    assert model.evaluate(policies[probe])[1][0, 1] > 1.2

# 2. Complete 7-method slate, 96-run Bayesian support check & paired advantage parity
rows = bundle["pathology_statistics"]
assert ceac_advantage_statistics(rows) == bundle["ceac_comparisons"]["rows"]
expected_methods = set(METHODS)
for kind, context, mode in {key[:3] for key in groups}:
    assert {key[3] for key in groups if key[:3] == (kind, context, mode)} == expected_methods
bayesian = [r for r in bundle["curves"] if r["result"]["method"] in ("ids", "ids_zero")]
assert len(bayesian) == 96
for run in bayesian:
    result, metrics = run["result"], run["metrics"]
    posterior = np.array([m["diagnostics"]["posterior"][1] for m in metrics])
    if result["method"] == "ids_zero":
        np.testing.assert_array_equal(posterior, 0)
        gap = .05 if result["true_mode"] == 1 else 0.
        np.testing.assert_allclose([m["cumulative_pseudo_regret"] for m in metrics],
                                   gap * np.arange(1, result["episodes"] + 1), atol=1e-8, rtol=0)
    else:
        assert np.all(np.isfinite(posterior)) and np.all(posterior > 0)
history = [r for r in bundle["history_audits"] if r["method"] == "ceac"]
assert len(history) == sum(r["method"] == "ceac" for r in bundle["endpoints"])

# 3. Print Multi-Step Path-Dependent Undertow & False Beacon Summary Table (k = 0 -> k = 3)
from ceac_pathdep.study import PATHDEP_LABELS, PATHDEP_ORDER
print("=== Path-Dependent Undertow & False Beacon Summary (Task 1, k = 0 -> k = 3, 32,768 Episodes) ===")
print(f"{'Environment':30s} | {'Algorithm':48s} | {'Gap (k=0 -> 3)':>16s} | {'Regret (k=0 -> 3)':>20s}")
print("-" * 124)
for k_key, k_lbl in [("undertow_pathdep", "Undertow PathDep (H=3)"), ("false_beacon_pathdep", "False Beacon PathDep (H=3)")]:
    for m in PATHDEP_ORDER:
        g0 = [r for r in pd_exp["rows"] if r["kind"] == k_key and r["method"] == m and r["dose"] == 0]
        g3 = [r for r in pd_exp["rows"] if r["kind"] == k_key and r["method"] == m and r["dose"] == 3]
        print(
            f"{k_lbl:30s} | {PATHDEP_LABELS[m][:48]:48s} | "
            f"{np.mean([x['late_gap'] for x in g0]):6.4f} -> {np.mean([x['late_gap'] for x in g3]):6.4f} | "
            f"{np.mean([x['cumulative_regret'] for x in g0]):8,.1f} -> {np.mean([x['cumulative_regret'] for x in g3]):8,.1f}"
        )""",
    )

    code(
        "part3-cli-entrypoints",
        """RUNNER_COMMANDS = {
    "grid_strength_suite": ".venv/bin/python -m ceac.run --output runs/my_pathology_strength --benchmarks pathology_rho005 pathology_rho025 pathology_rho050 pathology_rho075 --environments deepsea frozenlake --methods ceac oracle ids idac ucb_q pomd ids_zero --modes 0 1 --seeds 0 1 2 --episodes 2048 --workers 4",
    "pathdep_undertow_and_false_beacon": ".venv/bin/python -m ceac_pathdep_grids.study --workers 4 --episodes 32768",
    "rebuild_this_notebook": ".venv/bin/python scripts/update_diagnostics_notebook.py",
}
for name, cmd in RUNNER_COMMANDS.items():
    print(f"[{name}]\\n  {cmd}")""",
    )

    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "environment_organized_schema": 2,
        },
    )


def execute_notebook(nb: nbf.NotebookNode) -> int:
    from IPython.display import Markdown

    exec_ns = {"__name__": "__main__"}
    exec_idx = 0
    total_figs = 0

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        exec_idx += 1
        cell.execution_count = exec_idx
        cell.outputs = []
        pending_figs = []
        pending_md = []

        orig_show = plt.show

        def _capture_show(*_a, **_kw):
            for num in list(plt.get_fignums()):
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
                pending_figs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                plt.close(fig)

        def _capture_display(*objs, **_kw):
            for obj in objs:
                pending_md.append(obj.data if isinstance(obj, Markdown) else str(obj))

        exec_ns["display"] = _capture_display
        plt.show = _capture_show
        stdout_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf):
                exec(compile(cell.source, f"<{cell.id}>", "exec"), exec_ns)
        finally:
            plt.show = orig_show
            plt.close("all")

        txt = stdout_buf.getvalue()
        if txt:
            cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=txt))
        for b64 in pending_figs:
            cell.outputs.append(
                nbf.v4.new_output("display_data", data={"image/png": b64, "text/plain": "<Figure>"}, metadata={})
            )
        for md_txt in pending_md:
            cell.outputs.append(
                nbf.v4.new_output("display_data", data={"text/markdown": md_txt, "text/plain": md_txt[:80]}, metadata={})
            )
        total_figs += len(pending_figs)
    return total_figs


def main():
    nb_path = ROOT / "notebooks/Undertow_FalseBeacon_Diagnostics.ipynb"
    nb = build_diagnostics_notebook()
    total_figs = execute_notebook(nb)
    for c in nb.cells:
        c.metadata.pop("execution", None)
    nbf.validate(nb)
    nbf.write(nb, nb_path)
    code_cells = sum(c.cell_type == "code" for c in nb.cells)
    print(f"Built, executed, and validated {nb_path.name}: {code_cells} code cells, {total_figs} rendered single-environment figures.")


if __name__ == "__main__":
    main()
