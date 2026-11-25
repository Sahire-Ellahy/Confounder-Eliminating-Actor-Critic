#!/usr/bin/env python3
"""Declarative builder for notebooks/Undertow_FalseBeacon_Diagnostics.ipynb via `ceac_bench.reporting`."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceac_bench.reporting import (
    CodeBlock,
    FamilySection,
    MarkdownBlock,
    NotebookComposer,
    NotebookSpec,
    VariantSection,
)


def diagnostics_notebook_spec() -> NotebookSpec:
    return NotebookSpec(
        notebook_id="undertow_falsebeacon_diagnostics",
        intro=MarkdownBlock(
            "diag-overview",
            """# Undertow DeepSea & False Beacon Frozen Lake — Complete Environment-Organized Suite

This notebook is declaratively composed via `ceac_bench.reporting` and organizes every benchmark variant, held-out Reward & Evidence Critic TD / Bellman diagnostic, and multi-step path-dependent stress test **strictly by environment family**:

- **Color & Layout Convention (`ceac_bench.reporting.VisualTheme`)**:
  - **CEAC variants (shades of blue, boldface)**: Canonical CEAC (`ceac`, `ceac_natural`) in **Cobalt Blue** (`#0047AB` / `#1D4ED8`); Greedy Disagreement + Price-Ranked Replay CEAC (`ceac_ranked`, `ceac_ranked_natural`) in **Periwinkle** (`#6C7CFA` / `#96A2FF`).
  - **Bayesian IDS variants (shades of red)**: Restricted/null priors in **Dark Wine Red** (`#7F1D1D`), full-model MI in **Scarlet Red** (`#DC2626`), policy-tree optimal-action MI $I(\\Pi^\\star; Z)$ in **Coral Red** (`#FB7185`), and on-path quotient MI $I(C_{\\text{on-path}}^\\star; Z)$ in **Ruby Red** (`#B91C1C`).
  - **IDAC (`idac`)**: **Emerald Green** (`#15803D`); **Optimism (`pomd`, `ucb_q`)**: **Hot Pink / Vivid Fuchsia** (`#FF1493` / `#C026D3`).

1. **Part I — Undertow DeepSea Family (`deepsea` $8\\times 8$ Grid $\\to$ `undertow_junction` $\\to$ `undertow_pathdep` $H=3$)**
2. **Part II — False Beacon Frozen Lake Family (`frozenlake` Shore Grid $\\to$ `false_beacon_junction` $\\to$ `false_beacon_pathdep` $H=3$)**
3. **Part III — Consolidated Data-Integrity Audits, Summary Tables & Unified CLI Entrypoints**""",
        ),
        setup=CodeBlock(
            "diag-setup",
            """from pathlib import Path
import sys

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "pyproject.toml").is_file() and (p / "ceac_bench").is_dir()), None)
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

from ceac.model import FiniteModel
from ceac.pathology_report import (
    METHODS,
    ceac_advantage_statistics,
    draw_single_env_grid_traces,
    draw_single_env_rho_scaling,
    strength_groups,
)
from ceac_walkthrough.evidence import verified_bundle
from ceac_walkthrough.visuals import draw_undertow, draw_false_beacon, draw_signal_laws
from ceac_hardness.report import load_export as load_hardness_export
from ceac_walkthrough.mechanisms import load_mechanisms
from ceac_walkthrough.failure_report import (
    draw_critic_mechanism,
    draw_false_beacon_prior_reversal,
    draw_single_junction_burden,
)
from ceac_critic_diagnostics import draw_suite_critic_traces, load_suite_critic_audit
from ceac_critic_diagnostics.reconstruct_earlier import BURDEN_METHODS, GRID_METHODS, PRIOR_METHODS
from ceac_bench.reporting import (
    DoseTransitionMarkdownTable,
    SingleEnvCriticBellmanPlot,
    SingleEnvDoseScalingPlot,
    SingleEnvTimeTracePlot,
)
from ceac_bench.store import LegacyBundleAdapter

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

grid_pathdep_bundle = LegacyBundleAdapter.load_grid_pathdep(ROOT / "notebooks/evidence/pathdep_undertow_falsebeacon")

print(
    f"Verified bundles loaded via ceac_bench (0 training runs launched):\\n"
    f"  • Grid strength suite: {provenance['curve_runs']} traces ({len(grid_critic['cells'])} critic audits)\\n"
    f"  • Controlled junctions: {len(burden['runs'])} burden + {len(prior['runs'])} prior cells\\n"
    f"  • Path-Dependent Undertow & False Beacon StudyBundle v2: {len(grid_pathdep_bundle.cells)} cells"
)""",
        ),
        families=(
            FamilySection(
                family_id="part1-undertow",
                header_markdown="---\n## Part I — Undertow DeepSea Family (All DeepSea Variants)",
                variants=(
                    VariantSection(
                        variant_id="undertow-grid",
                        heading_markdown="### 1.1 Undertow DeepSea ($8 \\times 8$ Grid): Geometry, Overlaid Performance & Critic Bellman Diagnostics",
                        code_blocks=(
                            CodeBlock("part1-undertow-map", "fig = draw_undertow()\nplt.show()"),
                            CodeBlock(
                                "part1-undertow-grid-perf",
                                "fig = draw_single_env_grid_traces(groups, 'deepsea', mode=MODE, rho=0.75)\nplt.show()\n"
                                "fig = draw_single_env_rho_scaling(bundle['pathology_statistics'], 'deepsea', mode=MODE)\nplt.show()",
                            ),
                            CodeBlock(
                                "part1-undertow-grid-critics",
                                "fig = draw_suite_critic_traces(\n"
                                "    grid_critic['cells'], [('deepsea', 'Undertow DeepSea (8×8)')], target_dose=0.75,\n"
                                "    suptitle='Undertow DeepSea (8×8 Grid, ρ = 0.75, Mode 1) · Overlaid Held-Out Reward & Evidence Critic Error',\n"
                                "    method_order=GRID_METHODS,\n"
                                ")\nplt.show()",
                            ),
                        ),
                    ),
                    VariantSection(
                        variant_id="undertow-junction",
                        heading_markdown="### 1.2 Controlled Undertow Nuisance-Breadth Variant (`undertow_junction`, $B \\in \\{0, 4, 16, 64\\}$)",
                        code_blocks=(
                            CodeBlock(
                                "part1-undertow-junction-plots",
                                "fig = draw_single_junction_burden(burden, 'undertow_junction')\nplt.show()\n"
                                "fig = draw_critic_mechanism(mech_audit)\nplt.show()\n"
                                "fig = draw_suite_critic_traces(\n"
                                "    burden_critic['cells'], [('undertow_junction', 'Undertow Junction (B = 64)')], target_dose=64.0,\n"
                                "    suptitle='Controlled Undertow Junction (B = 64, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',\n"
                                "    method_order=BURDEN_METHODS,\n"
                                ")\nplt.show()",
                            ),
                        ),
                    ),
                    VariantSection(
                        variant_id="undertow-pathdep",
                        heading_markdown=(
                            "### 1.3 Multi-Step Path-Dependent Undertow DeepSea (`undertow_pathdep`, $H = 3$) — Breaking Optimal-Action IDS\n\n"
                            "Step-0 `dive_thermocline` ($-0.015$, `0` KL) unlocks $R = 8$ deep vents ($0.93$ vs. $0.10$) amidst $6k$ shallow echo plumes "
                            "and $(3^k - 1)$ off-path eddies, breaking **both** Russo–Van Roy policy-tree IDS $I(\\Pi^\\star; Z)$ and quotiented on-path IDS "
                            "$I(C_{\\text{on-path}}^\\star; Z)$ across $k = 0 \\to 3$ while periwinkle `ceac_ranked` stays flat near `~37` regret."
                        ),
                        code_blocks=(
                            CodeBlock(
                                "part1-undertow-pathdep-plots",
                                "for plotter in (\n"
                                "    SingleEnvDoseScalingPlot('undertow_pathdep'),\n"
                                "    SingleEnvTimeTracePlot('undertow_pathdep', dose=3),\n"
                                "    SingleEnvCriticBellmanPlot('undertow_pathdep', dose=3),\n"
                                "):\n"
                                "    fig = plotter(grid_pathdep_bundle)\n"
                                "    plt.show()",
                            ),
                        ),
                    ),
                ),
            ),
            FamilySection(
                family_id="part2-falsebeacon",
                header_markdown="---\n## Part II — False Beacon Frozen Lake Family (All Frozen Lake Variants)",
                variants=(
                    VariantSection(
                        variant_id="falsebeacon-grid",
                        heading_markdown="### 2.1 False Beacon Frozen Lake (Shore Grid): Geometry, Overlaid Performance & Critic Bellman Diagnostics",
                        code_blocks=(
                            CodeBlock("part2-falsebeacon-maps", "fig = draw_false_beacon()\nplt.show()\nfig = draw_signal_laws()\nplt.show()"),
                            CodeBlock(
                                "part2-falsebeacon-grid-perf",
                                "fig = draw_single_env_grid_traces(groups, 'frozenlake', mode=MODE, rho=0.75)\nplt.show()\n"
                                "fig = draw_single_env_rho_scaling(bundle['pathology_statistics'], 'frozenlake', mode=MODE)\nplt.show()",
                            ),
                            CodeBlock(
                                "part2-falsebeacon-grid-critics",
                                "fig = draw_suite_critic_traces(\n"
                                "    grid_critic['cells'], [('frozenlake', 'False Beacon Shore (FrozenLake)')], target_dose=0.75,\n"
                                "    suptitle='False Beacon Frozen Lake (ρ = 0.75, Mode 1) · Overlaid Held-Out Reward & Evidence Critic Error',\n"
                                "    method_order=GRID_METHODS,\n"
                                ")\nplt.show()",
                            ),
                        ),
                    ),
                    VariantSection(
                        variant_id="falsebeacon-junction",
                        heading_markdown="### 2.2 Controlled False Beacon Chamber-Breadth ($B \\in \\{0, 4, 16, 64\\}$) & Cue-Reversal ($d \\in [0, 1]$) Variants",
                        code_blocks=(
                            CodeBlock(
                                "part2-falsebeacon-junction-plots",
                                "fig = draw_single_junction_burden(burden, 'false_beacon_junction')\nplt.show()\n"
                                "fig = draw_false_beacon_prior_reversal(prior)\nplt.show()\n"
                                "fig = draw_suite_critic_traces(\n"
                                "    burden_critic['cells'], [('false_beacon_junction', 'False Beacon Junction (B = 64)')], target_dose=64.0,\n"
                                "    suptitle='Controlled False Beacon Junction (B = 64, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',\n"
                                "    method_order=BURDEN_METHODS,\n"
                                ")\nplt.show()\n"
                                "fig = draw_suite_critic_traces(\n"
                                "    prior_critic['cells'], [('false_beacon_junction', 'False Beacon Cue-Reversal (d = 1.0)')], target_dose=1.0,\n"
                                "    suptitle='Controlled False Beacon Cue-Reversal (d = 1.0, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',\n"
                                "    method_order=PRIOR_METHODS,\n"
                                ")\nplt.show()",
                            ),
                        ),
                    ),
                    VariantSection(
                        variant_id="falsebeacon-pathdep",
                        heading_markdown=(
                            "### 2.3 Multi-Step Path-Dependent False Beacon Shore (`false_beacon_pathdep`, $H = 3$) — Breaking Optimal-Action IDS\n\n"
                            "Step-0 `stage_icebreaker` ($-0.015$, `0` KL) unlocks $R = 8$ outer causeway channels ($0.95$ vs. $0.09$) amidst $6k$ shoreline "
                            "false-beacon glints and $(3^k - 1)$ reed slush corridors, breaking both `ids_action_tree` and `ids_action_onpath` while "
                            "periwinkle `ceac_ranked` stays flat near `~45` regret."
                        ),
                        code_blocks=(
                            CodeBlock(
                                "part2-falsebeacon-pathdep-plots",
                                "for plotter in (\n"
                                "    SingleEnvDoseScalingPlot('false_beacon_pathdep'),\n"
                                "    SingleEnvTimeTracePlot('false_beacon_pathdep', dose=3),\n"
                                "    SingleEnvCriticBellmanPlot('false_beacon_pathdep', dose=3),\n"
                                "):\n"
                                "    fig = plotter(grid_pathdep_bundle)\n"
                                "    plt.show()",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        closing_blocks=(
            MarkdownBlock("part3-summary-header", "---\n## Part III — Consolidated Integrity Assertions, Summary Tables & Unified CLI Entrypoints"),
            CodeBlock(
                "part3-integrity-and-tables",
                """models = {kind: FiniteModel(kind, benchmark="weak_signal_m4") for kind in ("deepsea", "frozenlake")}
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
        gap = 0.05 if result["true_mode"] == 1 else 0.0
        np.testing.assert_allclose(
            [m["cumulative_pseudo_regret"] for m in metrics],
            gap * np.arange(1, result["episodes"] + 1),
            atol=1e-8,
            rtol=0,
        )
    else:
        assert np.all(np.isfinite(posterior)) and np.all(posterior > 0)
history = [r for r in bundle["history_audits"] if r["method"] == "ceac"]
assert len(history) == sum(r["method"] == "ceac" for r in bundle["endpoints"])

print(DoseTransitionMarkdownTable()(grid_pathdep_bundle))""",
            ),
            CodeBlock(
                "part3-cli-entrypoints",
                """CLI_COMMANDS = {
    "list_suites": ".venv/bin/python -m ceac_bench.cli list-suites",
    "check_layer_boundaries": ".venv/bin/python -m ceac_bench.cli check-boundaries",
    "run_grid_pathdep_v2": ".venv/bin/python -m ceac_bench.cli run grid_pathdep --output runs/grid_pathdep_v2 --workers 4",
    "rebuild_this_notebook": ".venv/bin/python scripts/update_diagnostics_notebook.py",
}
for name, cmd in CLI_COMMANDS.items():
    print(f"[{name}]  {cmd}")""",
            ),
        ),
        metadata={"environment_organized_schema": 3},
    )


def main() -> None:
    nb_path = ROOT / "notebooks/Undertow_FalseBeacon_Diagnostics.ipynb"
    audit = NotebookComposer.build_and_execute(
        diagnostics_notebook_spec(),
        nb_path,
        audit_path=ROOT / "notebooks/diagnostics_execution.json",
        evidence_dirs=("hardness_burden", "hardness_prior", "critic_bellman_earlier", "pathdep_undertow_falsebeacon"),
    )
    print(f"Built, executed, and validated {nb_path.name}: {audit['code_cells']} code cells, {audit['rendered_figures']} rendered single-environment figures.")


if __name__ == "__main__":
    main()
