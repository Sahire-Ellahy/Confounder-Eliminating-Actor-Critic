"""Verified small-multiple reports, mechanism diagnostics, and portable evidence for ceac_pathologies."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import numpy as np
from .environments import EpistasisConfig, LockConfig
from .run import ROOT, atomic_json, digest


LABELS = {
    "idac": "IDAC · occupancy ratio",
    "ucb_q": "UCB Q-learning",
    "pomd": "Optimistic POMD",
    "ids_decoy": "IDS · decoy-lock prior",
    "ids_canonical": "IDS · canonical additive prior",
    "ids_additive": "IDS · no-interaction prior",
    "ids_product": "IDS · model-MI, uniform prior",
    "ids_empirical": "IDS · model-MI, data-grounded prior",
    "ids_action": "IDS · optimal-action MI (Russo–Van Roy)",
    "ceac": "CEAC · reward + evidence critics",
    "ceac_natural": "CEAC · guarded natural",
    "oracle": "Oracle Graves–Lai",
}

KINDS = {
    "combination_lock": "Combination Lock",
    "bidirectional_lock": "Bidirectional Combination Lock",
    "epistatic_signalling": "Epistatic Signalling Pathway",
    "gi_arrayed": "Arrayed pair screen (K562 GI map)",
    "gi_multiplexed": "Multiplexed screen variant (synthetic pooled blocks)",
}

DOSE_COLORS = ["#56a8cb", "#2b7bba", "#184e77", "#7f1d1d", "#3b0764"]

METHOD_COLORS = {
    "idac": "#C25300",
    "ucb_q": "#0072B2",
    "pomd": "#B34697",
    "ids_decoy": "#C93441",
    "ids_canonical": "#C93441",
    "ids_additive": "#C93441",
    "ids_product": "#595959",
    "ids_empirical": "#8C6D1F",
    "ids_action": "#111111",
    "ceac": "#007D60",
    "ceac_natural": "#1A9E78",
    "oracle": "#3E3163",
}

METHOD_MARKERS = {
    "idac": "s",
    "ucb_q": "^",
    "pomd": "D",
    "ids_decoy": "v",
    "ids_canonical": "v",
    "ids_additive": "v",
    "ids_product": "o",
    "ids_empirical": "h",
    "ids_action": "8",
    "ceac": "*",
    "ceac_natural": "P",
    "oracle": "x",
}

METHOD_ORDER = [
    "idac",
    "ucb_q",
    "pomd",
    "ids_decoy",
    "ids_canonical",
    "ids_additive",
    "ids_product",
    "ids_empirical",
    "ids_action",
    "ceac",
    "ceac_natural",
    "oracle",
]


def load_study(folder: Path | str) -> dict:
    folder = Path(folder)
    status = json.loads((folder / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError(f"Incomplete study: {folder}")
    registration = json.loads((folder / "registration.json").read_text())
    for name in ("registration", "results"):
        if digest(folder / f"{name}.json") != status[f"{name}_sha256"]:
            raise ValueError(f"Study {name} hash mismatch")
    index = json.loads((folder / "results.json").read_text())
    if len(index) != registration["jobs"] or len(index) != status["completed"]:
        raise ValueError("Missing study cells")

    runs, seen = [], set()
    for entry in index:
        ef = Path(entry["folder"])
        cell = ef if ef.is_absolute() else (ROOT / ef)
        state = json.loads((cell / "status.json").read_text())
        if state["status"] != "complete":
            raise ValueError(f"Incomplete cell: {cell}")
        for name, checksum in state["artifacts"].items():
            if digest(cell / name) != checksum:
                raise ValueError(f"Artifact mismatch: {cell / name}")
        manifest = json.loads((cell / "manifest.json").read_text())
        summary = json.loads((cell / "summary.json").read_text())
        if manifest["sources"] != registration["sources"] or manifest["identity"] != summary["identity"]:
            raise ValueError("Source or manifest identity mismatch")
        settings = summary["settings"]
        key = (
            settings["environment"]["kind"],
            int(settings["dose"]),
            int(settings["task_evaluator_only"]),
            settings["method"],
            int(settings["seed"]),
        )
        if key in seen:
            raise ValueError(f"Duplicate cell: {key}")
        seen.add(key)
        if summary["late_gap"] != entry["late_gap"] or summary["cumulative_regret"] != entry["cumulative_regret"]:
            raise ValueError("Study index and cell summary disagree")
        with gzip.open(cell / "trace.jsonl.gz", "rt") as f:
            trace = [json.loads(line) for line in f]
        if trace[-1]["episode"] != settings["episodes"] or trace[-1]["interactions"] != settings["physical_budget"]:
            raise ValueError("Incomplete trace budget")
        runs.append(
            dict(
                protocol=settings["protocol"],
                kind=settings["environment"]["kind"],
                dose=int(settings["dose"]),
                task=int(settings["task_evaluator_only"]),
                method=settings["method"],
                seed=int(settings["seed"]),
                summary=summary,
                trace=trace,
            )
        )

    cfg = registration["configuration"]
    protocol = registration["protocol"]
    spec = cfg[protocol]
    expected = {
        (k, int(d), int(t), m, int(s))
        for k in spec["kinds"]
        for d in spec["doses"]
        for t in spec["tasks"]
        for m in spec["methods"]
        for s in cfg["evaluation_seeds"]
    }
    if seen != expected:
        raise ValueError("Registered condition grid is incomplete or substituted")
    for run in runs:
        settings = run["summary"]["settings"]
        if protocol == "locks":
            want = asdict(LockConfig(kind=run["kind"], depth=run["dose"]))
        elif protocol == "epistasis":
            want = asdict(EpistasisConfig(kind=run["kind"], pleiotropy=run["dose"]))
        else:
            from ceac_gi.environment import build_config, relabel
            want = asdict(relabel(build_config(run["kind"], run["dose"]), run["seed"]))
        if settings["environment"] != json.loads(json.dumps(want)) or settings["episodes"] != registration["episodes"]:
            raise ValueError("Changed environment intervention or episode budget")
    return dict(registration=registration, runs=runs, source_folder=str(folder))


def endpoints(bundle: dict) -> list[dict]:
    groups = defaultdict(list)
    for run in bundle["runs"]:
        groups[(run["protocol"], run["kind"], run["task"], run["method"], run["dose"])].append(run)
    rows = []
    for (protocol, kind, task, method, dose), runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda r: r["seed"])
        gaps = np.array([r["summary"]["late_gap"] for r in runs], dtype=float)
        regrets = np.array([r["summary"]["cumulative_regret"] for r in runs], dtype=float)
        eps = runs[0]["summary"]["settings"]["episodes"]
        row = dict(
            protocol=protocol,
            kind=kind,
            task=task,
            method=method,
            dose=dose,
            seeds=len(runs),
            late_gap_mean=float(gaps.mean()),
            late_gap_sd=float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
            regret_mean=float(regrets.mean()),
            regret_sd=float(regrets.std(ddof=1)) if len(regrets) > 1 else 0.0,
            sink_fraction=float(np.mean([r["summary"]["events"]["sink"] / eps for r in runs])),
            decoy_fraction=float(np.mean([r["summary"]["events"]["decoy_unlock"] / eps for r in runs])),
            passenger_omics_fraction=float(np.mean([r["summary"]["events"]["passenger_omics"] / eps for r in runs])),
            pleiotropic_fraction=float(np.mean([r["summary"]["events"]["pleiotropic"] / eps for r in runs])),
            wrong_pair_fraction=float(np.mean([r["summary"]["events"].get("pair_test", 0) / eps for r in runs])),
            block_fraction=float(np.mean([r["summary"]["events"].get("block", 0) / eps for r in runs])),
            probe_fraction=float(np.mean([r["summary"]["events"]["probe"] / eps for r in runs])),
            pomd_zero_fraction=float(np.mean([r["summary"].get("pomd_zero_fraction", 0.0) for r in runs])),
            ceac_actor_updates=int(sum(r["summary"]["actor_updates"] for r in runs)),
            ceac_priced_actor_updates=int(sum(r["summary"]["priced_actor_updates"] for r in runs)),
            seed_ids=[r["seed"] for r in runs],
            seed_late_gaps=gaps.tolist(),
            seed_regrets=regrets.tolist(),
        )
        if method.startswith("ids_"):
            row["responder_posterior_mean"] = float(np.mean([r["summary"]["responder_posterior"] for r in runs]))
            row["prior_supported"] = all(r["summary"]["prior_contains_true_model"] for r in runs)
        rows.append(row)
    return rows


def paired_ceac_comparisons(bundle: dict) -> list[dict]:
    by_key = {
        (r["kind"], r["task"], r["dose"], r["method"], r["seed"]): r["summary"]
        for r in bundle["runs"]
    }
    kinds = sorted({r["kind"] for r in bundle["runs"]})
    tasks = sorted({r["task"] for r in bundle["runs"]})
    doses = sorted({r["dose"] for r in bundle["runs"]})
    seeds = sorted({r["seed"] for r in bundle["runs"]})
    methods = [m for m in METHOD_ORDER if m != "ceac" and any(r["method"] == m for r in bundle["runs"])]
    min_dose = doses[0]
    contrasts = []
    for kind in kinds:
        for task in tasks:
            for method in methods:
                base_diffs = np.array([
                    by_key[(kind, task, min_dose, method, s)]["cumulative_regret"]
                    - by_key[(kind, task, min_dose, "ceac", s)]["cumulative_regret"]
                    for s in seeds
                ])
                for dose in doses:
                    diffs = np.array([
                        by_key[(kind, task, dose, method, s)]["cumulative_regret"]
                        - by_key[(kind, task, dose, "ceac", s)]["cumulative_regret"]
                        for s in seeds
                    ])
                    added = diffs - base_diffs
                    gap_diffs = np.array([
                        by_key[(kind, task, dose, method, s)]["late_gap"]
                        - by_key[(kind, task, dose, "ceac", s)]["late_gap"]
                        for s in seeds
                    ])
                    contrasts.append(dict(
                        kind=kind,
                        task=task,
                        dose=dose,
                        comparator=method,
                        regret_saved_mean=float(diffs.mean()),
                        regret_saved_sd=float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                        added_advantage_mean=float(added.mean()),
                        added_advantage_sd=float(added.std(ddof=1)) if len(added) > 1 else 0.0,
                        late_gap_reduction_mean=float(gap_diffs.mean()),
                        ceac_wins_all_seeds=bool(np.all(diffs > 0)),
                        per_seed_regret_saved=diffs.tolist(),
                    ))
    return contrasts


def load_export(folder: Path | str) -> dict:
    """Load and SHA-256-verify a portable evidence export without touching raw /runs."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected_sha in manifest["files"].items():
        if digest(folder / name) != expected_sha:
            raise ValueError(f"Portable evidence checksum mismatch: {folder / name}")
    with gzip.open(folder / "evidence.json.gz", "rt") as f:
        return json.load(f)


def _apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})


def draw_learning(bundle: dict, kind: str, task: int = 1, metric: str = "cumulative_regret"):
    import matplotlib.pyplot as plt
    _apply_style()
    runs = [r for r in bundle["runs"] if r["kind"] == kind and r["task"] == task]
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in runs)]
    doses = sorted({r["dose"] for r in runs})
    protocol = bundle["registration"]["protocol"]
    cols = 3
    rows_n = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(13.6, 3.45 * rows_n), squeeze=False, sharex=True, sharey=True)
    stats = endpoints(bundle)
    dose_sym = "L" if protocol == "locks" else "k"
    badge_box = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.86)
    for idx, (ax, method) in enumerate(zip(axes.flat, methods)):
        for dose, color in zip(doses, DOSE_COLORS):
            cells = sorted([r for r in runs if r["method"] == method and r["dose"] == dose], key=lambda r: r["seed"])
            xs = [np.array([t["episode"] for t in r["trace"]], dtype=float) / 1000.0 for r in cells]
            if any(not np.array_equal(x, xs[0]) for x in xs):
                raise ValueError("Unequal episode grids across seeds")
            values = np.array([[t[metric] for t in r["trace"]] for r in cells], dtype=float)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
            ax.fill_between(xs[0], np.maximum(0.0, mean - sd), mean + sd, color=color, alpha=0.14, linewidth=0)
            ax.plot(xs[0], mean, color=color, lw=1.7, label=f"{dose_sym} = {dose}", zorder=3)
        lo_row = next(r for r in stats if r["kind"] == kind and r["task"] == task and r["method"] == method and r["dose"] == doses[0])
        hi_row = next(r for r in stats if r["kind"] == kind and r["task"] == task and r["method"] == method and r["dose"] == doses[-1])
        if metric == "window_gap":
            ax.text(0.97, 0.96, f"Gap: {lo_row['late_gap_mean']:.3f} → {hi_row['late_gap_mean']:.3f}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=8.5, color="#222222", bbox=badge_box, zorder=8)
        else:
            ax.text(0.03, 0.96, f"Regret ({dose_sym}={doses[-1]}): {hi_row['regret_mean']:,.0f}",
                    ha="left", va="top", transform=ax.transAxes, fontsize=8.5, color="#222222", bbox=badge_box, zorder=8)
        ax.set_title(LABELS[method], fontsize=10.5, pad=8)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, xs[0][-1])
        if idx + cols >= len(methods):
            ax.xaxis.set_tick_params(labelbottom=True)
        if metric == "window_gap":
            ax.set_ylim(-0.04, 1.08)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    fig.supxlabel("Training episodes (thousands)", fontsize=11, y=0.015)
    fig.supylabel("Expected policy gap" if metric == "window_gap" else "Cumulative expected regret", fontsize=11, x=0.01)
    subtitle = "lock depth sweep" if protocol == "locks" else "pleiotropic passenger & off-pathway complexity"
    fig.suptitle(f"{KINDS[kind]} · {subtitle} · task {task}", fontsize=13.5, y=0.99)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(doses), frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=10)
    fig.tight_layout(rect=(0.028, 0.04, 0.995, 0.90))
    return fig


def draw_response(bundle: dict):
    """Pathology dose on x: both final expected policy loss and cumulative regret (symlog)."""
    import matplotlib.pyplot as plt
    _apply_style()
    rows = endpoints(bundle)
    protocol = bundle["registration"]["protocol"]
    kinds = sorted({r["kind"] for r in rows})
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    doses = sorted({r["dose"] for r in rows})

    fig, axes = plt.subplots(len(kinds), 2, figsize=(12.8, 4.35 * len(kinds)), squeeze=False)
    xlab = {"locks": "Lock depth L (horizon H = L + 1)",
            "epistasis": "Pleiotropic passenger modules k (M = 128 · 2^k, B = 4^k - 1)",
            "gi_screen": "Variables with unknown main effects k (M = 46 · 2^k)"}[protocol]
    for row_idx, kind in enumerate(kinds):
        for col_idx, (field_mean, field_sd, ylabel, use_symlog) in enumerate([
            ("late_gap_mean", "late_gap_sd", "Final expected policy loss ↓", False),
            ("regret_mean", "regret_sd", "Cumulative pseudo-regret (symlog) ↓", True),
        ]):
            ax = axes[row_idx, col_idx]
            for method in methods:
                group = sorted([r for r in rows if r["kind"] == kind and r["task"] == 1 and r["method"] == method], key=lambda r: r["dose"])
                if not group:
                    continue
                x = np.array([r["dose"] for r in group], dtype=float)
                y = np.array([r[field_mean] for r in group], dtype=float)
                sd = np.array([r[field_sd] for r in group], dtype=float)
                color = METHOD_COLORS[method]
                ax.fill_between(x, np.maximum(0.0, y - sd), y + sd, color=color, alpha=0.12, linewidth=0)
                ax.plot(
                    x, y,
                    label=LABELS[method],
                    color=color,
                    lw=2.3 if method.startswith("ceac") else 1.6,
                    marker=METHOD_MARKERS[method],
                    markersize=9 if method == "ceac" else 6.5,
                    linestyle="--" if method == "ids_product" else (":" if method == "oracle" else "-"),
                    zorder=6 if method.startswith("ceac") else 3,
                )
            ax.set_title(f"{KINDS[kind]} · {'Final gap' if col_idx == 0 else 'Total learning cost'}", fontsize=11, pad=8)
            ax.set_xlabel(xlab, fontsize=10, labelpad=6)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(doses)
            if use_symlog:
                ax.set_yscale("symlog", linthresh=10.0)
                ax.set_ylim(bottom=0.0)
            else:
                ax.set_ylim(-0.03, 1.00)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    bottom_reserve = 0.18 if len(kinds) == 1 else 0.11
    fig.tight_layout(rect=(0.0, bottom_reserve, 1.0, 1.0))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, bottom_reserve - 0.02), ncol=3, frameon=False, fontsize=9.5)
    return fig


def draw_architecture(protocol: str):
    """Draw structural schematic of the sequential lock or epistatic signalling family."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch
    _apply_style()

    if protocol == "locks":
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8))
        # Panel A: Unidirectional Combination Lock
        ax = axes[0]
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title("Combination Lock (depth L, horizon H = L + 1)", fontsize=12, pad=8)
        xs = [0.10, 0.33, 0.56, 0.80]
        for idx, x in enumerate(xs):
            color = "#007D60" if idx == len(xs) - 1 else "#173e59"
            ax.add_patch(Circle((x, 0.74), 0.044, color=color))
            lbl = "start" if idx == 0 else ("gate" if idx == len(xs) - 1 else f"live {idx}")
            ax.text(x, 0.83, lbl, ha="center", fontsize=9.5)
            if idx < len(xs) - 1:
                ax.annotate("", xy=(xs[idx + 1] - 0.05, 0.74), xytext=(x + 0.05, 0.74),
                            arrowprops=dict(arrowstyle="->", color="#007D60", lw=2.0))
                ax.text((x + xs[idx + 1]) / 2, 0.77, f"key {idx % 2}\n(-c)", ha="center", fontsize=8, color="#007D60")
                sx = x + 0.14
                ax.annotate("", xy=(sx, 0.30), xytext=(x + 0.02, 0.68),
                            arrowprops=dict(arrowstyle="->", color="#C25300", lw=1.4, ls="--"))
                ax.add_patch(Circle((sx, 0.24), 0.038, color="#e3a179"))
        ax.text(0.46, 0.10, "3-lane diffusing dead-end sink (0 cost, 0 reward, high occupancy H)", ha="center", fontsize=9, color="#8a411c")
        ax.text(0.81, 0.64, "safe 0.50\ntarget 0.10/0.90\nprobe 0.48 (+rebate)", ha="center", va="top", fontsize=8.5, color="#005a44")

        # Panel B: Bidirectional ("Diabolical") Combination Lock
        ax = axes[1]
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title("Bidirectional Combination Lock (opposing chains)", fontsize=12, pad=8)
        ax.add_patch(Circle((0.48, 0.76), 0.045, color="#173e59"))
        ax.text(0.48, 0.86, "start", ha="center", fontsize=9.5)
        for x, lbl in [(0.26, "left 1..L-1"), (0.09, "decoy 0.50")]:
            ax.add_patch(Circle((x, 0.76), 0.042, color="#0072B2" if "decoy" in lbl else "#3d6b99"))
            ax.text(x, 0.86, lbl, ha="center", fontsize=8.5)
        ax.annotate("", xy=(0.31, 0.76), xytext=(0.43, 0.76), arrowprops=dict(arrowstyle="->", color="#0072B2", lw=2.0))
        ax.annotate("", xy=(0.14, 0.76), xytext=(0.21, 0.76), arrowprops=dict(arrowstyle="->", color="#0072B2", lw=2.0))
        ax.text(0.26, 0.64, "key 0 only (-c)", ha="center", fontsize=8, color="#0072B2")
        for x, lbl in [(0.70, "right 1..L-1"), (0.89, "target 0.1/0.9")]:
            ax.add_patch(Circle((x, 0.76), 0.042, color="#007D60" if "target" in lbl else "#2b7a66"))
            ax.text(x, 0.86, lbl, ha="center", fontsize=8.5)
        ax.annotate("", xy=(0.65, 0.76), xytext=(0.53, 0.76), arrowprops=dict(arrowstyle="->", color="#007D60", lw=2.0))
        ax.annotate("", xy=(0.84, 0.76), xytext=(0.75, 0.76), arrowprops=dict(arrowstyle="->", color="#007D60", lw=2.0))
        ax.text(0.70, 0.64, "key 1 only (-c)", ha="center", fontsize=8, color="#007D60")
        for sx in (0.28, 0.48, 0.68):
            ax.add_patch(Circle((sx, 0.24), 0.038, color="#e3a179"))
            ax.annotate("", xy=(sx, 0.29), xytext=(sx, 0.60), arrowprops=dict(arrowstyle="->", color="#C25300", lw=1.3, ls="--"))
        ax.text(0.48, 0.10, "Wrong keys on either chain drop irreversibly into dead sink", ha="center", fontsize=9, color="#8a411c")
    elif protocol == "gi_screen":
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 4.0))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title("Pairwise Interaction Screen on Published Horlbeck 2018 K562 Matrix (10 variables, 45 pairs)", fontsize=12, pad=8)
        ax.add_patch(Circle((0.11, 0.52), 0.050, color="#173e59"))
        ax.text(0.11, 0.37, "screen round", ha="center", fontsize=10)
        boxes = [
            (0.70, "#e8f6f1", "#007D60", "45 arrayed pair arms: hit ~ Φ((−3 − GI)/1.25) from real K562 scores;\nrare interacting pair (FDPS–PPP2R1A, GI = −5.24) pays ≈0.96 vs control 0.50"),
            (0.39, "#f3f3f3", "#595959", "10 single-variable arms: GI-scored reward 0.4999; k arms reveal\nsynthetic non-causal main-effect bits (decision-irrelevant)"),
            (0.08, "#fdf1e8", "#C25300", "Multiplexed variant only: 10 pooled 6-variable block arms (group-test readout,\nsuboptimal 0.20/0.30); arrayed variant contains no pooled blocks"),
        ]
        for y, fc, ec, text in boxes:
            ax.add_patch(FancyBboxPatch((0.36, y), 0.60, 0.22, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.4))
            ax.annotate("", xy=(0.35, y + 0.11), xytext=(0.17, 0.52), arrowprops=dict(arrowstyle="->", color=ec, lw=1.8))
            ax.text(0.66, y + 0.11, text, ha="center", va="center", fontsize=9, color="#222222")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 4.0))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title("Epistatic Signalling Discovery under Pleiotropic Confounding (G = 128 subtypes, k passenger modules)", fontsize=12, pad=8)
        ax.add_patch(Circle((0.11, 0.52), 0.050, color="#173e59"))
        ax.text(0.11, 0.37, "cell baseline", ha="center", fontsize=10)
        ax.add_patch(FancyBboxPatch((0.36, 0.70), 0.60, 0.22, boxstyle="round,pad=0.02", fc="#e8f6f1", ec="#007D60", lw=1.6))
        ax.annotate("", xy=(0.35, 0.81), xytext=(0.17, 0.56), arrowprops=dict(arrowstyle="->", color="#007D60", lw=2.2))
        ax.text(0.66, 0.81, "Causal Epistatic Cascade (K1 ∧ K2 synergy: 0.90 in 1/128 responder vs 0.10;\nphospho-probe: 0.48; baseline care: 0.50) · zero passenger dependence",
                ha="center", va="center", fontsize=9, color="#004d3b")
        ax.add_patch(FancyBboxPatch((0.36, 0.39), 0.60, 0.22, boxstyle="round,pad=0.02", fc="#f3f3f3", ec="#595959", lw=1.4))
        ax.annotate("", xy=(0.35, 0.50), xytext=(0.17, 0.52), arrowprops=dict(arrowstyle="->", color="#595959", lw=1.8, ls="--"))
        ax.text(0.66, 0.50, "k Passenger-Omics Assays (pay 0.4999; high Shannon MI about η ∈ {0,1}^k;\nZERO causal KL → traps model-MI IDS, ignored by action-MI IDS & CEAC)",
                ha="center", va="center", fontsize=9, color="#333333")
        ax.add_patch(FancyBboxPatch((0.36, 0.08), 0.60, 0.22, boxstyle="round,pad=0.02", fc="#fdf1e8", ec="#C25300", lw=1.4))
        ax.annotate("", xy=(0.35, 0.19), xytext=(0.17, 0.48), arrowprops=dict(arrowstyle="->", color="#C25300", lw=1.8))
        ax.text(0.66, 0.19, "4^k - 1 Off-Pathway Pleiotropic Arms (0 reward, 0 KL, distinct aberrant states;\ninflates occupancy entropy H(d^π) and S·A bonuses → traps IDAC, UCB-Q, POMD)",
                ha="center", va="center", fontsize=9, color="#7a3300")
    fig.tight_layout()
    return fig


def draw_mechanisms(bundle: dict):
    """Plot measured distraction/trap channels across doses for task 1."""
    import matplotlib.pyplot as plt
    _apply_style()
    rows = [r for r in endpoints(bundle) if r["task"] == 1]
    protocol = bundle["registration"]["protocol"]
    doses = sorted({r["dose"] for r in rows})
    methods = [m for m in METHOD_ORDER if m != "oracle" and any(r["method"] == m for r in rows)]

    if protocol == "locks":
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True)
        for ax, (kind, field, title) in zip(axes, [
            ("combination_lock", "sink_fraction", "Combination Lock · Dead-sink drop rate"),
            ("bidirectional_lock", "decoy_fraction", "Bidirectional Lock · Decoy-lock capture rate"),
        ]):
            for method in methods:
                grp = sorted([r for r in rows if r["kind"] == kind and r["method"] == method], key=lambda r: r["dose"])
                if not grp:
                    continue
                x = [r["dose"] for r in grp]
                y = [100.0 * r[field] for r in grp]
                ax.plot(x, y, label=LABELS[method], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                        lw=2.1 if method.startswith("ceac") else 1.5)
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlabel("Lock depth L", labelpad=6)
            ax.set_xticks(doses)
            ax.set_ylim(-3, 103)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Episodes entering pathology (%)")
    elif protocol == "gi_screen":
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True)
        for ax, kind in zip(axes, ("gi_arrayed", "gi_multiplexed")):
            for method in methods:
                grp = sorted([r for r in rows if r["kind"] == kind and r["method"] == method], key=lambda r: r["dose"])
                if not grp:
                    continue
                ax.plot([r["dose"] for r in grp], [100.0 * r["passenger_omics_fraction"] for r in grp],
                        label=LABELS[method], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                        lw=2.1 if method.startswith("ceac") else 1.5)
            ax.set_title(f"{KINDS[kind]} · main-effect profiling", fontsize=11, pad=8)
            ax.set_xlabel("Variables with unknown main effects k", labelpad=6)
            ax.set_xticks(doses)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Episodes spent profiling main effects (%)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
        for ax, (field, title, ylab) in zip(axes, [
            ("passenger_omics_fraction", "Confounder Trap · Passenger-omics assays", "Passenger-omics episodes (%)"),
            ("pleiotropic_fraction", "Coverage / Bonus Trap · Off-pathway arms", "Off-pathway pleiotropic episodes (%)"),
        ]):
            for method in methods:
                grp = sorted([r for r in rows if r["method"] == method], key=lambda r: r["dose"])
                if not grp:
                    continue
                x = [r["dose"] for r in grp]
                y = [100.0 * r[field] for r in grp]
                ax.plot(x, y, label=LABELS[method], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                        lw=2.1 if method.startswith("ceac") else 1.5)
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlabel("Pleiotropic passenger modules k", labelpad=6)
            ax.set_ylabel(ylab)
            ax.set_xticks(doses)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.15), ncol=3, frameon=False, fontsize=9)
    return fig


def summary_markdown(bundle: dict, rows: list[dict], contrasts: list[dict]) -> str:
    protocol = bundle["registration"]["protocol"]
    cfg = bundle["registration"]["configuration"]
    doses = sorted({r["dose"] for r in rows})
    title = {
        "locks": "# Sequential Combination & Bidirectional Locks",
        "epistasis": "# Epistatic Signalling Pathway Discovery under Pleiotropic Confounding",
        "gi_screen": "# Rare Synthetic-Lethal Pair Discovery on a Real Human GI Map (Horlbeck 2018, K562)",
    }[protocol]
    lines = [
        title,
        "",
        f"Verified **{len(bundle['runs'])}** completed cells ({len(cfg['evaluation_seeds'])} held-out seeds per condition, "
        f"{bundle['registration']['episodes']:,} episodes per run). Independent easy-case calibration passed before sweep execution.",
        "",
        "## Headline Dose Response (Task 1)",
        "",
        "![Dose response across pathology levels](dose_response.png)",
        "",
        "## Mechanism Diagnostics",
        "",
        "![Measured distraction and confounder mechanisms](mechanism_breakdown.png)",
        "",
        "## Within-Algorithm Learning Fans",
        "",
    ]
    for kind in sorted({r["kind"] for r in rows}):
        lines.append(f"![{KINDS[kind]} cumulative regret fans]({kind}_task1_cumulative_regret.png)")
        lines.append("")
    lines += [
        "## Weakest-to-Strongest Pathology Summary (Task 1)",
        "",
        "| Environment | Algorithm | Final gap: min → max | Regret: min → max | Primary distraction (max dose) |",
        "|---|---|---:|---:|---:|",
    ]
    t1 = [r for r in rows if r["task"] == 1]
    for kind in sorted({r["kind"] for r in t1}):
        for method in METHOD_ORDER:
            grp = sorted([r for r in t1 if r["kind"] == kind and r["method"] == method], key=lambda r: r["dose"])
            if not grp:
                continue
            lo, hi = grp[0], grp[-1]
            if protocol == "locks":
                dist_str = f"sink {100*hi['sink_fraction']:.1f}% / decoy {100*hi['decoy_fraction']:.1f}%"
            elif protocol == "gi_screen":
                dist_str = f"profiling {100*hi['passenger_omics_fraction']:.2f}% / wrong pairs {100*hi['wrong_pair_fraction']:.1f}% / blocks {100*hi['block_fraction']:.1f}%"
            else:
                dist_str = f"omics {100*hi['passenger_omics_fraction']:.2f}% / pleio {100*hi['pleiotropic_fraction']:.1f}%"
            lines.append(
                f"| {KINDS[kind]} | {LABELS[method]} | {lo['late_gap_mean']:.4f} → {hi['late_gap_mean']:.4f} "
                f"| {lo['regret_mean']:,.1f} → {hi['regret_mean']:,.1f} | {dist_str} |"
            )

    lines += [
        "",
        f"## Paired CEAC Advantage at Maximum Pathology (Dose = {doses[-1]}, Task 1)",
        "",
        "| Environment | Comparator | Regret saved vs CEAC (mean ± SD) | Added advantage over min dose | CEAC lower on all seeds |",
        "|---|---|---:|---:|---|",
    ]
    for c in contrasts:
        if c["task"] == 1 and c["dose"] == doses[-1]:
            lines.append(
                f"| {KINDS[c['kind']]} | {LABELS[c['comparator']]} | {c['regret_saved_mean']:+,.1f} ± {c['regret_saved_sd']:,.1f} "
                f"| {c['added_advantage_mean']:+,.1f} ± {c['added_advantage_sd']:,.1f} | {c['ceac_wins_all_seeds']} |"
            )
    lines += [
        "",
        "## Complete Endpoint Table",
        "",
        "| Environment | Task | Algorithm | Dose | Final gap (mean ± SD) | Cumulative regret (mean ± SD) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {KINDS[r['kind']]} | {r['task']} | {LABELS[r['method']]} | {r['dose']} "
            f"| {r['late_gap_mean']:.4f} ± {r['late_gap_sd']:.4f} | {r['regret_mean']:,.1f} ± {r['regret_sd']:,.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def export(folder: Path | str, output: Path | str):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = load_study(folder)
    rows = endpoints(bundle)
    contrasts = paired_ceac_comparisons(bundle)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    atomic_json(output / "endpoints.json", rows)
    atomic_json(output / "ceac_comparisons.json", contrasts)
    per_seed = [
        dict(
            protocol=r["protocol"],
            kind=r["kind"],
            task=r["task"],
            method=r["method"],
            dose=r["dose"],
            seed=r["seed"],
            late_gap=r["summary"]["late_gap"],
            cumulative_regret=r["summary"]["cumulative_regret"],
            episodes=r["summary"]["settings"]["episodes"],
            identity=r["summary"]["identity"],
        )
        for r in bundle["runs"]
    ]
    with (output / "seed_endpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (output / "endpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    (output / "assessment.md").write_text(summary_markdown(bundle, rows, contrasts))

    for kind, task in sorted({(r["kind"], r["task"]) for r in bundle["runs"]}):
        for metric in ("window_gap", "cumulative_regret"):
            fig = draw_learning(bundle, kind, task, metric)
            stem = f"{kind}_task{task}_{metric}"
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
            plt.close(fig)

    for fn, stem in [(draw_response, "dose_response"), (draw_mechanisms, "mechanism_breakdown")]:
        fig = fn(bundle)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    arch_fig = draw_architecture(bundle["registration"]["protocol"])
    for suffix in ("png", "pdf"):
        arch_fig.savefig(output / f"architecture.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(arch_fig)

    with gzip.open(output / "evidence.json.gz", "wt") as f:
        json.dump(bundle, f, allow_nan=False, separators=(",", ":"))
    atomic_json(
        output / "manifest.json",
        dict(
            schema=1,
            files={p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"},
            registration=bundle["registration"],
            raw_source_bundle_sha256=digest(Path(folder) / "source_bundle.tar.gz"),
        ),
    )
    print(f"Exported {len(bundle['runs'])} verified runs to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.study, args.output)


if __name__ == "__main__":
    main()
