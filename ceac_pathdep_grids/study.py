"""Unified low-memory runner, Held-Out Critic Bellman auditor, and primary-palette plotters for Path-Dependent Undertow & False Beacon."""
from __future__ import annotations
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
from collections import defaultdict, deque
from dataclasses import asdict
import gc
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

from ceac_bench.core import ConditionKey
from ceac_bench.problems import GridPathDepProblemBuilder
from ceac_bench.reporting.theme import style_ceac_legend
from ceac_bench.suites import SUITES
from ceac_critic_diagnostics.evaluator import (
    draw_suite_critic_dose_scaling,
    draw_suite_critic_traces,
    save_suite_critic_audit,
)
from ceac_pathdep.study import PATHDEP_COLORS, PATHDEP_LABELS, PATHDEP_MARKERS, PATHDEP_ORDER
from ceac_pathologies.run import ROOT, atomic_json, digest

GRID_PATHDEP_KINDS = [
    ("undertow_pathdep", "Path-Dependent Undertow DeepSea (Thermocline Lock, H = 3)"),
    ("false_beacon_pathdep", "Path-Dependent False Beacon Shore (Icebreaker Staging, H = 3)"),
]
GRID_PATHDEP_SHORT = {
    "undertow_pathdep": "Path-Dependent Undertow",
    "false_beacon_pathdep": "Path-Dependent False Beacon",
}
CHECKPOINTS_32K = (16, 64, 256, 1024, 4096, 16384, 32768)
AUDIT_METHODS = [m for m in PATHDEP_ORDER if m != "oracle"]


def run_grid_pathdep_cell(job) -> dict:
    """Delegate single-pass execution + held-out critic Bellman auditing directly to `ceac_bench`."""
    kind, method, seed, task, dose, episodes = job
    suite = SUITES["grid_pathdep"]
    builder = GridPathDepProblemBuilder(kind)
    factory = next(f for f in suite.learner_factories if f.method_id == method)
    cond = ConditionKey(
        suite_id="grid_pathdep",
        family_id=builder.metadata.family_id,
        variant_id=kind,
        dose=int(dose),
        task=int(task),
        seed=int(seed),
    )
    rec = suite.runner.run_cell(builder, factory, cond, int(episodes))

    ep_tr = rec.trace.get("episode", ())
    gap_tr = rec.trace.get("window_gap", ())
    reg_tr = rec.trace.get("cumulative_regret", ())
    tgt_tr = rec.trace.get("event_rate_target_route", tuple(0.0 for _ in ep_tr))
    ech_tr = rec.trace.get("event_rate_echo_reporter", tuple(0.0 for _ in ep_tr))
    pas_tr = rec.trace.get("event_rate_passenger_omics", tuple(0.0 for _ in ep_tr))
    edd_tr = rec.trace.get("event_rate_eddy_trap", tuple(0.0 for _ in ep_tr))
    trace = [
        dict(
            episode=int(ep),
            window_gap=float(wg),
            cumulative_regret=float(cr),
            target_fraction=float(tf),
            echo_fraction=float(ef),
            vane_fraction=round(float(pf + edf), 5),
        )
        for ep, wg, cr, tf, ef, pf, edf in zip(ep_tr, gap_tr, reg_tr, tgt_tr, ech_tr, pas_tr, edd_tr)
    ]

    cp_eps = rec.critic_audit.get("episode", ())
    cp_rw = rec.critic_audit.get("reward_bellman_error", ())
    cp_ev = rec.critic_audit.get("evidence_critic_error", ())
    cp_st = rec.critic_audit.get("stage_epistemic_error", ())
    cp_rows = [
        dict(
            episode=int(e),
            reward_bellman_error=float(rw),
            evidence_critic_error=float(ev),
            stage_epistemic_error=float(st),
        )
        for e, rw, ev, st in zip(cp_eps, cp_rw, cp_ev, cp_st)
    ]

    events = {
        name: int(rec.scalars.get(f"event_{name}", 0))
        for name in builder.metadata.event_names
    }
    bank_sz = rec.scalars.get("bank_policies") if method.startswith("ceac") else None
    return dict(
        kind=kind,
        method=method,
        seed=int(seed),
        task=int(task),
        dose=int(dose),
        episodes=int(episodes),
        late_gap=rec.late_gap,
        cumulative_regret=rec.cumulative_regret,
        bank_policies=bank_sz,
        events=events,
        elapsed_seconds=round(float(rec.scalars.get("elapsed_seconds", 0.0)), 2),
        trace=trace,
        checkpoints=cp_rows,
    )


def draw_single_grid_pathdep_scaling(rows: list[dict], kind_key: str):
    """Single-environment 1x2 scaling across k = 0 -> 3: Final Policy Gap (left) & Linear Cumulative Regret (right)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    sub = [r for r in rows if r["kind"] == kind_key]
    doses = sorted({r["dose"] for r in sub})
    methods = [m for m in PATHDEP_ORDER if any(r["method"] == m for r in sub)]
    short = GRID_PATHDEP_SHORT[kind_key]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)
    for c_idx, (field, ylab, title_sfx) in enumerate([
        ("late_gap", "Final expected policy gap ↓", "Final Policy Gap vs. k ↓"),
        ("cumulative_regret", "Cumulative expected regret (linear) ↓", "Cumulative Regret vs. k (Linear Scale) ↓"),
    ]):
        ax = axes[0, c_idx]
        for m in methods:
            xs, ys, sds = [], [], []
            for d in doses:
                grp = [r for r in sub if r["method"] == m and r["dose"] == d]
                if not grp:
                    continue
                v = np.array([g[field] for g in grp], dtype=float)
                xs.append(d)
                ys.append(float(v.mean()))
                sds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            if not xs:
                continue
            xs_a, ys_a, sds_a = np.array(xs), np.array(ys), np.array(sds)
            col = PATHDEP_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            is_opt_ids = "ids_action" in m
            ls = (
                "--" if m in {"ceac", "ceac_ranked_natural", "ids_product"}
                else (":" if m in {"oracle", "ids_null"} else ("-." if m in {"ucb_q", "ids_action_tree"} else "-"))
            )
            lw = 3.35 if is_ranked else (2.75 if is_ceac else (2.35 if is_opt_ids else 1.7))
            ms = 8.4 if is_ranked else (7.6 if is_ceac else (6.8 if is_opt_ids else 5.6))
            zord = 12 if is_ranked else (9 if is_opt_ids else (7 if is_ceac else 4))
            mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if PATHDEP_MARKERS[m] != "x" else {}
            ax.fill_between(xs_a, np.maximum(0.0, ys_a - sds_a), ys_a + sds_a, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(xs_a, ys_a, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls, marker=PATHDEP_MARKERS[m], markersize=ms, zorder=zord, **mk_kw)
        ax.set_xticks(doses)
        if c_idx == 0:
            ax.set_ylim(-0.025, 0.80)
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_title(f"{short} (H = 3) · {title_sfx}", fontsize=10.6, pad=8)
        ax.set_xlabel("Path-dependent confounding modules k", fontsize=9.6)
        ax.set_ylabel(ylab, fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{short} · Path-Dependent Confounding Breaks Optimal-Action IDS Across k = 0 → 3", fontsize=12.3, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_single_grid_pathdep_traces(rows: list[dict], kind_key: str, target_dose: int = 3):
    """Single-environment 1x2 overlaid time traces at k=3: Policy Gap (left) & Linear Cumulative Regret (right)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    sub = [r for r in rows if r["kind"] == kind_key and r["dose"] == target_dose]
    methods = [m for m in PATHDEP_ORDER if any(r["method"] == m for r in sub)]
    short = GRID_PATHDEP_SHORT[kind_key]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)
    for c_idx, (field, ylab, title_sfx, log_x) in enumerate([
        ("window_gap", "Rolling 256-episode policy gap ↓", f"Overlaid Policy Gap (k = {target_dose}) ↓", True),
        ("cumulative_regret", "Cumulative expected regret (linear) ↓", f"Overlaid Linear Regret (k = {target_dose}) ↓", False),
    ]):
        ax = axes[0, c_idx]
        for m_i, m in enumerate(methods):
            grp = [r for r in sub if r["method"] == m]
            if not grp:
                continue
            xs = np.array([pt["episode"] for pt in grp[0]["trace"]], dtype=float)
            if not log_x:
                xs = xs / 1000.0
            vals = np.array([[pt[field] for pt in g["trace"]] for g in grp], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(grp) > 1 else np.zeros_like(mean)
            col = PATHDEP_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            is_opt_ids = "ids_action" in m
            ls = (
                "--" if m in {"ceac", "ceac_ranked_natural", "ids_product"}
                else (":" if m in {"oracle", "ids_null"} else ("-." if m in {"ucb_q", "ids_action_tree"} else "-"))
            )
            lw = 3.35 if is_ranked else (2.75 if is_ceac else (2.35 if is_opt_ids else 1.7))
            ms = 8.2 if is_ranked else (7.4 if is_ceac else (6.6 if is_opt_ids else 5.5))
            zord = 12 if is_ranked else (9 if is_opt_ids else (7 if is_ceac else 4))
            if log_x:
                mk_idx = [int(min(len(xs) - 1, max(0, i + (m_i % 2)))) for i in np.unique(np.geomspace(1, len(xs), num=8).astype(int) - 1)]
            else:
                step_mk = max(1, len(xs) // 8)
                mk_idx = list(range((m_i * 2) % step_mk, len(xs), step_mk)) or [0]
            mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if PATHDEP_MARKERS[m] != "x" else {}
            ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(xs, mean, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls, marker=PATHDEP_MARKERS[m], markersize=ms, markevery=mk_idx, zorder=zord, **mk_kw)
        if log_x:
            ax.set_xscale("log")
            ax.set_xlim(256, 32768 * 1.08)
            ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
            ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k", "33k"])
            ax.minorticks_off()
            ax.set_ylim(-0.025, 0.80)
            ax.set_xlabel("Training episodes (log scale)", fontsize=9.6)
        else:
            ax.set_xlim(0.0, 33.0)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("Training episodes (thousands)", fontsize=9.6)
        ax.set_title(f"{short} (H = 3) · {title_sfx}", fontsize=10.6, pad=8)
        ax.set_ylabel(ylab, fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{short} (k = {target_dose}, Task 1) · Overlaid Policy Gap & Linear Regret Time Traces", fontsize=12.3, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_grid_pathdep_dose_response(rows: list[dict]):
    """Render clean 1x2 linear cumulative-regret scaling across k = 0 -> 3 for Path-Dependent Undertow & False Beacon."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)
    doses = sorted({r["dose"] for r in rows})
    methods = [m for m in PATHDEP_ORDER if any(r["method"] == m for r in rows)]

    for c_idx, (kind_key, _) in enumerate(GRID_PATHDEP_KINDS):
        ax = axes[0, c_idx]
        for m in methods:
            xs, ys, sds = [], [], []
            for d in doses:
                grp = [r for r in rows if r["kind"] == kind_key and r["method"] == m and r["dose"] == d]
                if not grp:
                    continue
                v = np.array([g["cumulative_regret"] for g in grp], dtype=float)
                xs.append(d)
                ys.append(float(v.mean()))
                sds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            if not xs:
                continue
            xs_a, ys_a, sds_a = np.array(xs), np.array(ys), np.array(sds)
            col = PATHDEP_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            is_opt_ids = "ids_action" in m
            ls = (
                "--" if m in {"ceac", "ceac_ranked_natural", "ids_product"}
                else (":" if m in {"oracle", "ids_null"} else ("-." if m in {"ucb_q", "ids_action_tree"} else "-"))
            )
            lw = 3.35 if is_ranked else (2.75 if is_ceac else (2.35 if is_opt_ids else 1.7))
            ms = 8.4 if is_ranked else (7.6 if is_ceac else (6.8 if is_opt_ids else 5.6))
            zord = 12 if is_ranked else (9 if is_opt_ids else (7 if is_ceac else 4))
            mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if PATHDEP_MARKERS[m] != "x" else {}
            ax.fill_between(xs_a, np.maximum(0.0, ys_a - sds_a), ys_a + sds_a, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(xs_a, ys_a, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls, marker=PATHDEP_MARKERS[m], markersize=ms, zorder=zord, **mk_kw)
        ax.set_ylim(bottom=0.0)
        ax.set_xticks(doses)
        ax.set_title(f"{GRID_PATHDEP_SHORT[kind_key]} (H=3) · Cumulative Regret vs. k (Linear)", fontsize=10.6, pad=8)
        ax.set_xlabel("Path-dependent confounding modules k (6k echo plumes/glints, 3^k − 1 off-path eddies)", fontsize=9.2)
        if c_idx == 0:
            ax.set_ylabel("Cumulative expected regret (linear) ↓", fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Path-Dependent Undertow DeepSea & False Beacon Shore · Scaling Across Confounding Modules k = 0 → 3", fontsize=12.4, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_grid_pathdep_overlaid_traces(rows: list[dict], target_dose: int = 3, metric: str = "cumulative_regret"):
    """Render clean 1x2 overlaid performance time traces (Path-Dependent Undertow vs False Beacon) at k=3."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    sub = [r for r in rows if r["dose"] == target_dose]
    methods = [m for m in PATHDEP_ORDER if any(r["method"] == m for r in sub)]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)

    is_reg = metric == "cumulative_regret"
    ylab = "Cumulative expected regret (linear) ↓" if is_reg else "Rolling 256-episode policy gap ↓"
    sfx = "Linear Cumulative Regret" if is_reg else "Expected Policy Gap"

    for c_idx, (kind_key, _) in enumerate(GRID_PATHDEP_KINDS):
        ax = axes[0, c_idx]
        for m_i, m in enumerate(methods):
            grp = [r for r in sub if r["kind"] == kind_key and r["method"] == m]
            if not grp:
                continue
            xs = np.array([pt["episode"] for pt in grp[0]["trace"]], dtype=float)
            if is_reg:
                xs = xs / 1000.0
            vals = np.array([[pt[metric] for pt in g["trace"]] for g in grp], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(grp) > 1 else np.zeros_like(mean)
            col = PATHDEP_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            is_opt_ids = "ids_action" in m
            ls = (
                "--" if m in {"ceac", "ceac_ranked_natural", "ids_product"}
                else (":" if m in {"oracle", "ids_null"} else ("-." if m in {"ucb_q", "ids_action_tree"} else "-"))
            )
            lw = 3.35 if is_ranked else (2.75 if is_ceac else (2.35 if is_opt_ids else 1.7))
            ms = 8.2 if is_ranked else (7.4 if is_ceac else (6.6 if is_opt_ids else 5.5))
            zord = 12 if is_ranked else (9 if is_opt_ids else (7 if is_ceac else 4))
            if is_reg:
                step_mk = max(1, len(xs) // 8)
                mk_idx = list(range((m_i * 2) % step_mk, len(xs), step_mk)) or [0]
            else:
                mk_idx = [int(min(len(xs) - 1, max(0, i + (m_i % 2)))) for i in np.unique(np.geomspace(1, len(xs), num=8).astype(int) - 1)]
            mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if PATHDEP_MARKERS[m] != "x" else {}
            ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(xs, mean, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls, marker=PATHDEP_MARKERS[m], markersize=ms, markevery=mk_idx, zorder=zord, **mk_kw)
        if is_reg:
            ax.set_xlim(0.0, 33.0)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("Training episodes (thousands)", fontsize=9.5)
        else:
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 0.80)
            ax.set_xlim(256, 32768 * 1.08)
            ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
            ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k", "33k"])
            ax.minorticks_off()
            ax.set_xlabel("Training episodes (log scale)", fontsize=9.5)
        ax.set_title(f"{GRID_PATHDEP_SHORT[kind_key]} (H=3) · {sfx}", fontsize=10.6, pad=8)
        if c_idx == 0:
            ax.set_ylabel(ylab, fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Path-Dependent Undertow & False Beacon · Overlaid {sfx} Time Traces (k = {target_dose}, Task 1)", fontsize=12.4, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def load_grid_pathdep_export(folder: Path | str) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Checksum mismatch in {folder / name}")
    with gzip.open(folder / "evidence.json.gz", "rt") as f:
        return json.load(f)


def run_and_export(out_dir: Path | str, workers: int = 4, episodes: int = 32768):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (kind, m, seed, 1, dose, episodes)
        for kind, _ in GRID_PATHDEP_KINDS
        for dose in (0, 1, 2, 3)
        for m in PATHDEP_ORDER
        for seed in (0, 1, 2)
    ]
    print(f"[grid-pathdep] Running {len(jobs)} unified sweep + critic-audit cells with {workers} spawn workers...", flush=True)
    t0 = time.monotonic()
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(run_grid_pathdep_cell, jobs, chunksize=1), 1):
            rows.append(res)
            if i % 24 == 0 or i == len(jobs):
                print(
                    f"[grid-pathdep] {i}/{len(jobs)} complete ({time.monotonic() - t0:.1f}s) — "
                    f"latest {res['kind']} {res['method']} k={res['dose']} s={res['seed']} "
                    f"gap={res['late_gap']:.4f} reg={res['cumulative_regret']:.1f} ({res['elapsed_seconds']}s)",
                    flush=True,
                )
    gc.collect()
    rows.sort(key=lambda r: (r["kind"], r["dose"], r["method"], r["seed"]))

    critic_cells = [
        dict(
            suite="grid_pathdep",
            kind=r["kind"],
            dose=r["dose"],
            task=r["task"],
            seed=r["seed"],
            method=r["method"],
            cumulative_regret=r["cumulative_regret"],
            checkpoints=r["checkpoints"],
        )
        for r in rows
        if r["method"] != "oracle"
    ]
    perf_rows = [{k: v for k, v in r.items() if k != "checkpoints"} for r in rows]

    with gzip.open(out_dir / "evidence.json.gz", "wt") as f:
        json.dump(dict(schema=1, episodes=episodes, rows=perf_rows), f, allow_nan=False, separators=(",", ":"))
    save_suite_critic_audit(
        out_dir,
        dict(schema=1, suite="grid_pathdep", checkpoints=list(CHECKPOINTS_32K), cells=critic_cells),
        "critic_bellman_audit.json.gz",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_kinds = [(k, GRID_PATHDEP_SHORT[k]) for k, _ in GRID_PATHDEP_KINDS]
    figs = [
        (draw_grid_pathdep_dose_response(perf_rows), "grid_pathdep_dose_response"),
        (draw_grid_pathdep_overlaid_traces(perf_rows, target_dose=3), "grid_pathdep_overlaid_performance"),
        (
            draw_suite_critic_traces(
                critic_cells,
                short_kinds,
                target_dose=3,
                suptitle="Path-Dependent Undertow & False Beacon (k = 3, Task 1) · Held-Out Reward & Evidence Critic Error",
                method_order=AUDIT_METHODS,
                ranked_ceac_present=True,
            ),
            "grid_pathdep_critic_bellman_traces",
        ),
        (
            draw_suite_critic_dose_scaling(
                critic_cells,
                short_kinds,
                xlab="Path-dependent confounding modules k (6k echo plumes/glints, 3^k − 1 off-path eddies)",
                suptitle="Path-Dependent Undertow & False Beacon · Late-Horizon Critic Error Scaling Across k ∈ {0, 1, 2, 3}",
                method_order=AUDIT_METHODS,
                ranked_ceac_present=True,
            ),
            "grid_pathdep_critic_dose_scaling",
        ),
    ]
    for fig, stem in figs:
        for sfx in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{sfx}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    md_lines = [
        "# Path-Dependent Undertow DeepSea & False Beacon Shore (`ceac_pathdep_grids`)",
        "",
        f"Evaluated **{len(rows)}** held-out cells (`seeds = 0, 1, 2`, `{episodes:,}` episodes / `{3*episodes:,}` physical steps per cell) across path-dependent confounding complexity `k ∈ {{0, 1, 2, 3}}`.",
        "",
        "![Dose response](grid_pathdep_dose_response.png)",
        "",
        "![Overlaid performance](grid_pathdep_overlaid_performance.png)",
        "",
        "![Critic Bellman traces](grid_pathdep_critic_bellman_traces.png)",
        "",
        "![Critic dose scaling](grid_pathdep_critic_dose_scaling.png)",
        "",
        "| Benchmark | Algorithm | Final Gap (k=0 → k=3) | Regret (k=0 → k=3) | Evidence Critic RMSE (k=3) |",
        "|---|---|---:|---:|---:|",
    ]
    for kind_key, kind_title in GRID_PATHDEP_KINDS:
        for m in PATHDEP_ORDER:
            g0 = [r for r in rows if r["kind"] == kind_key and r["method"] == m and r["dose"] == 0]
            g3 = [r for r in rows if r["kind"] == kind_key and r["method"] == m and r["dose"] == 3]
            if not g0 or not g3:
                continue
            ev_s = "—" if m == "oracle" else f"{np.mean([x['checkpoints'][-1]['evidence_critic_error'] for x in g3]):,.4f}"
            md_lines.append(
                f"| {kind_title} | {PATHDEP_LABELS[m]} "
                f"| {np.mean([x['late_gap'] for x in g0]):.4f} → {np.mean([x['late_gap'] for x in g3]):.4f} "
                f"| {np.mean([x['cumulative_regret'] for x in g0]):,.1f} → {np.mean([x['cumulative_regret'] for x in g3]):,.1f} "
                f"| {ev_s} |"
            )
    md_lines.append("")
    (out_dir / "assessment.md").write_text("\n".join(md_lines))

    files = {p.name: digest(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "manifest.json"}
    atomic_json(out_dir / "manifest.json", dict(schema=1, episodes=episodes, cells=len(rows), files=files))
    print(f"[grid-pathdep] Exported complete verified bundle + 4 figures to {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/evidence/pathdep_undertow_falsebeacon")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=32768)
    args = parser.parse_args()
    run_and_export(args.output, workers=args.workers, episodes=args.episodes)


if __name__ == "__main__":
    main()
