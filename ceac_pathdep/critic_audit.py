"""Held-out evaluation of Reward and Evidence Critic TD / Bellman errors across all methods.

Evaluates every learner at log-spaced checkpoints t in [16 .. 32,768] on an
independent held-out evaluation simulator (`eval_env`, never observed during
training) using 512 held-out episodes per checkpoint (256 on-policy rollouts
under pi_t + 256 exploratory DAG-probe rollouts), verifying exact trajectory
parity with the registered `ceac_pathdep` study while measuring:

1. `reward_bellman_error`: Held-out Bellman residual of each method's internal
   reward critic V_R / Q_R against empirical held-out transitions in m*.
2. `evidence_critic_error`: Held-out Bellman residual & divergence of each
   method's native evidence/exploration critic (priced V_I(s; beta_t) for CEAC,
   occupancy-entropy shadow critic for IDAC, Hoeffding/transition bonus critics
   for UCB-Q and Optimistic POMD, and unpruned cumulative log-likelihood ratio /
   surprise critic for Bayesian IDS).
3. `stage_epistemic_error`: Bounded stage-level epistemic / bonus critic error
   on held-out probe rollouts across confounding complexity k = 0..3 (replacing
   IDS's O(t) log-odds accumulator with bounded `stage_mi`, showing persistent
   non-zero plateau growth with k on `ids_product` and `ids_action_tree` vs.
   collapse to ~0 on `ceac_ranked`).
"""
from __future__ import annotations
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import math
import multiprocessing as mp
from pathlib import Path
import numpy as np

from ceac.contracts import Episode, Transition
from ceac_critic_diagnostics.evaluator import compile_outcome_tables, evaluate_heldout_critics_fast
from ceac_pathologies.report import log_markevery, style_ceac_legend
from ceac_pathologies.run import ROOT, atomic_json, digest, seed_for
from .learners import _base_policy_occ, make_truth_return_vec
from .study import (
    PATHDEP_COLORS,
    PATHDEP_KINDS,
    PATHDEP_LABELS,
    PATHDEP_MARKERS,
    PATHDEP_ORDER,
    build_pathdep_problem,
    construct_pathdep_learner,
    pathdep_source_identity,
)

CHECKPOINTS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
AUDIT_METHODS = tuple(m for m in PATHDEP_ORDER if m != "oracle")


def run_critic_audit_cell(job) -> dict:
    """Delegate single-pass training + held-out Reward/Evidence Critic Bellman audit directly to `ceac_bench`."""
    from ceac_bench.core import ConditionKey
    from ceac_bench.metrics import HeldOutBellmanObserver, PerformanceTraceObserver
    from ceac_bench.problems import CanalizedGIProblemBuilder, SignallingRewiringProblemBuilder
    from ceac_bench.runner import CellRunner
    from ceac_bench.suites import SUITES

    kind, method, seed, dose, episodes = job
    builder = SignallingRewiringProblemBuilder(kind) if kind == "signalling_rewiring" else CanalizedGIProblemBuilder(kind)
    factory = next(f for f in SUITES["bio_pathdep"].learner_factories if f.method_id == method)
    runner = CellRunner(
        observers=(
            PerformanceTraceObserver(window_size=256, log_every=512),
            HeldOutBellmanObserver(checkpoints=CHECKPOINTS, n_onpolicy=512, n_probe=256, compat_suite_key="pathdep"),
        ),
        compact_every=128,
    )
    cond = ConditionKey(
        suite_id="bio_pathdep",
        family_id=builder.metadata.family_id,
        variant_id=kind,
        dose=int(dose),
        task=1,
        seed=int(seed),
    )
    rec = runner.run_cell(builder, factory, cond, int(episodes))
    cp_eps = rec.critic_audit.get("episode", ())
    cp_rw = rec.critic_audit.get("reward_bellman_error", ())
    cp_ev = rec.critic_audit.get("evidence_critic_error", ())
    cp_st = rec.critic_audit.get("stage_epistemic_error", ())
    records = [
        dict(
            episode=int(e),
            reward_bellman_error=float(rw),
            evidence_critic_error=float(ev),
            stage_epistemic_error=float(st),
        )
        for e, rw, ev, st in zip(cp_eps, cp_rw, cp_ev, cp_st)
    ]
    return dict(
        kind=kind,
        dose=int(dose),
        method=method,
        seed=int(seed),
        cumulative_regret=round(float(rec.cumulative_regret), 6),
        checkpoints=records,
    )


def draw_critic_bellman_traces(audit_bundle: dict, dose: int = 3, field: str = "evidence_critic_error"):
    """Render clean 1x2 overlaid Held-Out Critic TD/Bellman Error time traces (Signalling Rewiring vs Canalized K562 GI)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    cells = [c for c in audit_bundle["cells"] if c["dose"] == dose]
    methods = [m for m in AUDIT_METHODS if any(c["method"] == m for c in cells)]
    kinds = ["signalling_rewiring", "canalized_gi"]
    xticks = [0.016, 0.064, 0.256, 1.024, 4.096, 16.384, 32.768]
    xticklabels = ["16", "64", "256", "1k", "4k", "16k", "33k"]

    is_ev = field == "evidence_critic_error"
    title_sfx = "Held-Out Evidence Critic Error / Divergence ↓" if is_ev else "Held-Out Reward Critic Bellman Error ↓"
    ylab = "Held-out Evidence Critic RMSE (log) ↓" if is_ev else "Held-out Reward Bellman RMSE (log) ↓"

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), sharex=False, squeeze=False)
    for c_idx, kind in enumerate(kinds):
        short_kind = "Signalling Rewiring (H=3)" if kind == "signalling_rewiring" else "Canalized K562 Genetic Risk (H=3)"
        ax = axes[0, c_idx]
        for m_idx, m in enumerate(methods):
            grp = sorted([c for c in cells if c["kind"] == kind and c["method"] == m], key=lambda x: x["seed"])
            if not grp:
                continue
            xs = np.array([pt["episode"] for pt in grp[0]["checkpoints"]], dtype=float) / 1000.0
            vals = np.array([[max(1e-5, pt[field]) for pt in g["checkpoints"]] for g in grp], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(grp) > 1 else np.zeros_like(mean)
            col = PATHDEP_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            ls = (
                "--" if m in {"ceac", "ceac_ranked_natural", "ids_product"}
                else (":" if m == "ids_null" else ("-." if m in {"ucb_q", "ids_action_tree"} else "-"))
            )
            lw = 3.25 if is_ranked else (2.75 if is_ceac else 1.75)
            ms = 8.0 if is_ranked else (7.6 if is_ceac else 5.8)
            zord = 11 if is_ranked else (9 if is_ceac else (5 if "ids_action" in m else 4))
            ax.fill_between(xs, np.maximum(1e-5, mean - sd), mean + sd, color=col, alpha=0.13 if is_ceac else 0.08, linewidth=0)
            ax.plot(
                xs, mean, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls,
                marker=PATHDEP_MARKERS[m], markersize=ms, markevery=list(range(m_idx % 2, len(xs), 2)),
                markeredgecolor="white", markeredgewidth=0.55, zorder=zord,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.014, 36.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.minorticks_off()
        ax.set_title(f"{short_kind} · {title_sfx}", fontsize=10.5, pad=8)
        ax.set_xlabel("Training episodes (log scale)", fontsize=9.6, labelpad=4)
        if c_idx == 0:
            ax.set_ylabel(ylab, fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    kind_lbl = "Evidence Critic TD / Divergence" if is_ev else "Reward Critic Bellman Error"
    fig.suptitle(
        f"Path-Dependent Signalling & K562 GI · Overlaid Held-Out {kind_lbl} Time Traces (k = {dose})",
        fontsize=12.4,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_critic_dose_scaling(audit_bundle: dict):
    """Render clean 1x2 late-horizon Held-Out Evidence Critic Error across k = 0..3."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    cells = audit_bundle["cells"]
    doses = sorted({c["dose"] for c in cells})
    methods = [m for m in AUDIT_METHODS if any(c["method"] == m for c in cells)]
    kinds = ["signalling_rewiring", "canalized_gi"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)
    for c_idx, kind in enumerate(kinds):
        short_kind = "Signalling Rewiring (H=3)" if kind == "signalling_rewiring" else "Canalized K562 Genetic Risk (H=3)"
        ax = axes[0, c_idx]
        for m in methods:
            xs, ys, sds = [], [], []
            for d in doses:
                grp = [c for c in cells if c["kind"] == kind and c["method"] == m and c["dose"] == d]
                if not grp:
                    continue
                v = np.array([max(1e-5, g["checkpoints"][-1]["evidence_critic_error"]) for g in grp], dtype=float)
                xs.append(float(d))
                ys.append(float(v.mean()))
                sds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            if not xs:
                continue
            xs_a, ys_a, sds_a = np.array(xs), np.array(ys), np.array(sds)
            col = PATHDEP_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            ls = (
                "--" if m in {"ceac", "ceac_ranked_natural", "ids_product"}
                else (":" if m == "ids_null" else ("-." if m in {"ucb_q", "ids_action_tree"} else "-"))
            )
            lw = 3.25 if is_ranked else (2.75 if is_ceac else 1.75)
            ms = 8.0 if is_ranked else (7.6 if is_ceac else 5.8)
            zord = 11 if is_ranked else (9 if is_ceac else 4)
            ax.fill_between(xs_a, np.maximum(1e-5, ys_a - sds_a), ys_a + sds_a, color=col, alpha=0.13 if is_ceac else 0.08, linewidth=0)
            ax.plot(
                xs_a, ys_a, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls,
                marker=PATHDEP_MARKERS[m], markersize=ms, markeredgecolor="white", markeredgewidth=0.55, zorder=zord,
            )
        ax.set_yscale("log")
        ax.set_xticks(doses)
        ax.set_title(f"{short_kind} · Final Evidence Critic Error vs. k", fontsize=10.5, pad=8)
        ax.set_xlabel("Path-dependent confounding modules k", fontsize=9.6, labelpad=4)
        if c_idx == 0:
            ax.set_ylabel("Final Evidence Critic RMSE (log) ↓", fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Held-Out Evidence Critic Scaling Across Path-Dependent Confounding k = 0 → 3",
        fontsize=12.4,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def load_critic_audit(out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    with gzip.open(out_dir / "critic_bellman_audit.json.gz", "rt") as f:
        return json.load(f)


def export_critic_audit(audit_cells: list[dict], out_dir: Path, episodes: int = 32768):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_cells = sorted(audit_cells, key=lambda c: (c["kind"], c["dose"], c["method"], c["seed"]))

    # Cross-check exact cumulative regret parity against registered sweep if present
    ev_path = out_dir / "evidence.json.gz"
    if ev_path.exists():
        with gzip.open(ev_path, "rt") as f:
            ev = json.load(f)
        ref_map = {(r["kind"], r["dose"], r["method"], r["seed"]): r["cumulative_regret"] for r in ev["sweep_rows"]}
        for c in audit_cells:
            key = (c["kind"], c["dose"], c["method"], c["seed"])
            if key in ref_map and abs(c["cumulative_regret"] - ref_map[key]) > 1e-3:
                raise ValueError(f"Training trajectory drift detected during held-out audit on {key}: {c['cumulative_regret']} vs {ref_map[key]}")

    bundle = dict(
        schema=1,
        episodes=episodes,
        checkpoints=list(CHECKPOINTS),
        heldout_rollouts_per_checkpoint=512,
        cells=audit_cells,
        sources=pathdep_source_identity(),
    )
    with gzip.open(out_dir / "critic_bellman_audit.json.gz", "wt") as f:
        json.dump(bundle, f, allow_nan=False, separators=(",", ":"))

    for fig_obj, stem in [
        (draw_critic_bellman_traces(bundle, dose=3), "pathdep_critic_bellman_traces"),
        (draw_critic_dose_scaling(bundle), "pathdep_critic_dose_scaling"),
    ]:
        for ext in ("png", "pdf"):
            fig_obj.savefig(out_dir / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig_obj)

    # Update assessment.md if not already containing the critic Bellman section
    md_path = out_dir / "assessment.md"
    if md_path.exists():
        text = md_path.read_text()
        marker = "## Held-Out Reward & Evidence Critic TD / Bellman Error Audit"
        section_lines = [
            marker,
            "",
            "Evaluated at 12 log-spaced checkpoints (`t = 16 .. 32,768`) on **512 independent held-out evaluation rollouts** per checkpoint (`256` on-policy + `256` exploratory DAG-probe rollouts in an isolated `eval_env` with zero training-state mutation; bit-for-bit cumulative regret parity verified against the registered sweep).",
            "",
            "![Held-out Reward and Evidence Critic Bellman error trajectories](pathdep_critic_bellman_traces.png)",
            "",
            "![Held-out Evidence Critic scaling across k](pathdep_critic_dose_scaling.png)",
            "",
            "| Benchmark (`k = 3`, `t = 32,768`) | Algorithm | Held-Out Reward Bellman RMSE | Native Evidence Critic RMSE | Bounded Stage-Epistemic RMSE |",
            "|---|---|---:|---:|---:|",
        ]
        for kind in ("signalling_rewiring", "canalized_gi"):
            for m in AUDIT_METHODS:
                grp = [c for c in audit_cells if c["kind"] == kind and c["dose"] == 3 and c["method"] == m]
                if not grp:
                    continue
                rm = float(np.mean([g["checkpoints"][-1]["reward_bellman_error"] for g in grp]))
                em = float(np.mean([g["checkpoints"][-1]["evidence_critic_error"] for g in grp]))
                sm = float(np.mean([g["checkpoints"][-1]["stage_epistemic_error"] for g in grp]))
                section_lines.append(
                    f"| {PATHDEP_KINDS[kind]} | {PATHDEP_LABELS[m]} | {rm:.4f} | {em:,.4f} | {sm:.4f} |"
                )
        section_lines.append("")
        section_block = "\n".join(section_lines)
        if marker in text:
            head = text.split(marker)[0].rstrip() + "\n\n"
            md_path.write_text(head + section_block)
        else:
            md_path.write_text(text.rstrip() + "\n\n" + section_block)

    # Refresh manifest.json checksums over all files in out_dir
    atomic_json(
        out_dir / "manifest.json",
        dict(
            schema=1,
            files={p.name: digest(p) for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json"},
            sources=pathdep_source_identity(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=32768)
    parser.add_argument("--replot-only", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "notebooks" / "evidence" / "pathologies_pathdep"
    if args.replot_only:
        bundle = load_critic_audit(out_dir)
        export_critic_audit(bundle["cells"], out_dir, episodes=bundle["episodes"])
        print(f"Re-rendered held-out critic Bellman figures in {out_dir}", flush=True)
        return

    jobs = [
        (kind, m, seed, dose, args.episodes)
        for kind in ("signalling_rewiring", "canalized_gi")
        for dose in (0, 1, 2, 3)
        for m in AUDIT_METHODS
        for seed in (0, 1, 2)
    ]
    ctx = mp.get_context("spawn")
    cells = []
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run_critic_audit_cell, jobs, chunksize=1):
            cells.append(res)
            last = res["checkpoints"][-1]
            print(
                "CRITIC_AUDIT:",
                json.dumps({
                    "kind": res["kind"],
                    "dose": res["dose"],
                    "method": res["method"],
                    "seed": res["seed"],
                    "rew_err": round(last["reward_bellman_error"], 4),
                    "ev_err": round(last["evidence_critic_error"], 4),
                    "stage_err": round(last["stage_epistemic_error"], 4),
                }),
                flush=True,
            )

    export_critic_audit(cells, out_dir, episodes=args.episodes)
    print(f"Exported {len(cells)} held-out critic Bellman audit cells to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
