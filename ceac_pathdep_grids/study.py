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

from ceac.contracts import Episode, Transition
from ceac.primitives import IDAC, IDACConfig, OptimisticPOMD, POMDConfig, PrimitiveSpec, UCBQ, UCBQConfig
from ceac_critic_diagnostics.evaluator import (
    SUITE_COLORS,
    SUITE_MARKERS,
    compile_outcome_tables,
    draw_suite_critic_dose_scaling,
    draw_suite_critic_traces,
    evaluate_heldout_critics_fast,
    save_suite_critic_audit,
)
from ceac_hardness.agents import CEACConfig, LaiGravesOracle
from ceac_pathdep.learners import FastPathDepBayesianIDS, MultiStepActionIDS, PathDepGreedyCEAC, _base_policy_occ, make_truth_return_vec
from ceac_pathdep.study import PATHDEP_COLORS, PATHDEP_LABELS, PATHDEP_MARKERS, PATHDEP_ORDER, PathDepCertifiedCEAC
from ceac_pathologies.agent import PathologiesCEACConfig
from ceac_pathologies.report import style_ceac_legend
from ceac_pathologies.run import ROOT, atomic_json, digest, seed_for
from ceac_ranked.agent import RankedCEACConfig
from .environments import (
    PathDepFalseBeaconConfig,
    PathDepFalseBeaconEnv,
    PathDepFalseBeaconModel,
    PathDepUndertowConfig,
    PathDepUndertowEnv,
    PathDepUndertowModel,
    relabel_grid_pathdep,
)

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


def build_grid_pathdep_problem(kind: str, dose: int, task: int, seed: int):
    if kind == "undertow_pathdep":
        cfg = relabel_grid_pathdep(PathDepUndertowConfig(complexity=int(dose)), int(seed))
        model = PathDepUndertowModel(cfg)
        pas_rng = np.random.default_rng(seed_for("grid_pathdep_pas", kind, seed, dose))
        pas = int(pas_rng.integers(cfg.modifier_states)) if cfg.complexity > 0 else 0
        truth = model.true_index(task, passenger=pas)
        ch, mod = model.hypotheses[truth]
        env = PathDepUndertowEnv(cfg, channel=ch, modifier=mod)
        return cfg, model, env, truth, pas
    if kind == "false_beacon_pathdep":
        cfg = relabel_grid_pathdep(PathDepFalseBeaconConfig(complexity=int(dose)), int(seed))
        model = PathDepFalseBeaconModel(cfg)
        pas_rng = np.random.default_rng(seed_for("grid_pathdep_pas", kind, seed, dose))
        pas = int(pas_rng.integers(cfg.modifier_states)) if cfg.complexity > 0 else 0
        truth = model.true_index(task, passenger=pas)
        ch, mod = model.hypotheses[truth]
        env = PathDepFalseBeaconEnv(cfg, channel=ch, modifier=mod)
        return cfg, model, env, truth, pas
    raise ValueError(kind)


def construct_grid_pathdep_learner(model, method: str, rng, episodes: int, truth: int):
    if method == "ceac":
        return PathDepCertifiedCEAC(model, rng, PathologiesCEACConfig(mass_bound=64.0, normalization=64.0, optimizer="reference"))
    if method == "ceac_natural":
        return PathDepCertifiedCEAC(model, rng, PathologiesCEACConfig(mass_bound=64.0, normalization=64.0, optimizer="guarded_natural"))
    if method == "ceac_ranked":
        return PathDepGreedyCEAC(model, rng, RankedCEACConfig(optimizer="reference"))
    if method == "ceac_ranked_natural":
        return PathDepGreedyCEAC(model, rng, RankedCEACConfig(optimizer="guarded_natural"))
    if method == "oracle":
        return LaiGravesOracle(model, truth, CEACConfig(mass_bound=64.0, normalization=64.0))
    if method == "ids_null":
        return FastPathDepBayesianIDS(model, rng, model.prior("null"))
    if method == "ids_product":
        return FastPathDepBayesianIDS(model, rng, model.prior("reference"))
    if method == "ids_action_tree":
        return MultiStepActionIDS(model, rng, model.prior("reference"), target="policy_tree")
    if method == "ids_action_onpath":
        return MultiStepActionIDS(model, rng, model.prior("reference"), target="onpath_class")
    elapsed = np.zeros(model.n, dtype=int)
    for t, sts in enumerate(model.levels):
        elapsed[sts] = t
    spec = PrimitiveSpec(model.mask, elapsed, model.horizon, model.initial, -0.02, 1.02)
    if method == "idac":
        return IDAC(spec, rng, IDACConfig())
    if method == "ucb_q":
        return UCBQ(spec, rng, episodes, UCBQConfig())
    if method == "pomd":
        return OptimisticPOMD(spec, episodes, POMDConfig())
    raise ValueError(method)


def run_grid_pathdep_cell(job) -> dict:
    """Single-pass execution returning BOTH performance trace AND held-out critic Bellman audit."""
    kind, method, seed, task, dose, episodes = job
    t0 = time.monotonic()
    cfg, model, env, truth, pas = build_grid_pathdep_problem(kind, dose, task, seed)
    compiled = compile_outcome_tables(model, truth=truth) if method != "oracle" else None

    rng = np.random.default_rng(seed_for("grid_pathdep", seed, kind, dose, task, method, "learner"))
    action_rng = np.random.default_rng(seed_for("grid_pathdep", seed, kind, dose, task, method, "action"))
    learner = construct_grid_pathdep_learner(model, method, rng, episodes, truth)
    env.reset(seed=seed_for("grid_pathdep", seed, kind, dose, task, "shared_environment"))

    cumulative = 0.0
    window = deque(maxlen=256)
    events = dict(safe=0, stage=0, unstaged=0, target_route=0, wrong_route=0, probe=0, echo_reporter=0, passenger_omics=0, eddy_trap=0)
    last_pol = None
    expected = None
    truth_r_vec = make_truth_return_vec(model, truth)
    trace = []
    cp_rows = []
    cp_set = set(c for c in CHECKPOINTS_32K if c <= episodes)
    log_every = 512

    for t in range(episodes):
        policy, behavior, prices = learner.select(t)
        if policy.identity != last_pol:
            expected = float(truth_r_vec @ _base_policy_occ(model, policy))
            last_pol = policy.identity
        gap = max(0.0, float(model.optimal_returns[truth] - expected))
        obs, _ = env.reset()
        trs = []
        ep_ev = set()
        for _ in range(model.horizon):
            s = model.index[obs]
            a = int(action_rng.choice(model.actions, p=policy.probabilities[s]))
            nxt, rew, done, _, info = env.step(a)
            ll = model.log_likelihood(s, a, nxt, rew, done)
            trs.append(Transition(s, a, rew, -1 if done else model.index[nxt], done, ll))
            if info["event"] in events:
                ep_ev.add(info["event"])
            obs = nxt
            if done:
                break
        for ev in ep_ev:
            events[ev] += 1
        learner.observe(Episode(t, 0, policy.identity, behavior, tuple(trs)), policy, prices)

        if (t + 1) % 128 == 0 and hasattr(learner, "replay") and hasattr(learner.replay, "strata"):
            if hasattr(learner.replay, "evidence") and hasattr(learner.replay.evidence, "strata"):
                learner.replay.evidence.strata.clear()
            if method in {"ceac", "ceac_natural"} and hasattr(learner.replay, "statistics"):
                keep = set(learner.bank._ids) | set(learner.candidates) | set(learner.frozen)
                learner.replay.strata.clear()
                for pid in [k_id for k_id in learner.replay.statistics if k_id not in keep]:
                    del learner.replay.statistics[pid]
            elif method in {"ceac_ranked", "ceac_ranked_natural"} and hasattr(learner.replay, "statistics"):
                keep = set(learner.bank._ids) | set(learner.replay.policies_by_id)
                learner.replay.strata.clear()
                for pid in [k_id for k_id in learner.replay.statistics if k_id not in keep]:
                    del learner.replay.statistics[pid]
            else:
                for bucket in learner.replay.strata.values():
                    bucket.clear()

        cumulative += gap
        window.append(gap)
        ep_num = t + 1
        if t == 0 or ep_num % log_every == 0 or ep_num == episodes:
            trace.append(dict(
                episode=ep_num,
                window_gap=round(float(np.mean(window)), 6),
                cumulative_regret=round(float(cumulative), 4),
                target_fraction=round(events["target_route"] / ep_num, 5),
                echo_fraction=round(events["echo_reporter"] / ep_num, 5),
                vane_fraction=round((events["passenger_omics"] + events["eddy_trap"]) / ep_num, 5),
            ))
        if method != "oracle" and ep_num in cp_set:
            diag = evaluate_heldout_critics_fast(
                suite_key="grid_pathdep",
                kind=kind,
                method=method,
                learner=learner,
                model=model,
                compiled=compiled,
                truth=truth,
                policy=policy,
                prices=prices,
                checkpoint=ep_num,
                seed=seed,
                dose=dose,
            )
            diag["episode"] = ep_num
            cp_rows.append(diag)

    bank_sz = len(learner.bank.policies) if hasattr(learner, "bank") and method.startswith("ceac") else None
    return dict(
        kind=kind,
        method=method,
        seed=int(seed),
        task=int(task),
        dose=int(dose),
        episodes=int(episodes),
        late_gap=float(np.mean(window)),
        cumulative_regret=float(cumulative),
        bank_policies=bank_sz,
        events=events,
        elapsed_seconds=round(time.monotonic() - t0, 2),
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
