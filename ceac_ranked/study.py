"""Head-to-head evaluation and reporting for GreedyPricedCEAC on the K562 GI screen."""
from __future__ import annotations
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ceac.contracts import Episode, Transition
from ceac_pathologies.report import LABELS as BASE_LABELS, KINDS
from ceac_pathologies.run import (
    ROOT,
    _sanitize_diagnostics,
    atomic_json,
    build_problem,
    construct_learner as base_construct_learner,
    digest,
    encoded,
    seed_for,
    source_identity as base_source_identity,
)
from .agent import GreedyPricedCEAC, RankedCEACConfig


RANKED_LABELS = {
    **BASE_LABELS,
    "ceac_ranked": "CEAC · greedy disagreement + price-ranked replay",
    "ceac_ranked_natural": "CEAC · greedy + price-ranked (guarded natural)",
}

RANKED_COLORS = {
    "idac": "#C25300",
    "ucb_q": "#0072B2",
    "pomd": "#B34697",
    "ids_additive": "#C93441",
    "ids_product": "#595959",
    "ids_empirical": "#8C6D1F",
    "ids_action": "#111111",
    "ceac": "#74A892",
    "ceac_natural": "#9CC5B0",
    "ceac_ranked": "#007D60",
    "ceac_ranked_natural": "#004D3B",
    "oracle": "#3E3163",
}

RANKED_ORDER = [
    "idac",
    "ucb_q",
    "pomd",
    "ids_additive",
    "ids_product",
    "ids_empirical",
    "ids_action",
    "ceac",
    "ceac_natural",
    "ceac_ranked",
    "ceac_ranked_natural",
    "oracle",
]


def ranked_source_identity() -> dict[str, str]:
    out = dict(base_source_identity())
    for p in sorted((ROOT / "ceac_ranked").glob("*.py")):
        if p.name != "study.py":
            out[str(p.relative_to(ROOT))] = digest(p)
    return out


def construct_any_learner(protocol: str, model, method: str, rng, episodes: int, truth: int):
    if method == "ceac_ranked":
        return GreedyPricedCEAC(model, rng, RankedCEACConfig(optimizer="reference"))
    if method == "ceac_ranked_natural":
        return GreedyPricedCEAC(model, rng, RankedCEACConfig(optimizer="guarded_natural"))
    return base_construct_learner(protocol, model, method, rng, episodes, truth)


def run_ranked_cell(
    protocol: str,
    config_dict: dict,
    method: str,
    seed: int,
    task: int,
    dose: int,
    episodes: int,
    output: Path | str,
    log_every: int = 512,
):
    cfg, model, env, truth, passenger = build_problem(protocol, config_dict, task, seed)
    ranked_cfg = RankedCEACConfig(optimizer="guarded_natural" if method.endswith("_natural") else "reference")
    settings = dict(
        protocol=protocol,
        environment=asdict(cfg),
        method=method,
        seed=int(seed),
        task_evaluator_only=int(task),
        realized_passenger_evaluator_only=int(passenger),
        dose=int(dose),
        episodes=int(episodes),
        horizon=int(model.horizon),
        physical_budget=int(model.horizon * episodes),
        log_every=int(log_every),
        num_models=int(model.num_models),
        ranked_config=asdict(ranked_cfg) if method.startswith("ceac_ranked") else None,
    )
    manifest = dict(schema=1, settings=settings, sources=ranked_source_identity())
    manifest["identity"] = hashlib.sha256(encoded(manifest).encode()).hexdigest()
    output = Path(output)
    if output.exists():
        old = json.loads((output / "manifest.json").read_text())
        if old != json.loads(encoded(manifest)):
            raise ValueError(f"Existing cell identity mismatch: {output}")
        status = json.loads((output / "status.json").read_text())
        if status["status"] == "complete":
            for name, checksum in status["artifacts"].items():
                if digest(output / name) != checksum:
                    raise ValueError(f"Cell artifact mismatch: {name}")
            return json.loads((output / "summary.json").read_text())
        raise ValueError(f"Incomplete cell: {output}")

    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "status.json", dict(status="running"))

    # Keep identical RNG streams for base methods; deterministic stream for ranked methods.
    rng_method = method
    rng = np.random.default_rng(seed_for(protocol, seed, cfg.kind, dose, task, rng_method, "learner"))
    action_rng = np.random.default_rng(seed_for(protocol, seed, cfg.kind, dose, task, rng_method, "action"))
    started = time.monotonic()
    learner = construct_any_learner(protocol, model, method, rng, episodes, truth)
    setup_seconds = time.monotonic() - started
    env.reset(seed=seed_for(protocol, seed, cfg.kind, dose, task, "shared_environment"))

    cumulative = 0.0
    window = deque(maxlen=256)
    events = dict(
        safe=0, target=0, synergy=0, probe=0, sink=0, decoy_unlock=0,
        single_ko=0, passenger_omics=0, pleiotropic=0,
        control=0, pair_test=0, sl_pair=0, block=0,
    )
    saturation = 0
    actor_updates = 0
    priced_updates = 0
    last_policy = None
    expected = None
    prior_supported = (
        bool(model.prior("reference" if method == "ids_action" else method.removeprefix("ids_"))[truth] > 0)
        if method.startswith("ids_")
        else True
    )

    with gzip.open(output / "trace.jsonl.gz", "wt") as target:
        for t in range(episodes):
            policy, behavior, prices = learner.select(t)
            if policy.identity != last_policy:
                expected = float(model.returns(policy)[truth])
                last_policy = policy.identity
            gap = max(0.0, float(model.optimal_returns[truth] - expected))
            observation, _ = env.reset()
            transitions = []
            ep_events = set()
            for _ in range(model.horizon):
                s = model.index[observation]
                action = int(action_rng.choice(model.actions, p=policy.probabilities[s]))
                nxt, reward, done, _, info = env.step(action)
                ll = model.log_likelihood(s, action, nxt, reward, done)
                transitions.append(Transition(s, action, reward, -1 if done else model.index[nxt], done, ll))
                ev = info["event"]
                if ev in events:
                    ep_events.add(ev)
                observation = nxt
                if done:
                    break
            for ev in ep_events:
                events[ev] += 1
            episode = Episode(t, 0, policy.identity, behavior, tuple(transitions))
            learner.observe(episode, policy, prices)
            if (t + 1) % 128 == 0 and hasattr(learner, "replay") and hasattr(learner.replay, "strata"):
                # Sufficient statistics are already accumulated in replay.statistics and evidence.scores;
                # evict raw Episode lists so worker RSS stays < 50 MB across 65,536 episodes.
                for bucket in learner.replay.strata.values():
                    bucket.clear()
                if hasattr(learner.replay, "evidence") and hasattr(learner.replay.evidence, "strata"):
                    learner.replay.evidence.strata.clear()
                if method in {"ceac", "ceac_natural"} and hasattr(learner.replay, "statistics"):
                    keep = set(learner.bank._ids) | set(learner.candidates) | set(learner.frozen)
                    for pid in [k for k in learner.replay.statistics if k not in keep]:
                        del learner.replay.statistics[pid]
            cumulative += gap
            window.append(gap)
            if method == "pomd":
                saturation += int(np.all(learner.q[learner.mask] == 0))
            if behavior == "actor":
                actor_updates += 1
                priced_updates += int(np.any(prices > 0))
            if t == 0 or (t + 1) % log_every == 0 or t + 1 == episodes:
                row = dict(
                    episode=t + 1,
                    interactions=model.horizon * (t + 1),
                    gap=gap,
                    window_gap=float(np.mean(window)),
                    cumulative_regret=cumulative,
                    events=dict(events),
                    actor_updates=actor_updates,
                    priced_actor_updates=priced_updates,
                    pomd_zero_fraction=saturation / (t + 1),
                )
                if method.startswith("ids_"):
                    row["responder_posterior"] = model.responder_probability(learner.posterior)
                target.write(json.dumps(row, allow_nan=False) + "\n")

    summary = dict(
        settings=settings,
        identity=manifest["identity"],
        cumulative_regret=cumulative,
        late_gap=float(np.mean(window)),
        events=events,
        actor_updates=actor_updates,
        priced_actor_updates=priced_updates,
        pomd_zero_fraction=saturation / episodes,
        seconds=time.monotonic() - started,
        setup_seconds=setup_seconds,
        prior_contains_true_model=prior_supported,
        public_class_contains_true_model=True,
        reachable_states=int(model.n),
        actions=int(model.actions),
        horizon=int(model.horizon),
        optimum=float(model.optimal_returns[truth]),
        learner_diagnostics=_sanitize_diagnostics(learner.diagnostics()),
    )
    atomic_json(output / "summary.json", summary)
    artifacts = ["manifest.json", "summary.json", "trace.jsonl.gz"]
    atomic_json(output / "status.json", dict(status="complete", artifacts={a: digest(output / a) for a in artifacts}))
    return summary


def _exec_job(job):
    protocol, cfg_dict, method, seed, task, dose, budget, folder = job
    res = run_ranked_cell(protocol, cfg_dict, method, seed, task, dose, budget, folder, log_every=512)
    return dict(
        folder=str(Path(folder).resolve().relative_to(ROOT)),
        kind=cfg_dict["kind"],
        dose=dose,
        task=task,
        method=method,
        seed=seed,
        late_gap=res["late_gap"],
        cumulative_regret=res["cumulative_regret"],
        bank_policies=res["learner_diagnostics"].get("bank_policies"),
        actor_updates=res["actor_updates"],
        priced_actor_updates=res["priced_actor_updates"],
        events=res["events"],
        seconds=res["seconds"],
    )


def load_gi_ranked_export(folder: Path | str) -> dict:
    """Load and SHA-256-verify the portable GI ranked evidence export."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected_sha in manifest["files"].items():
        if digest(folder / name) != expected_sha:
            raise ValueError(f"Portable GI evidence checksum mismatch: {folder / name}")
    with gzip.open(folder / "evidence.json.gz", "rt") as f:
        return json.load(f)


def draw_gi_ranked_response(summary_rows: list[dict]):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    kinds = sorted({r["kind"] for r in summary_rows})
    doses = sorted({r["dose"] for r in summary_rows})
    methods = [m for m in RANKED_ORDER if any(r["method"] == m for r in summary_rows)]

    fig, axes = plt.subplots(len(kinds), 2, figsize=(12.8, 4.4 * len(kinds)), squeeze=False)
    for r_i, kind in enumerate(kinds):
        for c_i, (field, sd_field, ylab, symlog) in enumerate([
            ("late_gap_mean", "late_gap_sd", "Final expected policy loss ↓", False),
            ("regret_mean", "regret_sd", "Cumulative pseudo-regret (symlog) ↓", True),
        ]):
            ax = axes[r_i, c_i]
            for m in methods:
                grp = sorted([r for r in summary_rows if r["kind"] == kind and r["task"] == 1 and r["method"] == m], key=lambda x: x["dose"])
                if not grp:
                    continue
                xs = np.array([x["dose"] for x in grp], dtype=float)
                ys = np.array([x[field] for x in grp], dtype=float)
                sds = np.array([x[sd_field] for x in grp], dtype=float)
                col = RANKED_COLORS[m]
                is_ranked = m.startswith("ceac_ranked")
                ls = "--" if m in {"ceac", "ceac_natural", "ids_product"} else (":" if m == "oracle" else "-")
                ax.fill_between(xs, np.maximum(0.0, ys - sds), ys + sds, color=col, alpha=0.12, linewidth=0)
                ax.plot(xs, ys, label=RANKED_LABELS[m], color=col, lw=2.5 if is_ranked else 1.5, linestyle=ls,
                        marker="o" if is_ranked else "s", markersize=6.5 if is_ranked else 5.0, zorder=7 if is_ranked else 3)
            ax.set_title(f"{KINDS[kind]} · {'Final gap' if c_i == 0 else 'Cumulative regret'}", fontsize=11, pad=8)
            ax.set_xlabel("Variables with unknown non-causal main effects k (M = 46 · 2^k)", fontsize=10, labelpad=6)
            ax.set_ylabel(ylab, fontsize=10)
            ax.set_xticks(doses)
            if symlog:
                ax.set_yscale("symlog", linthresh=10.0)
                ax.set_ylim(bottom=0.0)
            else:
                ax.set_ylim(-0.03, 1.00)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.105), ncol=3, frameon=False, fontsize=9)
    return fig


def draw_gi_ranked_mechanisms(summary_rows: list[dict]):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    doses = sorted({r["dose"] for r in summary_rows})
    methods = [m for m in RANKED_ORDER if any(r["method"] == m for r in summary_rows)]

    fig_m, axes_m = plt.subplots(1, 2, figsize=(12.8, 4.6))
    ceac_methods = [m for m in ("ceac", "ceac_natural", "ceac_ranked", "ceac_ranked_natural") if m in methods]
    for m in ceac_methods:
        grp = sorted([r for r in summary_rows if r["kind"] == "gi_arrayed" and r["task"] == 1 and r["method"] == m], key=lambda x: x["dose"])
        if grp and grp[0]["bank_policies_mean"] is not None:
            axes_m[0].plot(
                [x["dose"] for x in grp],
                [x["bank_policies_mean"] for x in grp],
                label=RANKED_LABELS[m],
                color=RANKED_COLORS[m],
                lw=2.4 if m.startswith("ceac_ranked") else 1.8,
                linestyle="-" if m.startswith("ceac_ranked") else "--",
                marker="o" if m.startswith("ceac_ranked") else "s",
            )
    axes_m[0].set_title("Arrayed Screen · Active Replay Bank Size ↓", fontsize=11, pad=8)
    axes_m[0].set_xlabel("Variables with unknown non-causal main effects k", labelpad=6)
    axes_m[0].set_ylabel("Registered LP bank columns")
    axes_m[0].set_xticks(doses)
    axes_m[0].set_ylim(0, 52)
    axes_m[0].grid(alpha=0.2)
    axes_m[0].spines[["top", "right"]].set_visible(False)

    for m in methods:
        if m == "oracle":
            continue
        grp = sorted([r for r in summary_rows if r["kind"] == "gi_arrayed" and r["task"] == 1 and r["method"] == m], key=lambda x: x["dose"])
        if grp:
            axes_m[1].plot(
                [x["dose"] for x in grp],
                [x["profiling_pct"] for x in grp],
                label=RANKED_LABELS[m],
                color=RANKED_COLORS[m],
                lw=2.2 if m.startswith("ceac_ranked") else 1.5,
                linestyle="--" if m in {"ceac", "ceac_natural", "ids_product"} else "-",
                marker="o" if m.startswith("ceac_ranked") else "s",
            )
    axes_m[1].set_title("Arrayed Screen · Non-Causal Main-Effect Profiling ↓", fontsize=11, pad=8)
    axes_m[1].set_xlabel("Variables with unknown non-causal main effects k", labelpad=6)
    axes_m[1].set_ylabel("Episodes spent on non-causal single arms (%)")
    axes_m[1].set_xticks(doses)
    axes_m[1].grid(alpha=0.2)
    axes_m[1].spines[["top", "right"]].set_visible(False)

    fig_m.tight_layout(rect=(0.0, 0.22, 1.0, 1.0))
    handles, labels = axes_m[1].get_legend_handles_labels()
    fig_m.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.19), ncol=3, frameon=False, fontsize=8.8)
    return fig_m


def _load_compact_traces(results: list[dict]) -> list[dict]:
    """Load compact columnar trajectories from each completed cell's trace.jsonl.gz."""
    traces = []
    for r in sorted(results, key=lambda x: (x["kind"], x["task"], x["dose"], x["method"], x["seed"])):
        cell_dir = Path(r["folder"])
        if not cell_dir.is_absolute():
            cell_dir = ROOT / cell_dir
        with gzip.open(cell_dir / "trace.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        traces.append(
            dict(
                kind=r["kind"],
                dose=int(r["dose"]),
                task=int(r["task"]),
                method=r["method"],
                seed=int(r["seed"]),
                episode=[int(t["episode"]) for t in rows],
                window_gap=[round(float(t["window_gap"]), 6) for t in rows],
                cumulative_regret=[round(float(t["cumulative_regret"]), 3) for t in rows],
                sl_pair=[int(t["events"].get("sl_pair", 0)) for t in rows],
                pair_test=[int(t["events"].get("pair_test", 0)) for t in rows],
                block=[int(t["events"].get("block", 0)) for t in rows],
                passenger_omics=[int(t["events"].get("passenger_omics", 0)) for t in rows],
                control=[int(t["events"].get("control", 0)) for t in rows],
            )
        )
    return traces


def _compact_window_rate(tr: dict, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    eps = tr["episode"]
    start = 1 if len(eps) > 1 and eps[0] == 1 else 0
    xs = np.array(eps[start:], dtype=float) / 1000.0
    rates = np.zeros(len(xs), dtype=float)
    for out_i, idx in enumerate(range(start, len(eps))):
        if out_i == 0:
            dt = float(eps[idx])
            de = sum(float(tr[k][idx]) for k in keys)
        else:
            dt = float(eps[idx] - eps[idx - 1])
            de = sum(float(tr[k][idx] - tr[k][idx - 1]) for k in keys)
        rates[out_i] = 100.0 * de / max(dt, 1.0)
    return xs, rates


def draw_gi_ranked_learning(gi_bundle: dict, kind: str, metric: str = "cumulative_regret"):
    """Small-multiple per-algorithm time traces across confounding doses k = 0..3."""
    import math
    import matplotlib.pyplot as plt
    from ceac_pathologies.report import DOSE_COLORS

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    traces = [t for t in gi_bundle["traces"] if t["kind"] == kind and t["task"] == 1]
    summary_rows = [r for r in gi_bundle["summary_rows"] if r["kind"] == kind and r["task"] == 1]
    methods = [m for m in RANKED_ORDER if any(t["method"] == m for t in traces)]
    doses = sorted({t["dose"] for t in traces})
    cols = 3
    rows_n = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(13.8, 3.25 * rows_n), squeeze=False, sharex=True, sharey=True)
    badge_box = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.88)

    for idx, (ax, m) in enumerate(zip(axes.flat, methods)):
        for dose, color in zip(doses, DOSE_COLORS):
            cells = sorted([t for t in traces if t["method"] == m and t["dose"] == dose], key=lambda x: x["seed"])
            xs = np.array(cells[0]["episode"], dtype=float) / 1000.0
            vals = np.array([c[metric] for c in cells], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
            ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=color, alpha=0.14, linewidth=0)
            ax.plot(xs, mean, color=color, lw=1.7, label=f"k = {dose}", zorder=3)
        lo_row = next(r for r in summary_rows if r["method"] == m and r["dose"] == doses[0])
        hi_row = next(r for r in summary_rows if r["method"] == m and r["dose"] == doses[-1])
        if metric == "window_gap":
            ax.text(
                0.97, 0.96, f"Gap: {lo_row['late_gap_mean']:.3f} → {hi_row['late_gap_mean']:.3f}",
                ha="right", va="top", transform=ax.transAxes, fontsize=8.3, color="#222222", bbox=badge_box, zorder=8,
            )
            ax.set_ylim(-0.04, 1.08)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            ax.text(
                0.03, 0.96, f"Regret (k={doses[-1]}): {hi_row['regret_mean']:,.0f}",
                ha="left", va="top", transform=ax.transAxes, fontsize=8.3, color="#222222", bbox=badge_box, zorder=8,
            )
        ax.set_title(RANKED_LABELS[m], fontsize=9.8, pad=7)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, xs[-1])
        if idx + cols >= len(methods):
            ax.xaxis.set_tick_params(labelbottom=True)
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)

    fig.supxlabel("Training episodes (thousands)", fontsize=11, y=0.012)
    fig.supylabel("Expected policy gap ↓" if metric == "window_gap" else "Cumulative expected regret ↓", fontsize=11, x=0.01)
    fig.suptitle(f"{KINDS[kind]} · Non-causal main-effect confounding sweep (k = 0..3) · Task 1", fontsize=13.2, y=0.992)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(doses), frameon=False, bbox_to_anchor=(0.5, 0.962), fontsize=9.8)
    fig.tight_layout(rect=(0.028, 0.032, 0.995, 0.935))
    return fig


def draw_gi_discovery_traces(gi_bundle: dict, dose: int = 3):
    """Render 3x2 visual discovery-effectiveness trajectories across arrayed and multiplexed GI screens."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    traces = [t for t in gi_bundle["traces"] if t["task"] == 1 and t["dose"] == dose]
    methods = [m for m in RANKED_ORDER if m != "oracle" and any(t["method"] == m for t in traces)]
    kinds = ["gi_arrayed", "gi_multiplexed"]
    row_specs = [
        (("sl_pair",), "True Interacting Pair (FDPS–PPP2R1A) Lock-On ↑", "Active pair selection (%)"),
        (("passenger_omics",), "Non-Causal Main-Effect Confounder Distraction ↓", "Main-effect profiling (%)"),
        (("pair_test", "block"), "Inactive Pair & Pooled-Block Screening Overhead ↓", "Inactive pair / block tests (%)"),
    ]
    xticks = [0.512, 1.0, 2.0, 5.0, 10.0, 25.0, 65.536]
    xticklabels = ["0.5k", "1k", "2k", "5k", "10k", "25k", "65k"]

    fig, axes = plt.subplots(3, 2, figsize=(13.6, 10.8), sharex=False, sharey=True)
    for r_idx, (keys, row_title, ylab) in enumerate(row_specs):
        for c_idx, kind in enumerate(kinds):
            ax = axes[r_idx, c_idx]
            for m in methods:
                cells = sorted([t for t in traces if t["kind"] == kind and t["method"] == m], key=lambda x: x["seed"])
                if not cells:
                    continue
                series = [_compact_window_rate(c, keys) for c in cells]
                xs = series[0][0]
                vals = np.array([s[1] for s in series], dtype=float)
                mean = vals.mean(axis=0)
                sd = vals.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
                col = RANKED_COLORS[m]
                is_ranked = m.startswith("ceac_ranked")
                ls = "--" if m in {"ceac", "ceac_natural", "ids_product"} else "-"
                lw = 2.4 if is_ranked else (1.9 if m == "ids_action" else 1.45)
                zord = 7 if is_ranked else (5 if m in {"ids_action", "ceac"} else 3)
                ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=col, alpha=0.11, linewidth=0)
                ax.plot(xs, mean, label=RANKED_LABELS[m], color=col, lw=lw, linestyle=ls, zorder=zord)
            ax.set_xscale("log")
            ax.set_xlim(0.50, 67.0)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.minorticks_off()
            ax.set_ylim(-3.0, 105.0)
            short_kind = "Arrayed K562 Screen" if kind == "gi_arrayed" else "Multiplexed Pooled-Block Screen"
            ax.set_title(f"{short_kind} · {row_title}", fontsize=10.3, pad=7)
            ax.set_xlabel("Training episodes (log scale)", fontsize=9.5, labelpad=4)
            if c_idx == 0:
                ax.set_ylabel(ylab, fontsize=9.8)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Horlbeck 2018 K562 Pair Screen · Discovery Process Effectiveness Over Time (k = {dose}, M = {46 * (2 ** dose):,})",
        fontsize=13.0,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.115, 1.0, 0.958))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.098), ncol=3, frameon=False, fontsize=8.8)
    return fig


def export_gi_comparison(study_dir: Path, cal_dir: Path | None, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    out_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads((study_dir / "results.json").read_text())
    cal_results = json.loads((cal_dir / "results.json").read_text()) if cal_dir and (cal_dir / "results.json").exists() else []

    groups = defaultdict(list)
    for r in results:
        groups[(r["kind"], r["task"], r["method"], r["dose"])].append(r)

    summary_rows = []
    for (kind, task, method, dose), grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda x: x["seed"])
        gaps = np.array([x["late_gap"] for x in grp], dtype=float)
        regs = np.array([x["cumulative_regret"] for x in grp], dtype=float)
        banks = [x["bank_policies"] for x in grp if x.get("bank_policies") is not None]
        eps = 65536
        summary_rows.append(
            dict(
                kind=kind,
                task=task,
                method=method,
                dose=dose,
                seeds=len(grp),
                late_gap_mean=float(gaps.mean()),
                late_gap_sd=float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                regret_mean=float(regs.mean()),
                regret_sd=float(regs.std(ddof=1)) if len(regs) > 1 else 0.0,
                bank_policies_mean=float(np.mean(banks)) if banks else None,
                profiling_pct=float(100.0 * np.mean([x["events"]["passenger_omics"] / eps for x in grp])),
                wrong_pair_pct=float(100.0 * np.mean([x["events"]["pair_test"] / eps for x in grp])),
                block_pct=float(100.0 * np.mean([x["events"]["block"] / eps for x in grp])),
            )
        )

    atomic_json(out_dir / "endpoints.json", summary_rows)
    with (out_dir / "endpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    kinds = sorted({r["kind"] for r in summary_rows})
    doses = sorted({r["dose"] for r in summary_rows})
    methods = [m for m in RANKED_ORDER if any(r["method"] == m for r in summary_rows)]

    fig = draw_gi_ranked_response(summary_rows)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"gi_ranked_dose_response.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig_m = draw_gi_ranked_mechanisms(summary_rows)
    for ext in ("png", "pdf"):
        fig_m.savefig(out_dir / f"gi_ranked_mechanisms.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig_m)

    traces = _load_compact_traces(results)
    bundle_for_plots = dict(summary_rows=summary_rows, traces=traces)

    fig_disc = draw_gi_discovery_traces(bundle_for_plots, dose=max(doses))
    for ext in ("png", "pdf"):
        fig_disc.savefig(out_dir / f"gi_discovery_traces.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig_disc)

    for kind in kinds:
        for metric in ("window_gap", "cumulative_regret"):
            fig_l = draw_gi_ranked_learning(bundle_for_plots, kind, metric=metric)
            for ext in ("png", "pdf"):
                fig_l.savefig(out_dir / f"{kind}_task1_{metric}.{ext}", dpi=180, bbox_inches="tight")
            plt.close(fig_l)

    # Paired seed-by-seed regret saved by ceac_ranked at k = max(doses)
    by_key = {(r["kind"], r["dose"], r["method"], r["seed"]): r for r in results if r["task"] == 1}
    seeds = sorted({r["seed"] for r in results})
    max_dose = max(doses)
    paired_rows = []
    for kind in kinds:
        for comp in methods:
            if comp == "ceac_ranked":
                continue
            diffs = np.array([
                by_key[(kind, max_dose, comp, s)]["cumulative_regret"]
                - by_key[(kind, max_dose, "ceac_ranked", s)]["cumulative_regret"]
                for s in seeds
            ])
            paired_rows.append(
                dict(
                    kind=kind,
                    dose=max_dose,
                    comparator=comp,
                    regret_saved_mean=float(diffs.mean()),
                    regret_saved_sd=float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                    ranked_wins_all_seeds=bool(np.all(diffs > 0)),
                )
            )
    atomic_json(out_dir / "paired_comparisons.json", paired_rows)

    # Markdown report
    lines = [
        "# Greedy Disagreement + Price-Ranked Replay CEAC on Horlbeck 2018 K562 Pair Screen",
        "",
        f"Verified **{len(results)}** held-out sweep cells (`seeds = 0, 1, 2`, `65,536` episodes/cell) across `k = 0, 1, 2, 3` non-causal main-effect confounders.",
        "",
        "> [!NOTE]",
        "> `ceac_ranked` and `ceac_ranked_natural` are explicitly labeled heuristic extensions of Section-3 CEAC that replace uniform seed-bank replay floors with (1) coordinate-free greedy maximally-disagreeing column selection and (2) dynamic evidence-price ranking & KKT debt-modulated replay reuse.",
        "",
        "## Headline Dose Response (`Task 1`, `k = 0 -> 3`)",
        "",
        "![Dose response across k](gi_ranked_dose_response.png)",
        "",
        "## Mechanism Diagnostics (Active Bank Compression & Confounder Immunity)",
        "",
        "![Active bank size and main-effect profiling](gi_ranked_mechanisms.png)",
        "",
        "## Visual Discovery-Process Time Traces (`k = 3`, `M = 368`)",
        "",
        "![K562 pair discovery and distraction trajectories](gi_discovery_traces.png)",
        "",
        "## Per-Algorithm Performance Time Traces (`k = 0 .. 3`)",
        "",
        "![Arrayed screen expected policy gap](gi_arrayed_task1_window_gap.png)",
        "",
        "![Arrayed screen cumulative regret](gi_arrayed_task1_cumulative_regret.png)",
        "",
        "![Multiplexed screen expected policy gap](gi_multiplexed_task1_window_gap.png)",
        "",
        "![Multiplexed screen cumulative regret](gi_multiplexed_task1_cumulative_regret.png)",
        "",
    ]
    if cal_results:
        lines += [
            "## Easy-Case Calibration Comparison (`dose = 0`, `seeds 1001, 1002`, Tasks 0 & 1)",
            "",
            "| Screen Variant | Task | Algorithm | Active Bank Columns | Final Gap (mean) | Cumulative Regret (mean) |",
            "|---|---:|---|---:|---:|---:|",
        ]
        c_groups = defaultdict(list)
        for r in cal_results:
            c_groups[(r["kind"], r["task"], r["method"])].append(r)
        for (kind, task, m), grp in sorted(c_groups.items()):
            b_str = f"{np.mean([x['bank_policies'] for x in grp]):.1f}" if grp[0].get("bank_policies") is not None else "—"
            lines.append(
                f"| {KINDS[kind]} | {task} | {RANKED_LABELS.get(m, m)} | {b_str} "
                f"| {np.mean([x['late_gap'] for x in grp]):.4f} | {np.mean([x['cumulative_regret'] for x in grp]):,.1f} |"
            )
        lines.append("")

    lines += [
        "## Held-Out Sweep Summary (`Task 1`, `k = 0 -> 3`, 65,536 Episodes)",
        "",
        "| Screen Variant | Algorithm | Active Bank (k=3) | Final Gap (k=0 → k=3) | Regret (k=0 → k=3) | Main-Effect Profiling (k=3) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for kind in kinds:
        for m in methods:
            grp = sorted([r for r in summary_rows if r["kind"] == kind and r["task"] == 1 and r["method"] == m], key=lambda x: x["dose"])
            if not grp:
                continue
            lo, hi = grp[0], grp[-1]
            b_str = f"{hi['bank_policies_mean']:.1f}" if hi["bank_policies_mean"] is not None else "—"
            lines.append(
                f"| {KINDS[kind]} | {RANKED_LABELS[m]} | {b_str} "
                f"| {lo['late_gap_mean']:.4f} → {hi['late_gap_mean']:.4f} "
                f"| {lo['regret_mean']:,.1f} → {hi['regret_mean']:,.1f} "
                f"| {hi['profiling_pct']:.2f}% |"
            )
    lines += [
        "",
        f"## Paired Regret Saved by `ceac_ranked` at Maximum Confounding (`k = {max_dose}`, Task 1)",
        "",
        "| Screen Variant | Comparator | Regret Saved vs `ceac_ranked` (mean ± SD) | `ceac_ranked` Lower on All Seeds |",
        "|---|---|---:|---|",
    ]
    for p in paired_rows:
        lines.append(
            f"| {KINDS[p['kind']]} | {RANKED_LABELS[p['comparator']]} "
            f"| {p['regret_saved_mean']:+,.1f} ± {p['regret_saved_sd']:,.1f} | {p['ranked_wins_all_seeds']} |"
        )
    lines.append("")
    (out_dir / "assessment.md").write_text("\n".join(lines))

    # Save portable evidence bundle + SHA-256 manifest so notebooks never depend on /runs.
    arch_fig = None
    from ceac_pathologies.report import draw_architecture
    arch_fig = draw_architecture("gi_screen")
    for ext in ("png", "pdf"):
        arch_fig.savefig(out_dir / f"architecture.{ext}", dpi=180, bbox_inches="tight")
    plt.close(arch_fig)

    portable = dict(
        schema=1,
        protocol="gi_screen",
        episodes=65536,
        summary_rows=summary_rows,
        paired_rows=paired_rows,
        calibration_rows=cal_results,
        sweep_rows=results,
        traces=traces,
        sources=ranked_source_identity(),
    )
    with gzip.open(out_dir / "evidence.json.gz", "wt") as f:
        json.dump(portable, f, allow_nan=False, separators=(",", ":"))
    atomic_json(
        out_dir / "manifest.json",
        dict(
            schema=1,
            files={p.name: digest(p) for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json"},
            sources=ranked_source_identity(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20261120_ranked_v1")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=65536)
    parser.add_argument("--export-only", action="store_true", help="Re-export figures and portable bundle from completed study")
    args = parser.parse_args()

    from ceac_gi.environment import build_config

    cal_dir = ROOT / "runs" / f"gi_ranked_cal_{args.tag}"
    sweep_dir = ROOT / "runs" / f"gi_ranked_sweep_{args.tag}"
    out_dir = ROOT / "notebooks" / "evidence" / "pathologies_gi_ranked"

    if args.export_only:
        export_gi_comparison(sweep_dir, cal_dir, out_dir)
        print(f"Exported GI ranked study to {out_dir}", flush=True)
        return

    cal_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    cal_methods = ["ceac", "ceac_natural", "ceac_ranked", "ceac_ranked_natural", "ids_action", "ids_empirical"]
    cal_jobs = []
    for kind in ("gi_arrayed", "gi_multiplexed"):
        env_dict = json.loads(json.dumps(asdict(build_config(kind, 0))))
        for task in (0, 1):
            for m in cal_methods:
                for seed in (1001, 1002):
                    folder = cal_dir / "cells" / kind / "dose_0" / f"task_{task}" / f"{m}_s{seed}"
                    cal_jobs.append(("gi_screen", env_dict, m, seed, task, 0, args.episodes, str(folder)))

    cal_results = []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        for fut in as_completed([pool.submit(_exec_job, j) for j in cal_jobs]):
            row = fut.result()
            cal_results.append(row)
            print("CAL:", json.dumps({k: row[k] for k in ("kind", "task", "method", "seed", "late_gap", "cumulative_regret", "bank_policies")}), flush=True)
    cal_results.sort(key=lambda r: (r["kind"], r["task"], r["method"], r["seed"]))
    atomic_json(cal_dir / "results.json", cal_results)

    sweep_methods = list(RANKED_ORDER)
    sweep_jobs = []
    for kind in ("gi_arrayed", "gi_multiplexed"):
        for dose in (0, 1, 2, 3):
            env_dict = json.loads(json.dumps(asdict(build_config(kind, dose))))
            for m in sweep_methods:
                for seed in (0, 1, 2):
                    folder = sweep_dir / "cells" / kind / f"dose_{dose}" / "task_1" / f"{m}_s{seed}"
                    sweep_jobs.append(("gi_screen", env_dict, m, seed, 1, dose, args.episodes, str(folder)))

    sweep_results = []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        for fut in as_completed([pool.submit(_exec_job, j) for j in sweep_jobs]):
            row = fut.result()
            sweep_results.append(row)
            print("SWEEP:", json.dumps({k: row[k] for k in ("kind", "dose", "method", "seed", "late_gap", "cumulative_regret", "bank_policies")}), flush=True)
    sweep_results.sort(key=lambda r: (r["kind"], r["dose"], r["method"], r["seed"]))
    atomic_json(sweep_dir / "results.json", sweep_results)

    export_gi_comparison(sweep_dir, cal_dir, out_dir)
    print(f"Exported GI ranked study to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
