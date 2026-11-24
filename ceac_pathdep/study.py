"""Low-memory runner, portable exporter, and primary-palette overlaid plots for ceac_pathdep."""
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
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

from ceac.contracts import Episode, Transition
from ceac.primitives import IDAC, IDACConfig, OptimisticPOMD, POMDConfig, PrimitiveSpec, UCBQ, UCBQConfig
from ceac_hardness.agents import CEACConfig, LaiGravesOracle
from ceac_hardness.ids import BayesianIDS
from ceac_pathologies.agent import ConfounderEliminatingAC, PathologiesCEACConfig
from ceac_pathologies.report import log_markevery, style_ceac_legend
from ceac_pathologies.run import (
    ROOT,
    _sanitize_diagnostics,
    atomic_json,
    digest,
    encoded,
    seed_for,
    source_identity as base_source_identity,
)
from ceac_ranked.agent import RankedCEACConfig
from .environments import (
    CanalizedGIConfig,
    CanalizedGIEnv,
    CanalizedGIModel,
    SignallingRewiringConfig,
    SignallingRewiringEnv,
    SignallingRewiringModel,
    build_canalized_gi_config,
    relabel_canalized_gi,
    relabel_signalling,
)
from .learners import FastPathDepBayesianIDS, MultiStepActionIDS, PathDepGreedyCEAC, _base_policy_occ, make_truth_return_vec


PATHDEP_KINDS = {
    "signalling_rewiring": "Path-Dependent Signalling Rewiring (EGFR/MEK Priming, H = 3)",
    "canalized_gi": "Canalized Genetic Risk Screen (K562 PPP2R1A Hub, H = 3)",
}

PATHDEP_LABELS = {
    "idac": "IDAC · occupancy ratio",
    "ucb_q": "UCB Q-learning",
    "pomd": "Optimistic POMD",
    "ids_null": "IDS · null / no-rewiring prior",
    "ids_product": "IDS · full-model MI I(θ,ξ; Z)",
    "ids_action_tree": "IDS · optimal-policy MI I(Π*; Z) (Russo–Van Roy)",
    "ids_action_onpath": "IDS · on-path quotient MI I(C*_onpath; Z)",
    "ceac": "CEAC · reward + evidence critics",
    "ceac_natural": "CEAC · guarded natural",
    "ceac_ranked": "CEAC · greedy disagreement + price-ranked replay",
    "ceac_ranked_natural": "CEAC · greedy + price-ranked (guarded natural)",
    "oracle": "Oracle Graves–Lai",
}

PATHDEP_COLORS = {
    "idac": "#15803D",                # Emerald / Forest Green (IDAC)
    "pomd": "#FF1493",                # Deep Hot Pink (Optimistic POMD)
    "ucb_q": "#C026D3",               # Vivid Fuchsia-Magenta (UCB-Q)
    "ids_null": "#7F1D1D",            # Dark Wine Red (restricted/null IDS)
    "ids_product": "#DC2626",         # Vivid Scarlet Red (full-model MI IDS)
    "ids_action_tree": "#FB7185",     # Coral / Rose Red (optimal-policy tree MI IDS)
    "ids_action_onpath": "#B91C1C",   # Ruby / Firebrick Red (quotiented on-path MI IDS)
    "ceac": "#0047AB",                # Canonical CEAC = Cobalt Blue
    "ceac_natural": "#1D4ED8",        # Canonical CEAC (guarded natural) = Royal Cobalt Blue
    "ceac_ranked": "#6C7CFA",         # Improved / Ranked CEAC = Periwinkle
    "ceac_ranked_natural": "#96A2FF", # Improved / Ranked CEAC (natural) = Lavender Periwinkle
    "oracle": "#374151",              # Charcoal Grey
}

PATHDEP_MARKERS = {
    "idac": "D",
    "ucb_q": ">",
    "pomd": "h",
    "ids_null": "<",
    "ids_product": "^",
    "ids_action_tree": "v",
    "ids_action_onpath": "s",
    "ceac": "*",
    "ceac_natural": "X",
    "ceac_ranked": "o",
    "ceac_ranked_natural": "P",
    "oracle": "x",
}

PATHDEP_ORDER = [
    "idac",
    "ucb_q",
    "pomd",
    "ids_null",
    "ids_product",
    "ids_action_tree",
    "ids_action_onpath",
    "ceac",
    "ceac_natural",
    "ceac_ranked",
    "ceac_ranked_natural",
    "oracle",
]


def pathdep_source_identity() -> dict[str, str]:
    out = dict(base_source_identity())
    for pkg in ("ceac_ranked", "ceac_pathdep"):
        for p in sorted((ROOT / pkg).glob("*.py")):
            if p.name not in {"study.py", "critic_audit.py"}:
                out[str(p.relative_to(ROOT))] = digest(p)
    return out


def build_pathdep_problem(kind: str, dose: int, task: int, seed: int):
    if kind == "signalling_rewiring":
        cfg = relabel_signalling(SignallingRewiringConfig(complexity=int(dose)), int(seed))
        model = SignallingRewiringModel(cfg)
        pas_rng = np.random.default_rng(seed_for("pathdep_passenger", kind, seed, dose))
        passenger = int(pas_rng.integers(cfg.modifier_states)) if cfg.complexity > 0 else 0
        truth = model.true_index(task, passenger=passenger)
        pw, mod = model.hypotheses[truth]
        env = SignallingRewiringEnv(cfg, pathway=pw, modifier=mod)
        return cfg, model, env, truth, passenger
    if kind == "canalized_gi":
        cfg = relabel_canalized_gi(build_canalized_gi_config(int(dose)), int(seed))
        model = CanalizedGIModel(cfg)
        pas_rng = np.random.default_rng(seed_for("pathdep_passenger", kind, seed, dose))
        passenger = int(pas_rng.integers(cfg.passenger_states)) if cfg.complexity > 0 else 0
        truth = model.true_index(task, passenger=passenger)
        q, mod = model.hypotheses[truth]
        env = CanalizedGIEnv(cfg, active_modifier=q, passenger=mod)
        return cfg, model, env, truth, passenger
    raise ValueError(f"Unknown pathdep kind: {kind}")


class PathDepCertifiedCEAC(ConfounderEliminatingAC):
    """ConfounderEliminatingAC using `model.canonical_path_policies()` on branching DAGs."""

    def __init__(self, model, rng, config: PathologiesCEACConfig | None = None):
        from ceac_hardness.design import PolicyBank, bounded_design
        from ceac_hardness.learning import DebtLedger
        from ceac_pathologies.agent import (
            GeneralPolicyReplay,
            NATURAL_CANDIDATE,
            NATURAL_SCHEDULE,
            SparseStabilityActor,
            SparseTwoCriticActor,
        )

        self.model = model
        self.rng = rng
        self.config = config or PathologiesCEACConfig()
        cfg = self.config
        if cfg.optimizer == "guarded_natural":
            self.actor = SparseStabilityActor(model, NATURAL_CANDIDATE, NATURAL_SCHEDULE)
        else:
            self.actor = SparseTwoCriticActor(model, cfg.actor_rate, cfg.critic_rate, cfg.temperature)
        self.replay = GeneralPolicyReplay(model, cfg.delta / 2.0)
        self.ledger = DebtLedger()
        self.bank = PolicyBank(model, model.seed_policies())
        self.seed_ids = tuple(p.identity for p in self.bank.policies)
        for p in self.bank.policies:
            self.replay.register(p)
            self.ledger.require_seed(p.identity)

        coeff_cache = {}
        seed_rets, seed_info = self.bank.arrays()
        coeffs = []
        alt_by_ref = defaultdict(list)
        for m, j in model.alternatives:
            alt_by_ref[m].append(j)
        for m in range(model.num_models):
            rows_m = alt_by_ref[m]
            sig = (
                round(float(model.optimal_returns[m]), 12),
                tuple(np.round(seed_rets[m], 12)),
                tuple(sorted(tuple(np.round(seed_info[m, j], 12)) for j in rows_m)),
            )
            if sig not in coeff_cache:
                cbank = PolicyBank(model, model.seed_policies())
                coeff_cache[sig] = bounded_design(model, cbank, (m,), cfg.mass_bound).cost
            coeffs.append(coeff_cache[sig])
        self.coefficients = np.array(coeffs, dtype=float)

        self.full_policies = list(model.canonical_path_policies())
        self.full_values = [model.evaluate(p) for p in self.full_policies]

        self._alt_set = set(model.alternatives)
        self.episodes = 0
        self.design = None
        self.prices = np.zeros((model.num_models, model.num_models))
        self.last_refs = ()
        self.certified = False
        self.refreshes = 0
        self.admissions = 0
        self.exact_admissions = 0
        self.events = []
        self.snapshots = []
        self.candidates = {}
        self.frozen = {}
        self.last_replay_scores = []
        self.last_reason = "initialization"
        self.refresh()


def construct_pathdep_learner(kind: str, model, method: str, rng, episodes: int, truth: int):
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
        t_name = "canonical" if kind == "signalling_rewiring" else "additive"
        return FastPathDepBayesianIDS(model, rng, model.prior(t_name))
    if method == "ids_product":
        return FastPathDepBayesianIDS(model, rng, model.prior("reference"))
    if method == "ids_action_tree":
        return MultiStepActionIDS(model, rng, model.prior("reference"), target="policy_tree")
    if method == "ids_action_onpath":
        return MultiStepActionIDS(model, rng, model.prior("reference"), target="onpath_class")

    elapsed = np.zeros(model.n, dtype=int)
    for t, states in enumerate(model.levels):
        elapsed[states] = t
    spec = PrimitiveSpec(model.mask, elapsed, model.horizon, model.initial, -0.02, 1.02)
    if method == "idac":
        return IDAC(spec, rng, IDACConfig())
    if method == "ucb_q":
        return UCBQ(spec, rng, episodes, UCBQConfig())
    if method == "pomd":
        return OptimisticPOMD(spec, episodes, POMDConfig())
    raise ValueError(f"Unknown method: {method}")


def run_pathdep_cell(
    kind: str,
    method: str,
    seed: int,
    task: int,
    dose: int,
    episodes: int,
    output: Path | str,
    log_every: int = 256,
):
    cfg, model, env, truth, passenger = build_pathdep_problem(kind, dose, task, seed)
    settings = dict(
        protocol="pathdep",
        kind=kind,
        environment=json.loads(json.dumps(asdict(cfg))),
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
    )
    manifest = dict(schema=1, settings=settings, sources=pathdep_source_identity())
    manifest["identity"] = hashlib.sha256(encoded(manifest).encode()).hexdigest()
    output = Path(output)
    if output.exists():
        import shutil
        status_file = output / "status.json"
        if status_file.exists() and json.loads(status_file.read_text()).get("status") == "complete":
            old = json.loads((output / "manifest.json").read_text())
            if old != json.loads(encoded(manifest)):
                raise ValueError(f"Existing cell identity mismatch: {output}")
            status = json.loads(status_file.read_text())
            for name, checksum in status["artifacts"].items():
                if digest(output / name) != checksum:
                    raise ValueError(f"Cell artifact mismatch: {name}")
            return json.loads((output / "summary.json").read_text())
        shutil.rmtree(output)

    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "status.json", dict(status="running"))

    rng = np.random.default_rng(seed_for("pathdep", seed, kind, dose, task, method, "learner"))
    action_rng = np.random.default_rng(seed_for("pathdep", seed, kind, dose, task, method, "action"))
    started = time.monotonic()
    learner = construct_pathdep_learner(kind, model, method, rng, episodes, truth)
    setup_seconds = time.monotonic() - started
    env.reset(seed=seed_for("pathdep", seed, kind, dose, task, "shared_environment"))

    cumulative = 0.0
    window = deque(maxlen=256)
    events = dict(
        safe=0, control=0, prime=0, decanalize=0, unprimed=0, buffered_ko=0,
        synergy=0, sl_pair=0, wrong_cotarget=0, pair_test=0, probe=0,
        echo_reporter=0, passenger_omics=0, pleiotropic=0,
    )
    actor_updates = 0
    priced_updates = 0
    last_policy = None
    expected = None
    truth_r_vec = make_truth_return_vec(model, truth)

    with gzip.open(output / "trace.jsonl.gz", "wt") as target:
        for t in range(episodes):
            policy, behavior, prices = learner.select(t)
            if policy.identity != last_policy:
                expected = float(truth_r_vec @ _base_policy_occ(model, policy))
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

            # Evict raw Episode lists every 128 episodes to keep worker RSS < 45 MB.
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
        seconds=time.monotonic() - started,
        setup_seconds=setup_seconds,
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


def _exec_pathdep_job(job):
    kind, method, seed, task, dose, budget, folder = job
    res = run_pathdep_cell(kind, method, seed, task, dose, budget, folder, log_every=256)
    return dict(
        folder=str(Path(folder).resolve().relative_to(ROOT)),
        kind=kind,
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


def load_pathdep_export(folder: Path | str) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected_sha in manifest["files"].items():
        if digest(folder / name) != expected_sha:
            raise ValueError(f"Portable pathdep checksum mismatch: {folder / name}")
    with gzip.open(folder / "evidence.json.gz", "rt") as f:
        return json.load(f)


def _load_compact_pathdep_traces(results: list[dict]) -> list[dict]:
    traces = []
    for r in sorted(results, key=lambda x: (x["kind"], x["task"], x["dose"], x["method"], x["seed"])):
        cell_dir = Path(r["folder"])
        if not cell_dir.is_absolute():
            cell_dir = ROOT / cell_dir
        with gzip.open(cell_dir / "trace.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        is_sig = r["kind"] == "signalling_rewiring"
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
                active_hit=[int(t["events"]["synergy" if is_sig else "sl_pair"]) for t in rows],
                echo_reporter=[int(t["events"]["echo_reporter"]) for t in rows],
                passenger_omics=[int(t["events"]["passenger_omics"]) for t in rows],
                pleiotropic=[int(t["events"]["pleiotropic"]) for t in rows],
                wrong_test=[int(t["events"]["wrong_cotarget" if is_sig else "pair_test"]) for t in rows],
            )
        )
    return traces


def _window_rate(tr: dict, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
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


def draw_pathdep_architecture():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.2))
    specs = [
        (
            "Signalling Pathway Rewiring (H = 3, EGFR/MEK Priming)",
            "basal\ncell",
            "Step 1 Priming (-0.015, 0 KL) → Step 2 Rewired Bypass Co-Target (0.92 vs 0.10)\nUnprimed / reversed-order co-target fails (0.25, intact feedback)",
            "6k Echoed Peripheral mRNA Reporters (0.49999; I(C*_onpath; Z) > 0\non peripheral clades, ZERO KL between Null & primary responder)",
            "k Lineage Assays + (3^k - 1) Off-Path Stress States (local rescue ξ_b\nsplits optimal policy tree Π* by 2^k, zero on-path return change)",
        ),
        (
            "Canalized Genetic Risk Screen (H = 3, K562 PPP2R1A Hub)",
            "intact\nhub",
            "Step 1 PPP2R1A De-Canalization (-0.015, 0 KL) → Step 2 Modifier Hit\nfrom Horlbeck 2018 K562 GI (FDPS = -5.24 pays ≈0.96 vs neutral modifiers)",
            "6k Echoed Peripheral eQTL Reporters (0.49999; traps quotiented\non-path IDS via positive scalar MI on peripheral K562 modifiers)",
            "Intact-Hub Knockdowns (buffered 0.4999; k reveal ξ) + (3^k - 1)\nOff-Path Complexes (traps Russo–Van Roy policy-tree IDS & model-free RL)",
        ),
    ]
    for ax, (title, root_lbl, b1, b2, b3) in zip(axes, specs):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.add_patch(Circle((0.085, 0.52), 0.048, color="#0038A8"))
        ax.text(0.085, 0.37, root_lbl, ha="center", va="top", fontsize=9.0)
        for y, fc, ec, text in [
            (0.70, "#e8f0fe", "#0038A8", b1),
            (0.39, "#fef2f2", "#D62828", b2),
            (0.08, "#fff7ed", "#F77F00", b3),
        ]:
            ax.add_patch(FancyBboxPatch((0.23, y), 0.74, 0.22, boxstyle="round,pad=0.018", fc=fc, ec=ec, lw=1.5))
            ax.annotate("", xy=(0.22, y + 0.11), xytext=(0.14, 0.52), arrowprops=dict(arrowstyle="->", color=ec, lw=1.8))
            ax.text(0.60, y + 0.11, text, ha="center", va="center", fontsize=7.9, color="#1a1a1a")
    fig.tight_layout()
    return fig


def draw_single_pathdep_scaling(summary_rows: list[dict], kind: str):
    """Single-environment 1x2 scaling across k = 0 -> 3: Final Policy Gap (left) & Linear Cumulative Regret (right)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    sub = [r for r in summary_rows if r["kind"] == kind and r["task"] == 1]
    doses = sorted({r["dose"] for r in sub})
    methods = [m for m in PATHDEP_ORDER if any(r["method"] == m for r in sub)]
    short_title = "Signalling Rewiring (H = 3)" if kind == "signalling_rewiring" else "Canalized K562 Genetic Risk (H = 3)"

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)
    for c_i, (field, sd_f, ylab, title_sfx) in enumerate([
        ("late_gap_mean", "late_gap_sd", "Final expected policy loss ↓", "Final Policy Gap vs. k ↓"),
        ("regret_mean", "regret_sd", "Cumulative expected regret (linear) ↓", "Cumulative Regret vs. k (Linear Scale) ↓"),
    ]):
        ax = axes[0, c_i]
        for m in methods:
            grp = sorted([r for r in sub if r["method"] == m], key=lambda x: x["dose"])
            if not grp:
                continue
            xs = np.array([x["dose"] for x in grp], dtype=float)
            ys = np.array([x[field] for x in grp], dtype=float)
            sds = np.array([x[sd_f] for x in grp], dtype=float)
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
            ax.fill_between(xs, np.maximum(0.0, ys - sds), ys + sds, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(xs, ys, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls, marker=PATHDEP_MARKERS[m], markersize=ms, zorder=zord, **mk_kw)
        ax.set_xticks(doses)
        if c_i == 0:
            ax.set_ylim(-0.025, 0.80)
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_title(f"{short_title} · {title_sfx}", fontsize=10.6, pad=8)
        ax.set_xlabel("Path-dependent confounding modules k", fontsize=9.6)
        ax.set_ylabel(ylab, fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{short_title} · Path-Dependent Confounding Breaks Optimal-Action IDS Across k = 0 → 3", fontsize=12.3, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_single_pathdep_traces(bundle: dict, kind: str, dose: int = 3):
    """Single-environment 1x2 overlaid time traces at k=3: Policy Gap (left) & Linear Cumulative Regret (right)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    traces = [t for t in bundle["traces"] if t["kind"] == kind and t["task"] == 1 and t["dose"] == dose]
    methods = [m for m in PATHDEP_ORDER if any(t["method"] == m for t in traces)]
    short_kind = "Signalling Rewiring (H = 3)" if kind == "signalling_rewiring" else "Canalized K562 Genetic Risk (H = 3)"

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)
    for c_idx, (metric, ylab, title_sfx, log_x) in enumerate([
        ("window_gap", "Rolling 256-episode policy gap ↓", f"Overlaid Policy Gap (k = {dose}) ↓", True),
        ("cumulative_regret", "Cumulative expected regret (linear) ↓", f"Overlaid Linear Regret (k = {dose}) ↓", False),
    ]):
        ax = axes[0, c_idx]
        for m_idx, m in enumerate(methods):
            cells = sorted([t for t in traces if t["method"] == m], key=lambda x: x["seed"])
            if not cells:
                continue
            start = 1 if len(cells[0]["episode"]) > 1 and cells[0]["episode"][0] == 1 else 0
            xs = np.array(cells[0]["episode"][start:], dtype=float) / 1000.0
            vals = np.array([c[metric][start:] for c in cells], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
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
                mk_idx = log_markevery(len(xs), m_idx)
            else:
                step_mk = max(1, len(xs) // 8)
                mk_idx = list(range((m_idx * 2) % step_mk, len(xs), step_mk)) or [0]
            mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if PATHDEP_MARKERS[m] != "x" else {}
            ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(xs, mean, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls, marker=PATHDEP_MARKERS[m], markersize=ms, markevery=mk_idx, zorder=zord, **mk_kw)
        if log_x:
            ax.set_xscale("log")
            ax.set_xlim(0.24, 34.0)
            ax.set_xticks([0.256, 0.5, 1.0, 2.0, 5.0, 12.0, 32.768])
            ax.set_xticklabels(["0.25k", "0.5k", "1k", "2k", "5k", "12k", "33k"])
            ax.minorticks_off()
            ax.set_ylim(-0.025, 0.80)
            ax.set_xlabel("Training episodes (log scale)", fontsize=9.6)
        else:
            ax.set_xlim(0.0, 33.0)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("Training episodes (thousands)", fontsize=9.6)
        ax.set_title(f"{short_kind} · {title_sfx}", fontsize=10.6, pad=8)
        ax.set_ylabel(ylab, fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{short_kind} (k = {dose}, Task 1) · Overlaid Policy Gap & Linear Regret Time Traces", fontsize=12.3, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_pathdep_dose_response(summary_rows: list[dict]):
    """Render clean 1x2 linear cumulative-regret scaling across path-dependent confounding k = 0 -> 3."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    kinds = ["signalling_rewiring", "canalized_gi"]
    doses = sorted({r["dose"] for r in summary_rows})
    methods = [m for m in PATHDEP_ORDER if any(r["method"] == m for r in summary_rows)]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), squeeze=False)
    for c_i, kind in enumerate(kinds):
        xlab = (
            "Path-dependent confounding modules k (6k echo clades, 3^k − 1 stress states)"
            if kind == "signalling_rewiring"
            else "Path-dependent confounding modules k (6k echo eQTLs, 3^k − 1 off-path states)"
        )
        ax = axes[0, c_i]
        for m in methods:
            grp = sorted([r for r in summary_rows if r["kind"] == kind and r["task"] == 1 and r["method"] == m], key=lambda x: x["dose"])
            if not grp:
                continue
            xs = np.array([x["dose"] for x in grp], dtype=float)
            ys = np.array([x["regret_mean"] for x in grp], dtype=float)
            sds = np.array([x["regret_sd"] for x in grp], dtype=float)
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
            ax.fill_between(xs, np.maximum(0.0, ys - sds), ys + sds, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(
                xs, ys, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls,
                marker=PATHDEP_MARKERS[m], markersize=ms, zorder=zord,
                **mk_kw,
            )
        short_title = "Signalling Rewiring (H=3)" if kind == "signalling_rewiring" else "Canalized K562 Genetic Risk (H=3)"
        ax.set_title(f"{short_title} · Cumulative Regret vs. k (Linear)", fontsize=10.8, pad=8)
        ax.set_xlabel(xlab, fontsize=9.4, labelpad=5)
        if c_i == 0:
            ax.set_ylabel("Cumulative expected regret (linear) ↓", fontsize=9.8)
        ax.set_xticks(doses)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Path-Dependent Signalling & Canalized K562 Genetic Risk · Scaling Across Confounding k = 0 → 3", fontsize=12.4, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_pathdep_overlaid_performance(bundle: dict, dose: int = 3, metric: str = "cumulative_regret"):
    """Render clean 1x2 overlaid performance time traces (Signalling Rewiring vs Canalized K562 GI) at k=3."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    traces = [t for t in bundle["traces"] if t["task"] == 1 and t["dose"] == dose]
    methods = [m for m in PATHDEP_ORDER if any(t["method"] == m for t in traces)]
    kinds = ["signalling_rewiring", "canalized_gi"]
    xticks = [0.256, 0.5, 1.0, 2.0, 5.0, 12.0, 32.768]
    xticklabels = ["0.25k", "0.5k", "1k", "2k", "5k", "12k", "33k"]

    is_reg = metric == "cumulative_regret"
    ylab = "Cumulative expected regret (linear) ↓" if is_reg else "Rolling 256-episode policy gap ↓"
    metric_lbl = "Linear Cumulative Regret" if is_reg else "Expected Policy Gap"

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), sharex=False, squeeze=False)
    for c_idx, kind in enumerate(kinds):
        short_kind = "Signalling Rewiring (H=3)" if kind == "signalling_rewiring" else "Canalized K562 Genetic Risk (H=3)"
        ax = axes[0, c_idx]
        for m_idx, m in enumerate(methods):
            cells = sorted([t for t in traces if t["kind"] == kind and t["method"] == m], key=lambda x: x["seed"])
            if not cells:
                continue
            start = 1 if len(cells[0]["episode"]) > 1 and cells[0]["episode"][0] == 1 else 0
            xs = np.array(cells[0]["episode"][start:], dtype=float) / 1000.0
            vals = np.array([c[metric][start:] for c in cells], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
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
                mk_idx = list(range((m_idx * 2) % step_mk, len(xs), step_mk)) or [0]
            else:
                mk_idx = log_markevery(len(xs), m_idx)
            mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if PATHDEP_MARKERS[m] != "x" else {}
            ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=col, alpha=0.13 if (is_ranked or is_opt_ids) else 0.07, linewidth=0)
            ax.plot(
                xs, mean, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls,
                marker=PATHDEP_MARKERS[m], markersize=ms, markevery=mk_idx,
                zorder=zord, **mk_kw,
            )
        if is_reg:
            ax.set_xlim(0.0, 33.0)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("Training episodes (thousands)", fontsize=9.6, labelpad=4)
        else:
            ax.set_xscale("log")
            ax.set_xlim(0.24, 34.0)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.minorticks_off()
            ax.set_ylim(-0.03, 0.80)
            ax.set_xlabel("Training episodes (log scale)", fontsize=9.6, labelpad=4)
        ax.set_title(f"{short_kind} · {metric_lbl}", fontsize=10.6, pad=8)
        if c_idx == 0:
            ax.set_ylabel(ylab, fontsize=9.8)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Path-Dependent Signalling & Genetic Risk · Overlaid {metric_lbl} Time Traces (k = {dose}, Task 1)", fontsize=12.4, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def draw_pathdep_discovery_traces(bundle: dict, dose: int = 3):
    """Render clean 1x2 overlaid optimal-path lock-on time traces (Signalling Rewiring vs Canalized K562 GI) at k=3."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    traces = [t for t in bundle["traces"] if t["task"] == 1 and t["dose"] == dose]
    methods = [m for m in PATHDEP_ORDER if m != "oracle" and any(t["method"] == m for t in traces)]
    kinds = ["signalling_rewiring", "canalized_gi"]
    xticks = [0.256, 0.5, 1.0, 2.0, 5.0, 12.0, 32.768]
    xticklabels = ["0.25k", "0.5k", "1k", "2k", "5k", "12k", "33k"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), sharex=False, sharey=True, squeeze=False)
    for c_idx, kind in enumerate(kinds):
        ax = axes[0, c_idx]
        for m_idx, m in enumerate(methods):
            cells = sorted([t for t in traces if t["kind"] == kind and t["method"] == m], key=lambda x: x["seed"])
            if not cells:
                continue
            series = [_window_rate(c, ("active_hit",)) for c in cells]
            xs = series[0][0]
            vals = np.array([s[1] for s in series], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
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
            ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=col, alpha=0.13 if is_ceac else 0.08, linewidth=0)
            ax.plot(
                xs, mean, label=PATHDEP_LABELS[m], color=col, lw=lw, linestyle=ls,
                marker=PATHDEP_MARKERS[m], markersize=ms, markevery=log_markevery(len(xs), m_idx),
                markeredgecolor="white", markeredgewidth=0.55, zorder=zord,
            )
        ax.set_xscale("log")
        ax.set_xlim(0.24, 34.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.minorticks_off()
        ax.set_ylim(-3.0, 105.0)
        short_kind = "Signalling Rewiring · Primed Co-Target Lock-On ↑" if kind == "signalling_rewiring" else "Canalized K562 GI · Hub-Decanalized Pair Lock-On ↑"
        ax.set_title(short_kind, fontsize=10.4, pad=8)
        ax.set_xlabel("Training episodes (log scale)", fontsize=9.5, labelpad=4)
        if c_idx == 0:
            ax.set_ylabel("Optimal path selection (%)", fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Path-Dependent Signalling & K562 Genetic Risk · Overlaid Discovery Time Traces (k = {dose}, Task 1)", fontsize=12.4, y=0.985)
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, frameon=False, fontsize=8.8)
    style_ceac_legend(leg)
    return fig


def export_pathdep_study(sweep_dir: Path, cal_dir: Path | None, out_dir: Path, episodes: int = 32768):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads((sweep_dir / "results.json").read_text())
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
        is_sig = kind == "signalling_rewiring"
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
                active_hit_pct=float(100.0 * np.mean([x["events"]["synergy" if is_sig else "sl_pair"] / episodes for x in grp])),
                echo_pct=float(100.0 * np.mean([x["events"]["echo_reporter"] / episodes for x in grp])),
                offpath_omics_pct=float(100.0 * np.mean([x["events"]["passenger_omics"] / episodes for x in grp])),
                pleiotropic_pct=float(100.0 * np.mean([x["events"]["pleiotropic"] / episodes for x in grp])),
            )
        )

    atomic_json(out_dir / "endpoints.json", summary_rows)
    with (out_dir / "endpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    traces = _load_compact_pathdep_traces(results)
    bundle = dict(
        schema=1,
        protocol="pathdep",
        episodes=episodes,
        summary_rows=summary_rows,
        calibration_rows=cal_results,
        sweep_rows=results,
        traces=traces,
        sources=pathdep_source_identity(),
    )

    max_dose = max(r["dose"] for r in summary_rows)
    by_key = {(r["kind"], r["dose"], r["method"], r["seed"]): r for r in results if r["task"] == 1}
    seeds = sorted({r["seed"] for r in results})
    methods = [m for m in PATHDEP_ORDER if any(r["method"] == m for r in summary_rows)]
    paired_rows = []
    for kind in ("signalling_rewiring", "canalized_gi"):
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
    bundle["paired_rows"] = paired_rows
    atomic_json(out_dir / "paired_comparisons.json", paired_rows)

    for fig_obj, stem in [
        (draw_pathdep_architecture(), "architecture"),
        (draw_pathdep_dose_response(summary_rows), "pathdep_dose_response"),
        (draw_pathdep_overlaid_performance(bundle, dose=max_dose), "pathdep_overlaid_performance"),
        (draw_pathdep_discovery_traces(bundle, dose=max_dose), "pathdep_discovery_traces"),
    ]:
        for ext in ("png", "pdf"):
            fig_obj.savefig(out_dir / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig_obj)

    lines = [
        "# Path-Dependent Signalling Rewiring & Canalized K562 Genetic Risk (`ceac_pathdep`)",
        "",
        f"Verified **{len(results)}** held-out sweep cells (`seeds = 0, 1, 2`, `{episodes:,}` episodes / `{3 * episodes:,}` physical steps per cell) "
        f"plus **{len(cal_results)}** easy-case calibration cells (`k = 0`, `seeds 1001, 1002`, `Tasks 0 & 1`).",
        "",
        "## Why CEAC Separates from Both Optimal-Policy MI `I(Π*; Z)` and On-Path Quotient MI `I(C*_onpath; Z)`",
        "",
        "- **`ids_action_tree` (`I(Π*; Z)`, Russo–Van Roy MDP optimal-policy MI)**: Off-path stress/non-hub states carry local rescue rules governed by latent modifiers `ξ ∈ {0, 1}^k`, splitting `Π*(θ, ξ)` by `2^k` and trapping policy-tree IDS into profiling non-causal modifier assays (`0%` on-path return impact).",
        "- **`ids_action_onpath` (`I(C*_onpath; Z)`, quotiented on-path class MI)**: Even after manually gifting IDS CEAC's on-path return quotient, scalar mutual information rewards cheap peripheral clade/eQTL reporters (`0.49998` return) that reduce tail entropy on peripheral candidates while providing **zero KL** between `Null` and the realized responder.",
        "- **`ceac_ranked` & `ceac`**: Track vector-valued pairwise log-likelihood ratios across undefeated on-path alternatives and propagate dual evidence prices `β` across the `-0.015` Step-1 priming / `PPP2R1A` hub-decanalization gate via `Q_I(s, a; β)`.",
        "",
        "## Architecture",
        "",
        "![Path-dependent signalling and canalized GI architecture](architecture.png)",
        "",
        "## Dose Response Across Path-Dependent Complexity (`k = 0 -> 3`)",
        "",
        "![Dose response across k](pathdep_dose_response.png)",
        "",
        "## Overlaid Performance & Discovery Time Traces (`k = 3`)",
        "",
        "![Overlaid performance time traces](pathdep_overlaid_performance.png)",
        "",
        "![Overlaid discovery and IDS-trap trajectories](pathdep_discovery_traces.png)",
        "",
        "## Held-Out Sweep Summary (`Task 1`, `k = 0 -> 3`)",
        "",
        "| Benchmark | Algorithm | Bank (k=3) | Final Gap (k=0 → k=3) | Regret (k=0 → k=3) | Echo% (k=3) | OffPath-Mod% (k=3) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for kind in ("signalling_rewiring", "canalized_gi"):
        for m in methods:
            grp = sorted([r for r in summary_rows if r["kind"] == kind and r["task"] == 1 and r["method"] == m], key=lambda x: x["dose"])
            if not grp:
                continue
            lo, hi = grp[0], grp[-1]
            b_str = f"{hi['bank_policies_mean']:.1f}" if hi["bank_policies_mean"] is not None else "—"
            lines.append(
                f"| {PATHDEP_KINDS[kind]} | {PATHDEP_LABELS[m]} | {b_str} "
                f"| {lo['late_gap_mean']:.4f} → {hi['late_gap_mean']:.4f} "
                f"| {lo['regret_mean']:,.1f} → {hi['regret_mean']:,.1f} "
                f"| {hi['echo_pct']:.2f}% | {hi['offpath_omics_pct']:.2f}% |"
            )
    lines += [
        "",
        f"## Paired Regret Saved by `ceac_ranked` at `k = {max_dose}` (`Task 1`)",
        "",
        "| Benchmark | Comparator | Regret Saved vs `ceac_ranked` (mean ± SD) | `ceac_ranked` Lower on All Seeds |",
        "|---|---|---:|---|",
    ]
    for p in paired_rows:
        lines.append(
            f"| {PATHDEP_KINDS[p['kind']]} | {PATHDEP_LABELS[p['comparator']]} "
            f"| {p['regret_saved_mean']:+,.1f} ± {p['regret_saved_sd']:,.1f} | {p['ranked_wins_all_seeds']} |"
        )
    lines.append("")
    (out_dir / "assessment.md").write_text("\n".join(lines))

    with gzip.open(out_dir / "evidence.json.gz", "wt") as f:
        json.dump(bundle, f, allow_nan=False, separators=(",", ":"))
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
    parser.add_argument("--tag", default="20261122_pathdep_v1")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=32768)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    cal_dir = ROOT / "runs" / f"pathdep_cal_{args.tag}"
    sweep_dir = ROOT / "runs" / f"pathdep_sweep_{args.tag}"
    out_dir = ROOT / "notebooks" / "evidence" / "pathologies_pathdep"

    if args.export_only:
        export_pathdep_study(sweep_dir, cal_dir, out_dir, episodes=args.episodes)
        print(f"Exported pathdep study to {out_dir}", flush=True)
        return

    cal_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    cal_methods = ["ceac", "ceac_natural", "ceac_ranked", "ceac_ranked_natural", "ids_product", "ids_action_tree", "ids_action_onpath"]
    cal_jobs = []
    for kind in ("signalling_rewiring", "canalized_gi"):
        for task in (0, 1):
            for m in cal_methods:
                for seed in (1001, 1002):
                    folder = cal_dir / "cells" / kind / "dose_0" / f"task_{task}" / f"{m}_s{seed}"
                    cal_jobs.append((kind, m, seed, task, 0, args.episodes, str(folder)))

    ctx = mp.get_context("spawn")
    cal_results = []
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_exec_pathdep_job, cal_jobs, chunksize=1):
            cal_results.append(row)
            print("CAL:", json.dumps({k: row[k] for k in ("kind", "task", "method", "seed", "late_gap", "cumulative_regret", "bank_policies")}), flush=True)
    cal_results.sort(key=lambda r: (r["kind"], r["task"], r["method"], r["seed"]))
    atomic_json(cal_dir / "results.json", cal_results)

    sweep_jobs = []
    for kind in ("signalling_rewiring", "canalized_gi"):
        for dose in (0, 1, 2, 3):
            for m in PATHDEP_ORDER:
                for seed in (0, 1, 2):
                    folder = sweep_dir / "cells" / kind / f"dose_{dose}" / "task_1" / f"{m}_s{seed}"
                    sweep_jobs.append((kind, m, seed, 1, dose, args.episodes, str(folder)))

    sweep_results = []
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_exec_pathdep_job, sweep_jobs, chunksize=1):
            sweep_results.append(row)
            print("SWEEP:", json.dumps({k: row[k] for k in ("kind", "dose", "method", "seed", "late_gap", "cumulative_regret", "bank_policies")}), flush=True)
    sweep_results.sort(key=lambda r: (r["kind"], r["dose"], r["method"], r["seed"]))
    atomic_json(sweep_dir / "results.json", sweep_results)

    export_pathdep_study(sweep_dir, cal_dir, out_dir, episodes=args.episodes)
    print(f"Exported pathdep study to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
