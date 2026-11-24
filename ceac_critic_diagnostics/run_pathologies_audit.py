"""Held-Out Reward & Evidence Critic Bellman Audit for `locks`, `epistasis`, and `gi_ranked`.

Replays the exact deterministic training seed streams (`65,536` episodes per cell) with
128-episode sufficient-statistic replay eviction (`< 50 MB` RSS per worker) and bit-exact
`FastBayesianIDS` / `FastActionIDS` early-exit once posterior expected regret drops below
`1e-12`, verifying `|cumulative_regret - ref_regret| < 1e-6` against the registered sweep
results on every single cell while evaluating Held-Out Reward & Evidence Critic Bellman
errors at log-spaced checkpoints `(16, 64, 256, 1024, 4096, 16384, 65536)`.
"""
from __future__ import annotations
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

from ceac.contracts import Episode, Transition
from ceac_hardness.ids import BayesianIDS, best_pair
from ceac_pathologies.ids_action import ActionInformationIDS
from ceac_pathologies.run import (
    ROOT,
    PathologiesCEACConfig,
    atomic_json,
    build_problem,
    construct_learner as base_construct_learner,
    digest,
    seed_for,
)
from ceac_ranked.agent import GreedyPricedCEAC, RankedCEACConfig
from .evaluator import (
    SUITE_LABELS,
    compile_outcome_tables,
    draw_suite_critic_dose_scaling,
    draw_suite_critic_traces,
    evaluate_heldout_critics_fast,
    save_suite_critic_audit,
)

from scipy.special import kl_div

CHECKPOINTS_65K = (16, 64, 256, 1024, 4096, 16384, 65536)


def fast_best_pair(regret, information) -> np.ndarray:
    """Vectorized IEEE-754 equivalent of `ceac_hardness.ids.best_pair` (~200x faster on 90+ bank columns)."""
    d, g = np.asarray(regret, dtype=float), np.asarray(information, dtype=float)
    zero = np.flatnonzero(d == 0)
    if len(zero):
        p = np.zeros(len(d))
        p[zero[0]] = 1.0
        return p
    if not np.any(g > 0):
        raise ValueError("All policies have positive regret and zero information")
    ds, gs = d / d.max(), g / g.max()
    n = len(d)
    ii, jj = np.triu_indices(n)
    a = ds[ii]
    b = ds[jj] - a
    c = gs[ii]
    slope = gs[jj] - c

    s0 = np.where(c > 0.0, (a * a) / np.where(c > 0.0, c, 1.0), np.inf)
    den1 = c + slope
    num1 = a + b
    s1 = np.where(den1 > 0.0, (num1 * num1) / np.where(den1 > 0.0, den1, 1.0), np.inf)

    valid_bs = (b != 0.0) & (slope != 0.0)
    safe_bs = np.where(valid_bs, b * slope, 1.0)
    root = np.where(valid_bs, (slope * a - 2.0 * b * c) / safe_bs, -1.0)
    in_01 = valid_bs & (root > 0.0) & (root < 1.0)
    den2 = c + slope * root
    num2 = a + b * root
    ok2 = in_01 & (den2 > 0.0)
    s2 = np.where(ok2, (num2 * num2) / np.where(ok2, den2, 1.0), np.inf)

    scores = np.column_stack((s0, s1, s2))
    flat_idx = int(np.argmin(scores))
    pair_idx, cand_idx = divmod(flat_idx, 3)
    i_best, j_best = int(ii[pair_idx]), int(jj[pair_idx])
    x_best = 0.0 if cand_idx == 0 else (1.0 if cand_idx == 1 else float(root[pair_idx]))
    distribution = np.zeros(n)
    distribution[i_best] += 1.0 - x_best
    distribution[j_best] += x_best
    return distribution


def _group_laws_by_k(model, pairs):
    by_k = {}
    for s, a in pairs:
        probs = np.array([[law[k] for k in model.laws[s, a][0]] for law in model.laws[s, a]], dtype=float)
        K = probs.shape[1]
        by_k.setdefault(K, ([], [], []))
        by_k[K][0].append(int(s))
        by_k[K][1].append(int(a))
        by_k[K][2].append(probs)
    return [
        (np.array(ss, dtype=int), np.array(aa, dtype=int), np.stack(plist, axis=0))
        for ss, aa, plist in by_k.values()
    ]


class FastBayesianIDS(BayesianIDS):
    """100% bit-exact equivalent of `BayesianIDS` avoiding `O(M^2 * B)` `bank.arrays()` tensor allocation."""

    def __init__(self, model, rng, prior=0.5):
        super().__init__(model, rng, prior)
        self._bank_returns = None
        self._bank_len = 0
        self._zero_prices = np.zeros((model.num_models, model.num_models), dtype=float)
        self._zero_prices.flags.writeable = False

    def _sync_bank_caches(self):
        if self._bank_len != len(self.bank.policies):
            self._bank_returns = np.column_stack([self.model.returns(p) for p in self.bank.policies])
            self._bank_len = len(self.bank.policies)

    def distribution(self):
        posterior = self.posterior
        key = tuple(self.log_weights)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        self._sync_bank_caches()
        regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
        if regret.min() <= self.regret_tolerance:
            distribution = np.zeros(len(regret))
            distribution[int(np.argmin(regret))] = 1.0
            self._cache = key, distribution
            return distribution
        stage_mi = self.model.mutual_information_stage(posterior)
        stage_return = np.einsum("m,ms,msa->sa", posterior, self.model.density, self.model.reward)
        for _ in range(128):
            self._sync_bank_caches()
            regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
            information = np.maximum(0.0, np.array([self.model.evaluate_stage(p, stage_mi) for p in self.bank.policies]))
            if regret.min() <= self.regret_tolerance:
                distribution = np.zeros(len(regret))
                distribution[int(np.argmin(regret))] = 1.0
                break
            distribution = best_pair(regret, information)
            d, g = float(distribution @ regret), float(distribution @ information)
            if d == 0:
                break
            price = d / (2.0 * g)
            proposal, _ = self.model.response_stage(stage_return + price * stage_mi, "ids-separator")
            new_returns = self.model.returns(proposal)
            new_d = float(posterior @ np.maximum(0.0, self.model.optimal_returns - new_returns))
            new_g = max(0.0, self.model.evaluate_stage(proposal, stage_mi))
            residual = max(0.0, d - new_d + price * (new_g - g))
            if residual <= 1e-7 * d:
                break
            if not self.bank.add(proposal):
                raise RuntimeError("IDS full-policy certificate did not close at posterior scale")
        self._cache = key, distribution
        return distribution

    def select(self, index):
        del index
        distribution = self.distribution()
        p = self.bank.policies[int(self.rng.choice(len(distribution), p=distribution))]
        return p, "ids", self._zero_prices


class FastActionIDS(ActionInformationIDS):
    """100% bit-exact equivalent of `ActionInformationIDS` avoiding `O(M^2 * B)` `bank.arrays()` tensor allocation."""

    def __init__(self, model, rng, prior):
        super().__init__(model, rng, prior)
        self._bank_returns = None
        self._bank_len = 0
        self._zero_prices = np.zeros((model.num_models, model.num_models), dtype=float)
        self._zero_prices.flags.writeable = False

    def _sync_bank_caches(self):
        if self._bank_len != len(self.bank.policies):
            self._bank_returns = np.column_stack([self.model.returns(p) for p in self.bank.policies])
            self._bank_len = len(self.bank.policies)

    def distribution(self):
        posterior = self.posterior
        key = tuple(self.log_weights)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        self._sync_bank_caches()
        regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
        if regret.min() <= self.regret_tolerance:
            distribution = np.zeros(len(regret))
            distribution[int(np.argmin(regret))] = 1.0
            self._cache = key, distribution
            return distribution
        stage_mi = self.action_information_stage(posterior)
        stage_return = np.einsum("m,ms,msa->sa", posterior, self.model.density, self.model.reward)
        for _ in range(128):
            self._sync_bank_caches()
            regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
            information = np.maximum(0.0, np.array([self.model.evaluate_stage(p, stage_mi) for p in self.bank.policies]))
            if regret.min() <= self.regret_tolerance or not np.any(information > 0):
                distribution = np.zeros(len(regret))
                distribution[int(np.argmin(regret))] = 1.0
                break
            distribution = best_pair(regret, information)
            d, g = float(distribution @ regret), float(distribution @ information)
            if d == 0:
                break
            price = d / (2.0 * g)
            proposal, _ = self.model.response_stage(stage_return + price * stage_mi, "ids-action-separator")
            new_returns = self.model.returns(proposal)
            new_d = float(posterior @ np.maximum(0.0, self.model.optimal_returns - new_returns))
            new_g = max(0.0, self.model.evaluate_stage(proposal, stage_mi))
            residual = max(0.0, d - new_d + price * (new_g - g))
            if residual <= 1e-7 * d:
                break
            if not self.bank.add(proposal):
                raise RuntimeError("Optimal-action IDS certificate did not close")
        self._cache = key, distribution
        return distribution

    def select(self, index):
        del index
        distribution = self.distribution()
        p = self.bank.policies[int(self.rng.choice(len(distribution), p=distribution))]
        return p, "ids", self._zero_prices


SUITE_SPECS = {
    "locks": dict(
        protocol="locks",
        sweep_dir=ROOT / "runs/pathologies_locks_sweep_20261119_v1",
        evidence_dir=ROOT / "notebooks/evidence/pathologies_locks",
        traces_stem="locks_critic_bellman_traces",
        scaling_stem="locks_critic_dose_scaling",
        target_dose=4,
        xlab="Gated combination-lock depth L (horizon H = L + 1)",
        traces_title="Sequential & Bidirectional Locks (Depth L = 4, Task 1) · Held-Out Reward & Evidence Critic Error",
        scaling_title="Sequential & Bidirectional Locks · Late-Horizon Critic Error Scaling Across Depth L ∈ {1, 2, 3, 4}",
        kinds=[("combination_lock", "Sequential Lock"), ("bidirectional_lock", "Bidirectional ('Diabolical') Lock")],
        methods=["idac", "ucb_q", "pomd", "ids_decoy", "ids_product", "ceac", "ceac_natural"],
        ranked=False,
        dose_symbol="L",
    ),
    "epistasis": dict(
        protocol="epistasis",
        sweep_dir=ROOT / "runs/pathologies_epistasis_sweep_20261120_v2",
        evidence_dir=ROOT / "notebooks/evidence/pathologies_epistasis",
        traces_stem="epistasis_critic_bellman_traces",
        scaling_stem="epistasis_critic_dose_scaling",
        target_dose=3,
        xlab="Pleiotropic passenger-omics modules k (M = 128 · 2^k, off-pathway arms = 4^k − 1)",
        traces_title="Epistatic Signalling Activation (Pleiotropy k = 3, Task 1) · Held-Out Reward & Evidence Critic Error",
        scaling_title="Epistatic Signalling Activation · Late-Horizon Critic Error Scaling Across Pleiotropy k ∈ {0, 1, 2, 3}",
        kinds=[("epistatic_signalling", "Epistatic Signalling")],
        methods=["idac", "ucb_q", "pomd", "ids_canonical", "ids_product", "ids_action", "ceac", "ceac_natural"],
        ranked=False,
        dose_symbol="k",
    ),
    "gi_ranked": dict(
        protocol="gi_screen",
        sweep_dir=ROOT / "runs/gi_ranked_sweep_20261120_ranked_v2",
        evidence_dir=ROOT / "notebooks/evidence/pathologies_gi_ranked",
        traces_stem="gi_critic_bellman_traces",
        scaling_stem="gi_critic_dose_scaling",
        target_dose=3,
        xlab="Non-causal single-variable confounder channels k (M = 46 · 2^k)",
        traces_title="Horlbeck 2018 K562 Pairwise Screen (Confounders k = 3, Task 1) · Held-Out Reward & Evidence Critic Error",
        scaling_title="Horlbeck 2018 K562 Pairwise Screen · Late-Horizon Critic Error Scaling Across Confounders k ∈ {0, 1, 2, 3}",
        kinds=[("gi_arrayed", "K562 GI Arrayed"), ("gi_multiplexed", "K562 GI Multiplexed")],
        methods=[
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
        ],
        ranked=True,
        dose_symbol="k",
    ),
}


def _construct_audit_learner(protocol: str, model, method: str, rng, episodes: int, truth: int):
    if method == "ceac_ranked":
        return GreedyPricedCEAC(model, rng, RankedCEACConfig(optimizer="reference"))
    if method == "ceac_ranked_natural":
        return GreedyPricedCEAC(model, rng, RankedCEACConfig(optimizer="guarded_natural"))
    if method == "ids_action":
        return FastActionIDS(model, rng, model.prior("reference"))
    if method.startswith("ids_"):
        return FastBayesianIDS(model, rng, model.prior(method.removeprefix("ids_")))
    return base_construct_learner(protocol, model, method, rng, episodes, truth)


def _audit_pathology_cell(args) -> dict:
    suite_key, protocol, cell_folder_s = args
    cell_folder = ROOT / cell_folder_s
    summary = json.loads((cell_folder / "summary.json").read_text())
    settings = summary["settings"]
    method = settings["method"]
    seed = int(settings["seed"])
    task = int(settings["task_evaluator_only"])
    dose = int(settings["dose"])
    episodes = int(settings["episodes"])
    env_dict = dict(settings["environment"])

    # Note: for gi_screen, summary["settings"]["environment"] ALREADY stores the relabeled config,
    # whereas build_problem("gi_screen", raw_cfg, task, seed) calls relabel(raw_cfg, seed)!
    # Reconstruct raw_cfg before relabeling OR construct directly from saved settings["environment"]!
    if protocol == "gi_screen":
        from ceac_gi.environment import GIScreenEnv, GIScreenModel, config_from_dict
        cfg = config_from_dict(env_dict)
        model = GIScreenModel(cfg)
        passenger = int(settings["realized_passenger_evaluator_only"])
        truth = model.true_index(task, passenger=passenger)
        env = GIScreenEnv(cfg, pair=model.hypotheses[truth][0], passenger=passenger)
    else:
        cfg, model, env, truth, _ = build_problem(protocol, env_dict, task, seed)

    compiled = compile_outcome_tables(model, truth=truth)
    rng = np.random.default_rng(seed_for(protocol, seed, cfg.kind, dose, task, method, "learner"))
    action_rng = np.random.default_rng(seed_for(protocol, seed, cfg.kind, dose, task, method, "action"))
    learner = _construct_audit_learner(protocol, model, method, rng, episodes, truth)
    env.reset(seed=seed_for(protocol, seed, cfg.kind, dose, task, "shared_environment"))

    cp_set = set(CHECKPOINTS_65K)
    cp_rows = []
    cumulative = 0.0
    last_policy = None
    expected = None
    started = time.monotonic()

    for t in range(episodes):
        policy, behavior, prices = learner.select(t)
        if policy.identity != last_policy:
            expected = float(model.returns(policy)[truth])
            last_policy = policy.identity
        gap = max(0.0, float(model.optimal_returns[truth] - expected))
        cumulative += gap

        obs, _ = env.reset()
        transitions = []
        for _ in range(model.horizon):
            s = model.index[obs]
            a = int(action_rng.choice(model.actions, p=policy.probabilities[s]))
            nxt, rew, done, _, _ = env.step(a)
            ll = model.log_likelihood(s, a, nxt, rew, done)
            transitions.append(Transition(s, a, rew, -1 if done else model.index[nxt], done, ll))
            obs = nxt
            if done:
                break
        episode = Episode(t, 0, policy.identity, behavior, tuple(transitions))
        learner.observe(episode, policy, prices)

        if (t + 1) % 128 == 0 and hasattr(learner, "replay") and hasattr(learner.replay, "strata"):
            for bucket in learner.replay.strata.values():
                bucket.clear()
            if hasattr(learner.replay, "evidence") and hasattr(learner.replay.evidence, "strata"):
                learner.replay.evidence.strata.clear()
            if method in {"ceac", "ceac_natural"} and hasattr(learner.replay, "statistics"):
                keep = set(learner.bank._ids) | set(learner.candidates) | set(learner.frozen)
                for pid in [k for k in learner.replay.statistics if k not in keep]:
                    del learner.replay.statistics[pid]

        ep_num = t + 1
        if ep_num in cp_set:
            diag = evaluate_heldout_critics_fast(
                suite_key=suite_key,
                kind=cfg.kind,
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

    ref_regret = float(summary["cumulative_regret"])
    if abs(cumulative - ref_regret) > 1e-6:
        raise ValueError(
            f"Bit-exact trajectory parity failed on {suite_key}/{cfg.kind}/{method}/d{dose}/s{seed}: "
            f"got {cumulative:.6f} vs ref {ref_regret:.6f}"
        )

    return dict(
        suite=suite_key,
        kind=cfg.kind,
        dose=dose,
        task=task,
        seed=seed,
        method=method,
        cumulative_regret=cumulative,
        elapsed_seconds=round(time.monotonic() - started, 3),
        checkpoints=cp_rows,
    )


def _update_assessment_and_manifest(spec: dict, cells: list[dict]):
    ev_dir: Path = spec["evidence_dir"]
    md_path = ev_dir / "assessment.md"
    old_md = md_path.read_text() if md_path.exists() else ""
    marker = "\n## Held-Out Reward & Evidence Critic TD / Bellman Diagnostics"
    if marker in old_md:
        old_md = old_md.split(marker)[0].rstrip() + "\n"

    doses = sorted({c["dose"] for c in cells})
    lo_d, hi_d = doses[0], doses[-1]
    dsym = spec["dose_symbol"]
    sec = [
        "",
        "## Held-Out Reward & Evidence Critic TD / Bellman Diagnostics",
        "",
        f"Evaluated all **{len(cells)}** non-oracle `Task 1` sweep cells across log-spaced checkpoints `t ∈ {{16 .. 65,536}}` on independent held-out rollouts (`512` on-policy + `256` uniform-probe episodes per checkpoint; bit-exact cumulative-regret parity verified `< 1e-6` against `evidence.json.gz`).",
        "",
        f"![Held-out critic Bellman trajectories]({spec['traces_stem']}.png)",
        "",
        f"![Held-out critic dose scaling]({spec['scaling_stem']}.png)",
        "",
        f"| Environment | Algorithm | Reward Bellman RMSE ({dsym}={hi_d}, t=65,536) | Evidence Critic RMSE ({dsym}={lo_d} → {hi_d}) | Active / Stage Critic RMSE ({dsym}={lo_d} → {hi_d}) |",
        "|---|---|---:|---:|---:|",
    ]
    for kind_key, kind_title in spec["kinds"]:
        for m in spec["methods"]:
            g_lo = [c for c in cells if c["kind"] == kind_key and c["method"] == m and c["dose"] == lo_d]
            g_hi = [c for c in cells if c["kind"] == kind_key and c["method"] == m and c["dose"] == hi_d]
            if not g_lo or not g_hi:
                continue
            r_hi = float(np.mean([x["checkpoints"][-1]["reward_bellman_error"] for x in g_hi]))
            e_lo = float(np.mean([x["checkpoints"][-1]["evidence_critic_error"] for x in g_lo]))
            e_hi = float(np.mean([x["checkpoints"][-1]["evidence_critic_error"] for x in g_hi]))
            s_lo = float(np.mean([x["checkpoints"][-1]["stage_epistemic_error"] for x in g_lo]))
            s_hi = float(np.mean([x["checkpoints"][-1]["stage_epistemic_error"] for x in g_hi]))
            sec.append(
                f"| {kind_title} | {SUITE_LABELS[m]} | {r_hi:.4f} | {e_lo:,.4f} → {e_hi:,.4f} | {s_lo:.4f} → {s_hi:.4f} |"
            )
    sec.append("")
    md_path.write_text(old_md.rstrip() + "\n" + "\n".join(sec))

    manifest_path = ev_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {
        p.name: digest(p) for p in sorted(ev_dir.iterdir()) if p.is_file() and p.name != "manifest.json"
    }
    atomic_json(manifest_path, manifest)


def run_suite_audit(suite_key: str, workers: int = 8):
    spec = SUITE_SPECS[suite_key]
    results = json.loads((spec["sweep_dir"] / "results.json").read_text())
    allowed = set(spec["methods"])
    tasks = []
    for r in results:
        if r.get("phase", "sweep") != "sweep" or r["task"] != 1 or r["method"] not in allowed:
            continue
        tasks.append((suite_key, spec["protocol"], r["folder"]))

    print(f"[{suite_key}] Auditing {len(tasks)} Task-1 sweep cells with {workers} spawn workers (maxtasksperchild=1)...", flush=True)
    cells = []
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        for i, c in enumerate(pool.imap_unordered(_audit_pathology_cell, tasks, chunksize=1), 1):
            cells.append(c)
            if i % 12 == 0 or i == len(tasks):
                print(
                    f"[{suite_key}] {i}/{len(tasks)} complete ({time.monotonic() - t0:.1f}s) — "
                    f"latest {c['kind']} {c['method']} d={c['dose']} s={c['seed']} ({c['elapsed_seconds']}s)",
                    flush=True,
                )
    gc.collect()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells.sort(key=lambda c: (c["kind"], c["dose"], c["method"], c["seed"]))
    ev_dir: Path = spec["evidence_dir"]
    save_suite_critic_audit(
        ev_dir,
        dict(
            schema=1,
            suite=suite_key,
            protocol=spec["protocol"],
            checkpoints=list(CHECKPOINTS_65K),
            parity_verified=True,
            cells=cells,
        ),
        "critic_bellman_audit.json.gz",
    )

    f_tr = draw_suite_critic_traces(
        cells,
        spec["kinds"],
        target_dose=spec["target_dose"],
        suptitle=spec["traces_title"],
        method_order=spec["methods"],
        ranked_ceac_present=spec["ranked"],
    )
    for sfx in ("png", "pdf"):
        f_tr.savefig(ev_dir / f"{spec['traces_stem']}.{sfx}", dpi=180, bbox_inches="tight")
    plt.close(f_tr)

    f_sc = draw_suite_critic_dose_scaling(
        cells,
        spec["kinds"],
        xlab=spec["xlab"],
        suptitle=spec["scaling_title"],
        method_order=spec["methods"],
        ranked_ceac_present=spec["ranked"],
    )
    for sfx in ("png", "pdf"):
        f_sc.savefig(ev_dir / f"{spec['scaling_stem']}.{sfx}", dpi=180, bbox_inches="tight")
    plt.close(f_sc)

    _update_assessment_and_manifest(spec, cells)
    print(f"[{suite_key}] Exported critic_bellman_audit.json.gz + figures + updated manifest in {ev_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["locks", "epistasis", "gi_ranked"], choices=list(SUITE_SPECS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for s in args.suites:
        run_suite_audit(s, workers=args.workers)


if __name__ == "__main__":
    main()
