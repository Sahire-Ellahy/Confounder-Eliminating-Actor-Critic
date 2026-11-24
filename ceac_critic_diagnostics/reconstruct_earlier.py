"""Exact saved-trajectory Held-Out Reward & Evidence Critic replay for FrozenLake & DeepSea suites.

Replays immutable saved trajectories (`trajectories.jsonl` / `episodes.jsonl.gz`) with zero
new training and zero LP solving across all three earlier pathological suites:
1. Grid Worlds (`runs/pathology_strength_20261110`): Undertow DeepSea 8x8 (`deepsea`) &
   False Beacon FrozenLake (`frozenlake`), rho in {0.05, 0.25, 0.50, 0.75}, true_mode=1, 2,048 episodes.
2. Primary Burden Junctions (`runs/ceac_primary_burden_20261111`): `undertow_junction` &
   `false_beacon_junction`, nuisance breadth B in {0, 4, 16, 64}, task=1, 65,536 episodes.
3. Primary Prior-Shift Junctions (`runs/ceac_primary_prior_20261111`): `false_beacon_junction`,
   prior shift rho in {0.0, 0.25, 0.50, 0.75, 1.00}, task=1, 16,384 episodes.
"""
from __future__ import annotations
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from ceac.breadth import make_model, register_breadth
from ceac.contracts import Episode, Policy, Transition
from ceac.ids import BayesianIDS as V90BayesianIDS
from ceac.learning import ActorCritic as V90ActorCritic
from ceac.primitives import IDAC, IDACConfig, OptimisticPOMD, POMDConfig, UCBQ, UCBQConfig
from ceac.run import primitive_specification, stable_seed
from ceac_hardness.environment import JunctionConfig
from ceac_hardness.ids import BayesianIDS as JunctionBayesianIDS
from ceac_hardness.model import JunctionModel
from ceac_hardness.run import ROOT, atomic_json, digest, primitive_spec as junction_primitive_spec, seed_for as junction_seed_for
from ceac_pseudocode.learning import TwoCriticActor
from .evaluator import (
    SUITE_LABELS,
    compile_outcome_tables,
    draw_suite_critic_dose_scaling,
    draw_suite_critic_traces,
    evaluate_heldout_critics_fast,
    save_suite_critic_audit,
)

register_breadth()

GRID_BENCHMARKS = {
    "pathology_rho005": 0.05,
    "pathology_rho025": 0.25,
    "pathology_rho050": 0.50,
    "pathology_rho075": 0.75,
}
GRID_CHECKPOINTS = (16, 32, 64, 128, 256, 512, 1024, 2048)
GRID_METHODS = ["idac", "ucb_q", "pomd", "ids_zero", "ids", "ceac"]
GRID_KINDS = [("deepsea", "Undertow DeepSea (8×8)"), ("frozenlake", "False Beacon Shore (FrozenLake)")]

BURDEN_CHECKPOINTS = (16, 64, 256, 1024, 4096, 16384, 65536)
BURDEN_METHODS = ["idac", "ucb_q", "pomd", "ids_product", "ceac"]
BURDEN_KINDS = [("undertow_junction", "Undertow Junction"), ("false_beacon_junction", "False Beacon Junction")]

PRIOR_CHECKPOINTS = (16, 64, 256, 1024, 4096, 16384)
PRIOR_METHODS = ["ids_correlated", "ids_product", "ceac"]
PRIOR_KINDS = [("false_beacon_junction", "False Beacon Junction (Prior Shift)")]


def _certified_value_vector(model, policy: Policy, weights: np.ndarray) -> np.ndarray:
    """Exact backward policy evaluation of `policy` under model-mixture `weights`."""
    S = model.n
    stage_r = np.einsum("m,ms,msa->sa", weights, model.density, model.reward)
    p_next = np.zeros((S, model.actions, S), dtype=float)
    for s in range(S):
        for a in np.flatnonzero(model.mask[s]):
            laws = model.laws[int(s), int(a)]
            for m, w in enumerate(weights):
                if w <= 0:
                    continue
                for (nxt, _, done), prob in laws[m].items():
                    if not done:
                        p_next[s, a, model.index[int(nxt)]] += w * prob
    vr = np.zeros(S, dtype=float)
    for lvl in reversed(model.levels):
        if len(lvl):
            qr_lvl = stage_r[lvl] + p_next[lvl] @ vr
            vr[lvl] = np.sum(policy.probabilities[lvl] * qr_lvl, axis=1)
    return vr


def _replay_grid_cell(args) -> dict:
    study_root_s, bench, rho, kind, seed, method = args
    study_root = Path(study_root_s)
    folder = study_root / bench / kind / "context0_mode1" / f"{method}_seed{seed}"
    res = json.loads((folder / "result.json").read_text())
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    trajs = [json.loads(line) for line in (folder / "trajectories.jsonl").read_text().splitlines()]

    model = make_model(kind, 0, bench)
    compiled = compile_outcome_tables(model, truth=1)
    spec = primitive_specification(model)
    rng = np.random.default_rng(stable_seed(seed, kind, method, 0, "learner"))

    pol_cache: dict[str, Policy] = {}

    def get_pol(pid: str) -> Policy:
        if pid not in pol_cache:
            probs = np.load(folder / "policies" / f"{pid}.npz")["probabilities"]
            pol_cache[pid] = Policy(probs, pid)
        return pol_cache[pid]

    if method == "ceac":
        init_p = model.response(np.array([0.5, 0.5]), np.zeros((2, 2)), "initial-mixture")[0]
        actor = V90ActorCritic(model, 1.0, 0.08, 0.3, init_p)
        learner = SimpleNamespace(actor=actor)
    elif method.startswith("ids"):
        learner = V90BayesianIDS(model, rng, 0.0 if method == "ids_zero" else 0.5)
    elif method == "idac":
        learner = IDAC(spec, rng, IDACConfig(policy_lr=0.1, value_lr=0.2, batch_episodes=16, tau=0.01, kappa=0.1))
    elif method == "ucb_q":
        learner = UCBQ(spec, rng, 2048, UCBQConfig(bonus_constant=1.0, delta=0.05))
    elif method == "pomd":
        learner = OptimisticPOMD(spec, 2048, POMDConfig(delta=0.05, policy_step=None))
    else:
        raise ValueError(method)

    cp_set = set(GRID_CHECKPOINTS)
    cp_rows = []
    for idx, (tr, mr) in enumerate(zip(trajs, metrics)):
        pol = get_pol(tr["policy_id"])
        prices = np.asarray(mr["prices"], dtype=float)
        ep = Episode(
            tr["index"],
            tr["context"],
            tr["policy_id"],
            tr["behavior"],
            tuple(
                Transition(x["state"], x["action"], x["reward"], x["next_state"], x["done"], tuple(x["log_likelihood"]))
                for x in tr["transitions"]
            ),
        )
        if method == "ceac":
            learner.actor.update(ep, pol, prices)
        elif method in {"idac", "ucb_q", "pomd"}:
            p_sel, _, _ = learner.select(idx)
            if p_sel.identity != tr["policy_id"]:
                raise ValueError(f"Grid policy parity mismatch: {folder} ep {idx}")
            learner.observe(ep, pol, prices)
        else:
            learner.observe(ep, pol, prices)

        ep_num = idx + 1
        if ep_num in cp_set:
            cert_vr = None
            if method == "ceac" and tr["behavior"] == "certified_deploy":
                w = np.asarray(mr["diagnostics"]["deployment"]["weights"], dtype=float)
                cert_vr = _certified_value_vector(model, pol, w)
            diag = evaluate_heldout_critics_fast(
                suite_key="grid_earlier",
                kind=kind,
                method=method,
                learner=learner,
                model=model,
                compiled=compiled,
                truth=1,
                policy=pol,
                prices=prices,
                checkpoint=ep_num,
                seed=seed,
                dose=rho,
                certified_vr=cert_vr,
            )
            diag["episode"] = ep_num
            cp_rows.append(diag)

    if method == "ceac":
        if learner.actor.snapshot().identity != res["final_actor_policy_id"]:
            raise ValueError(f"Final actor policy mismatch: {folder}")
    elif method.startswith("ids"):
        if not np.allclose(learner.posterior, res["diagnostics"]["posterior"], atol=1e-9):
            raise ValueError(f"Final IDS posterior mismatch: {folder}")

    return dict(
        suite="grid",
        benchmark=bench,
        kind=kind,
        dose=float(rho),
        task=1,
        seed=int(seed),
        method=method,
        pseudo_regret=float(res["pseudo_regret"]),
        checkpoints=cp_rows,
    )


def _replay_junction_cell(args) -> dict:
    suite_name, folder_s, dose_val, checkpoints = args
    folder = ROOT / folder_s
    summary = json.loads((folder / "summary.json").read_text())
    settings = summary["settings"]
    method = settings["method"]
    seed = int(settings["seed"])
    task = int(settings["task_evaluator_only"])
    episodes = int(settings["episodes"])

    cfg = JunctionConfig(**settings["environment"])
    q = (0.9 - 0.8 * settings["dose"] if task == 1 else 0.1 + 0.8 * settings["dose"]) if cfg.protocol == "prior_shift" else 0.5
    model = JunctionModel(cfg)
    truth = model.true_index(task, q)
    compiled = compile_outcome_tables(model, truth=truth)

    rng = np.random.default_rng(junction_seed_for(seed, cfg.kind, task, method, "learner"))
    prim_cfgs = settings.get("primitive_configs") or {}
    designs = []
    if method == "ceac":
        cc = settings["ceac_config"]
        actor = TwoCriticActor(model, cc["actor_rate"], cc["critic_rate"], cc["temperature"])
        learner = SimpleNamespace(actor=actor)
        designs = [e for e in json.loads((folder / "design_history.json").read_text()) if "design" in e]
    elif method.startswith("ids_"):
        learner = JunctionBayesianIDS(model, rng, model.prior(method.removeprefix("ids_")))
    elif method == "idac":
        learner = IDAC(junction_primitive_spec(model), rng, IDACConfig(**prim_cfgs.get("idac", {})))
    elif method == "ucb_q":
        learner = UCBQ(junction_primitive_spec(model), rng, episodes, UCBQConfig(**prim_cfgs.get("ucb_q", {})))
    elif method == "pomd":
        learner = OptimisticPOMD(junction_primitive_spec(model), episodes, POMDConfig(**prim_cfgs.get("pomd", {})))
    else:
        raise ValueError(method)

    cp_set = set(checkpoints)
    cp_rows = []
    cursor = -1
    zeros_m = np.zeros((model.num_models, model.num_models), dtype=float)

    with gzip.open(folder / "episodes.jsonl.gz", "rt") as handle:
        for idx, line in enumerate(handle):
            raw = json.loads(line)
            if raw["episode"] != idx:
                raise ValueError(f"Noncontiguous episode in {folder}")
            if method == "ceac":
                while cursor + 1 < len(designs) and designs[cursor + 1]["episode"] <= idx:
                    cursor += 1
                prices = np.asarray(designs[cursor]["design"]["beta"], dtype=float) if cursor >= 0 else zeros_m
                if raw["behavior"] == "actor":
                    pol = learner.actor.snapshot()
                    if pol.identity != raw["policy_id"]:
                        raise ValueError(f"Junction actor mismatch: {folder} ep {idx}")
                    ep = Episode(
                        idx,
                        0,
                        pol.identity,
                        "actor",
                        tuple(
                            Transition(x["state"], x["action"], x["reward"], x["next_state"], x["done"], tuple(x["log_likelihood"]))
                            for x in raw["transitions"]
                        ),
                    )
                    learner.actor.update(ep, pol, prices)
            else:
                pol, _, prices = learner.select(idx)
                if pol.identity != raw["policy_id"]:
                    raise ValueError(f"Junction comparator policy mismatch: {folder} ep {idx}")
                ep = Episode(
                    idx,
                    0,
                    pol.identity,
                    raw["behavior"],
                    tuple(
                        Transition(x["state"], x["action"], x["reward"], x["next_state"], x["done"], tuple(x["log_likelihood"]))
                        for x in raw["transitions"]
                    ),
                )
                learner.observe(ep, pol, prices)

            ep_num = idx + 1
            if ep_num in cp_set:
                eval_pol = learner.actor.snapshot() if method == "ceac" else pol
                diag = evaluate_heldout_critics_fast(
                    suite_key=suite_name,
                    kind=cfg.kind,
                    method=method,
                    learner=learner,
                    model=model,
                    compiled=compiled,
                    truth=truth,
                    policy=eval_pol,
                    prices=prices,
                    checkpoint=ep_num,
                    seed=seed,
                    dose=dose_val,
                )
                diag["episode"] = ep_num
                cp_rows.append(diag)

    if method == "ceac":
        if learner.actor.updates != summary["actor_updates"] or learner.actor.priced_updates != summary["priced_actor_updates"]:
            raise ValueError(f"Junction actor update count mismatch: {folder}")

    return dict(
        suite=suite_name,
        kind=cfg.kind,
        dose=float(dose_val),
        task=task,
        seed=seed,
        method=method,
        cumulative_regret=float(summary["cumulative_regret"]),
        checkpoints=cp_rows,
    )


def _format_table_section(title: str, cells: list[dict], kinds: list[tuple[str, str]], methods: list[str], dose_label: str) -> list[str]:
    doses = sorted({c["dose"] for c in cells})
    lo_d, hi_d = doses[0], doses[-1]
    ep_hi = cells[0]["checkpoints"][-1]["episode"]
    lines = [
        f"## {title}",
        "",
        f"| Environment | Algorithm | Reward Bellman RMSE ({dose_label}={hi_d:g}, t={ep_hi:,}) | Evidence Critic RMSE ({dose_label}={lo_d:g} → {hi_d:g}) | Bounded Stage-Critic RMSE ({dose_label}={lo_d:g} → {hi_d:g}) |",
        "|---|---|---:|---:|---:|",
    ]
    for kind_key, kind_title in kinds:
        for m in methods:
            g_lo = [c for c in cells if c["kind"] == kind_key and c["method"] == m and abs(c["dose"] - lo_d) < 1e-9]
            g_hi = [c for c in cells if c["kind"] == kind_key and c["method"] == m and abs(c["dose"] - hi_d) < 1e-9]
            if not g_lo or not g_hi:
                continue
            r_hi = float(np.mean([x["checkpoints"][-1]["reward_bellman_error"] for x in g_hi]))
            e_lo = float(np.mean([x["checkpoints"][-1]["evidence_critic_error"] for x in g_lo]))
            e_hi = float(np.mean([x["checkpoints"][-1]["evidence_critic_error"] for x in g_hi]))
            s_lo = float(np.mean([x["checkpoints"][-1]["stage_epistemic_error"] for x in g_lo]))
            s_hi = float(np.mean([x["checkpoints"][-1]["stage_epistemic_error"] for x in g_hi]))
            lines.append(
                f"| {kind_title} | {SUITE_LABELS[m]} | {r_hi:.4f} | {e_lo:,.4f} → {e_hi:,.4f} | {s_lo:.4f} → {s_hi:.4f} |"
            )
    lines.append("")
    return lines


def run_earlier_reconstruction(output_dir: Path | str, workers: int = 8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Grid worlds (Undertow DeepSea & False Beacon FrozenLake, true_mode=1, 144 cells)
    grid_root = ROOT / "runs/pathology_strength_20261110"
    grid_tasks = [
        (str(grid_root), bench, rho, kind, seed, method)
        for bench, rho in GRID_BENCHMARKS.items()
        for kind, _ in GRID_KINDS
        for seed in (0, 1, 2)
        for method in GRID_METHODS
    ]

    # 2. Primary Burden Junctions (task=1, 60 cells)
    burden_entries = json.loads((ROOT / "runs/ceac_primary_burden_20261111/results.json").read_text())
    burden_tasks = [
        ("junction_burden", e["folder"], float(e["burden"]), BURDEN_CHECKPOINTS)
        for e in burden_entries
        if e["task"] == 1 and e["method"] in BURDEN_METHODS
    ]

    # 3. Primary Prior-Shift Junctions (task=1, 45 cells)
    prior_entries = json.loads((ROOT / "runs/ceac_primary_prior_20261111/results.json").read_text())
    prior_tasks = [
        ("junction_prior", e["folder"], float(e["dose"]), PRIOR_CHECKPOINTS)
        for e in prior_entries
        if e["task"] == 1 and e["method"] in PRIOR_METHODS
    ]

    grid_cells, burden_cells, prior_cells = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs_g = [pool.submit(_replay_grid_cell, t) for t in grid_tasks]
        futs_b = [pool.submit(_replay_junction_cell, t) for t in burden_tasks]
        futs_p = [pool.submit(_replay_junction_cell, t) for t in prior_tasks]
        for i, fut in enumerate(as_completed(futs_g), 1):
            grid_cells.append(fut.result())
            if i % 36 == 0 or i == len(grid_tasks):
                print(f"[earlier-critic-audit] Grid {i}/{len(grid_tasks)} complete", flush=True)
        for i, fut in enumerate(as_completed(futs_b), 1):
            burden_cells.append(fut.result())
            if i % 20 == 0 or i == len(burden_tasks):
                print(f"[earlier-critic-audit] Burden Junction {i}/{len(burden_tasks)} complete", flush=True)
        for i, fut in enumerate(as_completed(futs_p), 1):
            prior_cells.append(fut.result())
            if i % 15 == 0 or i == len(prior_tasks):
                print(f"[earlier-critic-audit] Prior Junction {i}/{len(prior_tasks)} complete", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_cells.sort(key=lambda c: (c["kind"], c["dose"], c["method"], c["seed"]))
    burden_cells.sort(key=lambda c: (c["kind"], c["dose"], c["method"], c["seed"]))
    prior_cells.sort(key=lambda c: (c["kind"], c["dose"], c["method"], c["seed"]))

    save_suite_critic_audit(
        output_dir,
        dict(schema=1, training_launched=False, suite="grid_pathology_strength", checkpoints=list(GRID_CHECKPOINTS), cells=grid_cells),
        "grid_critic_audit.json.gz",
    )
    save_suite_critic_audit(
        output_dir,
        dict(schema=1, training_launched=False, suite="junction_burden", checkpoints=list(BURDEN_CHECKPOINTS), cells=burden_cells),
        "junction_burden_critic_audit.json.gz",
    )
    save_suite_critic_audit(
        output_dir,
        dict(schema=1, training_launched=False, suite="junction_prior", checkpoints=list(PRIOR_CHECKPOINTS), cells=prior_cells),
        "junction_prior_critic_audit.json.gz",
    )

    figures = [
        (
            draw_suite_critic_traces(
                grid_cells,
                GRID_KINDS,
                target_dose=0.75,
                suptitle="Undertow DeepSea & False Beacon Shore (ρ = 0.75, Mode 1) · Held-Out Reward & Evidence Critic Error",
                method_order=GRID_METHODS,
            ),
            "grid_deepsea_frozenlake_critic_traces",
        ),
        (
            draw_suite_critic_dose_scaling(
                grid_cells,
                GRID_KINDS,
                xlab="Physical diversion / false-beacon strength ρ",
                suptitle="Undertow DeepSea & False Beacon Shore · Late-Horizon Critic Error Scaling Across ρ ∈ {0.05, 0.25, 0.50, 0.75}",
                method_order=GRID_METHODS,
            ),
            "grid_critic_rho_scaling",
        ),
        (
            draw_suite_critic_traces(
                burden_cells,
                BURDEN_KINDS,
                target_dose=64.0,
                suptitle="Primary Controlled Junctions (Nuisance Breadth B = 64, Task 1) · Held-Out Reward & Evidence Critic Error",
                method_order=BURDEN_METHODS,
            ),
            "junction_burden_critic_traces",
        ),
        (
            draw_suite_critic_dose_scaling(
                burden_cells,
                BURDEN_KINDS,
                xlab="Nuisance action breadth B",
                suptitle="Primary Controlled Junctions · Late-Horizon Evidence Critic Divergence vs. Bounded CEAC Convergence (B = 0 → 64)",
                method_order=BURDEN_METHODS,
            ),
            "junction_burden_critic_scaling",
        ),
        (
            draw_suite_critic_traces(
                prior_cells,
                PRIOR_KINDS,
                target_dose=1.0,
                suptitle="False Beacon Prior-Shift Junction (Shift ρ = 1.0, Task 1) · Held-Out Reward & Evidence Critic Error",
                method_order=PRIOR_METHODS,
            ),
            "junction_prior_critic_traces",
        ),
    ]
    for fig, stem in figures:
        for sfx in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{sfx}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    md_lines = [
        "# Held-Out Reward & Evidence Critic TD / Bellman Diagnostics — Undertow & False Beacon Suites",
        "",
        "Reconstructed **249** completed cells (`144` Grid DeepSea/FrozenLake cells + `60` Controlled Burden Junction cells + `45` Prior-Shift Junction cells) directly from immutable saved trajectories (`trajectories.jsonl` / `episodes.jsonl.gz`) with **zero new training** and exact policy-ID / update-count parity.",
        "",
        "![Grid Undertow & False Beacon critic traces](grid_deepsea_frozenlake_critic_traces.png)",
        "",
        "![Grid Undertow & False Beacon rho scaling](grid_critic_rho_scaling.png)",
        "",
        "![Junction Burden B=64 critic traces](junction_burden_critic_traces.png)",
        "",
        "![Junction Burden B=0..64 critic scaling](junction_burden_critic_scaling.png)",
        "",
        "![Junction Prior-Shift rho=1.0 critic traces](junction_prior_critic_traces.png)",
        "",
    ]
    md_lines += _format_table_section("1. Grid Undertow DeepSea & False Beacon Shore (Mode 1, 2,048 Episodes)", grid_cells, GRID_KINDS, GRID_METHODS, "ρ")
    md_lines += _format_table_section("2. Primary Controlled Junctions — Nuisance Breadth Sweep (Task 1, 65,536 Episodes)", burden_cells, BURDEN_KINDS, BURDEN_METHODS, "B")
    md_lines += _format_table_section("3. Primary Prior-Shift Junctions — Cue Reversal Sweep (Task 1, 16,384 Episodes)", prior_cells, PRIOR_KINDS, PRIOR_METHODS, "ρ")
    (output_dir / "assessment.md").write_text("\n".join(md_lines))

    files = {p.name: digest(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "manifest.json"}
    atomic_json(
        output_dir / "manifest.json",
        dict(
            schema=1,
            training_launched=False,
            grid_cells=len(grid_cells),
            burden_cells=len(burden_cells),
            prior_cells=len(prior_cells),
            files=files,
        ),
    )
    print(f"[earlier-critic-audit] Exported verified bundle + 5 figures to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/evidence/critic_bellman_earlier")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_earlier_reconstruction(args.output, workers=args.workers)


if __name__ == "__main__":
    main()
