"""Reproducible cell execution for sequential locks and epistatic signalling discovery."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
from pathlib import Path
import gzip
import hashlib
import json
import time
import numpy as np
from ceac.contracts import Episode, Transition
from ceac.primitives import IDAC, IDACConfig, OptimisticPOMD, POMDConfig, PrimitiveSpec, UCBQ, UCBQConfig
from ceac_hardness.agents import CEACConfig, LaiGravesOracle
from ceac_hardness.ids import BayesianIDS
from .agent import ConfounderEliminatingAC, PathologiesCEACConfig
from .environments import EpistasisConfig, EpistasisEnv, LockConfig, LockEnv
from .ids_action import ActionInformationIDS
from .model import EpistasisModel, LockModel


ROOT = Path(__file__).resolve().parents[1]


def prior_treatment(method: str) -> str:
    """ids_action uses each model's declared reference prior (data-grounded for GI)."""
    return "reference" if method == "ids_action" else method.removeprefix("ids_")


def _sanitize_diagnostics(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_diagnostics(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_diagnostics(v) for v in value]
    return value


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def atomic_json(path: Path | str, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(encoded(value))
    temporary.replace(path)


def source_identity() -> dict[str, str]:
    paths = list((ROOT / "ceac_pathologies").glob("*.py")) + list((ROOT / "ceac_gi").glob("*.py"))
    paths += [ROOT / "ceac" / name for name in ("model.py", "contracts.py", "primitives.py")]
    paths += [ROOT / "ceac_hardness" / name for name in ("design.py", "ids.py", "learning.py", "agents.py")]
    paths += [ROOT / "ceac_pseudocode" / name for name in ("learning.py", "design.py")]
    paths += [ROOT / "ceac_stability" / "actor.py"]
    excluded = {"report.py", "experiment.py"}
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.name not in excluded}


def seed_for(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def primitive_spec(model, protocol: str) -> PrimitiveSpec:
    elapsed = np.zeros(model.n, dtype=int)
    for t, states in enumerate(model.levels):
        elapsed[states] = t
    if protocol == "locks":
        # Fixed reward scale across depths L=1..4 so normalization never confounds depth.
        r_lo, r_hi = -0.02, 1.02
    else:
        r_lo, r_hi = 0.0, 1.0
    return PrimitiveSpec(model.mask, elapsed, model.horizon, model.initial, r_lo, r_hi)


def build_problem(protocol: str, config_dict: dict, task: int, seed: int):
    if protocol == "locks":
        cfg = LockConfig(**config_dict) if isinstance(config_dict, dict) else config_dict
        model = LockModel(cfg)
        truth = model.true_index(task)
        env = LockEnv(cfg, task=task)
        return cfg, model, env, truth, 0
    if protocol == "epistasis":
        cfg = EpistasisConfig(**config_dict) if isinstance(config_dict, dict) else config_dict
        model = EpistasisModel(cfg)
        # Realized passenger background drawn deterministically per (seed, pleiotropy)
        pas_rng = np.random.default_rng(seed_for("realized_passenger", seed, cfg.pleiotropy))
        passenger = int(pas_rng.integers(cfg.passenger_states)) if cfg.pleiotropy > 0 else 0
        truth = model.true_index(task, passenger=passenger)
        subtype = (cfg.subtypes - 1) if task == 1 else 0
        env = EpistasisEnv(cfg, subtype=subtype, passenger=passenger)
        return cfg, model, env, truth, passenger
    if protocol == "gi_screen":
        from ceac_gi.environment import GIScreenEnv, GIScreenModel, config_from_dict, relabel
        cfg = config_from_dict(config_dict) if isinstance(config_dict, dict) else config_dict
        cfg = relabel(cfg, seed)
        model = GIScreenModel(cfg)
        pas_rng = np.random.default_rng(seed_for("realized_passenger", seed, cfg.nuisance))
        passenger = int(pas_rng.integers(cfg.passenger_states)) if cfg.nuisance > 0 else 0
        truth = model.true_index(task, passenger=passenger)
        env = GIScreenEnv(cfg, pair=model.hypotheses[truth][0], passenger=passenger)
        return cfg, model, env, truth, passenger
    raise ValueError(f"Unknown protocol: {protocol}")


def construct_learner(protocol: str, model, method: str, rng, episodes: int, truth: int, ceac_cfg: PathologiesCEACConfig | None = None):
    cfg = ceac_cfg or PathologiesCEACConfig(mass_bound=64.0, normalization=64.0)
    if method == "ceac":
        return ConfounderEliminatingAC(model, rng, PathologiesCEACConfig(**{**asdict(cfg), "optimizer": "reference"}))
    if method == "ceac_natural":
        return ConfounderEliminatingAC(model, rng, PathologiesCEACConfig(**{**asdict(cfg), "optimizer": "guarded_natural"}))
    if method == "oracle":
        oracle_cfg = CEACConfig(mass_bound=cfg.mass_bound, normalization=cfg.normalization)
        return LaiGravesOracle(model, truth, oracle_cfg)
    if method == "ids_action":
        return ActionInformationIDS(model, rng, model.prior("reference"))
    if method.startswith("ids_"):
        treatment = method.removeprefix("ids_")
        return BayesianIDS(model, rng, model.prior(treatment))
    spec = primitive_spec(model, protocol)
    if method == "idac":
        return IDAC(spec, rng, IDACConfig())
    if method == "ucb_q":
        return UCBQ(spec, rng, episodes, UCBQConfig())
    if method == "pomd":
        return OptimisticPOMD(spec, episodes, POMDConfig())
    raise ValueError(f"Unknown learner method: {method}")


def run_cell(
    protocol: str,
    config_dict: dict,
    method: str,
    seed: int,
    task: int,
    dose: int,
    episodes: int,
    output: Path | str,
    log_every: int = 256,
    save_episodes: bool = False,
):
    cfg, model, env, truth, passenger = build_problem(protocol, config_dict, task, seed)
    ceac_cfg = PathologiesCEACConfig(mass_bound=64.0, normalization=64.0,
                                     optimizer="guarded_natural" if method == "ceac_natural" else "reference")
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
        ceac_config=asdict(ceac_cfg),
    )
    manifest = dict(schema=1, settings=settings, sources=source_identity())
    manifest["identity"] = hashlib.sha256(encoded(manifest).encode()).hexdigest()
    output = Path(output)
    if output.exists():
        old = json.loads((output / "manifest.json").read_text())
        if old != json.loads(encoded(manifest)):
            raise ValueError("Existing cell has a different identity")
        status = json.loads((output / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError("Incomplete cell: use a fresh directory")
        for name, checksum in status["artifacts"].items():
            if digest(output / name) != checksum:
                raise ValueError(f"Cell artifact mismatch: {name}")
        return json.loads((output / "summary.json").read_text())

    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "status.json", dict(status="running"))

    rng = np.random.default_rng(seed_for(protocol, seed, cfg.kind, dose, task, method, "learner"))
    action_rng = np.random.default_rng(seed_for(protocol, seed, cfg.kind, dose, task, method, "action"))
    started = time.monotonic()
    learner = construct_learner(protocol, model, method, rng, episodes, truth, ceac_cfg)
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
    prior_supported = bool(model.prior(prior_treatment(method))[truth] > 0) if method.startswith("ids_") else True

    try:
        traj_ctx = gzip.open(output / "episodes.jsonl.gz", "wt") if save_episodes else None
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
                if traj_ctx is not None:
                    traj_ctx.write(
                        json.dumps(
                            dict(
                                episode=t,
                                policy_id=policy.identity,
                                behavior=behavior,
                                transitions=[
                                    dict(state=x.state, action=x.action, reward=x.reward, next_state=x.next_state, done=x.done)
                                    for x in transitions
                                ],
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
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
                        posterior = learner.posterior
                        row.update(
                            responder_posterior=model.responder_probability(posterior),
                            prior_contains_true_model=prior_supported,
                            posterior_underflow_entries=learner.posterior_underflow_entries,
                        )
                    target.write(json.dumps(row, allow_nan=False) + "\n")
        if traj_ctx is not None:
            traj_ctx.close()

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
        )
        if method.startswith("ids_"):
            summary["responder_posterior"] = model.responder_probability(learner.posterior)
        if hasattr(learner, "admissions"):
            summary["actor_admissions"] = learner.admissions
        summary["learner_diagnostics"] = _sanitize_diagnostics(learner.diagnostics())
        if hasattr(learner, "events"):
            atomic_json(output / "design_history.json", learner.events)
        atomic_json(output / "summary.json", summary)
        artifact_names = ["manifest.json", "summary.json", "trace.jsonl.gz"]
        if (output / "design_history.json").exists():
            artifact_names.append("design_history.json")
        if save_episodes and (output / "episodes.jsonl.gz").exists():
            artifact_names.append("episodes.jsonl.gz")
        atomic_json(
            output / "status.json",
            dict(status="complete", artifacts={name: digest(output / name) for name in artifact_names}),
        )
        return summary
    except Exception as exc:
        atomic_json(output / "status.json", dict(status="failed", error=repr(exc)))
        raise
