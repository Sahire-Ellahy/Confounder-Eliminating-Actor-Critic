"""One maintained runner: python -m ceac.run --help, from IDS/code."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import tarfile
import numpy as np
from .agents import CEAC, CEACConfig, LaiGravesOracle
from .contracts import Episode, Transition
from .primitives import PrimitiveSpec, IDAC, IDACConfig, UCBQ, UCBQConfig, OptimisticPOMD, POMDConfig
from .evidence import RunWriter, atomic_json, jsonable, manifest, verify_run
from .ids import BayesianIDS
from .breadth import (make_model, make_environment as environment, specification,
                      register_breadth, BREADTH_PROFILES, audit_breadth_model, audit_breadth_family)
from .benchmarks import BENCHMARKS, study_cells
from .benchmark_audit import audit_model, audit_strength_family


METHODS = ("ceac", "ceac_actor", "ids", "ids_rare", "ids_zero", "oracle", "idac", "ucb_q", "pomd")
register_breadth()

DEFAULT_METHODS = ("ceac", "oracle", "ids", "ids_zero", "idac", "ucb_q", "pomd")


def primitive_specification(model):
    """Public state/time/action interface and reward envelope, not model access."""
    elapsed = np.empty(model.n, dtype=int)
    for t, states in enumerate(model.levels):
        elapsed[states] = t
    rewards = [0.] + [float(reward) for pair in model.laws.values()
                      for law in pair for _, reward, _ in law]
    return PrimitiveSpec(model.mask, elapsed, model.cfg.max_steps, model.initial,
                         min(rewards), max(rewards))


def stable_seed(seed, kind, method, context, channel):
    material = json.dumps([seed, kind, method, context, channel]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def collect_episode(env, model, policy, episode_index, behavior, action_rng, simulation_seed):
    # The evaluator owns env. No env, config.true_modes, or info dictionary is
    # passed to a learner. Observable transition laws are scored by public model.
    observation, _ = env.reset(seed=simulation_seed,
                               options={"episode_index": model.context*env.config.block_length})
    transitions = []
    events = {"diagnostic": 0, "nuisance": 0, "risky": 0, "diversions": 0, "lost_steps": 0, "beacon_steps": 0}
    locations, beacon_locations = set(), set()
    for _ in range(model.cfg.max_steps):
        decoded = env.decode(observation)
        locations.add((decoded.row, decoded.column if model.kind == "deepsea" else decoded.col))
        state = model.index[int(observation)]
        action = int(action_rng.choice(model.actions, p=policy.probabilities[state]))
        next_obs, reward, term, trunc, info = env.step(action)
        event = info.get("diagnostic_event", "none")
        events["diagnostic"] += int(event in {"probe", "deep"})
        events["nuisance"] += int(event in {"tv", "reeds", "shallow", "beacon_field"})
        events["risky"] += int(event in {"challenge", "outer"})
        events["diversions"] += int(info.get("nuisance_activated", False))
        events["lost_steps"] += int(info.get("nuisance_delay_step", False))
        events["beacon_steps"] += int(info.get("beacon_field_step",False))
        if info.get("beacon_field_step",False):
            beacon_locations.add(tuple(info["position"]))
        done = bool(term or trunc)
        ll = model.log_likelihood(state, action, next_obs, reward, done)
        transitions.append(Transition(state, action, float(reward), -1 if done else model.index[int(next_obs)], done, ll))
        observation = next_obs
        if done:
            break
    else:
        raise RuntimeError("Simulator exceeded declared horizon")
    return (Episode(episode_index, model.context, policy.identity, behavior, tuple(transitions)),
            {"terminal_success": bool(info["terminal_success"]),
             "terminal_observation": int(observation), "events": events, "locations": locations, "beacon_locations": beacon_locations})


def execute(model, folder, method, seed, mode, episodes, config, resume=False, primitive_configs=None):
    if method not in METHODS:
        raise ValueError("Unsupported or retired comparator: " + str(method))
    primitive_configs = primitive_configs or {}
    primitive_config = (asdict(IDACConfig(**primitive_configs.get("idac", {}))) if method == "idac" else
                        asdict(UCBQConfig(**primitive_configs.get("ucb_q", {}))) if method == "ucb_q" else
                        asdict(POMDConfig(**primitive_configs.get("pomd", {}))) if method == "pomd" else None)
    if method == "ceac_actor":
        config = replace(config, deployment_rule="actor_only")
    prior = .001 if method == "ids_rare" else 0. if method == "ids_zero" else .5
    configuration = {"benchmark": model.specification, "environment": model.kind, "context": model.context, "method": method,
                     "seed": seed, "true_mode_evaluator_only": mode, "episodes": episodes,
                     "ids_prior": prior if method.startswith("ids") else None, "ceac": asdict(config),
                     "primitive_comparator": primitive_config}
    identity = manifest(configuration)
    if Path(folder).exists():
        if not resume:
            raise FileExistsError("Refusing to overwrite existing run: " + str(folder))
        return verify_run(folder, identity)["result"]
    method_seed = "ceac" if method == "ceac_actor" else method
    rng = np.random.default_rng(stable_seed(seed, model.kind, method_seed, model.context, "learner"))
    action_rng = np.random.default_rng(stable_seed(seed, model.kind, method_seed, model.context, "actions"))
    setup_started = time.monotonic()
    if method in {"ceac", "ceac_actor"}:
        learner = CEAC(model, rng, config)
    elif method.startswith("ids"):
        learner = BayesianIDS(model, rng, prior)
    elif method == "oracle":
        learner = LaiGravesOracle(model, mode, config)
    elif method == "idac":
        learner = IDAC(primitive_specification(model), rng, IDACConfig(**primitive_config))
    elif method == "pomd":
        learner = OptimisticPOMD(primitive_specification(model), episodes, POMDConfig(**primitive_config))
    elif method == "ucb_q":
        learner = UCBQ(primitive_specification(model), rng, episodes, UCBQConfig(**primitive_config))
    else:
        raise ValueError(method)
    setup_seconds = time.monotonic()-setup_started
    env = environment(model.kind, (mode, mode, mode), benchmark=model.benchmark)
    writer = RunWriter(folder, identity, model)
    started = time.monotonic()
    cumulative_return = cumulative_regret = cumulative_information = 0.
    cumulative_steps = 0
    cumulative_events = {"diagnostic": 0, "nuisance": 0, "risky": 0, "diversions": 0, "lost_steps": 0, "beacon_steps": 0}
    visited_states, visited_locations, visited_beacon = set(), set(), set()
    rows = []
    try:
        for index in range(episodes):
            policy, behavior, prices = learner.select(index)
            writer.policy(policy)
            simulation_seed = stable_seed(seed, model.kind, "shared_simulator", model.context, str(index))
            episode, evaluation = collect_episode(env, model, policy, index, behavior, action_rng, simulation_seed)
            expected, information = model.evaluate(policy)
            regret = max(0., model.optimal_returns[mode] - expected[mode])
            learner.observe(episode, policy, prices)
            cumulative_return += episode.total_return
            cumulative_regret += regret
            cumulative_information += information[mode, 1-mode]
            cumulative_steps += len(episode.transitions)
            for event, count in evaluation["events"].items():
                cumulative_events[event] += count
            visited_states.update(step.state for step in episode.transitions)
            visited_locations.update(evaluation["locations"])
            visited_beacon.update(evaluation["beacon_locations"])
            row = {"episode": index, "benchmark": model.benchmark, "context": model.context, "method": method, "behavior": behavior,
                   "return": episode.total_return, "expected_return": float(expected[mode]),
                   "optimal_return": float(model.optimal_returns[mode]), "pseudo_regret": float(regret),
                   "cumulative_return": cumulative_return, "cumulative_pseudo_regret": cumulative_regret,
                   "terminal_success": evaluation["terminal_success"], "steps": len(episode.transitions),
                   "policy_id": policy.identity, "prices": prices.tolist(),
                   "expected_model_information": float(information[mode, 1-mode]),
                   "cumulative_expected_model_information": float(cumulative_information),
                   "cumulative_primitive_steps": cumulative_steps,
                   "diagnostic_visits": evaluation["events"]["diagnostic"],
                   "nuisance_visits": evaluation["events"]["nuisance"],
                   "risky_visits": evaluation["events"]["risky"],
                   "cumulative_diagnostic_visits": cumulative_events["diagnostic"],
                   "cumulative_nuisance_visits": cumulative_events["nuisance"],
                   "cumulative_risky_visits": cumulative_events["risky"],
                   "cumulative_diversions": cumulative_events["diversions"],
                   "cumulative_lost_steps": cumulative_events["lost_steps"],
                   "cumulative_beacon_steps": cumulative_events["beacon_steps"],
                   "unique_beacon_cells": len(visited_beacon),
                   "unique_observed_states": len(visited_states),
                   "unique_physical_locations": len(visited_locations),
                   "diagnostics": learner.diagnostics()}
            writer.episode(episode, row)
            rows.append(row)
        # Persist every policy cited by a design certificate, even if its count
        # is zero. Frozen behavior policies above also retain unadmitted actors.
        for policy in learner.bank.policies:
            writer.policy(policy)
        result = {"benchmark": model.benchmark, "environment": model.kind, "context": model.context, "true_mode": mode, "method": method,
                  "seed": seed, "episodes": episodes, "total_return": cumulative_return,
                  "pseudo_regret": cumulative_regret, "mean_return": cumulative_return/episodes,
                  "success_rate": float(np.mean([r["terminal_success"] for r in rows])),
                  "last_window_expected_return": float(np.mean([r["expected_return"] for r in rows[-min(32, episodes):]])),
                  "elapsed_seconds": time.monotonic()-started, "setup_seconds": setup_seconds,
                  "primitive_steps": cumulative_steps, "event_counts": cumulative_events,
                  "unique_beacon_cells": len(visited_beacon),
                  "unique_observed_states": len(visited_states),
                  "unique_physical_locations": len(visited_locations),
                  "diagnostics": learner.diagnostics()}
        if method in {"ceac", "ceac_actor"}:
            for saved_deployment in learner.deployment_policies.values():
                writer.policy(saved_deployment)
            final_policy = learner.actor.snapshot(); writer.policy(final_policy); writer.policy(learner.initial_actor)
            result.update(initial_actor_expected_return=float(model.evaluate(learner.initial_actor)[0][mode]),
                          final_actor_expected_return=float(model.evaluate(final_policy)[0][mode]),
                          initial_actor_policy_id=learner.initial_actor.identity,
                          final_actor_policy_id=final_policy.identity,
                          final_deployment_expected_return=float(model.evaluate(learner.deployment.policy)[0][mode]),
                          final_deployment_policy_id=learner.deployment.policy.identity,
                          deployment_certified=learner.deployment.valid_for(learner.replay.plausible))
        result = jsonable(result)
        writer.finish(result, getattr(learner, "events", []))
    except BaseException as exc:
        writer.fail(exc)
        raise
    verify_run(folder, identity)
    return result



@lru_cache(maxsize=1)
def _worker_model(kind, context, benchmark):
    return make_model(kind, context, benchmark)


def _execute_cell(cell, episodes, config, resume, primitive_configs):
    benchmark, kind, context, mode, seed, method, folder = cell
    model = _worker_model(kind, context, benchmark)
    return execute(model, folder, method, seed, mode, episodes, config, resume, primitive_configs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environments", nargs="+", choices=("deepsea", "frozenlake"), default=["deepsea", "frozenlake"])
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(BENCHMARKS), default=["legacy"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mass-bound", type=float)
    parser.add_argument("--report-pdf", action="store_true", help="Render verified figures and PDF after all cells finish")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS),
                        help="Default slate includes both full-support and misspecified-prior Bayesian IDS")
    parser.add_argument("--contexts", nargs="+", type=int, choices=(0, 1, 2), default=[0])
    parser.add_argument("--modes", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--actor-init", choices=("model_mixture", "uniform"), default="model_mixture")
    parser.add_argument("--idac-batch-episodes", type=int, default=16)
    parser.add_argument("--idac-policy-lr", type=float, default=.1)
    parser.add_argument("--idac-value-lr", type=float, default=.2)
    parser.add_argument("--idac-tau", type=float, default=.01)
    parser.add_argument("--idac-kappa", type=float, default=.1)
    parser.add_argument("--pomd-policy-step", type=float, default=None)
    parser.add_argument("--ucb-bonus", type=float, default=1.)
    parser.add_argument("--comparator-delta", type=float, default=.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.workers < 1:
        parser.error("--episodes and --workers must be positive")
    if args.mass_bound is None:
        args.mass_bound = 128. if any(b != "legacy" for b in args.benchmarks) else 20.
    config = CEACConfig(mass_bound=args.mass_bound, normalization=args.mass_bound,
                        refresh_interval=args.refresh_interval, actor_initialization=args.actor_init,
                        confidence_delta=.05/len(set(args.contexts)))
    primitive_configs = {
        "idac": asdict(IDACConfig(policy_lr=args.idac_policy_lr, value_lr=args.idac_value_lr,
                    batch_episodes=args.idac_batch_episodes, tau=args.idac_tau, kappa=args.idac_kappa)),
        "ucb_q": asdict(UCBQConfig(bonus_constant=args.ucb_bonus, delta=args.comparator_delta)),
        "pomd": asdict(POMDConfig(delta=args.comparator_delta, policy_step=args.pomd_policy_step))}
    study_configuration = {k: v for k, v in vars(args).items() if k not in {"output", "resume"}}
    study_identity = manifest({"study": study_configuration, "ceac": asdict(config),
        "benchmark_specifications": [specification(b, k) for b in args.benchmarks for k in args.environments]})
    if args.output.exists():
        if not args.resume:
            parser.error("Output already exists; use --resume to verify/skip matching completed cells")
        saved = json.loads((args.output/"study.json").read_text())
        if saved != study_identity:
            raise ValueError("Study source/config/runtime identity changed; use a new output folder")
    else:
        args.output.mkdir(parents=True)
        atomic_json(args.output/"study.json", study_identity)
    source_bundle = args.output / "source_bundle.tar.gz"
    if not source_bundle.exists():
        pending_bundle = args.output / "source_bundle.tar.gz.pending"
        with tarfile.open(pending_bundle, "w:gz") as archive:
            for name, expected in study_identity["sources"].items():
                source = Path(__file__).resolve().parent / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError("Runtime source changed during study launch")
                archive.add(source, arcname="ceac/"+name)
            archive.add(args.output/"study.json", arcname="study.json")
        pending_bundle.replace(source_bundle)
    atomic_json(args.output/"status.json", {"status": "running"})
    results = []
    try:
        audit_path = args.output / "environment_audit.json"
        if not audit_path.exists():
            audits = []
            for benchmark in args.benchmarks:
                for kind in args.environments:
                    for context in args.contexts:
                        model = make_model(kind, context, benchmark)
                        audit = audit_breadth_model if benchmark in BREADTH_PROFILES else audit_model
                        audits.append(audit(model, config.mass_bound))
                        print(f"Preflight verified: {benchmark}/{kind}/context{context}", flush=True)
                        del model
            atomic_json(audit_path, {"study_identity": study_identity["identity"], "models": audits,
                                     "strength_checks": audit_strength_family(audits),
                                     "breadth_checks": audit_breadth_family(audits)})
        else:
            saved_audit = json.loads(audit_path.read_text())
            if saved_audit["study_identity"] != study_identity["identity"]:
                raise ValueError("Benchmark preflight belongs to a different study")
        cells = list(study_cells(args.output, study_configuration))
        results_by_index = {}
        def record(i, result):
            results_by_index[i] = result
            results[:] = [results_by_index[j] for j in sorted(results_by_index)]
            atomic_json(args.output/"results.json", results)
            atomic_json(args.output/"status.json", {"status": "running", "completed_cells": len(results),
                                                    "total_cells": len(cells)})
            print(f"[{len(results)}/{len(cells)}] {result['benchmark']}/{result['environment']} "
                  f"mode={result['true_mode']} {result['method']} seed={result['seed']}: "
                  f"regret={result['pseudo_regret']:.4f} last E[return]={result['last_window_expected_return']:.4f}",
                  flush=True)
        if args.workers == 1:
            for i, cell in enumerate(cells):
                record(i, _execute_cell(cell, args.episodes, config, args.resume, primitive_configs))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(_execute_cell, cell, args.episodes, config, args.resume, primitive_configs): i
                           for i, cell in enumerate(cells)}
                for future in as_completed(pending):
                    record(pending[future], future.result())

    except BaseException as exc:
        atomic_json(args.output/"status.json", {"status": "failed", "completed_cells": len(results), "error": str(exc)})
        raise
    atomic_json(args.output/"status.json", {"status": "complete", "completed_cells": len(results)})
    if args.report_pdf:
        from .report import render
        print(render(args.output, pdf=True), flush=True)


if __name__ == "__main__":
    main()
