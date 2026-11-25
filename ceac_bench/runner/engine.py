"""Unified single-pass `CellRunner`, `CalibrationGate`, and memory-bounded `SpawnStudyExecutor`."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import multiprocessing as mp
import os
import time
from typing import Any, Protocol, runtime_checkable
import numpy as np

from ceac.contracts import Episode, Transition
from ceac_bench.core.contracts import ConditionKey, SelectionStep
from ceac_bench.core.primitives import FrozenFunctionClass, PersistentCallable, RNGStreamSet
from ceac_bench.problems.interfaces import BoundProblem, FiniteDAGOracleEvaluator, ProblemBuilder
from ceac_bench.store.cache import CellCacheStore
from ceac_bench.store.schema import CellRecord
from .learners import Learner, LearnerFactory, OracleFactory


@runtime_checkable
class MetricObserver(PersistentCallable, Protocol):
    """Composable stateful observer session created fresh per cell run."""

    def start_session(self, problem: BoundProblem, learner: Learner, method_id: str, episodes: int) -> Any: ...


@dataclass(frozen=True)
class CellRunner(FrozenFunctionClass):
    """Single-pass execution loop shared across all discrete, DAG, and continuous environments."""

    observers: tuple[Any, ...] = ()
    compact_every: int = 128

    def run_cell(
        self,
        problem_builder: ProblemBuilder,
        learner_factory: LearnerFactory,
        condition: ConditionKey,
        episodes: int,
        *,
        cache: CellCacheStore | None = None,
    ) -> CellRecord:
        spec_payload = {
            "runner": self.to_spec(),
            "problem_builder": problem_builder.to_spec(),
            "learner_factory": learner_factory.to_spec(),
            "condition": {
                "suite_id": condition.suite_id,
                "family_id": condition.family_id,
                "variant_id": condition.variant_id,
                "dose": condition.dose,
                "task": condition.task,
                "seed": condition.seed,
            },
            "episodes": int(episodes),
        }
        cell_identity = FrozenFunctionClass.digest(self) + ":" + problem_builder.digest() + ":" + learner_factory.digest()
        from ceac_bench.core.primitives import canonical_json, digest_bytes

        full_identity = digest_bytes((cell_identity + canonical_json(spec_payload, indent=None)).encode("utf-8"))
        if cache is not None:
            cached = cache.try_load(condition, learner_factory.method_id, full_identity)
            if cached is not None:
                return cached

        t0 = time.monotonic()
        bound = problem_builder(condition)
        ns = bound.metadata.effective_namespace
        streams = RNGStreamSet.from_parts(
            namespace=ns,
            seed=condition.seed,
            variant_id=condition.variant_id,
            dose=condition.dose,
            task=condition.task,
            method_id=learner_factory.method_id,
        )

        learner_rng = streams.make_learner_rng()
        action_rng = streams.make_action_rng()
        if isinstance(learner_factory, OracleFactory):
            if not isinstance(bound.evaluator, FiniteDAGOracleEvaluator):
                raise TypeError("OracleFactory requires FiniteDAGOracleEvaluator.")
            learner = learner_factory.build_with_truth(bound.public_view, bound.evaluator.truth_index)
        else:
            learner = learner_factory(bound.public_view, int(episodes), learner_rng)

        bound.env.reset(seed=streams.env_seed)

        sessions = [obs.start_session(bound, learner, learner_factory.method_id, int(episodes)) for obs in self.observers]

        last_pol_id: str | None = None
        cached_gap = 0.0
        pub = bound.public_view
        horizon = int(pub.horizon)
        n_actions = int(pub.num_actions or 0)
        emit_ll = bool(getattr(learner_factory, "uses_candidate_likelihoods", True))

        for t in range(int(episodes)):
            selection: SelectionStep = learner.select(t)
            if selection.policy.identity != last_pol_id:
                cached_gap = bound.evaluator.policy_gap(selection.policy)
                last_pol_id = selection.policy.identity

            obs, _ = bound.env.reset()
            transitions: list[Transition] = []
            ep_events: set[str] = set()
            for _ in range(horizon):
                s = pub.map_observation_to_state(obs)
                a = selection.policy.sample_action(s, n_actions, action_rng)
                nxt, rew, done, _, info = bound.env.step(a)
                ll = pub.step_log_likelihood(s, a, nxt, rew, done) if emit_ll else ()
                s_val = int(s) if isinstance(s, (int, np.integer)) else s
                a_val = int(a) if isinstance(a, (int, np.integer)) else a
                if done:
                    nxt_s: Any = -1
                else:
                    nxt_mapped = pub.map_observation_to_state(nxt)
                    nxt_s = int(nxt_mapped) if isinstance(nxt_mapped, (int, np.integer)) else nxt_mapped
                transitions.append(Transition(s_val, a_val, float(rew), nxt_s, bool(done), ll))
                ev_tag = info.get("event") if isinstance(info, dict) else None
                if isinstance(ev_tag, str):
                    ep_events.add(ev_tag)
                obs = nxt
                if done:
                    break

            ep_obj = Episode(t, 0, selection.policy.identity, selection.behavior, tuple(transitions))
            step_info = learner.observe(ep_obj, selection)
            if self.compact_every > 0 and (t + 1) % self.compact_every == 0:
                learner.compact_memory(t + 1)

            ep_num = t + 1
            for sess in sessions:
                sess.on_episode(
                    episode_num=ep_num,
                    gap=cached_gap,
                    events=ep_events,
                    selection=selection,
                    step_info=step_info,
                    streams=streams,
                )

        elapsed = round(time.monotonic() - t0, 3)
        scalars: dict[str, Any] = {"elapsed_seconds": elapsed, **learner.diagnostics()}
        trace: dict[str, tuple[float, ...]] = {}
        critic_audit: dict[str, tuple[float, ...]] = {}
        late_gap = 0.0
        cumulative_regret = 0.0

        for sess in sessions:
            res = sess.finalize()
            if "late_gap" in res:
                late_gap = float(res["late_gap"])
            if "cumulative_regret" in res:
                cumulative_regret = float(res["cumulative_regret"])
            scalars.update(res.get("scalars", {}))
            trace.update(res.get("trace", {}))
            critic_audit.update(res.get("critic_audit", {}))

        from ceac_bench.core.primitives import sanitize_json_value

        record = CellRecord(
            condition=condition,
            method_id=learner_factory.method_id,
            episodes=int(episodes),
            late_gap=float(late_gap),
            cumulative_regret=float(cumulative_regret),
            scalars=sanitize_json_value(scalars),
            trace=trace,
            critic_audit=critic_audit,
        )
        if cache is not None:
            cache.save(record, expected_identity=full_identity, spec_payload=spec_payload)
        return record


@dataclass(frozen=True)
class CalibrationGate(FrozenFunctionClass):
    """Persistent easy-case validation gate executed before multi-dose sweeps."""

    easy_gap_tolerance: float = 0.05
    require_priced_ceac_updates: bool = True

    def validate(self, calibration_cells: tuple[CellRecord, ...], required_methods: set[str]) -> dict[str, Any]:
        failures: list[str] = []
        checked = 0
        for cell in calibration_cells:
            if cell.method_id not in required_methods:
                continue
            checked += 1
            if cell.late_gap > self.easy_gap_tolerance:
                failures.append(
                    f"{cell.condition.variant_id} {cell.method_id} task={cell.condition.task} seed={cell.condition.seed}: "
                    f"late_gap={cell.late_gap:.4f} > {self.easy_gap_tolerance:.4f}"
                )
            if self.require_priced_ceac_updates and cell.method_id.startswith("ceac"):
                au = int(cell.scalars.get("actor_updates", 0))
                pu = int(cell.scalars.get("priced_actor_updates", 0))
                if min(au, pu) <= 0:
                    failures.append(
                        f"{cell.condition.variant_id} {cell.method_id} s={cell.condition.seed}: zero priced actor updates ({au}, {pu})"
                    )
        return {"all_pass": len(failures) == 0 and checked > 0, "checked": checked, "failures": failures}


def _init_spawn_worker() -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


def _execute_cell_job(job: tuple[CellRunner, ProblemBuilder, LearnerFactory, ConditionKey, int, str | None]) -> CellRecord:
    runner, p_builder, l_factory, condition, episodes, cache_dir = job
    from pathlib import Path

    cache = CellCacheStore(Path(cache_dir)) if cache_dir is not None else None
    return runner.run_cell(p_builder, l_factory, condition, episodes, cache=cache)


@dataclass(frozen=True)
class SpawnStudyExecutor(FrozenFunctionClass):
    """Memory-disciplined `spawn` pool executor (`maxtasksperchild=1`, `chunksize=1`) for study grids."""

    workers: int = 4
    maxtasksperchild: int = 1
    log_interval: int = 24

    def execute_jobs(
        self,
        runner: CellRunner,
        jobs: list[tuple[ProblemBuilder, LearnerFactory, ConditionKey, int]],
        *,
        cache_dir: str | None = None,
        label: str = "ceac-bench",
    ) -> tuple[CellRecord, ...]:
        if not jobs:
            return ()
        packed = [(runner, pb, lf, cond, eps, cache_dir) for (pb, lf, cond, eps) in jobs]
        records: list[CellRecord] = []
        t0 = time.monotonic()

        if self.workers <= 1:
            _init_spawn_worker()
            for i, item in enumerate(packed, 1):
                rec = _execute_cell_job(item)
                records.append(rec)
                if i % self.log_interval == 0 or i == len(packed):
                    print(
                        f"[{label}] {i}/{len(packed)} ({time.monotonic() - t0:.1f}s) — "
                        f"{rec.condition.variant_id} {rec.method_id} d={rec.condition.dose} s={rec.condition.seed} "
                        f"gap={rec.late_gap:.4f} reg={rec.cumulative_regret:.1f}",
                        flush=True,
                    )
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=self.workers, initializer=_init_spawn_worker, maxtasksperchild=self.maxtasksperchild) as pool:
                for i, rec in enumerate(pool.imap_unordered(_execute_cell_job, packed, chunksize=1), 1):
                    records.append(rec)
                    if i % self.log_interval == 0 or i == len(packed):
                        print(
                            f"[{label}] {i}/{len(packed)} ({time.monotonic() - t0:.1f}s) — "
                            f"{rec.condition.variant_id} {rec.method_id} d={rec.condition.dose} s={rec.condition.seed} "
                            f"gap={rec.late_gap:.4f} reg={rec.cumulative_regret:.1f}",
                            flush=True,
                        )
        gc.collect()
        records.sort(key=lambda r: (r.condition.family_id, r.condition.variant_id, float(r.condition.dose), r.condition.task, r.method_id, r.condition.seed))
        return tuple(records)
