"""Organic Learner interface and persistent LearnerFactory function classes wrapping all methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
import numpy as np

from ceac.contracts import Episode
from ceac.primitives import IDAC, IDACConfig, OptimisticPOMD, POMDConfig, PrimitiveSpec, UCBQ, UCBQConfig
from ceac_bench.core.contracts import CriticSnapshot, PolicyView, SelectionStep
from ceac_bench.core.primitives import FrozenFunctionClass, PersistentCallable
from ceac_bench.problems.interfaces import PublicProblemView


@runtime_checkable
class Learner(Protocol):
    """Encapsulated sequential decision-making learner interface consumed by `CellRunner`."""

    def select(self, episode_idx: int) -> SelectionStep: ...

    def observe(self, episode: Episode, selection: SelectionStep) -> dict[str, Any]: ...

    def compact_memory(self, episode_idx: int) -> None: ...

    def critic_snapshot(self, selection: SelectionStep) -> CriticSnapshot: ...

    def diagnostics(self) -> dict[str, Any]: ...


@runtime_checkable
class LearnerFactory(PersistentCallable, Protocol):
    """Persistent function class constructing a `Learner` solely from `PublicProblemView`."""

    @property
    def method_id(self) -> str: ...

    @property
    def requires_calibration(self) -> bool: ...

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner: ...


def _make_primitive_spec(view: PublicProblemView) -> PrimitiveSpec:
    if view.action_mask is None or view.levels is None or view.num_states is None:
        raise TypeError("Tabular PrimitiveSpec requires finite state/action PublicProblemView.")
    elapsed = np.zeros(view.num_states, dtype=int)
    for t, sts in enumerate(view.levels):
        elapsed[list(sts)] = t
    lo, hi = view.metadata.reward_bounds
    return PrimitiveSpec(view.action_mask, elapsed, view.horizon, view.initial_state, float(lo), float(hi))


class _WrappedLearner:
    """Adapter encapsulating memory compaction and read-only CriticSnapshot extraction."""

    def __init__(self, method_id: str, inner: Any, view: PublicProblemView) -> None:
        self.method_id = method_id
        self.inner = inner
        self.view = view

    def select(self, episode_idx: int) -> SelectionStep:
        pol, behavior, prices = self.inner.select(episode_idx)
        return SelectionStep(policy=PolicyView.from_tabular(pol), behavior=str(behavior), prices=prices)

    def observe(self, episode: Episode, selection: SelectionStep) -> dict[str, Any]:
        raw_pol = selection.policy.to_tabular_policy()
        out = self.inner.observe(episode, raw_pol, selection.prices)
        return dict(out) if isinstance(out, dict) else {}

    def compact_memory(self, episode_idx: int) -> None:
        rep = getattr(self.inner, "replay", None)
        if rep is None or not hasattr(rep, "strata"):
            return
        ev = getattr(rep, "evidence", None)
        if ev is not None and hasattr(ev, "strata"):
            ev.strata.clear()
        stats = getattr(rep, "statistics", None)
        if self.method_id in {"ceac", "ceac_natural"} and stats is not None:
            keep = set(self.inner.bank._ids) | set(self.inner.candidates) | set(self.inner.frozen)
            rep.strata.clear()
            for pid in [k_id for k_id in stats if k_id not in keep]:
                del stats[pid]
        elif self.method_id in {"ceac_ranked", "ceac_ranked_natural"} and stats is not None:
            keep = set(self.inner.bank._ids) | set(rep.policies_by_id)
            rep.strata.clear()
            for pid in [k_id for k_id in stats if k_id not in keep]:
                del stats[pid]
        else:
            for bucket in rep.strata.values():
                bucket.clear()

    def diagnostics(self) -> dict[str, Any]:
        diag: dict[str, Any] = {}
        if hasattr(self.inner, "diagnostics") and callable(self.inner.diagnostics):
            raw = self.inner.diagnostics()
            if isinstance(raw, dict):
                diag.update(raw)
        if hasattr(self.inner, "bank") and hasattr(self.inner.bank, "policies"):
            diag.setdefault("bank_policies", len(self.inner.bank.policies))
        if self.method_id == "pomd" and hasattr(self.inner, "q") and hasattr(self.inner, "mask"):
            diag["pomd_saturated"] = int(np.all(self.inner.q[self.inner.mask] == 0))
        return diag

    def critic_snapshot(self, selection: SelectionStep) -> CriticSnapshot:
        S = int(self.view.num_states or 0)
        if self.method_id in {"ceac", "ceac_natural", "ceac_ranked", "ceac_ranked_natural"}:
            actor = self.inner.actor
            prices = selection.prices if selection.prices is not None else self.inner.prices
            vr = np.array(actor.reward_value[:S], dtype=float, copy=True)
            if hasattr(actor, "evidence_value"):
                vi = np.array(actor.evidence_value(prices)[:S], dtype=float, copy=True)
            else:
                vi = np.einsum("ij,ijs->s", prices, actor.row_values[:, :, :S])
            actor_pol = PolicyView.from_tabular(actor.snapshot())
            return CriticSnapshot(
                kind="ceac_two_critic",
                reward_v=vr,
                evidence_v=vi,
                epistemic_state={"prices": np.array(prices, dtype=float, copy=True), "eval_policy": actor_pol},
            )

        if self.method_id == "idac":
            spec = self.inner.spec
            return CriticSnapshot(
                kind="idac",
                reward_v=np.array(self.inner.cost_value, dtype=float, copy=True),
                evidence_v=np.array(self.inner.shadow_value, dtype=float, copy=True),
                epistemic_state={
                    "width": float(spec.width),
                    "reward_upper": float(spec.reward_upper),
                    "cost_floor": float(self.inner.config.cost_floor),
                    "horizon": int(spec.horizon),
                    "elapsed": np.array(spec.elapsed, dtype=int, copy=True),
                    "mu_cost": float(self.inner.mu_cost),
                    "mu_entropy": float(self.inner.mu_entropy),
                    "eval_policy": selection.policy,
                },
            )

        if self.method_id == "ucb_q":
            spec = self.inner.spec
            w = float(spec.width)
            lo = float(spec.reward_lower)
            H = int(spec.horizon)
            n_vis = np.maximum(1, self.inner.counts)
            bonus_all = w * self.inner.config.bonus_constant * np.sqrt((H ** 3) * self.inner.log_factor / n_vis)
            q_tot = lo + w * self.inner.q
            q_rew = q_tot - bonus_all
            return CriticSnapshot(
                kind="ucb_q",
                reward_q=q_rew,
                evidence_q=bonus_all,
                epistemic_state={
                    "horizon": H,
                    "elapsed": np.array(spec.elapsed, dtype=int, copy=True),
                    "eval_policy": selection.policy,
                },
            )

        if self.method_id == "pomd":
            spec = self.inner.spec
            w = float(spec.width)
            hi = float(spec.reward_upper)
            cb, tb = self.inner.bonuses()
            bonus = w * (cb[:S] + tb[:S])
            return CriticSnapshot(
                kind="pomd",
                reward_q=np.array(self.inner.q, dtype=float, copy=True),
                evidence_q=bonus,
                reward_v=np.array(self.inner.values, dtype=float, copy=True),
                epistemic_state={
                    "width": w,
                    "reward_upper": hi,
                    "horizon": int(spec.horizon),
                    "elapsed": np.array(spec.elapsed, dtype=int, copy=True),
                    "eval_policy": selection.policy,
                },
            )

        if self.method_id.startswith("ids"):
            model = self.view.candidate_model
            post = np.array(self.inner.posterior, dtype=float, copy=True)
            stage_i = (
                self.inner.action_information_stage(post)
                if hasattr(self.inner, "action_information_stage")
                else model.mutual_information_stage(post)
            )
            return CriticSnapshot(
                kind="bayesian_ids",
                evidence_q=np.array(stage_i, dtype=float, copy=True),
                epistemic_state={
                    "posterior": post,
                    "log_weights": np.array(self.inner.log_weights, dtype=float, copy=True),
                    "initial_support": np.array(self.inner.initial_support, dtype=bool, copy=True),
                    "eval_policy": selection.policy,
                },
            )

        return CriticSnapshot(kind="oracle", epistemic_state={"eval_policy": selection.policy})


@dataclass(frozen=True)
class IDACFactory(FrozenFunctionClass):
    method_id: str = "idac"
    requires_calibration: bool = True
    uses_candidate_likelihoods: bool = False

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        return _WrappedLearner(self.method_id, IDAC(_make_primitive_spec(view), rng, IDACConfig()), view)


@dataclass(frozen=True)
class UCBQFactory(FrozenFunctionClass):
    method_id: str = "ucb_q"
    requires_calibration: bool = True
    uses_candidate_likelihoods: bool = False

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        return _WrappedLearner(self.method_id, UCBQ(_make_primitive_spec(view), rng, int(episodes), UCBQConfig()), view)


@dataclass(frozen=True)
class POMDFactory(FrozenFunctionClass):
    method_id: str = "pomd"
    requires_calibration: bool = True
    uses_candidate_likelihoods: bool = False

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        return _WrappedLearner(self.method_id, OptimisticPOMD(_make_primitive_spec(view), int(episodes), POMDConfig()), view)


@dataclass(frozen=True)
class BayesianIDSFactory(FrozenFunctionClass):
    method_id: str = "ids_product"
    prior_name: str = "reference"  # "restricted" | "reference" | explicit prior name
    requires_calibration: bool = True
    uses_candidate_likelihoods: bool = True

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        model = view.candidate_model
        resolved = (
            view.metadata.restricted_prior_name
            if self.prior_name == "restricted"
            else (view.metadata.reference_prior_name if self.prior_name == "reference" else self.prior_name)
        )
        prior = model.prior(resolved)
        if hasattr(model, "_informative_pairs"):
            from ceac_pathdep.learners import FastPathDepBayesianIDS

            return _WrappedLearner(self.method_id, FastPathDepBayesianIDS(model, rng, prior), view)
        from ceac_hardness.ids import BayesianIDS

        return _WrappedLearner(self.method_id, BayesianIDS(model, rng, prior), view)


@dataclass(frozen=True)
class OptimalActionIDSFactory(FrozenFunctionClass):
    method_id: str = "ids_action_onpath"
    target: str = "onpath_class"  # "policy_tree" | "onpath_class" | "single_step"
    prior_name: str = "reference"
    requires_calibration: bool = True
    uses_candidate_likelihoods: bool = True

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        model = view.candidate_model
        resolved = (
            view.metadata.restricted_prior_name
            if self.prior_name == "restricted"
            else (view.metadata.reference_prior_name if self.prior_name == "reference" else self.prior_name)
        )
        prior = model.prior(resolved)
        if self.target == "single_step":
            from ceac_critic_diagnostics.run_pathologies_audit import FastActionIDS

            return _WrappedLearner(self.method_id, FastActionIDS(model, rng, prior), view)
        from ceac_pathdep.learners import MultiStepActionIDS

        return _WrappedLearner(self.method_id, MultiStepActionIDS(model, rng, prior, target=self.target), view)


@dataclass(frozen=True)
class CEACFactory(FrozenFunctionClass):
    method_id: str = "ceac"
    optimizer: str = "reference"
    mass_bound: float = 64.0
    normalization: float = 64.0
    requires_calibration: bool = True
    uses_candidate_likelihoods: bool = True

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        from ceac_pathologies.agent import ConfounderEliminatingAC, PathologiesCEACConfig

        model = view.candidate_model
        cfg = PathologiesCEACConfig(
            mass_bound=self.mass_bound,
            normalization=self.normalization,
            optimizer=self.optimizer,
        )
        if hasattr(model, "canonical_path_policies"):
            from ceac_pathdep.study import PathDepCertifiedCEAC

            return _WrappedLearner(self.method_id, PathDepCertifiedCEAC(model, rng, cfg), view)
        return _WrappedLearner(self.method_id, ConfounderEliminatingAC(model, rng, cfg), view)


@dataclass(frozen=True)
class GreedyRankedCEACFactory(FrozenFunctionClass):
    method_id: str = "ceac_ranked"
    optimizer: str = "reference"
    requires_calibration: bool = True
    uses_candidate_likelihoods: bool = True

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        from ceac_ranked.agent import GreedyPricedCEAC, RankedCEACConfig

        model = view.candidate_model
        cfg = RankedCEACConfig(optimizer=self.optimizer)
        if hasattr(model, "canonical_path_policies"):
            from ceac_pathdep.learners import PathDepGreedyCEAC

            return _WrappedLearner(self.method_id, PathDepGreedyCEAC(model, rng, cfg), view)
        return _WrappedLearner(self.method_id, GreedyPricedCEAC(model, rng, cfg), view)


@dataclass(frozen=True)
class OracleFactory(FrozenFunctionClass):
    """Offline Graves–Lai baseline (receives oracle truth index solely as an upper-bound comparator)."""

    method_id: str = "oracle"
    mass_bound: float = 64.0
    normalization: float = 64.0
    requires_calibration: bool = False
    uses_candidate_likelihoods: bool = True

    def __call__(self, view: PublicProblemView, episodes: int, rng: np.random.Generator) -> Learner:
        raise TypeError("OracleFactory requires `build_with_truth(view, truth)` via CellRunner.")

    def build_with_truth(self, view: PublicProblemView, truth: int) -> Learner:
        from ceac_hardness.agents import CEACConfig, LaiGravesOracle

        cfg = CEACConfig(mass_bound=self.mass_bound, normalization=self.normalization)
        return _WrappedLearner(self.method_id, LaiGravesOracle(view.candidate_model, int(truth), cfg), view)


def default_method_factories(*, pathdep: bool = True, include_ranked: bool = True, include_action_ids: bool = True) -> tuple[LearnerFactory, ...]:
    """Canonical learner slate for multi-step path-dependent or single-step suites."""
    factories: list[LearnerFactory] = [
        IDACFactory(),
        UCBQFactory(),
        POMDFactory(),
        BayesianIDSFactory(method_id="ids_null", prior_name="restricted", requires_calibration=False),
        BayesianIDSFactory(method_id="ids_product", prior_name="reference", requires_calibration=True),
    ]
    if pathdep:
        factories.extend([
            OptimalActionIDSFactory(method_id="ids_action_tree", target="policy_tree", prior_name="reference"),
            OptimalActionIDSFactory(method_id="ids_action_onpath", target="onpath_class", prior_name="reference"),
        ])
    elif include_action_ids:
        factories.append(OptimalActionIDSFactory(method_id="ids_action", target="single_step", prior_name="reference"))
    factories.extend([
        CEACFactory(method_id="ceac", optimizer="reference"),
        CEACFactory(method_id="ceac_natural", optimizer="guarded_natural"),
    ])
    if include_ranked:
        factories.extend([
            GreedyRankedCEACFactory(method_id="ceac_ranked", optimizer="reference"),
            GreedyRankedCEACFactory(method_id="ceac_ranked_natural", optimizer="guarded_natural"),
        ])
    factories.append(OracleFactory())
    return tuple(factories)
