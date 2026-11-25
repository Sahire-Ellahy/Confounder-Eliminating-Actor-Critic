"""Layer 1 organic interfaces separating public agent-visible structure from sealed oracle evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import numpy as np

from ceac_bench.core.contracts import ConditionKey, DomainMetadata, HeldOutBatch, PolicyView
from ceac_bench.core.primitives import FrozenFunctionClass, PersistentCallable


@runtime_checkable
class StepEnvironment(Protocol):
    """Minimal Gymnasium-compatible environment stepping interface."""

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class PublicProblemView:
    """Strictly public problem specification passed to Learners (zero ground-truth or passenger leak).

    For finite/tabular benchmarks, ``candidate_model`` exposes the public candidate hypothesis class.
    For pixel Atari or continuous MuJoCo/MJX benchmarks, ``candidate_model`` may be ``None`` while
    ``obs_shape`` / ``continuous_action_dim`` describe tensor spaces.
    """

    metadata: DomainMetadata
    horizon: int
    num_states: int | None = None
    num_actions: int | None = None
    action_mask: np.ndarray | None = None
    levels: tuple[tuple[int, ...], ...] | None = None
    initial_state: int = 0
    obs_shape: tuple[int, ...] | None = None
    continuous_action_dim: int | None = None
    candidate_model: Any = field(default=None, repr=False, compare=False)
    environment_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_finite_model(
        cls,
        metadata: DomainMetadata,
        model: Any,
        env_config: dict[str, Any],
    ) -> "PublicProblemView":
        mask = np.array(model.mask, dtype=bool, copy=True)
        mask.flags.writeable = False
        levels = tuple(tuple(int(s) for s in lvl) for lvl in model.levels)
        horizon = int(getattr(model, "horizon", len(levels)))
        return cls(
            metadata=metadata,
            horizon=horizon,
            num_states=int(model.n),
            num_actions=int(model.actions),
            action_mask=mask,
            levels=levels,
            initial_state=int(model.initial),
            candidate_model=model,
            environment_config=dict(env_config),
        )

    def map_observation_to_state(self, obs: Any) -> Any:
        if self.candidate_model is not None and hasattr(self.candidate_model, "index"):
            return int(self.candidate_model.index[obs])
        return obs

    def step_log_likelihood(self, state: Any, action: Any, next_obs: Any, reward: float, done: bool) -> tuple[float, ...]:
        if self.candidate_model is not None and hasattr(self.candidate_model, "log_likelihood"):
            return tuple(self.candidate_model.log_likelihood(int(state), int(action), next_obs, reward, done))
        return ()


class OracleEvaluator(FrozenFunctionClass):
    """Sealed ground-truth evaluator accessible ONLY to runners and metric observers, never learners."""

    def optimal_return(self) -> float:
        raise NotImplementedError

    def expected_return(self, policy: PolicyView) -> float:
        raise NotImplementedError

    def policy_gap(self, policy: PolicyView) -> float:
        return max(0.0, float(self.optimal_return() - self.expected_return(policy)))

    def sample_heldout_batch(
        self,
        policy: PolicyView,
        n_onpolicy: int,
        n_probe: int,
        rng: np.random.Generator,
    ) -> HeldOutBatch:
        raise NotImplementedError


@dataclass(frozen=True)
class FiniteDAGOracleEvaluator(OracleEvaluator):
    """Fast O(|S||A|) exact return and vectorized multinomial held-out rollout evaluator on finite DAGs."""

    variant_id: str
    truth_index: int = field(repr=False)
    realized_nuisance: int = field(default=0, repr=False)
    _model: Any = field(default=None, repr=False, compare=False)
    _truth_r_vec: np.ndarray = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _compiled_tables: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def build(
        cls,
        variant_id: str,
        model: Any,
        truth: int,
        realized_nuisance: int = 0,
    ) -> "FiniteDAGOracleEvaluator":
        S, A, M = int(model.n), int(model.actions), int(model.num_models)
        r_vec = np.zeros(S * A, dtype=float)
        tables: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        p_next_true = np.zeros((S, A, S), dtype=float)
        p_next_all = np.zeros((M, S, A, S), dtype=float)

        for s in range(S):
            for a in np.flatnonzero(model.mask[s]):
                s_i, a_i = int(s), int(a)
                laws = model.laws[s_i, a_i]
                law_t = laws[truth]
                r_vec[s_i * A + a_i] = sum(p * k[1] for k, p in law_t.items())

                keys = list(laws[0].keys())
                p_true = np.array([law_t[k] for k in keys], dtype=float)
                p_true /= p_true.sum()
                rews = np.array([float(k[1]) for k in keys], dtype=float)
                dests = np.array([S if k[2] else int(model.index[k[0]]) for k in keys], dtype=int)
                probs_m = np.array([[laws[m][k] for m in range(M)] for k in keys], dtype=float)
                ll_mat = np.log(probs_m)
                tables[s_i, a_i] = (p_true, rews, dests, ll_mat)
                for k_idx, d in enumerate(dests):
                    if d < S:
                        p_next_true[s_i, a_i, d] += p_true[k_idx]
                        p_next_all[:, s_i, a_i, d] += probs_m[k_idx]

        compiled = dict(tables=tables, p_next_true=p_next_true, p_next_all=p_next_all)
        return cls(
            variant_id=variant_id,
            truth_index=int(truth),
            realized_nuisance=int(realized_nuisance),
            _model=model,
            _truth_r_vec=r_vec,
            _compiled_tables=compiled,
        )

    @property
    def p_next_true(self) -> np.ndarray:
        return self._compiled_tables["p_next_true"]

    @property
    def p_next_all(self) -> np.ndarray:
        return self._compiled_tables["p_next_all"]

    def optimal_return(self) -> float:
        return float(self._model.optimal_returns[self.truth_index])

    def expected_return(self, policy: PolicyView) -> float:
        probs = policy.probabilities
        if probs is None:
            raise TypeError("FiniteDAGOracleEvaluator requires tabular policy probabilities.")
        model = self._model
        S, A = int(model.n), int(model.actions)
        H = int(getattr(model, "horizon", len(model.levels)))
        p_t = self._compiled_tables["p_next_true"]
        occ = np.zeros(S, dtype=float)
        occ[int(model.initial)] = 1.0
        sa_occ = np.zeros((S, A), dtype=float)
        for lvl in range(H):
            sts = model.levels[lvl]
            sa_lvl = occ[sts, None] * probs[sts]
            sa_occ[sts] = sa_lvl
            if lvl < H - 1:
                occ += np.einsum("ia,ias->s", sa_lvl, p_t[sts])
        return float(self._truth_r_vec @ sa_occ.ravel())

    def _multinomial_rollout(
        self,
        probs: np.ndarray,
        n_episodes: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model = self._model
        S, A, M = int(model.n), int(model.actions), int(model.num_models)
        tables = self._compiled_tables["tables"]
        sa_counts = np.zeros((S, A), dtype=int)
        sa_rew_sum = np.zeros((S, A), dtype=float)
        sa_next_counts = np.zeros((S, A, S + 1), dtype=int)
        sa_ll_sum = np.zeros((S, A, M), dtype=float)

        s_entering = np.zeros(S + 1, dtype=int)
        s_entering[int(model.initial)] = int(n_episodes)

        for level in model.levels:
            for s in level:
                s_i = int(s)
                cnt_s = int(s_entering[s_i])
                if cnt_s <= 0:
                    continue
                p_a = np.clip(probs[s_i], 0.0, 1.0)
                p_a /= p_a.sum()
                a_draws = rng.multinomial(cnt_s, p_a)
                for a_i in np.flatnonzero(a_draws > 0):
                    a_int = int(a_i)
                    cnt_sa = int(a_draws[a_int])
                    sa_counts[s_i, a_int] += cnt_sa
                    p_true, rews, dests, ll_mat = tables[s_i, a_int]
                    o_draws = rng.multinomial(cnt_sa, p_true)
                    for k_idx in np.flatnonzero(o_draws > 0):
                        c_o = int(o_draws[k_idx])
                        sa_rew_sum[s_i, a_int] += c_o * rews[k_idx]
                        sa_next_counts[s_i, a_int, dests[k_idx]] += c_o
                        sa_ll_sum[s_i, a_int] += c_o * ll_mat[k_idx]
                        if dests[k_idx] < S:
                            s_entering[dests[k_idx]] += c_o
        return sa_counts, sa_rew_sum, sa_next_counts, sa_ll_sum

    def sample_heldout_batch(
        self,
        policy: PolicyView,
        n_onpolicy: int,
        n_probe: int,
        rng: np.random.Generator,
    ) -> HeldOutBatch:
        probs_on = policy.probabilities
        if probs_on is None:
            raise TypeError("FiniteDAGOracleEvaluator requires tabular policy probabilities.")
        mask = self._model.mask
        probs_probe = mask / np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        c_on, r_on, nx_on, ll_on = self._multinomial_rollout(probs_on, n_onpolicy, rng)
        c_pr, r_pr, nx_pr, ll_pr = self._multinomial_rollout(probs_probe, n_probe, rng)
        return HeldOutBatch(
            onpolicy_counts=c_on,
            onpolicy_reward_sum=r_on,
            onpolicy_next_counts=nx_on,
            onpolicy_ll_sum=ll_on,
            probe_counts=c_pr,
            probe_reward_sum=r_pr,
            probe_next_counts=nx_pr,
            probe_ll_sum=ll_pr,
        )


@dataclass(frozen=True)
class BoundProblem:
    """Instantiated single-cell problem separating public learner view from sealed oracle evaluator."""

    condition: ConditionKey
    metadata: DomainMetadata
    env: StepEnvironment = field(repr=False, compare=False)
    public_view: PublicProblemView
    evaluator: OracleEvaluator


@runtime_checkable
class ProblemBuilder(PersistentCallable, Protocol):
    """Persistent function class mapping a ConditionKey -> BoundProblem."""

    @property
    def metadata(self) -> DomainMetadata: ...

    def __call__(self, condition: ConditionKey) -> BoundProblem: ...


@dataclass(frozen=True)
class EnvironmentFamily(FrozenFunctionClass):
    """Cohesive domain family grouping base, controlled-junction, and path-dependent problem builders."""

    family_id: str
    title: str
    builders: tuple[Any, ...]
