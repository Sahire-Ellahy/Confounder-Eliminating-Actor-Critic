"""Persistent value objects shared across Problem Setup, Runner, Metrics, and Reporting layers."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable
import numpy as np

from ceac.contracts import Policy


@dataclass(frozen=True)
class DomainMetadata:
    """Static description of an environment variant and its experimental intervention axis."""

    family_id: str
    variant_id: str
    title: str
    short_title: str
    horizon: int
    dose_symbol: str
    dose_label: str
    easy_dose: int | float = 0
    event_names: tuple[str, ...] = ()
    reward_bounds: tuple[float, float] = (0.0, 1.0)
    rng_namespace: str = ""
    restricted_prior_name: str = "null"
    reference_prior_name: str = "reference"

    @property
    def effective_namespace(self) -> str:
        return self.rng_namespace or self.family_id


@dataclass(frozen=True, order=True)
class ConditionKey:
    """Canonical coordinate of a single experimental cell."""

    suite_id: str
    family_id: str
    variant_id: str
    dose: int | float
    task: int
    seed: int


@dataclass(frozen=True)
class PolicyView:
    """Unified immutable policy representation supporting both tabular (S, A) and continuous/neural actors."""

    label: str = "policy"
    probabilities: np.ndarray | None = None
    act_fn: Callable[[Any, np.random.Generator], Any] | None = field(default=None, compare=False)
    explicit_id: str | None = None
    _raw_policy: Policy | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.probabilities is None and self.act_fn is None:
            raise ValueError("PolicyView requires either tabular `probabilities` or callable `act_fn`.")
        if self.probabilities is not None:
            arr = np.array(self.probabilities, dtype=float, copy=True)
            arr.flags.writeable = False
            object.__setattr__(self, "probabilities", arr)

    @classmethod
    def from_tabular(cls, policy: Policy) -> "PolicyView":
        return cls(
            label=policy.label,
            probabilities=policy.probabilities,
            explicit_id=policy.identity,
            _raw_policy=policy,
        )

    @property
    def identity(self) -> str:
        if self.explicit_id is not None:
            return self.explicit_id
        if self.probabilities is not None:
            return hashlib.sha256(self.probabilities.tobytes()).hexdigest()
        return hashlib.sha256(self.label.encode("utf-8")).hexdigest()

    def to_tabular_policy(self) -> Policy:
        if self._raw_policy is not None:
            return self._raw_policy
        if self.probabilities is None:
            raise TypeError("Continuous/callable PolicyView has no tabular probability matrix.")
        return Policy(self.probabilities, label=self.label)

    def sample_action(self, obs_or_state: Any, num_actions: int, rng: np.random.Generator) -> Any:
        if self.probabilities is not None:
            return int(rng.choice(num_actions, p=self.probabilities[int(obs_or_state)]))
        assert self.act_fn is not None
        return self.act_fn(obs_or_state, rng)


@dataclass(frozen=True)
class SelectionStep:
    """Decision returned by a Learner before rolling out episode t."""

    policy: PolicyView
    behavior: str
    prices: np.ndarray | None = None


@dataclass(frozen=True)
class CriticSnapshot:
    """Read-only lens into a learner's reward and evidence/epistemic critics for held-out Bellman auditing.

    Supports both finite `(S, A)` state-action tables (`reward_q`, `evidence_q`) and continuous/pixel
    neural critic callables (`reward_fn`, `evidence_fn`) so upcoming Atari and Dexterous Manipulation
    benchmarks share the exact same observer pipeline.
    """

    kind: str  # "ceac_two_critic" | "bayesian_ids" | "ucb_q" | "pomd" | "idac" | "oracle" | "neural_ceac"
    reward_q: np.ndarray | None = None
    evidence_q: np.ndarray | None = None
    reward_v: np.ndarray | None = None
    evidence_v: np.ndarray | None = None
    epistemic_state: dict[str, Any] = field(default_factory=dict)
    reward_fn: Callable[[Any, Any], float] | None = field(default=None, compare=False)
    evidence_fn: Callable[[Any, Any], float] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class HeldOutBatch:
    """Aggregated held-out evaluation rollout statistics sampled without mutating training state."""

    onpolicy_counts: np.ndarray
    onpolicy_reward_sum: np.ndarray
    onpolicy_next_counts: np.ndarray
    onpolicy_ll_sum: np.ndarray
    probe_counts: np.ndarray
    probe_reward_sum: np.ndarray
    probe_next_counts: np.ndarray
    probe_ll_sum: np.ndarray
    raw_transitions: tuple[Any, ...] = ()

    @property
    def mix_counts(self) -> np.ndarray:
        return self.onpolicy_counts + self.probe_counts

    @property
    def mix_reward_sum(self) -> np.ndarray:
        return self.onpolicy_reward_sum + self.probe_reward_sum

    @property
    def mix_next_counts(self) -> np.ndarray:
        return self.onpolicy_next_counts + self.probe_next_counts
