"""Persistent policy and trajectory contracts shared by all algorithms."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class Policy:
    probabilities: np.ndarray
    label: str = "policy"

    def __post_init__(self):
        p = np.array(self.probabilities, dtype=float, copy=True)
        if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid policy probabilities")
        if not np.allclose(p.sum(axis=1), 1., atol=1e-12):
            raise ValueError("Policy rows must sum to one")
        p.flags.writeable = False
        object.__setattr__(self, "probabilities", p)

    @property
    def identity(self):
        return hashlib.sha256(self.probabilities.tobytes()).hexdigest()


@dataclass(frozen=True)
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    log_likelihood: tuple[float, ...]


@dataclass(frozen=True)
class Episode:
    index: int
    context: int
    policy_id: str
    behavior: str
    transitions: tuple[Transition, ...]

    @property
    def total_return(self):
        return sum(t.reward for t in self.transitions)

    @property
    def log_likelihood(self):
        return np.sum([t.log_likelihood for t in self.transitions], axis=0)
