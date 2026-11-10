#!/usr/bin/env python3
"""Gymnasium-compatible Shifting Diagnostic DeepSea environment.

The environment keeps the standard DeepSea reachability primitive: an episode
reaches the terminal reward only after a coherent sequence of rightward moves.
At the context-dependent gate, the deployed action has the same law under two
candidate modes.  A probe or challenge action reveals the mode.  Thus the
wrong mode is a decision-theoretic confounder for the deployed policy.

Gymnasium is optional at runtime.  When it is unavailable, the module exposes a
small compatibility layer with the same ``reset``/``step`` and ``spaces`` API.
Installing Gymnasium makes ``ShiftingDeepSeaEnv`` a regular ``gymnasium.Env``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised when Gymnasium is installed.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - the repository tests use this branch.
    class _Env:
        metadata: Dict[str, Any] = {}

        def __init__(self) -> None:
            self.np_random = np.random.default_rng()

        def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)
            del options
            return None

    class _Discrete:
        def __init__(self, n: int, seed: int | None = None) -> None:
            if n <= 0:
                raise ValueError("n must be positive")
            self.n = int(n)
            self._rng = np.random.default_rng(seed)

        def seed(self, seed: int | None = None) -> None:
            self._rng = np.random.default_rng(seed)

        def sample(self, mask: np.ndarray | None = None) -> int:
            if mask is None:
                return int(self._rng.integers(self.n))
            valid = np.flatnonzero(np.asarray(mask, dtype=np.int8))
            if valid.size == 0:
                raise ValueError("mask has no valid actions")
            return int(self._rng.choice(valid))

        def contains(self, x: Any) -> bool:
            try:
                value = int(x)
            except (TypeError, ValueError):
                return False
            return 0 <= value < self.n

    class _Spaces:
        Discrete = _Discrete

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


LEFT = 0
RIGHT = 1
PROBE = 2
CHALLENGE = 3
TV = 4
ACTION_NAMES = {
    LEFT: "left",
    RIGHT: "right",
    PROBE: "probe",
    CHALLENGE: "challenge",
    TV: "tv",
}


@dataclass(frozen=True)
class DeepSeaConfig:
    size: int = 8
    gates: Tuple[int, ...] = (2, 4, 6)
    context_schedule: Tuple[int, ...] = (0, 1, 2, 0, 1, 2)
    block_length: int = 300
    true_modes: Tuple[int, ...] = (0, 1, 0)
    terminal_reward: float = 1.0
    right_cost: float = 0.002
    probe_cost: float = 0.015
    invalid_cost: float = 0.050
    tv_cost: float = 0.005
    challenge_scale: float = 0.080
    probe_prob_mode0: float = 0.15
    probe_prob_mode1: float = 0.85
    challenge_prob_mode0: float = 0.25
    challenge_prob_mode1: float = 0.75
    tv_bins: int = 8
    distributed_nuisance: bool = False

    def __post_init__(self) -> None:
        if self.size < 3:
            raise ValueError("size must be at least three")
        if len(self.gates) != len(self.true_modes):
            raise ValueError("gates and true_modes must have the same length")
        if any(g < 0 or g >= self.size for g in self.gates):
            raise ValueError("all gates must lie inside the DeepSea horizon")
        if any(c < 0 or c >= len(self.gates) for c in self.context_schedule):
            raise ValueError("context_schedule contains an invalid context")
        if any(m not in (0, 1) for m in self.true_modes):
            raise ValueError("true_modes must be binary")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")

    @property
    def n_contexts(self) -> int:
        return len(self.gates)

    @property
    def total_episodes(self) -> int:
        return self.block_length * len(self.context_schedule)

    @property
    def max_steps(self) -> int:
        # One extra step permits one probe without preventing terminal arrival.
        return self.size + 1

    def probe_probability(self, mode: int) -> float:
        return self.probe_prob_mode1 if int(mode) == 1 else self.probe_prob_mode0

    def challenge_probability(self, mode: int) -> float:
        return self.challenge_prob_mode1 if int(mode) == 1 else self.challenge_prob_mode0


@dataclass(frozen=True)
class DecodedState:
    context: int
    row: int
    column: int
    probe_used: int
    signal: int  # -1 means no signal, otherwise 0 or 1.
    tv_symbol: int  # -1 means no symbol, otherwise 0, ..., tv_bins-1.
    elapsed: int


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str


POLICY_SPECS: Tuple[PolicySpec, ...] = (
    PolicySpec("exploit", "exploit"),
    PolicySpec("probe", "probe"),
    PolicySpec("challenge", "challenge"),
    PolicySpec("probe-adaptive", "probe-adaptive"),
    PolicySpec("tv", "tv"),
)
POLICY_INDEX = {spec.name: i for i, spec in enumerate(POLICY_SPECS)}


class ShiftingDeepSeaEnv(gym.Env):
    """Contextual DeepSea with policy-changing diagnostic alternatives.

    Observation is a single integer encoding
    ``(context, row, column, probe_used, signal, tv_symbol, elapsed)``.
    The context schedule is deterministic and visible.  Each context has a
    fixed latent mode, so revisiting a context tests whether an algorithm uses
    information retained in its replay state.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: DeepSeaConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeepSeaConfig()
        self.action_space = spaces.Discrete(5)
        self._dims = (
            self.config.n_contexts,
            self.config.size + 1,
            self.config.size + 1,
            2,
            3,
            self.config.tv_bins + 1,
            self.config.max_steps + 1,
        )
        self._n_states = int(np.prod(self._dims, dtype=np.int64))
        self.observation_space = spaces.Discrete(self._n_states)
        self.episode_index = 0
        self.context = 0
        self.row = 0
        self.column = 0
        self.probe_used = 0
        self.signal = -1
        self.tv_symbol = -1
        self.elapsed = 0
        self._terminated = False
        self._episode_return = 0.0

    @property
    def n_states(self) -> int:
        return self._n_states

    def context_for_episode(self, episode_index: int) -> int:
        block = min(
            max(int(episode_index), 0) // self.config.block_length,
            len(self.config.context_schedule) - 1,
        )
        return int(self.config.context_schedule[block])

    def mode_for_context(self, context: int) -> int:
        return int(self.config.true_modes[int(context)])

    def gate_for_context(self, context: int) -> int:
        return int(self.config.gates[int(context)])

    def encode(self, state: DecodedState) -> int:
        signal_code = int(state.signal) + 1
        tv_code = int(state.tv_symbol) + 1
        values = (
            int(state.context),
            int(state.row),
            int(state.column),
            int(state.probe_used),
            signal_code,
            tv_code,
            int(state.elapsed),
        )
        return int(np.ravel_multi_index(values, self._dims))

    def decode(self, observation: int) -> DecodedState:
        values = np.unravel_index(int(observation), self._dims)
        return DecodedState(
            context=int(values[0]),
            row=int(values[1]),
            column=int(values[2]),
            probe_used=int(values[3]),
            signal=int(values[4]) - 1,
            tv_symbol=int(values[5]) - 1,
            elapsed=int(values[6]),
        )

    def _state(self) -> DecodedState:
        return DecodedState(
            context=self.context,
            row=self.row,
            column=self.column,
            probe_used=self.probe_used,
            signal=self.signal,
            tv_symbol=self.tv_symbol,
            elapsed=self.elapsed,
        )

    def _observation(self) -> int:
        return self.encode(self._state())

    def at_gate(self, state: DecodedState | None = None) -> bool:
        s = state or self._state()
        return s.row == self.gate_for_context(s.context) and s.column == s.row

    def valid_action_mask(self, observation: int | None = None) -> np.ndarray:
        state = self.decode(observation) if observation is not None else self._state()
        mask = np.ones(self.action_space.n, dtype=np.int8)
        if not self.at_gate(state):
            mask[PROBE] = 0
            mask[CHALLENGE] = 0
        elif state.probe_used:
            mask[PROBE] = 0
        # Legacy: one nuisance branch. Refined Undertow instances expose
        # a mode-independent sonar/drift action at every depth; each use
        # sacrifices horizontal progress and cannot reveal the task mode.
        if not self.config.distributed_nuisance and not (state.row == 1 and state.column == state.row):
            mask[TV] = 0
        return mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Tuple[int, Dict[str, Any]]:
        try:
            super().reset(seed=seed)
        except TypeError:  # compatibility layer
            super().reset(seed=seed, options=options)
        if seed is not None and hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        options = dict(options or {})
        if "episode_index" in options:
            self.episode_index = int(options["episode_index"])
        self.context = int(options.get("context", self.context_for_episode(self.episode_index)))
        self.row = 0
        self.column = 0
        self.probe_used = 0
        self.signal = -1
        self.tv_symbol = -1
        self.elapsed = 0
        self._terminated = False
        self._episode_return = 0.0
        observation = self._observation()
        info = self._base_info()
        info["action_mask"] = self.valid_action_mask(observation)
        return observation, info

    def _base_info(self) -> Dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "context": int(self.context),
            "gate": int(self.gate_for_context(self.context)),
            "mode": int(self.mode_for_context(self.context)),
            "row": int(self.row),
            "column": int(self.column),
            "elapsed": int(self.elapsed),
        }

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("step() called after termination; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        action = int(action)
        mode = self.mode_for_context(self.context)
        was_at_gate = self.at_gate()
        diagnostic_event = "none"
        diagnostic_outcome: int | None = None
        valid = bool(self.valid_action_mask()[action])
        reward = 0.0

        self.tv_symbol = -1
        if action == PROBE and valid:
            diagnostic_event = "probe"
            p = self.config.probe_probability(mode)
            diagnostic_outcome = int(self.np_random.random() < p)
            self.signal = diagnostic_outcome
            self.probe_used = 1
            self.elapsed += 1
            reward -= self.config.probe_cost
        else:
            self.elapsed += 1
            if action == RIGHT:
                self.row += 1
                self.column = min(self.config.size, self.column + 1)
                reward -= self.config.right_cost
            elif action == CHALLENGE and valid and was_at_gate:
                diagnostic_event = "challenge"
                p = self.config.challenge_probability(mode)
                diagnostic_outcome = int(self.np_random.random() < p)
                self.row += 1
                self.column = min(self.config.size, self.column + 1)
                reward -= self.config.right_cost
                reward += self.config.challenge_scale * (2 * diagnostic_outcome - 1)
            elif action == TV:
                diagnostic_event = "tv"
                diagnostic_outcome = int(self.np_random.integers(self.config.tv_bins))
                self.tv_symbol = diagnostic_outcome
                self.row += 1
                reward -= self.config.tv_cost
            else:
                # LEFT or an invalid diagnostic action advances the clock but
                # loses horizontal progress, preserving the DeepSea pathology.
                self.row += 1
                self.column = max(0, self.column - 1)
                if action in (PROBE, CHALLENGE):
                    reward -= self.config.invalid_cost

        terminated = self.row >= self.config.size
        truncated = self.elapsed >= self.config.max_steps and not terminated
        if terminated and self.column >= self.config.size:
            reward += self.config.terminal_reward
        self._terminated = bool(terminated or truncated)
        self._episode_return += reward

        observation = self._observation()
        info = self._base_info()
        info.update(
            {
                "action_name": ACTION_NAMES[action],
                "valid_action": valid,
                "diagnostic_event": diagnostic_event,
                "diagnostic_outcome": diagnostic_outcome,
                "at_gate_before_action": was_at_gate,
                "terminal_success": bool(terminated and self.column >= self.config.size),
                "episode_return": float(self._episode_return),
                "action_mask": self.valid_action_mask(observation) if not self._terminated else np.ones(5, dtype=np.int8),
            }
        )
        if self._terminated:
            self.episode_index += 1
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> str:
        marker = ["."] * (self.config.size + 1)
        marker[min(self.column, self.config.size)] = "A"
        return (
            f"context={self.context} mode={self.mode_for_context(self.context)} "
            f"row={self.row} gate={self.gate_for_context(self.context)} "
            f"[{''.join(marker)}] signal={self.signal}"
        )


def policy_action(policy_name: str, state: DecodedState, cfg: DeepSeaConfig) -> int:
    """Action for one of the fixed complete-policy experiments."""
    gate = cfg.gates[state.context]
    at_gate = state.row == gate and state.column == state.row
    if policy_name == "exploit":
        return RIGHT
    if policy_name == "challenge":
        return CHALLENGE if at_gate else RIGHT
    if policy_name == "probe":
        if at_gate and not state.probe_used:
            return PROBE
        return RIGHT
    if policy_name == "probe-adaptive":
        if at_gate and not state.probe_used:
            return PROBE
        if at_gate and state.probe_used and state.signal == 1:
            return CHALLENGE
        return RIGHT
    if policy_name == "tv":
        if state.row == 1 and state.column == state.row:
            return TV
        return RIGHT
    raise KeyError(f"unknown policy {policy_name}")


def event_log_probability(
    cfg: DeepSeaConfig,
    mode: int,
    event: str,
    outcome: int | None,
) -> float:
    """Log likelihood of a model-dependent event under one context mode."""
    if event == "probe":
        if outcome not in (0, 1):
            raise ValueError("probe outcome must be binary")
        p = cfg.probe_probability(mode)
        return math.log(p if outcome == 1 else 1.0 - p)
    if event == "challenge":
        if outcome not in (0, 1):
            raise ValueError("challenge outcome must be binary")
        p = cfg.challenge_probability(mode)
        return math.log(p if outcome == 1 else 1.0 - p)
    # Navigation and TV have the same law under both candidate modes.
    return 0.0


def transition_log_likelihood_ratio(
    cfg: DeepSeaConfig,
    reference_mode: int,
    challenger_mode: int,
    event: str,
    outcome: int | None,
) -> float:
    return event_log_probability(cfg, reference_mode, event, outcome) - event_log_probability(
        cfg, challenger_mode, event, outcome
    )


def bernoulli_kl(p: float, q: float) -> float:
    eps = 1e-12
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def policy_signature_distribution(
    cfg: DeepSeaConfig,
    policy_name: str,
    mode: int,
) -> Dict[Tuple[Any, ...], float]:
    """Exact model-dependent observation law for a fixed complete policy."""
    if policy_name == "exploit":
        return {("none",): 1.0}
    if policy_name == "tv":
        return {("tv", symbol): 1.0 / cfg.tv_bins for symbol in range(cfg.tv_bins)}
    if policy_name == "probe":
        p = cfg.probe_probability(mode)
        return {("probe", 0): 1.0 - p, ("probe", 1): p}
    if policy_name == "challenge":
        p = cfg.challenge_probability(mode)
        return {("challenge", 0): 1.0 - p, ("challenge", 1): p}
    if policy_name == "probe-adaptive":
        p_probe = cfg.probe_probability(mode)
        p_challenge = cfg.challenge_probability(mode)
        return {
            ("probe", 0): 1.0 - p_probe,
            ("probe", 1, "challenge", 0): p_probe * (1.0 - p_challenge),
            ("probe", 1, "challenge", 1): p_probe * p_challenge,
        }
    raise KeyError(policy_name)


def distribution_kl(p: Mapping[Tuple[Any, ...], float], q: Mapping[Tuple[Any, ...], float]) -> float:
    value = 0.0
    for key in set(p) | set(q):
        pk = float(p.get(key, 0.0))
        qk = float(q.get(key, 0.0))
        if pk <= 0.0:
            continue
        if qk <= 0.0:
            return float("inf")
        value += pk * math.log(pk / qk)
    return value


def policy_information(cfg: DeepSeaConfig, policy_name: str, reference_mode: int) -> float:
    challenger = 1 - int(reference_mode)
    return distribution_kl(
        policy_signature_distribution(cfg, policy_name, reference_mode),
        policy_signature_distribution(cfg, policy_name, challenger),
    )


def expected_policy_return(cfg: DeepSeaConfig, policy_name: str, mode: int) -> float:
    """Expected episode return for a coherent fixed policy.

    All fixed policies except TV traverse the rightward corridor.  This closed
    form is used by finite-model CEAC and IDS; the realized experiment still
    executes every policy through ``env.step``.
    """
    base = cfg.terminal_reward - cfg.size * cfg.right_cost
    challenge_bonus = cfg.challenge_scale * (2.0 * cfg.challenge_probability(mode) - 1.0)
    if policy_name == "exploit":
        return base
    if policy_name == "challenge":
        return base + challenge_bonus
    if policy_name == "probe":
        return base - cfg.probe_cost
    if policy_name == "probe-adaptive":
        return base - cfg.probe_cost + cfg.probe_probability(mode) * challenge_bonus
    if policy_name == "tv":
        return -cfg.tv_cost - (cfg.size - 1) * cfg.right_cost
    raise KeyError(policy_name)


def optimal_policy_name(cfg: DeepSeaConfig, mode: int) -> str:
    names = [spec.name for spec in POLICY_SPECS if spec.name != "tv"]
    return max(names, key=lambda name: expected_policy_return(cfg, name, mode))


def optimal_expected_return(cfg: DeepSeaConfig, mode: int) -> float:
    return expected_policy_return(cfg, optimal_policy_name(cfg, mode), mode)


def policy_gap(cfg: DeepSeaConfig, policy_name: str, mode: int) -> float:
    return max(0.0, optimal_expected_return(cfg, mode) - expected_policy_return(cfg, policy_name, mode))


def mutual_information_binary(
    prior_mode1: float,
    distribution0: Mapping[Tuple[Any, ...], float],
    distribution1: Mapping[Tuple[Any, ...], float],
) -> float:
    p1 = float(np.clip(prior_mode1, 1e-12, 1.0 - 1e-12))
    p0 = 1.0 - p1
    mixture: Dict[Tuple[Any, ...], float] = {}
    for key in set(distribution0) | set(distribution1):
        mixture[key] = p0 * distribution0.get(key, 0.0) + p1 * distribution1.get(key, 0.0)
    return p0 * distribution_kl(distribution0, mixture) + p1 * distribution_kl(distribution1, mixture)


def trajectory_signature(transition_infos: Sequence[Mapping[str, Any]]) -> Tuple[Any, ...]:
    events: List[Any] = []
    for info in transition_infos:
        event = str(info.get("diagnostic_event", "none"))
        outcome = info.get("diagnostic_outcome")
        if event == "probe":
            events.extend(("probe", int(outcome)))
        elif event == "challenge":
            events.extend(("challenge", int(outcome)))
        elif event == "tv":
            events.extend(("tv", int(outcome)))
    return tuple(events) if events else ("none",)


def trajectory_log_likelihood(
    cfg: DeepSeaConfig,
    mode: int,
    transition_infos: Sequence[Mapping[str, Any]],
) -> float:
    return float(
        sum(
            event_log_probability(
                cfg,
                mode,
                str(info.get("diagnostic_event", "none")),
                info.get("diagnostic_outcome"),
            )
            for info in transition_infos
        )
    )


__all__ = [
    "ACTION_NAMES",
    "CHALLENGE",
    "DeepSeaConfig",
    "DecodedState",
    "LEFT",
    "POLICY_INDEX",
    "POLICY_SPECS",
    "PROBE",
    "PolicySpec",
    "RIGHT",
    "ShiftingDeepSeaEnv",
    "TV",
    "distribution_kl",
    "event_log_probability",
    "expected_policy_return",
    "mutual_information_binary",
    "optimal_expected_return",
    "optimal_policy_name",
    "policy_action",
    "policy_gap",
    "policy_information",
    "policy_signature_distribution",
    "trajectory_log_likelihood",
    "trajectory_signature",
    "transition_log_likelihood_ratio",
]
