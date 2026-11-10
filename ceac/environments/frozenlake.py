#!/usr/bin/env python3
"""Gym-compatible Shifting-Shore FrozenLake benchmark.

The benchmark isolates a frequentist evidence-purchasing pathology.

There are two latent shoreline modes:

* mode 0 (normal shore): the inner bridge is optimal;
* mode 1 (shifted shore): the inner bridge has *the same trajectory law*, but
  an outer causeway becomes strictly better.

Hence mode 1 is a policy-changing confounder for the deployed inner-bridge
policy: exploiting the normal-shore optimum supplies no evidence that the
shoreline has shifted.  A deep survey or an outer-causeway traversal is needed
to distinguish the two modes.

The environment exposes a standard ``reset``/``step`` interface.  Gymnasium is
optional; a small compatibility layer is provided for repository tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

try:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
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
            self.n = int(n)
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
DOWN = 1
RIGHT = 2
UP = 3
ACTION_NAMES = {LEFT: "left", DOWN: "down", RIGHT: "right", UP: "up"}


@dataclass(frozen=True)
class FrozenShoreConfig:
    rows: int = 7
    cols: int = 9
    block_length: int = 400
    context_schedule: Tuple[int, ...] = (0, 1, 2, 0, 1, 2)
    true_modes: Tuple[int, ...] = (0, 1, 0)
    step_cost: float = 0.001
    safe_reward: float = 0.700
    inner_reward: float = 1.000
    outer_reward: float = 1.200
    inner_success_normal: float = 0.930
    inner_success_shifted: float = 0.930
    outer_success_normal: float = 0.550
    outer_success_shifted: float = 0.950
    deep_signal_normal: float = 0.150
    deep_signal_shifted: float = 0.850
    shallow_signal_probability: float = 0.500
    shallow_fee: float = 0.015
    deep_fee: float = 0.040
    reeds_fee: float = 0.005
    reed_bins: int = 12
    invalid_cost: float = 0.020
    nuisance_probability: float | None = None

    def __post_init__(self) -> None:
        if self.nuisance_probability is not None and not 0 < self.nuisance_probability < 1:
            raise ValueError("Physical nuisance probability must be strictly between zero and one")
        if len(self.true_modes) != 3:
            raise ValueError("the default benchmark expects three visible shore contexts")
        if any(mode not in (0, 1) for mode in self.true_modes):
            raise ValueError("true_modes must be binary")
        if any(context < 0 or context >= len(self.true_modes) for context in self.context_schedule):
            raise ValueError("context_schedule contains an invalid context")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")

    @property
    def n_contexts(self) -> int:
        return len(self.true_modes)

    @property
    def total_episodes(self) -> int:
        return self.block_length * len(self.context_schedule)

    @property
    def max_steps(self) -> int:
        return 16

    def inner_success(self, mode: int) -> float:
        return self.inner_success_shifted if int(mode) == 1 else self.inner_success_normal

    def outer_success(self, mode: int) -> float:
        return self.outer_success_shifted if int(mode) == 1 else self.outer_success_normal

    def deep_signal_probability(self, mode: int) -> float:
        return self.deep_signal_shifted if int(mode) == 1 else self.deep_signal_normal


@dataclass(frozen=True)
class DecodedState:
    context: int
    row: int
    col: int
    shallow_signal: int  # -1, 0, 1
    deep_signal: int  # -1, 0, 1
    reed_symbol: int  # -1, ..., reed_bins - 1
    elapsed: int


@dataclass(frozen=True)
class PolicySpec:
    name: str
    actions: Tuple[int, ...]


# Deterministic complete-policy routes.  The route, rather than a single
# primitive action, is the experimental unit in the Graves--Lai program.
POLICY_SPECS: Tuple[PolicySpec, ...] = (
    PolicySpec("shore", (RIGHT,) * 8),
    PolicySpec("inner-bridge", (RIGHT,) * 3 + (UP,) * 3 + (RIGHT,) * 5),
    PolicySpec("outer-causeway", (RIGHT,) * 6 + (UP,) * 5 + (RIGHT,) * 2),
    PolicySpec("shallow-survey", (UP,) + (RIGHT,) * 2 + (DOWN,) + (RIGHT,) * 6),
    PolicySpec("deep-survey", (DOWN,) + (RIGHT,) * 4 + (UP,) + (RIGHT,) * 4),
    PolicySpec("reed-scan", (RIGHT,) * 3 + (UP,) * 2 + (RIGHT,) + (DOWN,) * 2 + (RIGHT,) * 4),
)
POLICY_ACTIONS = {spec.name: spec.actions for spec in POLICY_SPECS}
POLICY_NAMES = tuple(spec.name for spec in POLICY_SPECS)
POLICY_INDEX = {name: index for index, name in enumerate(POLICY_NAMES)}

START = (5, 0)
SAFE_DOCK = (5, 8)
INNER_DOCK = (2, 8)
OUTER_DOCK = (0, 8)
SHALLOW_BUOY = (4, 2)
DEEP_BUOY = (6, 4)
REEDS = (3, 4)

# Open corridors are the union of all six policy routes.  Invalid moves leave
# the agent in place and charge a small cost; fixed policies never use them.
def _route_cells(actions: Sequence[int]) -> set[Tuple[int, int]]:
    row, col = START
    cells = {(row, col)}
    for action in actions:
        if action == LEFT:
            col -= 1
        elif action == RIGHT:
            col += 1
        elif action == UP:
            row -= 1
        elif action == DOWN:
            row += 1
        cells.add((row, col))
    return cells


OPEN_CELLS: frozenset[Tuple[int, int]] = frozenset(
    set().union(*(_route_cells(spec.actions) for spec in POLICY_SPECS))
)


class ShiftingShoreFrozenLakeEnv(gym.Env):
    """FrozenLake-style grid with a policy-changing shifted-shore mode."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: FrozenShoreConfig | None = None) -> None:
        super().__init__()
        self.config = config or FrozenShoreConfig()
        self.action_space = spaces.Discrete(4)
        self._dims = (
            self.config.n_contexts,
            self.config.rows,
            self.config.cols,
            3,
            3,
            self.config.reed_bins * (3 if self.config.nuisance_probability is not None else 1) + 1,
            self.config.max_steps + 1,
        )
        self._n_states = int(np.prod(self._dims, dtype=np.int64))
        self.observation_space = spaces.Discrete(self._n_states)
        self.episode_index = 0
        self.context = 0
        self.mode = 0
        self.row, self.col = START
        self.shallow_signal = -1
        self.deep_signal = -1
        self.reed_symbol = -1
        self.elapsed = 0
        self._terminated = False

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

    def encode(self, state: DecodedState) -> int:
        values = (
            int(state.context),
            int(state.row),
            int(state.col),
            int(state.shallow_signal) + 1,
            int(state.deep_signal) + 1,
            int(state.reed_symbol) + 1,
            int(state.elapsed),
        )
        return int(np.ravel_multi_index(values, self._dims))

    def decode(self, observation: int) -> DecodedState:
        values = np.unravel_index(int(observation), self._dims)
        return DecodedState(
            context=int(values[0]),
            row=int(values[1]),
            col=int(values[2]),
            shallow_signal=int(values[3]) - 1,
            deep_signal=int(values[4]) - 1,
            reed_symbol=int(values[5]) - 1,
            elapsed=int(values[6]),
        )

    def _state(self) -> DecodedState:
        return DecodedState(
            context=self.context,
            row=self.row,
            col=self.col,
            shallow_signal=self.shallow_signal,
            deep_signal=self.deep_signal,
            reed_symbol=self.reed_symbol,
            elapsed=self.elapsed,
        )

    def _observation(self) -> int:
        return self.encode(self._state())

    def valid_action_mask(self, observation: int | None = None) -> np.ndarray:
        state = self._state() if observation is None else self.decode(observation)
        mask = np.zeros(4, dtype=np.int8)
        if (self.config.nuisance_probability is not None
                and self.config.reed_bins <= state.reed_symbol < 2*self.config.reed_bins):
            mask[RIGHT] = 1  # The next physical step is the forced beacon dwell.
            return mask
        for action, (dr, dc) in {
            LEFT: (0, -1),
            DOWN: (1, 0),
            RIGHT: (0, 1),
            UP: (-1, 0),
        }.items():
            candidate = (state.row + dr, state.col + dc)
            if candidate in OPEN_CELLS:
                mask[action] = 1
        return mask

    def _base_info(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "mode": self.mode,
            "position": (self.row, self.col),
            "elapsed": self.elapsed,
            "action_mask": self.valid_action_mask(),
            "shallow_signal": self.shallow_signal,
            "deep_signal": self.deep_signal,
            "reed_symbol": self.reed_symbol,
        }

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
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = dict(options or {})
        self.episode_index = int(options.get("episode_index", self.episode_index))
        self.context = self.context_for_episode(self.episode_index)
        self.mode = self.mode_for_context(self.context)
        self.row, self.col = START
        self.shallow_signal = -1
        self.deep_signal = -1
        self.reed_symbol = -1
        self.elapsed = 0
        self._terminated = False
        info = self._base_info()
        info.update(
            {
                "shore_label": "shifted" if self.mode == 1 else "normal",
                "optimal_policy": optimal_policy_name(self.config, self.mode),
            }
        )
        return self._observation(), info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("step called after termination")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        action = int(action)
        action_mask_before = self.valid_action_mask()
        reward = -self.config.step_cost
        event = "none"
        outcome: int | None = None
        invalid = not bool(action_mask_before[action])
        dwell = (self.config.nuisance_probability is not None
                 and self.config.reed_bins <= self.reed_symbol < 2*self.config.reed_bins)
        activated = False
        if dwell:
            self.reed_symbol += self.config.reed_bins  # Retain that the dwell was completed.
            event = "beacon_dwell"
        elif invalid:
            reward -= self.config.invalid_cost
        else:
            if action == LEFT:
                self.col -= 1
            elif action == RIGHT:
                self.col += 1
            elif action == UP:
                self.row -= 1
            elif action == DOWN:
                self.row += 1

        self.elapsed += 1
        terminated = False
        truncated = self.elapsed >= self.config.max_steps

        position = (self.row, self.col)
        if position == SHALLOW_BUOY and self.shallow_signal < 0:
            event = "shallow"
            outcome = int(self.np_random.random() < self.config.shallow_signal_probability)
            self.shallow_signal = outcome
            reward -= self.config.shallow_fee
        elif position == DEEP_BUOY and self.deep_signal < 0:
            event = "deep"
            probability = self.config.deep_signal_probability(self.mode)
            outcome = int(self.np_random.random() < probability)
            self.deep_signal = outcome
            reward -= self.config.deep_fee
        elif position == REEDS and self.reed_symbol < 0:
            event = "reeds"
            n = self.config.reed_bins
            rho = self.config.nuisance_probability
            self.reed_symbol = (int(self.np_random.integers(n)) if rho is None else
                int(self.np_random.choice(2*n, p=np.r_[np.full(n,(1-rho)/n),np.full(n,rho/n)])))
            activated = rho is not None and self.reed_symbol >= n
            outcome = self.reed_symbol % n
            reward -= self.config.reeds_fee

        terminal_success = False
        if position == SAFE_DOCK:
            event = "shore"
            reward += self.config.safe_reward
            terminated = True
            terminal_success = True
        elif position == INNER_DOCK:
            event = "inner"
            probability = self.config.inner_success(self.mode)
            outcome = int(self.np_random.random() < probability)
            reward += self.config.inner_reward * outcome
            terminated = True
            terminal_success = bool(outcome)
        elif position == OUTER_DOCK:
            event = "outer"
            probability = self.config.outer_success(self.mode)
            outcome = int(self.np_random.random() < probability)
            reward += self.config.outer_reward * outcome
            terminated = True
            terminal_success = bool(outcome)

        self._terminated = bool(terminated or truncated)
        info = self._base_info()
        info.update(
            {
                "action_name": ACTION_NAMES[action],
                "action_mask_before": action_mask_before,
                "invalid_action": invalid,
                "diagnostic_event": event,
                "diagnostic_outcome": outcome,
                "nuisance_activated": activated,
                "nuisance_delay_step": dwell,
                "terminal_success": terminal_success,
                "shore_label": "shifted" if self.mode == 1 else "normal",
                "optimal_policy": optimal_policy_name(self.config, self.mode),
            }
        )
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> str:
        rows = []
        for row in range(self.config.rows):
            tokens = []
            for col in range(self.config.cols):
                position = (row, col)
                if position == (self.row, self.col):
                    token = "A"
                elif position == START:
                    token = "S"
                elif position == SAFE_DOCK:
                    token = "D"
                elif position == INNER_DOCK:
                    token = "I"
                elif position == OUTER_DOCK:
                    token = "O"
                elif position == SHALLOW_BUOY:
                    token = "q"
                elif position == DEEP_BUOY:
                    token = "Q"
                elif position == REEDS:
                    token = "R"
                elif position in OPEN_CELLS:
                    token = "."
                else:
                    token = "#"
                tokens.append(token)
            rows.append("".join(tokens))
        return "\n".join(rows)


def policy_action(policy_name: str, state: DecodedState, cfg: FrozenShoreConfig | None = None) -> int:
    actions = POLICY_ACTIONS[policy_name]
    completed_dwell = (cfg is not None and cfg.nuisance_probability is not None
                       and state.reed_symbol >= 2*cfg.reed_bins)
    route_index = state.elapsed - int(completed_dwell)
    if route_index >= len(actions):
        # The only way to exceed a fixed route is an invalid external action;
        # continue toward the safe dock rather than introducing hidden logic.
        return RIGHT
    return int(actions[route_index])


def bernoulli_kl(p: float, q: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    q = float(np.clip(q, 1e-12, 1.0 - 1e-12))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def event_log_probability(cfg: FrozenShoreConfig, mode: int, event: str, outcome: int | None) -> float:
    if event in {"none", "shore", "reeds", "shallow"}:
        if event == "shallow" and outcome is not None:
            p = cfg.shallow_signal_probability
            return math.log(p if int(outcome) == 1 else 1.0 - p)
        return 0.0
    if outcome is None:
        return 0.0
    if event == "deep":
        p = cfg.deep_signal_probability(mode)
    elif event == "inner":
        p = cfg.inner_success(mode)
    elif event == "outer":
        p = cfg.outer_success(mode)
    else:
        return 0.0
    return math.log(p if int(outcome) == 1 else 1.0 - p)


def trajectory_log_likelihood(
    cfg: FrozenShoreConfig,
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


def policy_signature_distribution(
    cfg: FrozenShoreConfig,
    policy_name: str,
    mode: int,
) -> Dict[Tuple[Any, ...], float]:
    if policy_name == "inner-bridge":
        p = cfg.inner_success(mode)
        return {("inner", 1): p, ("inner", 0): 1.0 - p}
    if policy_name == "outer-causeway":
        p = cfg.outer_success(mode)
        return {("outer", 1): p, ("outer", 0): 1.0 - p}
    if policy_name == "deep-survey":
        p = cfg.deep_signal_probability(mode)
        return {("deep", 1): p, ("deep", 0): 1.0 - p}
    if policy_name == "shallow-survey":
        p = cfg.shallow_signal_probability
        return {("shallow", 1): p, ("shallow", 0): 1.0 - p}
    if policy_name == "reed-scan":
        return {("reeds", symbol): 1.0 / cfg.reed_bins for symbol in range(cfg.reed_bins)}
    return {("shore",): 1.0}


def distribution_kl(
    p: Mapping[Tuple[Any, ...], float],
    q: Mapping[Tuple[Any, ...], float],
) -> float:
    value = 0.0
    for outcome, probability in p.items():
        if probability <= 0.0:
            continue
        denominator = float(q.get(outcome, 0.0))
        if denominator <= 0.0:
            return float("inf")
        value += float(probability) * math.log(float(probability) / denominator)
    return float(value)


def policy_information(
    cfg: FrozenShoreConfig,
    policy_name: str,
    reference_mode: int,
    alternative_mode: int,
) -> float:
    return distribution_kl(
        policy_signature_distribution(cfg, policy_name, reference_mode),
        policy_signature_distribution(cfg, policy_name, alternative_mode),
    )


def expected_policy_return(cfg: FrozenShoreConfig, policy_name: str, mode: int) -> float:
    length = len(POLICY_ACTIONS[policy_name])
    cost = cfg.step_cost * length
    if policy_name == "shore":
        return cfg.safe_reward - cost
    if policy_name == "inner-bridge":
        return cfg.inner_reward * cfg.inner_success(mode) - cost
    if policy_name == "outer-causeway":
        return cfg.outer_reward * cfg.outer_success(mode) - cost
    if policy_name == "shallow-survey":
        return cfg.safe_reward - cost - cfg.shallow_fee
    if policy_name == "deep-survey":
        return cfg.safe_reward - cost - cfg.deep_fee
    if policy_name == "reed-scan":
        return cfg.safe_reward - cost - cfg.reeds_fee
    raise KeyError(policy_name)


def optimal_policy_name(cfg: FrozenShoreConfig, mode: int) -> str:
    values = {name: expected_policy_return(cfg, name, mode) for name in POLICY_NAMES}
    return max(values, key=values.get)


def optimal_expected_return(cfg: FrozenShoreConfig, mode: int) -> float:
    return expected_policy_return(cfg, optimal_policy_name(cfg, mode), mode)


def policy_gap(cfg: FrozenShoreConfig, policy_name: str, mode: int) -> float:
    return optimal_expected_return(cfg, mode) - expected_policy_return(cfg, policy_name, mode)


def mutual_information(
    prior: np.ndarray,
    distributions: Sequence[Mapping[Tuple[Any, ...], float]],
) -> float:
    prior = np.asarray(prior, dtype=float)
    prior = prior / float(prior.sum())
    outcomes = set().union(*(distribution.keys() for distribution in distributions))
    mixture = {
        outcome: float(sum(prior[m] * distributions[m].get(outcome, 0.0) for m in range(len(prior))))
        for outcome in outcomes
    }
    value = 0.0
    for m, distribution in enumerate(distributions):
        for outcome, probability in distribution.items():
            if probability <= 0.0:
                continue
            denominator = mixture[outcome]
            if denominator > 0.0:
                value += prior[m] * probability * math.log(probability / denominator)
    return float(value)


__all__ = [
    "ACTION_NAMES",
    "DEEP_BUOY",
    "DecodedState",
    "FrozenShoreConfig",
    "INNER_DOCK",
    "LEFT",
    "DOWN",
    "RIGHT",
    "UP",
    "OPEN_CELLS",
    "OUTER_DOCK",
    "POLICY_ACTIONS",
    "POLICY_INDEX",
    "POLICY_NAMES",
    "POLICY_SPECS",
    "REEDS",
    "SAFE_DOCK",
    "SHALLOW_BUOY",
    "START",
    "ShiftingShoreFrozenLakeEnv",
    "distribution_kl",
    "event_log_probability",
    "expected_policy_return",
    "mutual_information",
    "optimal_expected_return",
    "optimal_policy_name",
    "policy_action",
    "policy_gap",
    "policy_information",
    "policy_signature_distribution",
    "trajectory_log_likelihood",
]
