"""Sequential combination locks and epistatic signalling discovery environments.

Two controlled pathology families:
1. LockEnv (`combination_lock`, `bidirectional_lock`): sweeps lock depth L >= 1
   (physical horizon H = L + 1) while holding useful route returns, decision gap,
   and diagnostic KL strictly invariant across depth. At L = 1 both variants
   share the calibrated two-step easy baseline with uniform log(2) occupancy entropy.
2. EpistasisEnv (`epistatic_signalling`): models discovering a rare synthetic-rescue
   epistatic interaction (1 of G genetic subtypes) amidst k off-pathway pleiotropic
   modules (eta in {0,1}^k, inducing both high-MI zero-value passenger readouts
   and 4^k - 1 aberrant pleiotropic cell states). At k = 0 all passenger/pleiotropic
   distractions vanish, leaving a calibrated two-state baseline.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


@dataclass(frozen=True)
class LockConfig:
    kind: str = "combination_lock"
    depth: int = 1
    safe_reward: float = 0.50
    probe_reward: float = 0.48
    target_probabilities: tuple[float, float] = (0.10, 0.90)
    probe_probabilities: tuple[float, float] = (0.15, 0.85)
    step_cost: float = 0.004
    sink_lanes: int = 3

    def __post_init__(self):
        if self.kind not in {"combination_lock", "bidirectional_lock"}:
            raise ValueError(f"Unknown lock kind: {self.kind}")
        if int(self.depth) != self.depth or self.depth < 1:
            raise ValueError("Lock depth must be a positive integer")
        if int(self.sink_lanes) != self.sink_lanes or self.sink_lanes < 2:
            raise ValueError("Sink manifold requires at least two lanes")
        if not 0.0 < self.probe_reward < self.safe_reward < 1.0:
            raise ValueError("Invalid lock route rewards")
        if self.step_cost < 0.0 or self.step_cost * max(0, self.depth - 1) >= 0.10:
            raise ValueError("Invalid anti-shaped step cost")
        if any(not 0.0 < p < 1.0 for p in self.target_probabilities + self.probe_probabilities):
            raise ValueError("Common positive outcome support required")

    @property
    def horizon(self) -> int:
        return self.depth + 1


class LockEnv:
    """Sequential gated lock or bidirectional (decoy vs target) combination lock."""
    TERMINAL = 10000

    def __init__(self, config: LockConfig | None = None, task: int = 0):
        self.config = config or LockConfig()
        if task not in (0, 1):
            raise ValueError("Task must be 0 or 1")
        self.task = int(task)
        self.action_space = SimpleNamespace(n=4)
        self._build_states()
        self.np_random = np.random.default_rng()
        self.observation = 0
        self.terminated = False

    def _build_states(self):
        c = self.config
        L = c.depth
        descriptors: list[tuple[int, str, int]] = []
        if L == 1:
            descriptors.append((0, "start", 0))
            descriptors.append((1, "gate", 0))
        elif c.kind == "combination_lock":
            descriptors.append((0, "live", 0))
            for t in range(1, L):
                descriptors.append((t, "live", 0))
                for b in range(c.sink_lanes):
                    descriptors.append((t, "sink", b))
            descriptors.append((L, "gate", 0))
            for b in range(c.sink_lanes):
                descriptors.append((L, "sink", b))
        else:
            descriptors.append((0, "start", 0))
            for t in range(1, L):
                descriptors.append((t, "left", 0))
                descriptors.append((t, "right", 0))
                for b in range(c.sink_lanes):
                    descriptors.append((t, "sink", b))
            descriptors.append((L, "left_gate", 0))
            descriptors.append((L, "right_gate", 0))
            for b in range(c.sink_lanes):
                descriptors.append((L, "sink", b))
        self._descriptors = tuple(descriptors)
        self._code_of = {desc: idx for idx, desc in enumerate(self._descriptors)}
        self.states = tuple(range(len(self._descriptors)))

    def describe(self, observation: int) -> tuple[int, str, int]:
        return self._descriptors[int(observation)]

    def elapsed(self, observation: int) -> int:
        return self._descriptors[int(observation)][0]

    def valid_action_mask(self, observation: int) -> np.ndarray:
        if observation not in self.states:
            raise ValueError("Not an active lock state")
        t, role, _ = self._descriptors[observation]
        mask = np.zeros(4, dtype=bool)
        if t < self.config.depth and role in {"start", "live", "left", "right"}:
            mask[:] = True
        else:
            mask[0] = True
        return mask

    def _sink_diffusion(self, next_t: int) -> dict[tuple[int, float, bool], float]:
        lanes = self.config.sink_lanes
        w = 1.0 / lanes
        return {(self._code_of[(next_t, "sink", b)], 0.0, False): w for b in range(lanes)}

    def _terminal_chamber_law(self, gate_code: int, action: int, rebate: float, next_t: int):
        c = self.config
        if action == 0:
            return {(gate_code, c.safe_reward + rebate, False): 1.0}
        if action == 1:
            p = c.target_probabilities[self.task]
            return {(gate_code, 0.0 + rebate, False): 1.0 - p,
                    (gate_code, 1.0 + rebate, False): p}
        if action == 2:
            q = c.probe_probabilities[self.task]
            return {(gate_code, round(c.probe_reward - 0.02 + rebate, 10), False): (1.0 - q) / 2.0,
                    (gate_code, round(c.probe_reward + rebate, 10), False): q,
                    (gate_code, round(c.probe_reward + 0.02 + rebate, 10), False): (1.0 - q) / 2.0}
        if c.depth == 1:
            return {(gate_code, 0.0, False): 1.0}
        return self._sink_diffusion(next_t)

    def law(self, observation: int, action: int) -> dict[tuple[int, float, bool], float]:
        if not self.valid_action_mask(observation)[action]:
            raise ValueError("Masked action")
        c = self.config
        L = c.depth
        t, role, _ = self._descriptors[observation]
        if t == L:
            return {(self.TERMINAL, 0.0, True): 1.0}
        if L == 1:
            gate = self._code_of[(1, "gate", 0)]
            return self._terminal_chamber_law(gate, action, 0.0, 1)
        if role == "sink":
            return self._sink_diffusion(t + 1)
        rebate = round((L - 1) * c.step_cost, 10)
        if c.kind == "combination_lock":
            if t < L - 1:
                key = t % 2
                if action == key:
                    return {(self._code_of[(t + 1, "live", 0)], -c.step_cost, False): 1.0}
                wrong = [a for a in range(4) if a != key]
                lane = wrong.index(action) % c.sink_lanes
                return {(self._code_of[(t + 1, "sink", lane)], 0.0, False): 1.0}
            gate = self._code_of[(L, "gate", 0)]
            return self._terminal_chamber_law(gate, action, rebate, L)
        # Bidirectional combination lock (L >= 2)
        if role == "start":
            if action == 0:
                return {(self._code_of[(1, "left", 0)], -c.step_cost, False): 1.0}
            if action == 1:
                return {(self._code_of[(1, "right", 0)], -c.step_cost, False): 1.0}
            return self._sink_diffusion(1)
        if role == "left":
            if t < L - 1:
                if action == 0:
                    return {(self._code_of[(t + 1, "left", 0)], -c.step_cost, False): 1.0}
                lane = (action - 1) % c.sink_lanes
                return {(self._code_of[(t + 1, "sink", lane)], 0.0, False): 1.0}
            if action == 0:
                gate = self._code_of[(L, "left_gate", 0)]
                return {(gate, round(c.safe_reward + rebate, 10), False): 1.0}
            lane = (action - 1) % c.sink_lanes
            return {(self._code_of[(L, "sink", lane)], 0.0, False): 1.0}
        if role == "right":
            if t < L - 1:
                if action == 1:
                    return {(self._code_of[(t + 1, "right", 0)], -c.step_cost, False): 1.0}
                wrong = [0, 2, 3]
                lane = wrong.index(action) % c.sink_lanes
                return {(self._code_of[(t + 1, "sink", lane)], 0.0, False): 1.0}
            right_gate = self._code_of[(L, "right_gate", 0)]
            if action in (1, 2):
                return self._terminal_chamber_law(right_gate, action, rebate, L)
            return self._sink_diffusion(L)
        raise AssertionError("Unreachable lock state")

    def reset(self, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.observation = 0
        self.terminated = False
        return 0, {}

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Reset before stepping a terminal lock episode")
        prev = self.observation
        action = int(action)
        law = self.law(prev, action)
        outcomes = list(law)
        chosen = outcomes[int(self.np_random.choice(len(outcomes), p=list(law.values())))]
        nxt, reward, done = chosen
        self.observation = nxt
        self.terminated = done
        t, role, _ = self._descriptors[prev]
        L = self.config.depth
        event = "step"
        if done:
            event = "finish"
        elif nxt != self.TERMINAL and self._descriptors[nxt][1] == "sink":
            event = "sink"
        elif L == 1 and t == 0:
            event = ("safe", "target", "probe", "sink")[action]
        elif t == L - 1:
            if role == "left" and action == 0:
                event = "decoy_unlock"
            elif role in {"live", "right"}:
                event = {0: "safe", 1: "target", 2: "probe"}.get(action, "sink")
        return nxt, reward, done, False, {"event": event, "physical_steps": 1}


@dataclass(frozen=True)
class EpistasisConfig:
    kind: str = "epistatic_signalling"
    pleiotropy: int = 0
    subtypes: int = 128
    safe_reward: float = 0.50
    synergy_probabilities: tuple[float, float] = (0.10, 0.90)
    probe_reward: float = 0.48
    probe_probabilities: tuple[float, float] = (0.15, 0.85)
    passenger_reward: float = 0.4999
    passenger_signal_halfwidth: float = 0.022

    def __post_init__(self):
        if self.kind != "epistatic_signalling":
            raise ValueError(f"Unknown epistasis kind: {self.kind}")
        if int(self.pleiotropy) != self.pleiotropy or not 0 <= self.pleiotropy <= 5:
            raise ValueError("Pleiotropy module count k must lie in {0..5}")
        if int(self.subtypes) != self.subtypes or self.subtypes < 2:
            raise ValueError("Requires at least two genetic subtypes")
        if not 0.0 < self.probe_reward < self.passenger_reward < self.safe_reward < 1.0:
            raise ValueError("Invalid signalling assay reward hierarchy")
        if not 0.0 < self.passenger_signal_halfwidth < 0.45:
            raise ValueError("Invalid passenger signal halfwidth")
        if any(not 0.0 < p < 1.0 for p in self.synergy_probabilities + self.probe_probabilities):
            raise ValueError("Common positive outcome support required")

    @property
    def pleiotropic_branches(self) -> int:
        return (4 ** self.pleiotropy) - 1

    @property
    def passenger_states(self) -> int:
        return 1 << self.pleiotropy

    @property
    def horizon(self) -> int:
        return 2


class EpistasisEnv:
    """Two-step cellular signalling & rare genetic epistasis discovery environment.

    Actions at state 0 (cell baseline):
      0: cytostatic_care (deterministic safe_reward = 0.50)
      1: epistatic_synergy (K1 & K2 co-activation: Bern(0.90) in rare responder
         subtype G-1, Bern(0.10) in all G-1 canalized subtypes 0..G-2)
      2: phospho_probe (targeted pathway assay: mean 0.48, subtle causal readout)
      3 .. 3+k-1: passenger_omics_i (mean 0.4999 across ALL models; symmetric
         ternary readout of passenger bit eta_i with zero causal information)
      3+k .. 3+k+(4^k-1)-1: pleiotropic_ko_j (zero reward, zero KL across all
         subtypes due to canalization, transitions to aberrant state 2+j)
    """
    TERMINAL = 10000

    def __init__(self, config: EpistasisConfig | None = None, subtype: int = 0, passenger: int = 0):
        self.config = config or EpistasisConfig()
        c = self.config
        if not 0 <= int(subtype) < c.subtypes or not 0 <= int(passenger) < c.passenger_states:
            raise ValueError("Invalid realized (subtype, passenger) genotype")
        self.subtype = int(subtype)
        self.passenger = int(passenger)
        self.responder = int(self.subtype == c.subtypes - 1)
        n_actions = 3 + c.pleiotropy + c.pleiotropic_branches
        self.action_space = SimpleNamespace(n=n_actions)
        self.states = tuple(range(2 + c.pleiotropic_branches))
        self.np_random = np.random.default_rng()
        self.observation = 0
        self.terminated = False

    def elapsed(self, observation: int) -> int:
        return int(observation != 0)

    def valid_action_mask(self, observation: int) -> np.ndarray:
        if observation not in self.states:
            raise ValueError("Not an active cellular state")
        mask = np.zeros(self.action_space.n, dtype=bool)
        if observation == 0:
            mask[:] = True
        else:
            mask[0] = True
        return mask

    def law(self, observation: int, action: int) -> dict[tuple[int, float, bool], float]:
        if not self.valid_action_mask(observation)[action]:
            raise ValueError("Masked action")
        c = self.config
        if observation > 0:
            return {(self.TERMINAL, 0.0, True): 1.0}
        if action == 0:
            return {(1, c.safe_reward, False): 1.0}
        if action == 1:
            p = c.synergy_probabilities[self.responder]
            return {(1, 0.0, False): 1.0 - p, (1, 1.0, False): p}
        if action == 2:
            q = c.probe_probabilities[self.responder]
            return {(1, round(c.probe_reward - 0.02, 10), False): (1.0 - q) / 2.0,
                    (1, c.probe_reward, False): q,
                    (1, round(c.probe_reward + 0.02, 10), False): (1.0 - q) / 2.0}
        if action < 3 + c.pleiotropy:
            mod = action - 3
            bit = (self.passenger >> mod) & 1
            d = c.passenger_signal_halfwidth
            q = 0.5 + d if bit == 1 else 0.5 - d
            return {(1, round(c.passenger_reward - 0.02, 10), False): (1.0 - q) / 2.0,
                    (1, c.passenger_reward, False): q,
                    (1, round(c.passenger_reward + 0.02, 10), False): (1.0 - q) / 2.0}
        branch = action - (3 + c.pleiotropy)
        return {(2 + branch, 0.0, False): 1.0}

    def reset(self, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.observation = 0
        self.terminated = False
        return 0, {}

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Reset before stepping a terminal cellular episode")
        prev = self.observation
        action = int(action)
        law = self.law(prev, action)
        outcomes = list(law)
        chosen = outcomes[int(self.np_random.choice(len(outcomes), p=list(law.values())))]
        nxt, reward, done = chosen
        self.observation = nxt
        self.terminated = done
        c = self.config
        if prev > 0:
            event = "finish"
        elif action == 0:
            event = "safe"
        elif action == 1:
            event = "synergy"
        elif action == 2:
            event = "probe"
        elif action < 3 + c.pleiotropy:
            event = "passenger_omics"
        else:
            event = "pleiotropic"
        return nxt, reward, done, False, {"event": event, "physical_steps": 1}
