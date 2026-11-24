"""Path-Dependent Undertow DeepSea & False Beacon Shore (FrozenLake) environments (H = 3).

Extends the original two-mode Undertow and False Beacon benchmarks with genuine
multi-step path dependence across complexity k in {0, 1, 2, 3}:

1. `undertow_pathdep` ("Path-Dependent Undertow DeepSea", H = 3):
   - Step-0 `dive_thermocline` pays -0.015 depth cost (0 immediate KL) to enter
     `deep_trench` (t = 1), unlocking R = 8 deep hydrothermal vent routes (0.93 vs 0.10)
     and `trench_sonar_ping` (0.27); unballasted plunges from the surface are swept
     away by the undertow (0.24, 0 KL), while `shelf_cruise` pays 0.50.
   - Sweeping k in {0, 1, 2, 3} adds:
     (a) 4k shallow acoustic/chemical echo plumes (0.49999) informative only about
         4k marginal shelf-seep vents (0.5004 vs 0.495) with zero KL on deep vents;
     (b) k current-vane drift sensors (0.49998) + (3^k - 1) off-path undertow eddies
         whose Step-1 trim rescue (0.08/0.20, total <= 0.26 < 0.50) depends on xi_{b mod k}.

2. `false_beacon_pathdep` ("Path-Dependent False Beacon Shore", H = 3):
   - Step-0 `stage_icebreaker` pays -0.015 staging cost (0 immediate KL) to reach
     `outer_soundings` (t = 1), unlocking R = 8 outer causeway channels (0.95 vs 0.09)
     and `deep_ice_core` (0.26); unreinforced crossings hit thin shore ice (0.23, 0 KL),
     while `inner_dock` pays 0.50.
   - Sweeping k in {0, 1, 2, 3} adds:
     (a) 4k shoreline false-beacon optical glints (0.49999) informative only about
         4k marginal coves (0.5004 vs 0.495) with zero KL on outer causeways;
     (b) k reed-bed wind buoys (0.49998) + (3^k - 1) off-path reed slush corridors
         whose Step-1 steering recovery depends on latent shear xi_{b mod k}.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from types import SimpleNamespace
import numpy as np
from ceac.contracts import Policy
from ceac_pathologies.model import DAGProductModel


@dataclass(frozen=True)
class _GridPathDepBaseConfig:
    kind: str = "undertow_pathdep"
    complexity: int = 0
    primary_channels: int = 8
    target_channel: int = 1
    safe_reward: float = 0.50
    stage_cost: float = 0.015
    unstaged_reward: float = 0.24
    channel_probabilities: tuple[float, float] = (0.10, 0.93)
    peripheral_probabilities: tuple[float, float] = (0.12, 0.5004)
    probe_reward: float = 0.27
    probe_target_prob: float = 0.36
    echo_subclades_per_module: int = 6
    echo_reward: float = 0.49999
    echo_halfwidth: float = 0.0115
    echo_prior_per_subclade: float = 0.048
    modifier_reward: float = 0.49998
    modifier_halfwidth: float = 0.0110
    responder_prior_mass: float = 0.0055

    def __post_init__(self):
        if self.kind not in {"undertow_pathdep", "false_beacon_pathdep"}:
            raise ValueError(f"Unknown grid pathdep kind: {self.kind}")
        if int(self.complexity) != self.complexity or not 0 <= self.complexity <= 3:
            raise ValueError("Complexity k must lie in {0..3}")
        if self.primary_channels < 2 or not 1 <= self.target_channel <= self.primary_channels:
            raise ValueError("Invalid primary_channels or target_channel")
        if not 0.0 < self.probe_reward < self.modifier_reward < self.echo_reward < self.safe_reward < 1.0:
            raise ValueError("Invalid grid pathdep reward ordering")

    @property
    def horizon(self) -> int:
        return 3

    @property
    def echo_count(self) -> int:
        return self.echo_subclades_per_module * self.complexity

    @property
    def total_channels(self) -> int:
        return self.primary_channels + self.echo_count

    @property
    def offpath_eddies(self) -> int:
        return (3 ** self.complexity) - 1

    @property
    def modifier_states(self) -> int:
        return 1 << self.complexity


@dataclass(frozen=True)
class PathDepUndertowConfig(_GridPathDepBaseConfig):
    kind: str = "undertow_pathdep"
    unstaged_reward: float = 0.24
    channel_probabilities: tuple[float, float] = (0.10, 0.93)
    probe_reward: float = 0.27
    probe_target_prob: float = 0.36


@dataclass(frozen=True)
class PathDepFalseBeaconConfig(_GridPathDepBaseConfig):
    kind: str = "false_beacon_pathdep"
    unstaged_reward: float = 0.23
    channel_probabilities: tuple[float, float] = (0.09, 0.95)
    probe_reward: float = 0.26
    probe_target_prob: float = 0.38


class _GridPathDepEnv:
    TERMINAL = 10000

    def __init__(self, config: _GridPathDepBaseConfig, channel: int = 0, modifier: int = 0):
        self.config = c = config
        if not 0 <= int(channel) <= c.total_channels or not 0 <= int(modifier) < c.modifier_states:
            raise ValueError("Invalid (channel, modifier) hypothesis")
        self.channel = int(channel)
        self.modifier = int(modifier)
        self._build_layout()
        self.np_random = np.random.default_rng()
        self.observation = 0
        self.terminated = False

    def _build_layout(self):
        c = self.config
        k = c.complexity
        n_echo = c.echo_count
        R_tot = c.total_channels
        self.n_root_actions = 3 + n_echo + k + c.offpath_eddies
        self.n_staged_actions = R_tot + 2
        n_actions = max(self.n_root_actions, self.n_staged_actions, 2)
        self.action_space = SimpleNamespace(n=n_actions)

        descriptors: list[tuple[int, str, int]] = [(0, "start", 0)]
        descriptors.append((1, "safe_dock", 0))
        descriptors.append((1, "staged", 0))
        descriptors.append((1, "unstaged_sweep", 0))
        for i in range(n_echo):
            descriptors.append((1, "echo_plume", i))
        for i in range(k):
            descriptors.append((1, "vane_hold", i))
        for b in range(c.offpath_eddies):
            descriptors.append((1, "eddy_corridor", b))
        descriptors.append((2, "gate", 0))
        for b in range(c.offpath_eddies):
            descriptors.append((2, "eddy_end", b))

        self._descriptors = tuple(descriptors)
        self._code_of = {d: idx for idx, d in enumerate(self._descriptors)}
        self.states = tuple(range(len(self._descriptors)))

    def elapsed(self, observation: int) -> int:
        return self._descriptors[int(observation)][0]

    def valid_action_mask(self, observation: int) -> np.ndarray:
        t, role, _ = self._descriptors[int(observation)]
        mask = np.zeros(self.action_space.n, dtype=bool)
        if t == 0:
            mask[: self.n_root_actions] = True
        elif t == 1 and role == "staged":
            mask[: self.n_staged_actions] = True
        elif t == 1 and role == "eddy_corridor":
            mask[:2] = True
        else:
            mask[0] = True
        return mask

    def _echo_law(self, next_code: int, mean_rew: float, echo_idx: int):
        c = self.config
        target_ch = c.primary_channels + 1 + echo_idx
        d = c.echo_halfwidth
        q = 0.5 + d if self.channel == target_ch else 0.5 - d
        return {
            (next_code, round(mean_rew - 0.02, 10), False): (1.0 - q) / 2.0,
            (next_code, round(mean_rew, 10), False): q,
            (next_code, round(mean_rew + 0.02, 10), False): (1.0 - q) / 2.0,
        }

    def law(self, observation: int, action: int) -> dict[tuple[int, float, bool], float]:
        if not self.valid_action_mask(observation)[action]:
            raise ValueError("Masked grid pathdep action")
        c = self.config
        k = c.complexity
        n_echo = c.echo_count
        R_tot = c.total_channels
        t, role, idx = self._descriptors[int(observation)]
        gate = self._code_of[(2, "gate", 0)]

        if t == 2:
            return {(self.TERMINAL, 0.0, True): 1.0}

        if t == 0:
            if action == 0:
                return {(self._code_of[(1, "safe_dock", 0)], c.safe_reward, False): 1.0}
            if action == 1:
                return {(self._code_of[(1, "staged", 0)], -c.stage_cost, False): 1.0}
            if action == 2:
                return {(self._code_of[(1, "unstaged_sweep", 0)], c.unstaged_reward, False): 1.0}
            if action < 3 + n_echo:
                e_i = action - 3
                return {(self._code_of[(1, "echo_plume", e_i)], 0.0, False): 1.0}
            if action < 3 + n_echo + k:
                m_i = action - (3 + n_echo)
                nxt = self._code_of[(1, "vane_hold", m_i)]
                bit = (self.modifier >> m_i) & 1
                d = c.modifier_halfwidth
                q = 0.5 + d if bit == 1 else 0.5 - d
                return {
                    (nxt, round(c.modifier_reward - 0.02, 10), False): (1.0 - q) / 2.0,
                    (nxt, round(c.modifier_reward, 10), False): q,
                    (nxt, round(c.modifier_reward + 0.02, 10), False): (1.0 - q) / 2.0,
                }
            b = action - (3 + n_echo + k)
            return {(self._code_of[(1, "eddy_corridor", b)], 0.06, False): 1.0}

        if role in {"safe_dock", "unstaged_sweep", "vane_hold"}:
            return {(gate, 0.0, False): 1.0}
        if role == "echo_plume":
            return self._echo_law(gate, c.echo_reward, idx)
        if role == "eddy_corridor":
            end_b = self._code_of[(2, "eddy_end", idx)]
            bit = (self.modifier >> (idx % k)) & 1
            p = 0.85 if (action == bit) else 0.15
            return {(end_b, 0.08, False): 1.0 - p, (end_b, 0.20, False): p}
        if role == "staged":
            rebate = c.stage_cost
            if action == 0:
                return {(gate, round(0.485 + rebate, 10), False): 1.0}
            if 1 <= action <= R_tot:
                active = int(self.channel == action)
                pair_p = c.channel_probabilities if action <= c.primary_channels else c.peripheral_probabilities
                p = pair_p[active]
                return {
                    (gate, round(0.0 + rebate, 10), False): 1.0 - p,
                    (gate, round(1.0 + rebate, 10), False): p,
                }
            n_out = c.primary_channels + 1
            hi = c.probe_target_prob
            lo = (1.0 - hi) / (n_out - 1)
            eff_ch = self.channel if self.channel <= c.primary_channels else 0
            probs = [hi if j == eff_ch else lo for j in range(n_out)]
            out = {}
            base = c.probe_reward + rebate
            for j, pj in enumerate(probs):
                delta = round(0.004 * (j + 1), 10)
                out[(gate, round(base - delta, 10), False)] = 0.5 * pj
                out[(gate, round(base + delta, 10), False)] = 0.5 * pj
            return out
        raise AssertionError("Unreachable grid pathdep state")

    def reset(self, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.observation = 0
        self.terminated = False
        return 0, {}

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Reset before stepping a terminal episode")
        prev = self.observation
        action = int(action)
        law = self.law(prev, action)
        keys = list(law)
        nxt, reward, done = keys[int(self.np_random.choice(len(keys), p=list(law.values())))]
        self.observation = nxt
        self.terminated = done
        c = self.config
        k = c.complexity
        n_echo = c.echo_count
        t, role, _ = self._descriptors[prev]
        event = "step"
        if done:
            event = "finish"
        elif t == 0:
            if action == 0:
                event = "safe"
            elif action == 1:
                event = "stage"
            elif action == 2:
                event = "unstaged"
            elif action < 3 + n_echo:
                event = "echo_reporter"
            elif action < 3 + n_echo + k:
                event = "passenger_omics"
            else:
                event = "eddy_trap"
        elif t == 1 and role == "staged":
            if action == 0:
                event = "safe"
            elif 1 <= action <= c.total_channels:
                event = "target_route" if action == self.channel else "wrong_route"
            else:
                event = "probe"
        return nxt, reward, done, False, {"event": event, "physical_steps": 1}


class PathDepUndertowEnv(_GridPathDepEnv):
    def __init__(self, config: PathDepUndertowConfig | None = None, channel: int = 0, modifier: int = 0):
        super().__init__(config or PathDepUndertowConfig(), channel, modifier)


class PathDepFalseBeaconEnv(_GridPathDepEnv):
    def __init__(self, config: PathDepFalseBeaconConfig | None = None, channel: int = 0, modifier: int = 0):
        super().__init__(config or PathDepFalseBeaconConfig(), channel, modifier)


def relabel_grid_pathdep(cfg: _GridPathDepBaseConfig, seed: int) -> _GridPathDepBaseConfig:
    salt = 41 if cfg.kind == "undertow_pathdep" else 53
    rng = np.random.default_rng([2026, salt, int(seed)])
    target = int(rng.integers(1, cfg.primary_channels + 1))
    cls = PathDepUndertowConfig if cfg.kind == "undertow_pathdep" else PathDepFalseBeaconConfig
    return cls(**{**asdict(cfg), "target_channel": target})


class _GridPathDepModel(DAGProductModel):
    ENV_CLS = _GridPathDepEnv

    def __init__(self, config: _GridPathDepBaseConfig):
        self.cfg = config
        self.kind = config.kind
        self.context = 0
        self.target_channel = int(config.target_channel)
        self.hypotheses = tuple(
            (ch, mod)
            for ch in range(config.total_channels + 1)
            for mod in range(config.modifier_states)
        )
        self._models = [self.ENV_CLS(config, ch, mod) for ch, mod in self.hypotheses]
        self.num_models = len(self.hypotheses)
        self.actions = self._models[0].action_space.n
        self.initial_observation = 0
        self.benchmark = f"{config.kind}_k{config.complexity}"
        self.specification = asdict(config)
        self._compile_dag(config.horizon)
        self._finalize_optima()

    def canonical_path_policies(self) -> list[Policy]:
        env0 = self._models[0]
        staged_s = self.index[env0._code_of[(1, "staged", 0)]]
        k = self.cfg.complexity
        n_echo = self.cfg.echo_count
        out = []
        for a0 in [0, 2] + list(range(3, 3 + n_echo + k)):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, a0] = 1.0
            out.append(Policy(p, f"root-{a0}"))
        for a1 in range(env0.n_staged_actions):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, 1] = 1.0
            p[staged_s, :] = 0.0
            p[staged_s, a1] = 1.0
            out.append(Policy(p, f"staged-{a1}"))
        for b in range(self.cfg.offpath_eddies):
            sb = self.index[env0._code_of[(1, "eddy_corridor", b)]]
            a0 = 3 + n_echo + k + b
            for a1 in (0, 1):
                p = np.zeros(self.mask.shape, dtype=float)
                p[:, 0] = 1.0
                p[self.initial, :] = 0.0
                p[self.initial, a0] = 1.0
                p[sb, :] = 0.0
                p[sb, a1] = 1.0
                out.append(Policy(p, f"eddy-{b}-{a1}"))
        return out

    def seed_policies(self) -> list[Policy]:
        env0 = self._models[0]
        staged_s = self.index[env0._code_of[(1, "staged", 0)]]
        out = []
        p0 = np.zeros(self.mask.shape, dtype=float)
        p0[:, 0] = 1.0
        out.append(Policy(p0, "safe"))
        for a1 in [env0.n_staged_actions - 1] + list(range(1, self.cfg.total_channels + 1)):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, 1] = 1.0
            p[staged_s, :] = 0.0
            p[staged_s, a1] = 1.0
            out.append(Policy(p, f"seed-staged-{a1}"))
        return out

    def prior(self, treatment: str) -> np.ndarray:
        m_states = self.cfg.modifier_states
        if treatment == "null":
            w = np.array([1.0 if ch == 0 else 0.0 for ch, _ in self.hypotheses], dtype=float)
            return w / w.sum()
        if treatment in {"product", "reference", "empirical"}:
            n_echo = self.cfg.echo_count
            echo_each = self.cfg.echo_prior_per_subclade
            rem = 1.0 - n_echo * echo_each
            resp_mass = self.cfg.responder_prior_mass * rem
            null_mass = rem - resp_mass
            n_prim = self.cfg.primary_channels
            w = np.array(
                [
                    (null_mass / m_states)
                    if ch == 0
                    else ((resp_mass / (n_prim * m_states)) if ch <= n_prim else (echo_each / m_states))
                    for ch, _ in self.hypotheses
                ],
                dtype=float,
            )
            return w / w.sum()
        raise ValueError(f"Unknown grid pathdep prior: {treatment}")

    def true_index(self, task: int, passenger: int = 0) -> int:
        ch = self.target_channel if task == 1 else 0
        return self.hypotheses.index((ch, int(passenger)))

    def responder_probability(self, posterior: np.ndarray) -> float:
        return float(sum(p for p, (ch, _) in zip(posterior, self.hypotheses) if ch == self.target_channel))


class PathDepUndertowModel(_GridPathDepModel):
    ENV_CLS = PathDepUndertowEnv


class PathDepFalseBeaconModel(_GridPathDepModel):
    ENV_CLS = PathDepFalseBeaconEnv
