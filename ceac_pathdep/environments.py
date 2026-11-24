"""Path-dependent multi-step signalling rewiring and canalized K562 genetic risk environments.

Both families have physical horizon H = 3 (t = 0 -> t = 1 -> t = 2 -> done) where
the optimal intervention is strictly path-dependent (state-contingent):

1. SignallingRewiringEnv (`signalling_rewiring`):
   Models time-staggered oncogenic signalling rewiring (e.g. Step-1 EGFR/MEK priming
   relieving negative feedback to expose a genotype-specific Step-2 bypass co-target
   among R = 4 post-translational kinases + k transcriptionally echoed clades).
   - Applying co-targets at t = 0 without priming fails due to intact feedback (0.25, 0 KL).
   - Step-1 priming incurs step tension -0.015 (0 KL at t = 0) and enters `primed_state`,
     where matching `co_target_r` pays 0.92 (vs 0.10 mismatched / null) and `primed_probe`
     yields a clean phospho-proteomic readout (0.47, high KL across all pathways).
   - Sweeping k in {0, 1, 2, 3} adds:
     (a) k two-step echoed coarse mRNA reporters (total return 0.49998; noisy signal on
         peripheral clade e_i echoed at t = 0 and t = 1, zero KL between Null and primary
         post-translational responders);
     (b) k basal modifier assays (0.4999) + (3^k - 1) off-path stress branches whose
         Step-2 local rescue depends on counterfactual bit xi_{b mod k} in {0, 1}
         (max total return 0.22 < 0.50, never visited on any optimal trajectory).

2. CanalizedGIEnv (`canalized_gi`, grounded in Horlbeck et al. 2018 K562 GI matrix):
   Models cryptic genetic risk modifiers buffered by master phosphatase hub `PPP2R1A`:
   - Perturbing candidate modifiers at t = 0 while `PPP2R1A` is intact is buffered (0.4999,
     zero causal modifier KL; k non-causal single arms reveal off-path modifier bits xi).
   - Knocking down `PPP2R1A` at t = 0 (`decanalize_hub`, -0.015 step cost, 0 KL at t = 0)
     transitions into `decanalized_state` at t = 1, where testing candidate modifier m
     produces synthetic-lethal/rescue hits governed by the published K562 GI(PPP2R1A, m)
     score (real strong partner ANAPC13/FDPS vs 9 neutral K562 modifiers + k peripheral
     eQTL-echoed K562 modifiers).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import erf, sqrt
from types import SimpleNamespace
import numpy as np
from ceac.contracts import Policy
from ceac_pathologies.model import DAGProductModel


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass(frozen=True)
class SignallingRewiringConfig:
    kind: str = "signalling_rewiring"
    complexity: int = 0
    primary_pathways: int = 8
    target_pathway: int = 1
    safe_reward: float = 0.50
    prime_cost: float = 0.015
    unprimed_reward: float = 0.25
    synergy_probabilities: tuple[float, float] = (0.10, 0.92)
    peripheral_synergy_probabilities: tuple[float, float] = (0.12, 0.5004)
    probe_reward: float = 0.26
    probe_target_prob: float = 0.36
    echo_subclades_per_module: int = 6
    echo_reward: float = 0.49999
    echo_halfwidth: float = 0.0115
    echo_prior_per_subclade: float = 0.048
    modifier_reward: float = 0.49998
    modifier_halfwidth: float = 0.0110
    responder_prior_mass: float = 0.0055

    def __post_init__(self):
        if self.kind != "signalling_rewiring":
            raise ValueError(f"Unknown signalling kind: {self.kind}")
        if int(self.complexity) != self.complexity or not 0 <= self.complexity <= 4:
            raise ValueError("Complexity dose k must lie in {0..4}")
        if self.primary_pathways < 2 or not 1 <= self.target_pathway <= self.primary_pathways:
            raise ValueError("Invalid primary_pathways or target_pathway")
        if not 0.0 < self.prime_cost < 0.10:
            raise ValueError("Invalid priming step cost")
        if not 0.0 < self.probe_reward < self.modifier_reward < self.echo_reward < self.safe_reward < 1.0:
            raise ValueError("Invalid signalling reward hierarchy")

    @property
    def horizon(self) -> int:
        return 3

    @property
    def echo_count(self) -> int:
        return self.echo_subclades_per_module * self.complexity

    @property
    def total_pathways(self) -> int:
        return self.primary_pathways + self.echo_count

    @property
    def stress_branches(self) -> int:
        return (3 ** self.complexity) - 1

    @property
    def modifier_states(self) -> int:
        return 1 << self.complexity


class SignallingRewiringEnv:
    """Three-step sequential priming & bypass selection environment."""
    TERMINAL = 10000

    def __init__(self, config: SignallingRewiringConfig | None = None, pathway: int = 0, modifier: int = 0):
        self.config = c = config or SignallingRewiringConfig()
        if not 0 <= int(pathway) <= c.total_pathways or not 0 <= int(modifier) < c.modifier_states:
            raise ValueError("Invalid (pathway, modifier) hypothesis")
        # pathway == 0 is Null (canalized non-responder); 1..R are primary bypasses; R+1..R+k are echoed clades
        self.pathway = int(pathway)
        self.modifier = int(modifier)
        self._build_layout()
        self.np_random = np.random.default_rng()
        self.observation = 0
        self.terminated = False

    def _build_layout(self):
        c = self.config
        k = c.complexity
        n_echo = c.echo_count
        R_tot = c.total_pathways
        # Root actions at t = 0:
        # 0: cytostatic_care
        # 1: prime_cascade
        # 2: unprimed_simultaneous (reversed-order control)
        # 3 .. 3+n_echo-1: echo_reporter_0 .. n_echo-1
        # 3+n_echo .. 3+n_echo+k-1: modifier_assay_0 .. k-1
        # 3+n_echo+k .. 3+n_echo+k+(3^k-1)-1: off_path_stress_b
        self.n_root_actions = 3 + n_echo + k + c.stress_branches
        # Inside primed_state (t = 1):
        # 0: recover_care, 1..R_tot: co_target_1..R_tot, R_tot+1: primed_phospho_panel
        self.n_primed_actions = R_tot + 2
        n_actions = max(self.n_root_actions, self.n_primed_actions, 2)
        self.action_space = SimpleNamespace(n=n_actions)

        descriptors: list[tuple[int, str, int]] = [(0, "basal", 0)]
        descriptors.append((1, "safe_hold", 0))
        descriptors.append((1, "primed", 0))
        descriptors.append((1, "unprimed_hold", 0))
        for i in range(n_echo):
            descriptors.append((1, "echo_stage", i))
        for i in range(k):
            descriptors.append((1, "modifier_hold", i))
        for b in range(c.stress_branches):
            descriptors.append((1, "stress_branch", b))
        descriptors.append((2, "gate", 0))
        for b in range(c.stress_branches):
            descriptors.append((2, "stress_end", b))

        self._descriptors = tuple(descriptors)
        self._code_of = {d: idx for idx, d in enumerate(self._descriptors)}
        self.states = tuple(range(len(self._descriptors)))

    def describe(self, observation: int) -> tuple[int, str, int]:
        return self._descriptors[int(observation)]

    def elapsed(self, observation: int) -> int:
        return self._descriptors[int(observation)][0]

    def valid_action_mask(self, observation: int) -> np.ndarray:
        t, role, _ = self._descriptors[int(observation)]
        mask = np.zeros(self.action_space.n, dtype=bool)
        if t == 0:
            mask[: self.n_root_actions] = True
        elif t == 1 and role == "primed":
            mask[: self.n_primed_actions] = True
        elif t == 1 and role == "stress_branch":
            mask[:2] = True
        else:
            mask[0] = True
        return mask

    def _echo_law(self, next_code: int, mean_rew: float, echo_idx: int):
        c = self.config
        target_pathway = c.primary_pathways + 1 + echo_idx
        d = c.echo_halfwidth
        q = 0.5 + d if self.pathway == target_pathway else 0.5 - d
        return {
            (next_code, round(mean_rew - 0.02, 10), False): (1.0 - q) / 2.0,
            (next_code, round(mean_rew, 10), False): q,
            (next_code, round(mean_rew + 0.02, 10), False): (1.0 - q) / 2.0,
        }

    def law(self, observation: int, action: int) -> dict[tuple[int, float, bool], float]:
        if not self.valid_action_mask(observation)[action]:
            raise ValueError("Masked signalling action")
        c = self.config
        k = c.complexity
        R_tot = c.total_pathways
        t, role, idx = self._descriptors[int(observation)]
        gate = self._code_of[(2, "gate", 0)]

        if t == 2:
            return {(self.TERMINAL, 0.0, True): 1.0}

        if t == 0:
            n_echo = c.echo_count
            if action == 0:
                return {(self._code_of[(1, "safe_hold", 0)], c.safe_reward, False): 1.0}
            if action == 1:
                return {(self._code_of[(1, "primed", 0)], -c.prime_cost, False): 1.0}
            if action == 2:
                return {(self._code_of[(1, "unprimed_hold", 0)], c.unprimed_reward, False): 1.0}
            if action < 3 + n_echo:
                e_i = action - 3
                nxt = self._code_of[(1, "echo_stage", e_i)]
                return {(nxt, 0.0, False): 1.0}
            if action < 3 + n_echo + k:
                m_i = action - (3 + n_echo)
                nxt = self._code_of[(1, "modifier_hold", m_i)]
                bit = (self.modifier >> m_i) & 1
                d = c.modifier_halfwidth
                q = 0.5 + d if bit == 1 else 0.5 - d
                return {
                    (nxt, round(c.modifier_reward - 0.02, 10), False): (1.0 - q) / 2.0,
                    (nxt, round(c.modifier_reward, 10), False): q,
                    (nxt, round(c.modifier_reward + 0.02, 10), False): (1.0 - q) / 2.0,
                }
            b = action - (3 + n_echo + k)
            return {(self._code_of[(1, "stress_branch", b)], 0.06, False): 1.0}

        # t == 1 transitions
        if role in {"safe_hold", "unprimed_hold", "modifier_hold"}:
            return {(gate, 0.0, False): 1.0}
        if role == "echo_stage":
            return self._echo_law(gate, c.echo_reward, idx)
        if role == "stress_branch":
            end_b = self._code_of[(2, "stress_end", idx)]
            bit = (self.modifier >> (idx % k)) & 1
            p = 0.85 if (action == bit) else 0.15
            return {(end_b, 0.08, False): 1.0 - p, (end_b, 0.20, False): p}
        if role == "primed":
            rebate = c.prime_cost
            if action == 0:
                return {(gate, round(0.485 + rebate, 10), False): 1.0}
            if 1 <= action <= R_tot:
                active = int(self.pathway == action)
                pair_p = c.synergy_probabilities if action <= c.primary_pathways else c.peripheral_synergy_probabilities
                p = pair_p[active]
                return {
                    (gate, round(0.0 + rebate, 10), False): 1.0 - p,
                    (gate, round(1.0 + rebate, 10), False): p,
                }
            # action == R_tot + 1: primed_phospho_panel (post-translational phosphorylation panel over 0..R_primary)
            n_out = c.primary_pathways + 1
            hi = c.probe_target_prob
            lo = (1.0 - hi) / (n_out - 1)
            eff_pw = self.pathway if self.pathway <= c.primary_pathways else 0
            probs = [hi if j == eff_pw else lo for j in range(n_out)]
            # Center rewards symmetrically around c.probe_reward + rebate so mean == c.probe_reward + rebate
            # regardless of which outcome has mass hi! Wait: if rewards differ across j, mean shifts slightly
            # unless pair-symmetric around self.pathway OR we pair outcomes (j, -j)!
            # Pairing 2*(R_tot + 1) symmetric +/- offsets around (probe_reward + rebate) guarantees exact
            # expected return (probe_reward + rebate) under EVERY pathway hypothesis!
            out = {}
            base = c.probe_reward + rebate
            for j, pj in enumerate(probs):
                delta = round(0.004 * (j + 1), 10)
                out[(gate, round(base - delta, 10), False)] = 0.5 * pj
                out[(gate, round(base + delta, 10), False)] = 0.5 * pj
            return out
        raise AssertionError("Unreachable signalling state")

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
        t, role, _ = self._descriptors[prev]
        event = "step"
        if done:
            event = "finish"
        elif t == 0:
            n_echo = c.echo_count
            if action == 0:
                event = "safe"
            elif action == 1:
                event = "prime"
            elif action == 2:
                event = "unprimed"
            elif action < 3 + n_echo:
                event = "echo_reporter"
            elif action < 3 + n_echo + k:
                event = "passenger_omics"
            else:
                event = "pleiotropic"
        elif t == 1 and role == "primed":
            if action == 0:
                event = "safe"
            elif 1 <= action <= c.total_pathways:
                event = "synergy" if action == self.pathway else "wrong_cotarget"
            else:
                event = "probe"
        return nxt, reward, done, False, {"event": event, "physical_steps": 1}


def relabel_signalling(cfg: SignallingRewiringConfig, seed: int) -> SignallingRewiringConfig:
    """Seeded public assignment of the realized primary bypass kinase slot (1..R)."""
    rng = np.random.default_rng([2026, 31, int(seed)])
    target = int(rng.integers(1, cfg.primary_pathways + 1))
    return SignallingRewiringConfig(**{**asdict(cfg), "target_pathway": target})


class SignallingRewiringModel(DAGProductModel):
    """Compiled DAG model over (pathway in 0..R+k) x (modifier in 0..2^k-1)."""

    def __init__(self, config: SignallingRewiringConfig | None = None):
        self.cfg = config or SignallingRewiringConfig()
        self.kind = self.cfg.kind
        self.context = 0
        self.target_pathway = int(self.cfg.target_pathway)
        self.hypotheses = tuple(
            (pw, mod)
            for pw in range(self.cfg.total_pathways + 1)
            for mod in range(self.cfg.modifier_states)
        )
        self._models = [SignallingRewiringEnv(self.cfg, pw, mod) for pw, mod in self.hypotheses]
        self.num_models = len(self.hypotheses)
        self.actions = self._models[0].action_space.n
        self.initial_observation = 0
        self.benchmark = f"signalling_rewiring_k{self.cfg.complexity}"
        self.specification = asdict(self.cfg)
        self._compile_dag(self.cfg.horizon)
        self._finalize_optima()

    def canonical_path_policies(self) -> list[Policy]:
        """Minimal exact basis of deterministic reachable-path policies (avoids Cartesian blowup)."""
        env0 = self._models[0]
        primed_s = self.index[env0._code_of[(1, "primed", 0)]]
        out = []
        # Root actions other than prime_cascade (1) and stress branches
        k = self.cfg.complexity
        n_echo = self.cfg.echo_count
        for a0 in [0, 2] + list(range(3, 3 + n_echo + k)):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, a0] = 1.0
            out.append(Policy(p, f"root-{a0}"))
        # Primed sub-actions
        for a1 in range(env0.n_primed_actions):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, 1] = 1.0
            p[primed_s, :] = 0.0
            p[primed_s, a1] = 1.0
            out.append(Policy(p, f"primed-{a1}"))
        # Off-path stress branches (2 sub-actions each)
        for b in range(self.cfg.stress_branches):
            sb = self.index[env0._code_of[(1, "stress_branch", b)]]
            a0 = 3 + n_echo + k + b
            for a1 in (0, 1):
                p = np.zeros(self.mask.shape, dtype=float)
                p[:, 0] = 1.0
                p[self.initial, :] = 0.0
                p[self.initial, a0] = 1.0
                p[sb, :] = 0.0
                p[sb, a1] = 1.0
                out.append(Policy(p, f"stress-{b}-{a1}"))
        return out

    def seed_policies(self) -> list[Policy]:
        """Safe baseline, primed phospho-panel diagnostic, and primed bypass co-targets."""
        env0 = self._models[0]
        primed_s = self.index[env0._code_of[(1, "primed", 0)]]
        out = []
        # 0: safe baseline
        p0 = np.zeros(self.mask.shape, dtype=float)
        p0[:, 0] = 1.0
        out.append(Policy(p0, "safe"))
        # Primed probe + primed co-targets
        for a1 in [env0.n_primed_actions - 1] + list(range(1, self.cfg.total_pathways + 1)):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, 1] = 1.0
            p[primed_s, :] = 0.0
            p[primed_s, a1] = 1.0
            out.append(Policy(p, f"seed-primed-{a1}"))
        return out

    def prior(self, treatment: str) -> np.ndarray:
        m_states = self.cfg.modifier_states
        if treatment == "canonical":
            w = np.array([1.0 if pw == 0 else 0.0 for pw, _ in self.hypotheses], dtype=float)
            return w / w.sum()
        if treatment in {"product", "reference", "empirical"}:
            n_echo = self.cfg.echo_count
            echo_each = self.cfg.echo_prior_per_subclade
            rem = 1.0 - n_echo * echo_each
            resp_mass = self.cfg.responder_prior_mass * rem
            null_mass = rem - resp_mass
            n_prim = self.cfg.primary_pathways
            w = np.array(
                [
                    (null_mass / m_states)
                    if pw == 0
                    else ((resp_mass / (n_prim * m_states)) if pw <= n_prim else (echo_each / m_states))
                    for pw, _ in self.hypotheses
                ],
                dtype=float,
            )
            return w / w.sum()
        raise ValueError(f"Unknown signalling prior: {treatment}")

    def true_index(self, task: int, passenger: int = 0) -> int:
        pw = self.target_pathway if task == 1 else 0
        return self.hypotheses.index((pw, int(passenger)))

    def responder_probability(self, posterior: np.ndarray) -> float:
        return float(sum(p for p, (pw, _) in zip(posterior, self.hypotheses) if pw == self.target_pathway))


# ============================================================================
# Experiment 2: Canalized Genetic Risk Factors on Horlbeck 2018 K562 GI Matrix
# ============================================================================

def select_canalized_panel(primary_modifiers: int = 10, max_echo_modifiers: int = 18, seed: int = 2026) -> dict:
    """Select capacitor hub PPP2R1A + 1 strong K562 partner + neutral K562 candidates."""
    from ceac_gi.data import SOURCE, load_k562_map

    names, mat = load_k562_map()
    hub_name = "PPP2R1A"
    hub_idx = names.index(hub_name)
    row = mat[hub_idx].copy()
    row[hub_idx] = 0.0

    # Strongest moderate-to-strong partner FDPS (-5.241) or ANAPC13 (-12.743); use FDPS (-5.2413) for
    # consistent realistic assay SNR ~0.96 matching ceac_gi, with neutral candidates > -1.8.
    partner_name = "FDPS"
    partner_idx = names.index(partner_name)
    strong_val = float(row[partner_idx])
    assert strong_val <= -5.0

    rng = np.random.default_rng(seed)
    total_needed = primary_modifiers + max_echo_modifiers
    chosen = [partner_idx]
    for idx in rng.permutation(len(names)):
        idx = int(idx)
        if idx in (hub_idx, partner_idx):
            continue
        if -1.6 < row[idx] < 1.6:
            chosen.append(idx)
        if len(chosen) == total_needed:
            break
    if len(chosen) < total_needed:
        raise RuntimeError("Insufficient neutral K562 hub modifiers")
    mod_genes = tuple(names[i] for i in chosen)
    hub_gi = tuple(float(row[i]) for i in chosen)
    return dict(
        hub_gene=hub_name,
        primary_genes=mod_genes[:primary_modifiers],
        primary_gi=hub_gi[:primary_modifiers],
        echo_pool_genes=mod_genes[primary_modifiers:],
        echo_pool_gi=hub_gi[primary_modifiers:],
        sl_index=0,
        strong_value=strong_val,
        source=SOURCE,
    )


@dataclass(frozen=True)
class CanalizedGIConfig:
    kind: str = "canalized_gi"
    complexity: int = 0
    hub_gene: str = "PPP2R1A"
    modifier_genes: tuple[str, ...] = ()
    hub_gi_values: tuple[float, ...] = ()
    primary_count: int = 10
    sl_modifier_index: int = 0
    strong_value: float = -5.2413
    neutral_value: float = 0.0
    threshold: float = -3.0
    sigma: float = 1.25
    assay_floor: float = 0.01
    control_reward: float = 0.50
    hub_cost: float = 0.015
    buffered_single_reward: float = 0.49998
    modifier_halfwidth: float = 0.0110
    echo_subclades_per_module: int = 6
    echo_reward: float = 0.49999
    echo_halfwidth: float = 0.0115
    echo_inactive_gi: float = -1.60
    echo_active_gi: float = -3.0012
    echo_prior_per_subclade: float = 0.048
    panel_active_rate: float = 0.0055
    source: str = ""

    def __post_init__(self):
        if self.kind != "canalized_gi":
            raise ValueError(f"Unknown canalized GI kind: {self.kind}")
        if not 0 <= self.complexity <= 3:
            raise ValueError("Complexity k must lie in {0..3}")
        if len(self.modifier_genes) != self.primary_count + self.echo_count:
            raise ValueError("Modifier gene tuple must equal primary_count + echo_count")
        if len(self.hub_gi_values) != len(self.modifier_genes):
            raise ValueError("Hub GI score length mismatch")
        if not 0 <= self.sl_modifier_index < self.primary_count:
            raise ValueError("SL modifier index out of primary range")

    @property
    def echo_count(self) -> int:
        return self.echo_subclades_per_module * self.complexity

    @property
    def horizon(self) -> int:
        return 3

    @property
    def total_modifiers(self) -> int:
        return len(self.modifier_genes)

    @property
    def offpath_branches(self) -> int:
        return (3 ** self.complexity) - 1

    @property
    def passenger_states(self) -> int:
        return 1 << self.complexity

    @property
    def nuisance_indices(self) -> tuple[int, ...]:
        return tuple([i for i in range(self.primary_count) if i != self.sl_modifier_index][: self.complexity])


def build_canalized_gi_config(complexity: int = 0) -> CanalizedGIConfig:
    k = int(complexity)
    n_echo = 6 * k
    panel = select_canalized_panel(primary_modifiers=10, max_echo_modifiers=18)
    genes = panel["primary_genes"] + panel["echo_pool_genes"][:n_echo]
    gis = panel["primary_gi"] + panel["echo_pool_gi"][:n_echo]
    return CanalizedGIConfig(
        complexity=k,
        hub_gene=panel["hub_gene"],
        modifier_genes=genes,
        hub_gi_values=gis,
        primary_count=10,
        sl_modifier_index=panel["sl_index"],
        strong_value=panel["strong_value"],
        source=panel["source"],
    )


def relabel_canalized_gi(cfg: CanalizedGIConfig, seed: int) -> CanalizedGIConfig:
    """Public seeded permutation of the primary post-translational modifier slots."""
    rng = np.random.default_rng([2026, 17, int(seed)])
    p_n = cfg.primary_count
    perm = rng.permutation(p_n)
    sl_gene = cfg.modifier_genes[cfg.sl_modifier_index]
    prim_genes = tuple(cfg.modifier_genes[i] for i in perm)
    prim_gis = tuple(cfg.hub_gi_values[i] for i in perm)
    echo_genes = cfg.modifier_genes[p_n:]
    echo_gis = cfg.hub_gi_values[p_n:]
    new_sl = prim_genes.index(sl_gene)
    return CanalizedGIConfig(
        **{
            **asdict(cfg),
            "modifier_genes": prim_genes + echo_genes,
            "hub_gi_values": prim_gis + echo_gis,
            "sl_modifier_index": new_sl,
        }
    )


class CanalizedGIEnv:
    """Three-step capacitor-hub de-canalization & K562 risk-modifier screen."""
    TERMINAL = 10000

    def __init__(self, config: CanalizedGIConfig, active_modifier: int = -1, passenger: int = 0):
        self.config = c = config
        if not -1 <= int(active_modifier) < c.total_modifiers or not 0 <= int(passenger) < c.passenger_states:
            raise ValueError("Invalid (active_modifier, passenger) hypothesis")
        self.active_modifier = int(active_modifier)
        self.passenger = int(passenger)

        vals = np.array(c.hub_gi_values, dtype=float)
        if c.echo_count > 0:
            vals[c.primary_count :] = c.echo_inactive_gi
        if self.active_modifier != c.sl_modifier_index:
            vals[c.sl_modifier_index] = c.neutral_value
        if self.active_modifier >= 0:
            vals[self.active_modifier] = c.strong_value if self.active_modifier < c.primary_count else c.echo_active_gi
        self.gi = vals

        self._build_layout()
        self.np_random = np.random.default_rng()
        self.observation = 0
        self.terminated = False

    def _build_layout(self):
        c = self.config
        n_echo = c.echo_count
        G = c.total_modifiers
        self.n_root_actions = 2 + G + n_echo + c.offpath_branches
        self.n_decanalized_actions = 1 + G
        n_actions = max(self.n_root_actions, self.n_decanalized_actions, 2)
        self.action_space = SimpleNamespace(n=n_actions)

        descriptors: list[tuple[int, str, int]] = [(0, "intact_hub", 0)]
        descriptors.append((1, "control_hold", 0))
        descriptors.append((1, "decanalized", 0))
        descriptors.append((1, "buffered_hold", 0))
        for i in range(n_echo):
            descriptors.append((1, "echo_eqtl", i))
        for b in range(c.offpath_branches):
            descriptors.append((1, "offpath_complex", b))
        descriptors.append((2, "gate", 0))
        for b in range(c.offpath_branches):
            descriptors.append((2, "offpath_end", b))

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
        elif t == 1 and role == "decanalized":
            mask[: self.n_decanalized_actions] = True
        elif t == 1 and role == "offpath_complex":
            mask[:2] = True
        else:
            mask[0] = True
        return mask

    def _hit(self, gi: float) -> float:
        c = self.config
        return float(np.clip(_phi((c.threshold - gi) / c.sigma), c.assay_floor, 1.0 - c.assay_floor))

    def _echo_law(self, next_code: int, mean_rew: float, echo_idx: int):
        c = self.config
        target_mod = c.primary_count + echo_idx
        d = c.echo_halfwidth
        q = 0.5 + d if self.active_modifier == target_mod else 0.5 - d
        return {
            (next_code, round(mean_rew - 0.02, 10), False): (1.0 - q) / 2.0,
            (next_code, round(mean_rew, 10), False): q,
            (next_code, round(mean_rew + 0.02, 10), False): (1.0 - q) / 2.0,
        }

    def law(self, observation: int, action: int) -> dict[tuple[int, float, bool], float]:
        if not self.valid_action_mask(observation)[action]:
            raise ValueError("Masked canalized GI action")
        c = self.config
        k = c.complexity
        G = c.total_modifiers
        t, role, idx = self._descriptors[int(observation)]
        gate = self._code_of[(2, "gate", 0)]

        if t == 2:
            return {(self.TERMINAL, 0.0, True): 1.0}

        if t == 0:
            if action == 0:
                return {(self._code_of[(1, "control_hold", 0)], c.control_reward, False): 1.0}
            if action == 1:
                return {(self._code_of[(1, "decanalized", 0)], -c.hub_cost, False): 1.0}
            if action < 2 + G:
                gene_idx = action - 2
                nxt = self._code_of[(1, "buffered_hold", 0)]
                if gene_idx in c.nuisance_indices:
                    bit_pos = c.nuisance_indices.index(gene_idx)
                    bit = (self.passenger >> bit_pos) & 1
                    d = c.modifier_halfwidth
                    q = 0.5 + d if bit == 1 else 0.5 - d
                    return {
                        (nxt, round(c.buffered_single_reward - 0.02, 10), False): (1.0 - q) / 2.0,
                        (nxt, round(c.buffered_single_reward, 10), False): q,
                        (nxt, round(c.buffered_single_reward + 0.02, 10), False): (1.0 - q) / 2.0,
                    }
                return {(nxt, c.buffered_single_reward, False): 1.0}
            n_echo = c.echo_count
            if action < 2 + G + n_echo:
                e_i = action - (2 + G)
                nxt = self._code_of[(1, "echo_eqtl", e_i)]
                return {(nxt, 0.0, False): 1.0}
            b = action - (2 + G + n_echo)
            return {(self._code_of[(1, "offpath_complex", b)], 0.06, False): 1.0}

        # t == 1 transitions
        if role in {"control_hold", "buffered_hold"}:
            return {(gate, 0.0, False): 1.0}
        if role == "echo_eqtl":
            return self._echo_law(gate, c.echo_reward, idx)
        if role == "offpath_complex":
            end_b = self._code_of[(2, "offpath_end", idx)]
            bit = (self.passenger >> (idx % k)) & 1
            p = 0.85 if (action == bit) else 0.15
            return {(end_b, 0.08, False): 1.0 - p, (end_b, 0.20, False): p}
        if role == "decanalized":
            rebate = c.hub_cost
            if action == 0:
                return {(gate, round(0.485 + rebate, 10), False): 1.0}
            m_idx = action - 1
            p = self._hit(float(self.gi[m_idx]))
            return {
                (gate, round(0.0 + rebate, 10), False): 1.0 - p,
                (gate, round(1.0 + rebate, 10), False): p,
            }
        raise AssertionError("Unreachable canalized GI state")

    def reset(self, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.observation = 0
        self.terminated = False
        return 0, {}

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("Reset first")
        prev = self.observation
        action = int(action)
        law = self.law(prev, action)
        keys = list(law)
        nxt, reward, done = keys[int(self.np_random.choice(len(keys), p=list(law.values())))]
        self.observation = nxt
        self.terminated = done
        c = self.config
        k = c.complexity
        G = c.total_modifiers
        t, role, _ = self._descriptors[prev]
        event = "step"
        if done:
            event = "finish"
        elif t == 0:
            if action == 0:
                event = "control"
            elif action == 1:
                event = "decanalize"
            elif action < 2 + G:
                g_i = action - 2
                event = "passenger_omics" if g_i in c.nuisance_indices else "buffered_ko"
            elif action < 2 + G + c.echo_count:
                event = "echo_reporter"
            else:
                event = "pleiotropic"
        elif t == 1 and role == "decanalized":
            if action == 0:
                event = "control"
            else:
                m_i = action - 1
                event = "sl_pair" if m_i == self.active_modifier else "pair_test"
        return nxt, reward, done, False, {"event": event, "physical_steps": 1}


class CanalizedGIModel(DAGProductModel):
    """Compiled DAG model over (active_modifier in -1..G-1) x (passenger in 0..2^k-1)."""

    def __init__(self, config: CanalizedGIConfig):
        self.cfg = config
        self.kind = config.kind
        self.context = 0
        self.hypotheses = tuple(
            (q, e)
            for q in range(-1, config.total_modifiers)
            for e in range(config.passenger_states)
        )
        self._models = [CanalizedGIEnv(config, q, e) for q, e in self.hypotheses]
        self.num_models = len(self.hypotheses)
        self.actions = self._models[0].action_space.n
        self.initial_observation = 0
        self.benchmark = f"canalized_gi_k{config.complexity}"
        self.specification = asdict(config)
        self._compile_dag(config.horizon)
        self._finalize_optima()

    def canonical_path_policies(self) -> list[Policy]:
        env0 = self._models[0]
        dec_s = self.index[env0._code_of[(1, "decanalized", 0)]]
        n_echo = self.cfg.echo_count
        G = self.cfg.total_modifiers
        out = []
        for a0 in [0] + list(range(2, 2 + G + n_echo)):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, a0] = 1.0
            out.append(Policy(p, f"root-{a0}"))
        for a1 in range(env0.n_decanalized_actions):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, 1] = 1.0
            p[dec_s, :] = 0.0
            p[dec_s, a1] = 1.0
            out.append(Policy(p, f"decanalized-{a1}"))
        for b in range(self.cfg.offpath_branches):
            sb = self.index[env0._code_of[(1, "offpath_complex", b)]]
            a0 = 2 + G + n_echo + b
            for a1 in (0, 1):
                p = np.zeros(self.mask.shape, dtype=float)
                p[:, 0] = 1.0
                p[self.initial, :] = 0.0
                p[self.initial, a0] = 1.0
                p[sb, :] = 0.0
                p[sb, a1] = 1.0
                out.append(Policy(p, f"offpath-{b}-{a1}"))
        return out

    def seed_policies(self) -> list[Policy]:
        env0 = self._models[0]
        dec_s = self.index[env0._code_of[(1, "decanalized", 0)]]
        out = []
        p0 = np.zeros(self.mask.shape, dtype=float)
        p0[:, 0] = 1.0
        out.append(Policy(p0, "control"))
        for a1 in range(1, env0.n_decanalized_actions):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[self.initial, :] = 0.0
            p[self.initial, 1] = 1.0
            p[dec_s, :] = 0.0
            p[dec_s, a1] = 1.0
            out.append(Policy(p, f"seed-mod-{a1}"))
        return out

    def prior(self, treatment: str) -> np.ndarray:
        e_states = self.cfg.passenger_states
        if treatment == "additive":
            w = np.array([1.0 if q < 0 else 0.0 for q, _ in self.hypotheses], dtype=float)
            return w / w.sum()
        if treatment == "product":
            return np.ones(self.num_models, dtype=float) / self.num_models
        if treatment in {"empirical", "reference"}:
            n_echo = self.cfg.echo_count
            echo_each = self.cfg.echo_prior_per_subclade
            rem = 1.0 - n_echo * echo_each
            act_mass = self.cfg.panel_active_rate * rem
            null_mass = rem - act_mass
            n_prim = self.cfg.primary_count
            w = np.array(
                [
                    (null_mass / e_states)
                    if q < 0
                    else ((act_mass / (n_prim * e_states)) if q < n_prim else (echo_each / e_states))
                    for q, _ in self.hypotheses
                ],
                dtype=float,
            )
            return w / w.sum()
        raise ValueError(f"Unknown canalized GI prior: {treatment}")

    def true_index(self, task: int, passenger: int = 0) -> int:
        q = self.cfg.sl_modifier_index if task == 1 else -1
        return self.hypotheses.index((q, int(passenger)))

    def responder_probability(self, posterior: np.ndarray) -> float:
        return float(sum(p for p, (q, _) in zip(posterior, self.hypotheses) if q == self.cfg.sl_modifier_index))
