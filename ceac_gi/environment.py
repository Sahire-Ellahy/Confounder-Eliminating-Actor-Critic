"""Pairwise knockout screen on a real Horlbeck-2018 K562 gene panel.

Episode = one perturbation of a 10-gene panel, then a forced finish step.
Actions at the root (state 0):
  0                      control (reward 0.50 under every hypothesis)
  1 .. G                 single knockouts. Rewards score the GI (deviation from
                         the multiplicative expectation), so singles never carry
                         therapeutic value (0.4999). For the first k "profiled"
                         non-SL genes, singles reveal a SYNTHETIC main-effect bit.
  G+1 .. G+C(G,2)        pair knockouts (i, j). Hit probability
                         clip(Phi((threshold - GI_ij)/sigma), floor, 1-floor);
                         reward 1 on a synthetic-lethal hit, else 0.
  then B pooled blocks   multiplex knockout of a public gene block; hit if the
                         block's most negative GI crosses threshold; toxic
                         (reward 0.30 hit / 0.20 miss).
Hypotheses: (q, eta) with q in {null} U pairs and eta in {0,1}^k. Under q the
panel's REAL GI values hold everywhere except: the real SL pair is set to a
declared neutral value (0) unless q is that pair, and q carries the real strong
value. The true task-1 hypothesis is therefore exactly the published panel.
Task 0 (negative control) is the counterfactual null with the SL pair neutral.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import erf, sqrt
from types import SimpleNamespace
import numpy as np
from ceac.contracts import Policy
from ceac_pathologies.model import DAGProductModel

# Fraction of random 10-gene K562-map panels with >=1 pair at GI <= -5
# (ceac_gi.data.panel_base_rate(10, draws=5000) = 0.0918); frozen before any run.
EMPIRICAL_SL_PANEL_RATE = 0.0918


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def public_blocks(size: int, block_size: int = 6, count: int = 10, seed: int = 11) -> tuple[tuple[int, ...], ...]:
    """Public pooled design: every pair covered and every pair's membership signature unique."""
    rng = np.random.default_rng(seed)
    pairs = list(combinations(range(size), 2))
    for _ in range(20000):
        blocks = [tuple(sorted(int(g) for g in rng.choice(size, block_size, replace=False))) for _ in range(count)]
        sig = [tuple(int(a in b and c in b) for b in blocks) for a, c in pairs]
        if all(any(s) for s in sig) and len(set(sig)) == len(sig):
            return tuple(blocks)
    raise RuntimeError("No identifying pooled design found")


@dataclass(frozen=True)
class GIScreenConfig:
    kind: str = "gi_arrayed"
    nuisance: int = 0
    genes: tuple = ()
    values: tuple = ()
    sl_pair_index: int = 0
    strong_value: float = -5.0
    neutral_value: float = 0.0
    threshold: float = -3.0
    sigma: float = 1.25
    assay_floor: float = 0.01
    control_reward: float = 0.50
    single_reward: float = 0.4999
    single_halfwidth: float = 0.04
    block_rewards: tuple = (0.20, 0.30)
    blocks: tuple = ()
    source: str = ""

    def __post_init__(self):
        g = len(self.genes)
        if self.kind not in {"gi_arrayed", "gi_multiplexed"}:
            raise ValueError(f"Unknown GI screen kind: {self.kind}")
        if (self.kind == "gi_arrayed") != (len(self.blocks) == 0):
            raise ValueError("Arrayed screens have no pooled blocks; multiplexed screens require them")
        if g < 4 or len(self.values) != g * (g - 1) // 2:
            raise ValueError("Panel genes/values mismatch")
        if not 0 <= self.nuisance <= g - 2:
            raise ValueError("Nuisance genes must exclude the SL pair")
        if not 0 <= self.sl_pair_index < len(self.values):
            raise ValueError("Invalid SL pair index")
        if not 0.0 < self.single_halfwidth < 0.45 or not 0.0 < self.assay_floor < 0.5:
            raise ValueError("Invalid assay parameters")

    @property
    def horizon(self) -> int:
        return 2

    @property
    def pairs(self):
        return list(combinations(range(len(self.genes)), 2))

    @property
    def nuisance_genes(self) -> tuple[int, ...]:
        sl = set(self.pairs[self.sl_pair_index])
        return tuple([i for i in range(len(self.genes)) if i not in sl][: self.nuisance])

    @property
    def passenger_states(self) -> int:
        return 1 << self.nuisance


def build_config(kind: str, nuisance: int, **overrides) -> GIScreenConfig:
    from .data import SOURCE, select_panel
    panel = select_panel(10)
    blocks = public_blocks(len(panel["genes"])) if kind == "gi_multiplexed" else ()
    return GIScreenConfig(kind=kind, nuisance=int(nuisance), genes=panel["genes"], values=panel["values"],
                          sl_pair_index=panel["sl_pair_index"], strong_value=panel["strong_value"],
                          blocks=blocks, source=SOURCE, **overrides)


def config_from_dict(d: dict) -> GIScreenConfig:
    d = dict(d)
    for key in ("genes", "values", "block_rewards"):
        d[key] = tuple(d[key])
    d["blocks"] = tuple(tuple(int(x) for x in b) for b in d["blocks"])
    return GIScreenConfig(**d)


def relabel(cfg: GIScreenConfig, seed: int) -> GIScreenConfig:
    """Seeded public permutation of panel gene order. Labels are arbitrary, so this
    removes index/tie-break artifacts (e.g. the SL pair always being pair 0) for every
    method. Pair values are re-read from the verified map; the block design stays positional."""
    from .data import panel_values
    rng = np.random.default_rng([2018, 11, int(seed)])
    order = rng.permutation(len(cfg.genes))
    genes = tuple(cfg.genes[i] for i in order)
    sl_genes = set(cfg.genes[i] for i in cfg.pairs[cfg.sl_pair_index])
    pairs = list(combinations(range(len(genes)), 2))
    sl = next(k for k, (a, b) in enumerate(pairs) if {genes[a], genes[b]} == sl_genes)
    values = panel_values(genes)
    return GIScreenConfig(**{**asdict(cfg), "genes": genes, "values": values, "sl_pair_index": sl,
                             "blocks": cfg.blocks, "block_rewards": cfg.block_rewards})


class GIScreenEnv:
    TERMINAL = 10000

    def __init__(self, config: GIScreenConfig, pair: int = -1, passenger: int = 0):
        self.config = c = config
        if not -1 <= pair < len(c.values) or not 0 <= passenger < c.passenger_states:
            raise ValueError("Invalid realized hypothesis")
        self.pair, self.passenger = int(pair), int(passenger)
        g = len(c.genes)
        self.n_pairs = len(c.values)
        self.action_space = SimpleNamespace(n=1 + g + self.n_pairs + len(c.blocks))
        self.states = (0, 1)
        vals = np.array(c.values, dtype=float)
        if pair != c.sl_pair_index:
            vals[c.sl_pair_index] = c.neutral_value
        if pair >= 0:
            vals[pair] = c.strong_value
        self.gi = vals
        self.np_random = np.random.default_rng()
        self.observation = 0
        self.terminated = False

    def elapsed(self, observation):
        return int(observation != 0)

    def valid_action_mask(self, observation):
        mask = np.zeros(self.action_space.n, dtype=bool)
        if observation == 0:
            mask[:] = True
        else:
            mask[0] = True
        return mask

    def _hit(self, gi: float) -> float:
        c = self.config
        return float(np.clip(_phi((c.threshold - gi) / c.sigma), c.assay_floor, 1.0 - c.assay_floor))

    def law(self, observation, action):
        c = self.config
        if not self.valid_action_mask(observation)[action]:
            raise ValueError("Masked action")
        if observation == 1:
            return {(self.TERMINAL, 0.0, True): 1.0}
        g = len(c.genes)
        if action == 0:
            return {(1, c.control_reward, False): 1.0}
        if action <= g:
            gene = action - 1
            if gene in c.nuisance_genes:
                bit = (self.passenger >> c.nuisance_genes.index(gene)) & 1
                q = 0.5 + c.single_halfwidth if bit else 0.5 - c.single_halfwidth
                return {(1, round(c.single_reward - 0.02, 10), False): (1 - q) / 2,
                        (1, c.single_reward, False): q,
                        (1, round(c.single_reward + 0.02, 10), False): (1 - q) / 2}
            return {(1, c.single_reward, False): 1.0}
        if action <= g + self.n_pairs:
            p = self._hit(self.gi[action - g - 1])
            return {(1, 0.0, False): 1 - p, (1, 1.0, False): p}
        block = c.blocks[action - g - 1 - self.n_pairs]
        members = [k for k, (a, b) in enumerate(c.pairs) if a in block and b in block]
        p = self._hit(float(self.gi[members].min()))
        lo, hi = c.block_rewards
        return {(1, lo, False): 1 - p, (1, hi, False): p}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.observation, self.terminated = 0, False
        return 0, {}

    def step(self, action):
        if self.terminated:
            raise RuntimeError("Reset first")
        prev, action = self.observation, int(action)
        law = self.law(prev, action)
        keys = list(law)
        nxt, reward, done = keys[int(self.np_random.choice(len(keys), p=list(law.values())))]
        self.observation, self.terminated = nxt, done
        g = len(self.config.genes)
        if prev:
            event = "finish"
        elif action == 0:
            event = "control"
        elif action <= g:
            event = "passenger_omics" if (action - 1) in self.config.nuisance_genes else "single_ko"
        elif action <= g + self.n_pairs:
            event = "sl_pair" if (action - g - 1) == self.pair else "pair_test"
        else:
            event = "block"
        return nxt, reward, done, False, {"event": event, "physical_steps": 1}


class GIScreenModel(DAGProductModel):
    def __init__(self, config: GIScreenConfig):
        self.cfg = config
        self.kind = config.kind
        self.context = 0
        self.hypotheses = tuple((q, e) for q in range(-1, len(config.values)) for e in range(config.passenger_states))
        self._models = [GIScreenEnv(config, q, e) for q, e in self.hypotheses]
        self.num_models = len(self.hypotheses)
        self.actions = self._models[0].action_space.n
        self.initial_observation = 0
        self.benchmark = f"{config.kind}_k{config.nuisance}"
        self.specification = asdict(config)
        self._compile_dag(config.horizon)
        self._finalize_optima()

    def seed_policies(self):
        """Control plus the minimal tests covering every policy-changing row: all pooled
        blocks (multiplexed) or, in an arrayed screen, every pair test (only the test of
        pair q separates the null from q). The arrayed bank size is a registered CEAC cost."""
        g, n_pairs = len(self.cfg.genes), len(self.cfg.values)
        if self.cfg.kind == "gi_multiplexed":
            actions = [0] + [1 + g + n_pairs + b for b in range(len(self.cfg.blocks))]
        else:
            actions = [0] + [1 + g + q for q in range(n_pairs)]
        out = []
        for a in actions:
            p = np.zeros(self.mask.shape)
            p[:, 0] = 1.0
            p[0, :] = 0.0
            p[0, a] = 1.0
            out.append(Policy(p, f"seed-{a}"))
        return out

    def prior(self, treatment: str) -> np.ndarray:
        e = self.cfg.passenger_states
        if treatment == "product":
            return np.ones(self.num_models) / self.num_models
        if treatment in {"empirical", "additive", "reference"}:
            null_mass = 1.0 if treatment == "additive" else 1.0 - EMPIRICAL_SL_PANEL_RATE
            w = np.array([null_mass / e if q < 0 else (1 - null_mass) / (e * len(self.cfg.values))
                          for q, _ in self.hypotheses])
            return w / w.sum()
        raise ValueError(treatment)

    def true_index(self, task: int, passenger: int = 0) -> int:
        q = self.cfg.sl_pair_index if task == 1 else -1
        return self.hypotheses.index((q, int(passenger)))

    def responder_probability(self, posterior) -> float:
        return float(sum(p for p, (q, _) in zip(posterior, self.hypotheses) if q == self.cfg.sl_pair_index))
