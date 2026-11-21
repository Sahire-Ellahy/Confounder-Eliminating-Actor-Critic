"""General finite-horizon DAG product-model compiler with vectorized evaluation."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
import math
import numpy as np
from scipy.special import kl_div
from ceac.model import FiniteModel
from ceac.contracts import Policy
from .environments import LockConfig, LockEnv, EpistasisConfig, EpistasisEnv


class DAGProductModel(FiniteModel):
    """Exact finite-horizon DAG compiler for multi-step locks and multi-hypothesis biology."""

    def _compile_dag(self, horizon: int):
        count = self.num_models
        env = self._models[0]
        self.horizon = int(horizon)
        self.state_ids = np.array(env.states, dtype=int)
        self.n = len(self.state_ids)
        self.index = {int(s): i for i, s in enumerate(self.state_ids)}
        self.initial = self.index[0]
        self.mask = np.array([env.valid_action_mask(int(s)) for s in self.state_ids], dtype=bool)
        self.levels = [
            np.array([i for i, s in enumerate(self.state_ids) if env.elapsed(int(s)) == t], dtype=int)
            for t in range(self.horizon)
        ]
        self.reward = np.zeros((count, self.n, self.actions))
        self._pair_kl: dict[tuple[int, int], np.ndarray] = {}
        self._zero_kl = np.zeros((count, count), dtype=float)
        self._zero_kl.flags.writeable = False
        self.density = np.full((count, self.n), np.nan)
        self.density[:, self.initial] = 1.0
        self.laws = {}
        edge, destination, prob = [], [], []
        for s, obs in enumerate(self.state_ids):
            for action in np.flatnonzero(self.mask[s]):
                laws = [e.law(int(obs), int(action)) for e in self._models]
                keys0 = laws[0].keys()
                if any(
                    law.keys() != keys0
                    or any(p <= 0.0 for p in law.values())
                    or not np.isclose(sum(law.values()), 1.0, atol=1e-12)
                    for law in laws
                ):
                    raise ValueError("Candidate outcome support must be common, positive, and normalized")
                self.laws[s, int(action)] = laws
                p_mat = np.array([[law[k] for k in keys0] for law in laws], dtype=float)
                r_vec = np.array([k[1] for k in keys0], dtype=float)
                self.reward[:, s, action] = p_mat @ r_vec
                if not np.allclose(p_mat, p_mat[0:1], atol=1e-14):
                    log_p = np.log(p_mat)
                    kl_block = np.maximum(0.0, (p_mat * log_p).sum(axis=1)[:, None] - (p_mat @ log_p.T))
                    kl_block[kl_block < 1e-14] = 0.0  # floating residue for identical laws
                    np.fill_diagonal(kl_block, 0.0)
                    if kl_block.max() > 1e-12:
                        self._pair_kl[(s, int(action))] = kl_block
                grouped = {}
                for key in keys0:
                    nxt, _, done = key
                    dest = self.n if done else self.index[nxt]
                    edge.append(s * self.actions + int(action))
                    destination.append(dest)
                    prob.append([law[key] for law in laws])
                    if not done:
                        if env.elapsed(nxt) <= env.elapsed(int(obs)):
                            raise ValueError("Non-DAG transition")
                        grouped.setdefault(dest, np.zeros(count))
                        grouped[dest] += [law[key] for law in laws]
                for dest, mass in grouped.items():
                    ratio = self.density[:, s] * mass / mass[0]
                    if np.isnan(self.density[0, dest]):
                        self.density[:, dest] = ratio
                    elif not np.allclose(self.density[:, dest], ratio, atol=1e-12, rtol=1e-10):
                        raise ValueError("Policy-dependent occupancy ratio: unsupported response oracle")
        self.edge = np.array(edge, dtype=int)
        self.destination = np.array(destination, dtype=int)
        self.prob = np.array(prob, dtype=float).T  # shape: (num_models, num_edges)
        if not np.isfinite(self.density).all():
            raise ValueError("Unreachable public state")

        # Verify that no forgotten reward/outcome evidence precedes a future informative state.
        future = np.zeros(self.n + 1, dtype=bool)
        for states in reversed(self.levels):
            for s in states:
                for a in np.flatnonzero(self.mask[s]):
                    future[s] |= (int(s), int(a)) in self._pair_kl
                    future[s] |= any(future[self.index[k[0]]] for k in self.laws[s, int(a)][0] if not k[2])
        for (s, a), laws in self.laws.items():
            for key in laws[0]:
                if key[2]:
                    continue
                dest = self.index[key[0]]
                ratios = self.density[:, s] * np.array([law[key] for law in laws]) / laws[0][key]
                if future[dest] and not np.allclose(ratios, self.density[:, dest], atol=1e-12, rtol=1e-10):
                    raise ValueError("Forgotten evidence affects future predictive beliefs")

        # Precompute non-terminal transition operator(s) for fast vectorized evaluation.
        nonterm = self.destination < self.n
        self._shared_nonterm = bool(
            np.allclose(self.prob[:, nonterm], self.prob[0:1, nonterm], atol=1e-12, rtol=1e-10)
        )
        if self._shared_nonterm:
            op = np.zeros((self.n * self.actions, self.n), dtype=float)
            np.add.at(op, (self.edge[nonterm], self.destination[nonterm]), self.prob[0, nonterm])
            self._nonterm_op = op.reshape(self.n, self.actions, self.n)
        else:
            ops = np.zeros((count, self.n * self.actions, self.n), dtype=float)
            for m in range(count):
                np.add.at(ops[m], (self.edge[nonterm], self.destination[nonterm]), self.prob[m, nonterm])
            self._nonterm_op = ops.reshape(count, self.n, self.actions, self.n)

        # Precompute exact all-outcome episodic return range [r_min, r_max].
        low, high = np.zeros(self.n), np.zeros(self.n)
        for states in reversed(self.levels):
            for s in states:
                lows, highs = [], []
                for a in np.flatnonzero(self.mask[s]):
                    for nxt, rew, done in self.laws[s, int(a)][0]:
                        lows.append(rew + (0.0 if done else low[self.index[nxt]]))
                        highs.append(rew + (0.0 if done else high[self.index[nxt]]))
                low[s], high[s] = min(lows), max(highs)
        self.return_range = (float(low[self.initial]), float(high[self.initial]))

        self._informative_pairs = tuple(sorted(self._pair_kl))
        self._evaluations = {}
        self._mi_cache = None

    def stage_kl(self, s: int, a: int) -> np.ndarray:
        return self._pair_kl.get((int(s), int(a)), self._zero_kl)

    def response(self, reward_weights, information_weights, label="separator", secondary_reward_weights=None):
        rw = np.asarray(reward_weights, dtype=float)
        iw = np.asarray(information_weights, dtype=float)
        stage = np.einsum("m,ms,msa->sa", rw, self.density, self.reward)
        if np.any(iw):
            for s, a in self._informative_pairs:
                stage[s, a] += float(np.sum(iw * self.density[:, s:s + 1] * self._pair_kl[(s, a)]))
        secondary = (
            None
            if secondary_reward_weights is None
            else np.einsum("m,ms,msa->sa", secondary_reward_weights, self.density, self.reward)
        )
        return self.response_stage(stage, label, secondary)

    def _finalize_optima(self):
        eye = np.eye(self.num_models)
        zeros = np.zeros((self.num_models, self.num_models))
        self.optimal_policies = [self.response(eye[m], zeros, "optimal")[0] for m in range(self.num_models)]
        # Only returns are needed here; avoid caching M x M information blocks for every model.
        opt_evals = np.array([self.returns(p) for p in self.optimal_policies])
        self.optimal_returns = np.array([opt_evals[m, m] for m in range(self.num_models)])
        raw_alts = defaultdict(list)
        for m in range(self.num_models):
            for j in range(self.num_models):
                if m != j and self.optimal_returns[j] - opt_evals[m, j] > 1e-10:
                    raw_alts[m].append(j)
        undominated = []
        info_blocks = [self._pair_kl[sa] for sa in self._informative_pairs]
        for m, js in raw_alts.items():
            if len(js) <= 1 or not info_blocks:
                undominated.extend((m, j) for j in js)
                continue
            vecs = np.column_stack([blk[m, js] for blk in info_blocks])
            keep = np.ones(len(js), dtype=bool)
            for idx_a in range(len(js)):
                if not keep[idx_a]:
                    continue
                diff = vecs - vecs[idx_a]
                strictly_larger = np.all(diff >= -1e-12, axis=1) & (np.max(diff, axis=1) > 1e-10)
                keep[strictly_larger] = False
            for idx, j in enumerate(js):
                if keep[idx]:
                    undominated.append((m, j))
        self.all_policy_changing_alternatives = tuple((m, j) for m, js in raw_alts.items() for j in js)
        self.alternatives = tuple(undominated)

    def returns(self, policy: Policy) -> np.ndarray:
        self.validate_policy(policy)
        values = np.zeros((self.num_models, self.n), dtype=float)
        probs = policy.probabilities
        for t in range(self.horizon - 1, -1, -1):
            states = self.levels[t]
            if not len(states):
                continue
            imm = np.einsum("sa,msa->ms", probs[states], self.reward[:, states, :])
            if t == self.horizon - 1:
                values[:, states] = imm
            elif self._shared_nonterm:
                p_trans = np.einsum("sa,san->sn", probs[states], self._nonterm_op[states])
                values[:, states] = imm + values @ p_trans.T
            else:
                p_trans = np.einsum("sa,msan->msn", probs[states], self._nonterm_op[:, states])
                values[:, states] = imm + np.einsum("msn,mn->ms", p_trans, values)
        return values[:, self.initial].copy()

    def evaluate(self, policy: Policy) -> tuple[np.ndarray, np.ndarray]:
        self.validate_policy(policy)
        key = policy.identity
        if key in self._evaluations:
            return self._evaluations[key]
        returns = self.returns(policy)
        probs = policy.probabilities
        # Forward state-action occupancy under each model, contracted ONLY on informative (s, a) pairs
        if self._shared_nonterm:
            occ = np.zeros(self.n, dtype=float)
            occ[self.initial] = 1.0
            for t in range(self.horizon - 1):
                states = self.levels[t]
                if not len(states):
                    continue
                p_trans = np.einsum("sa,san->sn", probs[states], self._nonterm_op[states])
                occ += occ[states] @ p_trans
            information = np.zeros((self.num_models, self.num_models), dtype=float)
            for s, a in self._informative_pairs:
                w = occ[s] * probs[s, a]
                if w > 0.0:
                    information += w * self._pair_kl[(s, a)]
        else:
            occ = np.zeros((self.num_models, self.n), dtype=float)
            occ[:, self.initial] = 1.0
            for t in range(self.horizon - 1):
                states = self.levels[t]
                if not len(states):
                    continue
                p_trans = np.einsum("sa,msan->msn", probs[states], self._nonterm_op[:, states])
                occ += np.einsum("ms,msn->mn", occ[:, states], p_trans)
            information = np.zeros((self.num_models, self.num_models), dtype=float)
            for s, a in self._informative_pairs:
                w = occ[:, s] * probs[s, a]
                information += w[:, None] * self._pair_kl[(s, a)]
        np.fill_diagonal(information, 0.0)
        # Bound cache memory (~200 MB per process) regardless of model-class size.
        if len(self._evaluations) >= max(8, int(2e8 // (8 * self.num_models * self.num_models))):
            self._evaluations.clear()
        self._evaluations[key] = (returns, information)
        return returns, information

    def log_likelihood(self, state: int, action: int, next_observation: int, reward: float, done: bool):
        key = (int(next_observation), float(reward), bool(done))
        laws = self.laws[int(state), int(action)]
        if key not in laws[0]:
            # Guard tiny float rounding in multi-step rebate arithmetic
            matches = [k for k in laws[0] if k[0] == key[0] and k[2] == key[2] and abs(k[1] - key[1]) < 1e-9]
            if len(matches) != 1:
                raise ValueError(f"Observed transition {key} is outside the declared model family")
            key = matches[0]
        return tuple(math.log(law[key]) for law in laws)

    def mutual_information_stage(self, posterior: np.ndarray) -> np.ndarray:
        posterior = np.asarray(posterior, dtype=float)
        if posterior.shape != (self.num_models,) or (posterior < 0).any() or not np.isclose(posterior.sum(), 1.0):
            raise ValueError("Invalid posterior")
        key = tuple(posterior)
        if self._mi_cache is not None and self._mi_cache[0] == key:
            return self._mi_cache[1]
        stage = np.zeros(self.mask.shape, dtype=float)
        masses = posterior[:, None] * self.density
        for s, a in self._informative_pairs:
            laws = self.laws[s, a]
            weights = masses[:, s]
            total = float(weights.sum())
            if total <= 0.0:
                continue
            belief = weights / total
            probabilities = np.array([[law[k] for k in laws[0]] for law in laws], dtype=float)
            mixture = belief @ probabilities
            stage[s, a] = max(0.0, float(weights @ kl_div(probabilities, mixture[None, :]).sum(axis=1)))
        self._mi_cache = (key, stage)
        return stage

    def decision_classes(self) -> tuple[tuple[int, ...], ...]:
        """Partition candidate models into mutually non-conflicting optimal-policy classes."""
        alt_set = set(getattr(self, "all_policy_changing_alternatives", self.alternatives))
        groups: list[list[int]] = []
        for m in range(self.num_models):
            placed = False
            for group in groups:
                if all((m, g) not in alt_set and (g, m) not in alt_set for g in group):
                    group.append(m)
                    placed = True
                    break
            if not placed:
                groups.append([m])
        return tuple(tuple(g) for g in groups)


class LockModel(DAGProductModel):
    """Binary model family for unidirectional and bidirectional combination locks."""

    def __init__(self, config: LockConfig | None = None):
        self.cfg = config or LockConfig()
        self.kind = self.cfg.kind
        self.context = 0
        self.hypotheses = ((0, 0), (1, 0))
        self._models = [LockEnv(self.cfg, task) for task in (0, 1)]
        self.num_models = 2
        self.actions = self._models[0].action_space.n
        self.initial_observation = 0
        self.benchmark = f"{self.cfg.kind}_L{self.cfg.depth}"
        self.specification = asdict(self.cfg)
        self._compile_dag(self.cfg.horizon)
        self._finalize_optima()

    def seed_policies(self) -> list[Policy]:
        c = self.cfg
        L = c.depth
        policies = []
        for route_idx, name in enumerate(("incumbent", "challenge", "diagnostic")):
            p = np.zeros(self.mask.shape, dtype=float)
            for s, obs in enumerate(self.state_ids):
                t, role, _ = self._models[0].describe(int(obs))
                if not self.mask[s].sum() > 1:
                    p[s, int(np.flatnonzero(self.mask[s])[0])] = 1.0
                elif L == 1:
                    p[s, route_idx] = 1.0
                elif c.kind == "combination_lock":
                    p[s, (t % 2) if t < L - 1 else route_idx] = 1.0
                else:
                    # Bidirectional lock: incumbent unlocks Left (key 0);
                    # challenge (1) and diagnostic (2) unlock Right (key 1, then 1 or 2).
                    if role == "start":
                        p[s, 0 if route_idx == 0 else 1] = 1.0
                    elif role == "left":
                        p[s, 0] = 1.0
                    elif role == "right":
                        p[s, 1 if t < L - 1 else route_idx] = 1.0
                    else:
                        p[s, 0] = 1.0
            policies.append(Policy(p, name))
        return policies

    def prior(self, treatment: str) -> np.ndarray:
        if treatment in {"product", "reference"}:
            return np.array([0.5, 0.5], dtype=float)
        if treatment == "decoy":
            return np.array([1.0, 0.0], dtype=float)
        if treatment == "rare":
            return np.array([0.999, 0.001], dtype=float)
        raise ValueError(f"Unknown lock prior treatment: {treatment}")

    def true_index(self, task: int, passenger: int = 0) -> int:
        del passenger
        if task not in (0, 1):
            raise ValueError("Lock task must be 0 or 1")
        return int(task)

    def responder_probability(self, posterior: np.ndarray) -> float:
        return float(posterior[1])


class EpistasisModel(DAGProductModel):
    """Product family over G genetic subtypes x 2^k passenger modules."""

    def __init__(self, config: EpistasisConfig | None = None):
        self.cfg = config or EpistasisConfig()
        self.kind = self.cfg.kind
        self.context = 0
        self.hypotheses = tuple(
            (subtype, passenger)
            for subtype in range(self.cfg.subtypes)
            for passenger in range(self.cfg.passenger_states)
        )
        self._models = [EpistasisEnv(self.cfg, sub, pas) for sub, pas in self.hypotheses]
        self.num_models = len(self.hypotheses)
        self.actions = self._models[0].action_space.n
        self.initial_observation = 0
        self.benchmark = f"epistasis_k{self.cfg.pleiotropy}"
        self.specification = asdict(self.cfg)
        self._compile_dag(self.cfg.horizon)
        self._finalize_optima()

    def seed_policies(self) -> list[Policy]:
        policies = []
        for action, name in enumerate(("cytostatic", "synergy", "phospho_probe")):
            p = np.zeros(self.mask.shape, dtype=float)
            p[:, 0] = 1.0
            p[0, :] = 0.0
            p[0, action] = 1.0
            policies.append(Policy(p, name))
        return policies

    def prior(self, treatment: str) -> np.ndarray:
        if treatment in {"product", "reference"}:
            return np.ones(self.num_models, dtype=float) / self.num_models
        if treatment == "canonical":
            # Canonical additive-genetics prior: single-gene silence excludes
            # synergistic rescue (subtype == G - 1), spreading mass uniformly
            # across all (G - 1) * 2^k canalized non-responder hypotheses.
            weights = np.array(
                [1.0 if sub < self.cfg.subtypes - 1 else 0.0 for sub, _ in self.hypotheses],
                dtype=float,
            )
            return weights / weights.sum()
        raise ValueError(f"Unknown epistasis prior treatment: {treatment}")

    def true_index(self, task: int, passenger: int = 0) -> int:
        if task not in (0, 1):
            raise ValueError("Task must be 0 (canalized control) or 1 (rare epistatic responder)")
        target_subtype = (self.cfg.subtypes - 1) if task == 1 else 0
        hits = [
            m for m, (sub, pas) in enumerate(self.hypotheses)
            if sub == target_subtype and pas == int(passenger)
        ]
        if len(hits) != 1:
            raise ValueError("True genotype absent from public model class")
        return hits[0]

    def responder_probability(self, posterior: np.ndarray) -> float:
        target = self.cfg.subtypes - 1
        return float(sum(p for p, (sub, _) in zip(posterior, self.hypotheses) if sub == target))
