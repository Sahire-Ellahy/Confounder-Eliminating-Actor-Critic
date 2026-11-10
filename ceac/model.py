"""Exact finite-model interface and certified shared-policy response oracle.

Planning sees every *declared* hypothesis, never the realized mode. The two
benchmarks admit a policy-independent state-occupancy likelihood ratio. We
verify this property before using a common-policy Bellman response. General
MDPs that fail the check need a different response oracle, not model-wise max.
"""
from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import math
import numpy as np
from scipy.special import kl_div
from .contracts import Policy
from .benchmarks import specification
from .environments import deepsea, frozenlake


class OutcomeRNG:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, 0

    def random(self):
        self.calls += 1
        return 0.0 if self.outcome == 1 else np.nextafter(1., 0.)

    def choice(self, upper, p):
        self.calls += 1
        if len(p) != upper or not 0 <= self.outcome < upper:
            raise ValueError("Invalid enumerated categorical outcome")
        return self.outcome

    def integers(self, upper):
        self.calls += 1
        if not 0 <= self.outcome < upper:
            raise ValueError("Invalid enumerated outcome")
        return self.outcome


def environment(kind, modes=(0, 1, 0), block_length=64, benchmark="legacy"):
    """Runner-only construction of a realized simulator."""
    module = deepsea if kind == "deepsea" else frozenlake if kind == "frozenlake" else None
    if module is None:
        raise ValueError(kind)
    cfg_type = module.DeepSeaConfig if kind == "deepsea" else module.FrozenShoreConfig
    factory = module.ShiftingDeepSeaEnv if kind == "deepsea" else module.ShiftingShoreFrozenLakeEnv
    return factory(cfg_type(true_modes=tuple(modes), block_length=block_length,
                            **specification(benchmark, kind)["configuration_overrides"]))


class FiniteModel:
    """Public binary model family, built independently of the realized simulator."""
    def __init__(self, kind, context=0, benchmark="legacy"):
        self.kind, self.context, self.benchmark = kind, int(context), benchmark
        self.specification = specification(benchmark, kind)
        self.module = deepsea if kind == "deepsea" else frozenlake if kind == "frozenlake" else None
        if self.module is None:
            raise ValueError(kind)
        self._models = [environment(kind, (m, m, m), benchmark=benchmark) for m in (0, 1)]
        self.cfg = self._models[0].config  # candidate zero, not the realized configuration
        self.actions = self._models[0].action_space.n
        self.num_models = 2
        self.initial_observation = self._models[0].reset(
            seed=0, options={"episode_index": self.context * self.cfg.block_length})[0]
        self._compile()
        self.optimal_policies = [self.response(np.eye(2)[m], np.zeros((2, 2)))[0] for m in (0, 1)]
        self.optimal_returns = np.array([self.evaluate(p)[0][m] for m, p in enumerate(self.optimal_policies)])
        self.alternatives = tuple((m, j) for m in (0, 1) for j in (0, 1) if m != j
                                  and self.optimal_returns[j] - self.evaluate(self.optimal_policies[m])[0][j] > 1e-10)
        self._mi_cache = None

    def _branch(self, observation, action, mode, outcome):
        env = self._models[mode]
        for key, value in vars(env.decode(observation)).items():
            setattr(env, key, value)
        env.episode_index = 0
        env._terminated = False
        if self.kind == "deepsea":
            env._episode_return = 0.
        else:
            env.mode = mode
        env.np_random = OutcomeRNG(outcome)
        nxt, reward, term, trunc, info = env.step(action)
        if env.np_random.calls > 1:
            raise ValueError("Multiple random events per step require a new kernel enumerator")
        return (int(nxt), float(reward), bool(term or trunc)), info

    def _law(self, observation, action, mode):
        first, info = self._branch(observation, action, mode, 0)
        event = info["diagnostic_event"]
        if event in ("tv", "reeds"):
            n = self.cfg.tv_bins if event == "tv" else self.cfg.reed_bins
            rho = self.cfg.nuisance_probability
            outcomes = ([(j, 1. / n) for j in range(n)] if rho is None else
                        [(j, ((1-rho) if j < n else rho)/n) for j in range(2*n)])
        else:
            probs = ({"probe": self.cfg.probe_probability, "challenge": self.cfg.challenge_probability}
                     if self.kind == "deepsea" else
                     {"deep": self.cfg.deep_signal_probability, "inner": self.cfg.inner_success,
                      "outer": self.cfg.outer_success, "shallow": lambda _: self.cfg.shallow_signal_probability})
            if event not in probs:
                return {first: 1.}
            p = probs[event](mode)
            outcomes = [(0, 1 - p), (1, p)]
        law = {}
        for outcome, weight in outcomes:
            key, _ = self._branch(observation, action, mode, outcome)
            law[key] = law.get(key, 0.) + weight
        if not math.isclose(sum(law.values()), 1., abs_tol=1e-12):
            raise ValueError("Unnormalized transition law")
        return law

    def _compile(self):
        pending, visited, laws, masks = [self.initial_observation], set(), {}, {}
        env = self._models[0]
        while pending:
            obs = pending.pop()
            if obs in visited:
                continue
            visited.add(obs)
            mask = env.valid_action_mask(obs).astype(bool)
            masks[obs] = mask
            for action in np.flatnonzero(mask):
                pair = [self._law(obs, int(action), m) for m in (0, 1)]
                if pair[0].keys() != pair[1].keys() or any(p <= 0 for law in pair for p in law.values()):
                    raise ValueError("Exact finite-model lane requires common positive outcome support")
                laws[obs, int(action)] = pair
                for nxt, _, done in pair[0]:
                    if not done:
                        if env.decode(nxt).elapsed <= env.decode(obs).elapsed:
                            raise ValueError("Response oracle needs a finite-horizon DAG")
                        pending.append(nxt)
        self.state_ids = np.array(sorted(visited, key=lambda s: (env.decode(s).elapsed, s)), dtype=np.int64)
        self.index = {int(s): i for i, s in enumerate(self.state_ids)}
        self.initial = self.index[self.initial_observation]
        self.n = len(self.state_ids)
        self.mask = np.array([masks[s] for s in self.state_ids])
        self.levels = [np.array([i for i, s in enumerate(self.state_ids) if env.decode(s).elapsed == t], dtype=int)
                       for t in range(self.cfg.max_steps)]
        self.reward = np.zeros((2, self.n, self.actions))
        self.kl = np.zeros((2, 2, self.n, self.actions))
        self.density = np.full((2, self.n), np.nan)
        self.density[:, self.initial] = 1.
        sources, destinations, probs = [], [], []
        self.laws = {}
        for s, obs in enumerate(self.state_ids):
            for a in np.flatnonzero(self.mask[s]):
                pair = laws[int(obs), int(a)]
                self.laws[s, int(a)] = pair
                for m in (0, 1):
                    self.reward[m, s, a] = sum(w * key[1] for key, w in pair[m].items())
                    self.kl[m, 1-m, s, a] = sum(w * math.log(w / pair[1-m][key]) for key, w in pair[m].items())
                grouped = {}
                for key in pair[0]:
                    nxt, _, done = key
                    dest = self.n if done else self.index[nxt]
                    sources.append(s * self.actions + int(a)); destinations.append(dest)
                    probs.append([pair[m][key] for m in (0, 1)])
                    if not done:
                        grouped.setdefault(dest, np.zeros(2))
                        grouped[dest] += [pair[m][key] for m in (0, 1)]
                for dest, mass in grouped.items():
                    candidate = self.density[:, s] * mass / mass[0]
                    if np.isnan(self.density[0, dest]):
                        self.density[:, dest] = candidate
                    elif not np.allclose(self.density[:, dest], candidate, rtol=1e-10, atol=1e-12):
                        raise ValueError("Shared-state occupancy ratio is path-dependent; use a structured response oracle")
        self.edge = np.array(sources, dtype=int)
        self.destination = np.array(destinations, dtype=int)
        self.prob = np.array(probs).T
        if not np.isfinite(self.density).all():
            raise ValueError("Unreachable or uninitialized public states")
        # MI may use state beliefs only if forgotten reward evidence has no
        # future informative descendant. This is checked, not assumed.
        future = np.zeros(self.n + 1, dtype=bool)
        for s in range(self.n - 1, -1, -1):
            for a in np.flatnonzero(self.mask[s]):
                pair = self.laws[s, int(a)]
                future[s] |= self.kl[:, :, s, a].max() > 1e-12
                future[s] |= any(future[self.index[k[0]]] for k in pair[0] if not k[2])
        for (s, a), pair in self.laws.items():
            for key in pair[0]:
                if key[2]:
                    continue
                dest = self.index[key[0]]
                full_ratio = self.density[1, s] * pair[1][key] / pair[0][key]
                if future[dest] and not math.isclose(full_ratio, self.density[1, dest], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("IDS requires history-dependent beliefs beyond this state representation")
        self._informative_pairs = tuple((s,a) for s,a in self.laws if self.kl[:,:,s,a].max() > 0.)
        self._evaluations = {}

    def _continuation(self, values, mode):
        return np.bincount(self.edge, weights=self.prob[mode] * values[self.destination],
                           minlength=self.n * self.actions).reshape(self.n, self.actions)

    def response_stage(self, stage, label="separator", secondary_stage=None):
        """Maximize a public common-policy additive score in base-model measure."""
        values = np.zeros(self.n + 1)
        chosen = np.zeros(self.n, dtype=int)
        secondary_values = np.zeros(self.n+1) if secondary_stage is not None else None
        for states in reversed(self.levels):
            if not len(states):
                continue
            q = stage[states] + self._continuation(values, 0)[states]
            q = np.where(self.mask[states], q, -np.inf)
            if secondary_stage is None:
                chosen[states] = np.argmax(q, axis=1)
            else:
                # Lexicographic tie resolution; primary residual is bounded
                # by horizon * 1e-12, below the global oracle certificate tol.
                ties = self.mask[states] & (q >= q.max(axis=1,keepdims=True)-1e-12)
                secondary_q = secondary_stage[states] + self._continuation(secondary_values,0)[states]
                chosen[states] = np.argmax(np.where(ties,secondary_q,-np.inf),axis=1)
                secondary_values[states] = secondary_q[np.arange(len(states)),chosen[states]]
            values[states] = q[np.arange(len(states)), chosen[states]]
        p = np.zeros_like(self.mask, dtype=float)
        p[np.arange(self.n), chosen] = 1.
        return Policy(p, label), float(values[self.initial])

    def response(self, reward_weights, information_weights, label="separator", secondary_reward_weights=None):
        rw, iw = np.asarray(reward_weights), np.asarray(information_weights)
        stage = np.einsum("m,ms,msa->sa", rw, self.density, self.reward)
        stage += np.einsum("mj,ms,mjsa->sa", iw, self.density, self.kl)
        secondary = (None if secondary_reward_weights is None else
                     np.einsum("m,ms,msa->sa", secondary_reward_weights, self.density, self.reward))
        return self.response_stage(stage, label, secondary)

    def evaluate_stage(self, policy, stage):
        self.validate_policy(policy)
        value = np.zeros(self.n + 1)
        for states in reversed(self.levels):
            q = stage[states] + self._continuation(value, 0)[states]
            value[states] = np.sum(policy.probabilities[states] * q, axis=1)
        return float(value[self.initial])

    def validate_policy(self, policy):
        if policy.probabilities.shape != self.mask.shape or np.any(policy.probabilities[~self.mask] > 0):
            raise ValueError("Policy violates the declared common action mask")

    def evaluate(self, policy):
        self.validate_policy(policy)
        identity = policy.identity
        if identity in self._evaluations:
            return self._evaluations[identity]
        returns, information = np.zeros(2), np.zeros((2, 2))
        for m in (0, 1):
            vr, vi = np.zeros(self.n + 1), np.zeros(self.n + 1)
            for states in reversed(self.levels):
                qr = self.reward[m, states] + self._continuation(vr, m)[states]
                qi = self.kl[m, 1-m, states] + self._continuation(vi, m)[states]
                vr[states] = np.sum(policy.probabilities[states] * qr, axis=1)
                vi[states] = np.sum(policy.probabilities[states] * qi, axis=1)
            returns[m], information[m, 1-m] = vr[self.initial], vi[self.initial]
        self._evaluations[identity] = returns, information
        return returns, information

    def mutual_information_stage(self, posterior):
        posterior = np.asarray(posterior, dtype=float)
        if posterior.shape != (2,) or (posterior < 0).any() or not np.isclose(posterior.sum(), 1.):
            raise ValueError("Invalid posterior")
        key = tuple(posterior)
        if self._mi_cache is not None and self._mi_cache[0] == key:
            return self._mi_cache[1]
        stage = np.zeros(self.mask.shape, dtype=float)
        if np.count_nonzero(posterior) == 2:
            masses = posterior[:, None] * self.density
            for s, a in self._informative_pairs:
                pair = self.laws[s,a]
                weights = masses[:, s]
                belief = weights / weights.sum()
                p = np.array([[law[k] for k in pair[0]] for law in pair])
                mixture = belief @ p
                divergence = kl_div(p, mixture[None, :]).sum(axis=1)
                stage[s, a] = max(0., float(weights @ divergence))
        self._mi_cache = key, stage
        return stage

    def log_likelihood(self, state, action, next_observation, reward, done):
        key = int(next_observation), float(reward), bool(done)
        pair = self.laws[int(state), int(action)]
        if any(key not in law for law in pair):
            raise ValueError("Observed transition is outside the declared model family")
        return tuple(math.log(law[key]) for law in pair)

    def seed_policies(self):
        names = ([p.name for p in deepsea.POLICY_SPECS] if self.kind == "deepsea" else frozenlake.POLICY_NAMES)
        policies = []
        for name in names:
            probabilities = np.zeros(self.mask.shape, dtype=float)
            for s, obs in enumerate(self.state_ids):
                state = self._models[0].decode(int(obs))
                a = (deepsea.policy_action(name, state, self.cfg) if self.kind == "deepsea"
                     else frozenlake.policy_action(name, state, self.cfg))
                if not self.mask[s, a]:
                    a = int(np.flatnonzero(self.mask[s])[0])
                probabilities[s, a] = 1.
            policies.append(Policy(probabilities, name))
        if self.kind == "deepsea" and self.benchmark != "legacy":
            # A coherent broad-coverage policy, shared by all policy-bank
            # methods. Its count stays fixed as the nuisance alphabet grows.
            p = np.zeros(self.mask.shape)
            p[:, deepsea.TV] = 1.
            policies.append(Policy(p, "sonar-sweep"))
        return policies
