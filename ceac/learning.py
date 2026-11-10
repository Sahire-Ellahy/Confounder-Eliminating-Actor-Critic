"""On-policy actor/two-critic update, likelihood evidence and single-credit debt."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from scipy.special import logsumexp
from .contracts import Policy


class LikelihoodEvidence:
    """Finite anytime confidence set from pairwise likelihood-ratio martingales.

Under any true declared model, Ville + a union bound gives P(ever excluded)
<= delta. All candidate log scores start equally; this is not an IDS prior.
"""
    def __init__(self, models=2, delta=.05):
        if models < 2 or not 0 < delta < 1:
            raise ValueError("Invalid confidence specification")
        self.scores = np.zeros(models)
        self.threshold = math.log((models - 1) / delta)
        self.delta = delta
        self.strata = defaultdict(list)
        self._seen = set()

    @property
    def plausible(self):
        return tuple(np.flatnonzero(self.scores.max() - self.scores <= self.threshold))

    @property
    def mixture(self):
        return np.exp(self.scores - logsumexp(self.scores))

    def append(self, episode):
        key = episode.context, episode.index
        if key in self._seen:
            raise ValueError("An episode cannot be credited to replay twice")
        self._seen.add(key)
        self.strata[episode.policy_id].append(episode)
        self.scores += episode.log_likelihood
        self.scores -= self.scores.max()

    def stratum_statistics(self, policy_id, reference, challenger):
        episodes = self.strata[policy_id]
        if not episodes:
            return {"count": 0, "return_mean": None, "llr_mean": None}
        return {"count": len(episodes), "return_mean": float(np.mean([e.total_return for e in episodes])),
                "llr_mean": float(np.mean([e.log_likelihood[reference]-e.log_likelihood[challenger] for e in episodes]))}


class DebtLedger:
    def __init__(self):
        self.quotas = defaultdict(float)
        self.counts = defaultdict(int)
        self.statistical_required = defaultdict(int)
        self.statistical_assigned = defaultdict(int)
        self.scale = 0.
        self._seen = set()

    def require_seed(self, identity):
        self.statistical_required[identity] = max(1, self.statistical_required[identity])

    def request_fresh(self, identity):
        self.statistical_required[identity] = max(self.statistical_required[identity], self.statistical_assigned[identity] + 1)

    def add_design(self, design, scale):
        if scale < self.scale - 1e-12 or not np.isfinite(scale):
            raise ValueError("Evidence scale cannot decrease")
        minimum = min(1., design.row_minimum)
        if minimum <= 0:
            raise ValueError("Uncertified zero-coverage design")
        increment = max(0., scale - self.scale) / minimum
        for identity, weight in zip(design.policy_ids, design.eta):
            if weight < -1e-10:
                raise ValueError("Negative design mass")
            self.quotas[identity] += increment * max(0., float(weight))
        self.scale = float(scale)

    def select(self):
        stats = {p: n - self.statistical_assigned[p] for p, n in self.statistical_required.items()}
        if stats and max(stats.values()) > 0:
            return max(stats, key=stats.get), "statistical"
        debt = {p: q - self.counts[p] for p, q in self.quotas.items()}
        if debt and max(debt.values()) > 1e-12:
            return max(debt, key=debt.get), "design"
        return None, "actor"

    def credit(self, episode):
        key = episode.context, episode.index
        if key in self._seen:
            raise ValueError("Double credit is forbidden")
        self._seen.add(key)
        p = episode.policy_id
        self.counts[p] += 1
        # Every trajectory of this immutable policy is compatible with the
        # stationary certificate stratum; one physical episode earns one count.
        self.statistical_assigned[p] += 1

    def record(self):
        ids = set(self.quotas) | set(self.statistical_required)
        return {p: {"quota": self.quotas[p], "count": self.counts[p],
                    "statistical_required": self.statistical_required[p],
                    "statistical_assigned": self.statistical_assigned[p]} for p in sorted(ids)}


def evidence_scale(t, inflation=1.):
    t = max(float(t), 1.)
    gamma = inflation * math.sqrt(math.log(math.log(max(t, math.exp(math.e)))) /
                                   math.log(max(t, math.exp(2.))))
    return (1 + gamma) * math.log(max(t, 2.))


class ActorCritic:
    """Frozen-episode score updates with unpriced, reference-indexed critics.

V_I(beta)=sum_(m,j) beta[m,j] V[m,j]. This makes beta=0 recover the reward-only
update exactly, including after previous nonzero prices or reference changes.
"""
    def __init__(self, model, temperature=1., actor_rate=.08, critic_rate=.3, initial_policy=None):
        if min(temperature, actor_rate, critic_rate) <= 0:
            raise ValueError("Positive actor/critic parameters required")
        self.model, self.temperature = model, float(temperature)
        self.actor_rate, self.critic_rate = actor_rate, critic_rate
        if initial_policy is None:
            self.logits = np.zeros(model.mask.shape)
        else:
            uniform = model.mask / model.mask.sum(axis=1, keepdims=True)
            p = .9 * initial_policy.probabilities + .1 * uniform
            self.logits = np.zeros_like(p)
            self.logits[model.mask] = temperature * np.log(p[model.mask])
        self.reward_value = np.zeros(model.n)
        self.row_values = np.zeros((2, 2, model.n))
        self.updates = 0

    def snapshot(self):
        logits = np.where(self.model.mask, self.logits/self.temperature, -np.inf)
        p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return Policy(p, "actor-snapshot")

    def update(self, episode, policy, prices):
        if episode.behavior != "actor":
            return False
        if episode.policy_id != policy.identity:
            raise ValueError("Actor episode does not match its frozen behavior policy")
        if policy.identity != self.snapshot().identity:
            raise ValueError("Off-policy actor replay is not allowed")
        prices = np.asarray(prices)
        if prices.shape != (2, 2) or (prices < 0).any():
            raise ValueError("Invalid frozen episode prices")
        actor_delta = np.zeros_like(self.logits)
        critic_r, critic_i = [], []
        for step in episode.transitions:
            s, nxt = step.state, step.next_state
            dr = step.reward - self.reward_value[s]
            di = np.subtract.outer(step.log_likelihood, step.log_likelihood) - self.row_values[:, :, s]
            if not step.done:
                dr += self.reward_value[nxt]
                di += self.row_values[:, :, nxt]
            score = -policy.probabilities[s].copy(); score[step.action] += 1.
            score /= self.temperature
            actor_delta[s] += (dr + float(np.sum(prices * di))) * score
            critic_r.append((s, dr)); critic_i.append((s, di))
        actor_step = self.actor_rate / (1 + self.updates/100.)**.75
        critic_step = self.critic_rate / (1 + self.updates/100.)**.6
        # All TD errors and scores above precede *either* parameter update.
        self.logits += actor_step * actor_delta
        for s, dr in critic_r:
            self.reward_value[s] += critic_step * dr
        for s, di in critic_i:
            self.row_values[:, :, s] += critic_step * di
        self.updates += 1
        if not np.isfinite(self.logits).all():
            raise FloatingPointError("Actor update diverged")
        return True
