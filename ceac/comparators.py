"""Confidence and occupancy controls with explicit information-access contracts.

The OIR/IDAC arm is a finite episodic, complete-policy adaptation of the
occupancy-information-ratio objective, not a claim to reproduce an original
infinite-horizon neural benchmark. It shares an enriched public policy bank
with coherent policy-UCB and uniform coverage, including the missing survey
route. Those three arms never use model likelihoods or true-model returns to
update their decisions.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp
from .design import PolicyBank, bounded_design
from .learning import LikelihoodEvidence


def _behavior_signature(model, policy):
    pending, visited, rows = [model.initial], set(), []
    while pending:
        s = pending.pop()
        if s in visited:
            continue
        visited.add(s)
        for a in np.flatnonzero(policy.probabilities[s]):
            rows.append((s, int(a), float(policy.probabilities[s, a])))
            for nxt, _, done in model.laws[s, int(a)][0]:
                if not done:
                    pending.append(model.index[nxt])
    return tuple(sorted(rows))


def public_comparison_bank(model):
    """Model-class-wide offline enrichment, independent of the realized mode."""
    if hasattr(model, "_comparison_bank"):
        return PolicyBank(model, model._comparison_bank)
    bank = PolicyBank(model, model.seed_policies()+model.optimal_policies)
    bank.add(model.response(np.array([.5, .5]), np.zeros((2, 2)), "mixture-deployment")[0])
    for references in ((0,), (1,), (0, 1)):
        bounded_design(model, bank, references, 20. if model.benchmark == "legacy" else 128.)
    bank.add(model.response(np.zeros(2), np.ones((2, 2)), "maximum-model-information",
                            secondary_reward_weights=(None if model.benchmark == "legacy" else np.array([.5,.5])))[0])
    seen, policies = set(), []
    for policy in bank.policies:
        signature = _behavior_signature(model, policy)
        if signature not in seen:
            policies.append(policy); seen.add(signature)
    model._comparison_bank = tuple(policies)
    return PolicyBank(model, policies)


def trajectory_return_range(model):
    """All-outcome episodic reward bounds, not expected-return bounds."""
    if hasattr(model, "_return_range"):
        return model._return_range
    lower, upper = np.zeros(model.n), np.zeros(model.n)
    for s in range(model.n-1, -1, -1):
        lows, highs = [], []
        for a in np.flatnonzero(model.mask[s]):
            for nxt, reward, done in model.laws[s, int(a)][0]:
                lows.append(reward + (0. if done else lower[model.index[nxt]]))
                highs.append(reward + (0. if done else upper[model.index[nxt]]))
        lower[s], upper[s] = min(lows), max(highs)
    model._return_range = float(lower[model.initial]), float(upper[model.initial])
    return model._return_range


class ModelConfidence:
    """Optimism over an anytime likelihood-confidence set, full policy class."""
    def __init__(self, model, rng, delta=.05):
        self.model, self.rng = model, rng
        self.evidence = LikelihoodEvidence(delta=delta)
        self.bank = PolicyBank(model, model.optimal_policies)
        self.chosen = None

    def select(self, index):
        del index
        plausible = self.evidence.plausible
        best = max(self.model.optimal_returns[m] for m in plausible)
        candidates = [m for m in plausible if self.model.optimal_returns[m] == best]
        self.chosen = int(self.rng.choice(candidates))
        return self.model.optimal_policies[self.chosen], "optimistic_confidence", np.zeros((2, 2))

    def observe(self, episode, policy, prices):
        del policy, prices
        self.evidence.append(episode)

    def diagnostics(self):
        return {"plausible": self.evidence.plausible, "optimistic_model": self.chosen,
                "information_access": "declared_models_and_own_likelihood_evidence"}


class UniformCoverage:
    def __init__(self, model, rng):
        self.bank, self.rng = public_comparison_bank(model), rng
        self.counts = np.zeros(len(self.bank.policies), dtype=int)

    def select(self, index):
        del index
        i = int(self.rng.integers(len(self.bank.policies)))
        return self.bank.policies[i], "uniform_coverage", np.zeros((2, 2))

    def observe(self, episode, policy, prices):
        del prices
        if policy.identity != episode.policy_id:
            raise ValueError("Behavior policy identity mismatch")
        self.counts[self.bank._ids[episode.policy_id]] += 1

    def diagnostics(self):
        return {"policy_ids": tuple(p.identity for p in self.bank.policies), "counts": self.counts.tolist(),
                "policy_class": "shared_enriched_public_bank", "information_access": "trajectory_only"}


class PolicyUCB(UniformCoverage):
    """Complete-policy UCB with an anytime Hoeffding union bound.

The radius uses the entire episodic reward range and count-dependent error
spending delta/(L * pi^2 * n^2/6). No hand-tuned 0.18 bonus is called a CI.
"""
    def __init__(self, model, rng, delta=.05):
        super().__init__(model, rng)
        self.delta = delta
        self.return_sum = np.zeros(len(self.bank.policies))
        self.lower, self.upper = trajectory_return_range(model)
        self.radius = np.full(len(self.bank.policies), self.upper-self.lower)

    def select(self, index):
        del index
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            chosen = int(untried[0])
        else:
            n = self.counts.astype(float)
            self.radius = (self.upper-self.lower)*np.sqrt(
                np.log(len(n)*math.pi**2*n*n/(3*self.delta))/(2*n))
            ucb = self.return_sum/n + self.radius
            candidates = np.flatnonzero(ucb == ucb.max())
            chosen = int(self.rng.choice(candidates))
        return self.bank.policies[chosen], "policy_ucb", np.zeros((2, 2))

    def observe(self, episode, policy, prices):
        super().observe(episode, policy, prices)
        if not self.lower-1e-10 <= episode.total_return <= self.upper+1e-10:
            raise ValueError("Observed return exceeds the declared confidence range")
        self.return_sum[self.bank._ids[episode.policy_id]] += episode.total_return

    def diagnostics(self):
        return dict(super().diagnostics(), confidence_delta=self.delta,
                    return_range=[self.lower, self.upper], radius=self.radius.tolist())


def occupancy_ratio_gradient(distribution, costs, occupancy, kappa):
    """Exact logit gradient of C(q)/(kappa+H(sum_i q_i d_i))."""
    mixture = distribution @ occupancy
    if np.any(mixture <= 0) or kappa <= 0:
        raise ValueError("Positive occupancy support and entropy offset are required")
    entropy = -float(mixture @ np.log(mixture))
    denominator = kappa + entropy
    cost = float(distribution @ costs)
    entropy_gradient = -occupancy @ (np.log(mixture)+1.)
    gradient_q = (costs*denominator-cost*entropy_gradient)/denominator**2
    gradient_logits = distribution*(gradient_q-float(distribution @ gradient_q))
    return gradient_logits, cost/denominator, entropy


class OIRIDAC(UniformCoverage):
    """Finite-policy occupancy-ratio adaptation; empirical returns/occupancies.

Occupancy is the expected distribution of a uniformly drawn pre-action state
within an episode. A total pseudocount of 0.25 per policy is spread across the
public state alphabet, rather than adding 0.25 to thousands of states each.
The episode-length normalization is explicit and differs from an infinite-
horizon stationary occupancy objective.
"""
    def __init__(self, model, rng, actor_step=.2, kappa=.25, occupancy_prior=.25, cost_reference=1.25):
        super().__init__(model, rng)
        if min(actor_step, kappa, occupancy_prior) <= 0:
            raise ValueError("Positive OIR settings required")
        if cost_reference <= trajectory_return_range(model)[1]:
            raise ValueError("OIR cost reference must exceed all possible episode returns")
        self.model = model
        self.actor_step, self.kappa, self.cost_reference = actor_step, kappa, cost_reference
        n = len(self.bank.policies)
        self.logits, self.return_sum = np.zeros(n), np.zeros(n)
        self.occupancy = np.full((n, model.n), occupancy_prior/model.n)
        self.last_ratio, self.last_entropy = None, None

    @property
    def distribution(self):
        return np.exp(self.logits-logsumexp(self.logits))

    def select(self, index):
        del index
        untried = np.flatnonzero(self.counts == 0)
        chosen = int(untried[0]) if len(untried) else int(self.rng.choice(len(self.logits), p=self.distribution))
        return self.bank.policies[chosen], "oir_policy_adaptation", np.zeros((2, 2))

    def observe(self, episode, policy, prices):
        super().observe(episode, policy, prices)
        i = self.bank._ids[episode.policy_id]
        self.return_sum[i] += episode.total_return
        states = np.array([step.state for step in episode.transitions], dtype=int)
        self.occupancy[i] += np.bincount(states, minlength=self.model.n)/len(states)
        if np.any(self.counts == 0):
            return
        costs = self.cost_reference-self.return_sum/self.counts
        occupancy = self.occupancy/self.occupancy.sum(axis=1, keepdims=True)
        gradient, self.last_ratio, self.last_entropy = occupancy_ratio_gradient(
            self.distribution, costs, occupancy, self.kappa)
        self.logits -= self.actor_step*gradient

    def diagnostics(self):
        return dict(super().diagnostics(), objective="episodic_cost_over_occupancy_entropy",
                    adaptation="finite_complete_policy_OIR_IDAC", policy_mass=self.distribution.tolist(),
                    ratio=self.last_ratio, occupancy_entropy=self.last_entropy,
                    cost_reference=self.cost_reference, kappa=self.kappa, actor_step=self.actor_step)
