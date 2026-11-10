"""Trajectory-only, primitive-action comparators with explicit source provenance.

IDAC ports InfoRatioACBASE's two-critic softmax update. UCB-Q implements
Algorithm 1 of Jin et al. (2018); optimistic POMD follows Shani et al. (2020),
Algorithm 2. None receives a FiniteModel or task laws.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
from scipy.special import softmax
from .contracts import Policy


@dataclass(frozen=True)
class PrimitiveSpec:
    mask: np.ndarray
    elapsed: np.ndarray
    horizon: int
    initial: int
    reward_lower: float
    reward_upper: float

    def __post_init__(self):
        mask = np.array(self.mask, dtype=bool, copy=True)
        elapsed = np.array(self.elapsed, dtype=int, copy=True)
        if (mask.ndim != 2 or elapsed.shape != (len(mask),) or not mask.any(axis=1).all()
                or self.horizon < 1 or np.any(elapsed < 0) or np.any(elapsed >= self.horizon)
                or not 0 <= self.initial < len(mask) or elapsed[self.initial] != 0
                or not np.isfinite([self.reward_lower, self.reward_upper]).all()
                or not self.reward_lower <= 0 <= self.reward_upper
                or self.reward_lower >= self.reward_upper):
            raise ValueError("Invalid public primitive-action specification")
        mask.flags.writeable = elapsed.flags.writeable = False
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "elapsed", elapsed)

    @property
    def n(self):
        return len(self.mask)

    @property
    def width(self):
        return self.reward_upper-self.reward_lower

    @property
    def extended_mask(self):
        padding = np.zeros((self.horizon, self.mask.shape[1]), dtype=bool)
        padding[:, 0] = True
        return np.vstack((self.mask, padding))

    def padded_episode(self, episode):
        """Zero-reward absorbing padding; final successor resets to initial.

        The padding is analytic bookkeeping, never charged as a simulator call.
        Fixing length H preserves reward ordering under an affine reward/cost
        transformation and defines a continuing process for differential IDAC.
        """
        rows = []
        transitions = episode.transitions
        if not transitions or len(transitions) > self.horizon or not transitions[-1].done:
            raise ValueError("Expected a complete finite-horizon episode")
        if transitions[0].state != self.initial:
            raise ValueError("Unexpected initial state")
        for t, step in enumerate(transitions):
            if (not 0 <= step.state < self.n or not 0 <= step.action < self.mask.shape[1]
                    or self.elapsed[step.state] != t or not self.mask[step.state, step.action]
                    or step.done != (t == len(transitions)-1)):
                raise ValueError("Trajectory violates the public time/action contract")
            if not self.reward_lower-1e-12 <= step.reward <= self.reward_upper+1e-12:
                raise ValueError("Reward is outside the declared primitive range")
            if not step.done:
                if step.next_state != transitions[t+1].state:
                    raise ValueError("Disconnected trajectory")
                successor = step.next_state
            else:
                successor = self.initial if t+1 == self.horizon else self.n+t+1
            rows.append((step.state, step.action, step.reward, successor))
        for t in range(len(transitions), self.horizon):
            successor = self.initial if t+1 == self.horizon else self.n+t+1
            rows.append((self.n+t, 0, 0., successor))
        return rows


@dataclass(frozen=True)
class IDACConfig:
    policy_lr: float = .1
    value_lr: float = .2
    tau: float = .01
    kappa: float = .1
    entropy_eps: float = .0001
    batch_episodes: int = 16
    initial_variance: float = 1.
    cost_floor: float = .01

    def __post_init__(self):
        if (not all(np.isfinite(v) and v > 0 for v in
                    (self.policy_lr, self.value_lr, self.tau, self.kappa, self.entropy_eps, self.cost_floor))
                or self.tau > 1 or self.batch_episodes < 1 or self.initial_variance < 0
                or not np.isfinite(self.initial_variance)):
            raise ValueError("Invalid IDAC configuration")


@dataclass(frozen=True)
class UCBQConfig:
    bonus_constant: float = 1.
    delta: float = .05

    def __post_init__(self):
        if not np.isfinite(self.bonus_constant) or self.bonus_constant <= 0 or not 0 < self.delta < 1:
            raise ValueError("Invalid UCB-Q configuration")


class PrimitiveLearner:
    def __init__(self, spec):
        self.spec = spec
        # Runner persists all behavior snapshots; no oracle-enriched policy bank.
        self.bank = SimpleNamespace(policies=[])
        self.behavior_policy = None
        self.observed_episodes = 0
        self.observed_steps = 0

    def _publish(self, probabilities, label):
        self.behavior_policy = Policy(probabilities[:self.spec.n], label)
        self.bank.policies = [self.behavior_policy]
        return self.behavior_policy, label, np.zeros((2, 2))

    def _check(self, episode, policy):
        if self.behavior_policy is None or episode.policy_id != policy.identity or policy.identity != self.behavior_policy.identity:
            raise ValueError("Observed trajectory differs from the frozen behavior policy")
        rows = self.spec.padded_episode(episode)
        self.observed_episodes += 1
        self.observed_steps += len(episode.transitions)
        return rows


class IDAC(PrimitiveLearner):
    """Tabular softmax/two-critic port of upstream InfoRatioACBASE.update.

    Batched empirical occupancy, moving means, differential TD and the ratio
    policy gradient follow the source. The reset/padding adapter is explicit.
    """
    def __init__(self, spec, rng, config=None):
        super().__init__(spec)
        self.config = config or IDACConfig()
        self.mask = spec.extended_mask
        scale = math.sqrt(self.config.initial_variance)
        self.logits = rng.normal(0, scale, self.mask.shape)
        self.cost_value = rng.normal(0, scale, len(self.mask))
        self.shadow_value = rng.normal(0, scale, len(self.mask))
        self.mu_cost = self.mu_entropy = 0.
        self.pending = []
        self.updates = 0
        self.last_batch_cost = self.last_batch_entropy = None
        self.last_gradient_norm = None
        self.padded_steps = 0

    @property
    def probabilities(self):
        return softmax(np.where(self.mask, self.logits, -np.inf), axis=1)

    def select(self, index):
        del index
        return self._publish(self.probabilities, "idac")

    def update_batch(self, states, actions, costs, next_states):
        """One source-equivalent update, exposed for reference parity tests."""
        states, actions, next_states = (np.asarray(x, dtype=int) for x in (states, actions, next_states))
        costs = np.asarray(costs, dtype=float)
        if not len(states) or not (states.shape == actions.shape == costs.shape == next_states.shape):
            raise ValueError("Invalid IDAC rollout arrays")
        if (not np.isfinite(costs).all() or np.any(states < 0) or np.any(states >= len(self.mask))
                or np.any(next_states < 0) or np.any(next_states >= len(self.mask))
                or np.any(actions < 0) or np.any(actions >= self.mask.shape[1])
                or not self.mask[states, actions].all()):
            raise ValueError("IDAC rollout has invalid values/actions")
        counts = np.bincount(states, minlength=len(self.mask))
        shadow = -np.log(counts[states]/len(states))
        cfg = self.config
        mean_cost = float(costs.mean())
        mean_entropy = max(cfg.entropy_eps, float(shadow.mean()))
        self.mu_cost = (1-cfg.tau)*self.mu_cost + cfg.tau*mean_cost
        self.mu_entropy = (1-cfg.tau)*self.mu_entropy + cfg.tau*mean_entropy
        denominator = self.mu_entropy+cfg.kappa
        td_cost = costs-self.mu_cost+self.cost_value[next_states]-self.cost_value[states]
        td_shadow = shadow-self.mu_entropy+self.shadow_value[next_states]-self.shadow_value[states]
        weights = (td_cost*denominator-self.mu_cost*td_shadow)/(len(states)*denominator**2)
        score = -self.probabilities[states].copy()
        score[np.arange(len(states)), actions] += 1.
        gradient = np.zeros_like(self.logits)
        np.add.at(gradient, states, weights[:, None]*score)
        self.cost_value += cfg.value_lr*np.bincount(states, weights=td_cost, minlength=len(self.mask))/len(states)
        self.shadow_value += cfg.value_lr*np.bincount(states, weights=td_shadow, minlength=len(self.mask))/len(states)
        self.logits -= cfg.policy_lr*gradient
        if not all(np.isfinite(a).all() for a in (self.logits,self.cost_value,self.shadow_value)):
            raise FloatingPointError("Nonfinite IDAC update; no silent clipping or fallback")
        self.last_batch_cost, self.last_batch_entropy = mean_cost, float(shadow.mean())
        self.last_gradient_norm = float(np.linalg.norm(gradient))
        self.updates += 1

    def observe(self, episode, policy, prices):
        del prices
        rows = self._check(episode, policy)
        self.pending.extend(rows)
        self.padded_steps += len(rows)-len(episode.transitions)
        if self.observed_episodes % self.config.batch_episodes == 0:
            states, actions, rewards, successors = map(np.asarray, zip(*self.pending))
            costs = (self.spec.reward_upper-rewards)/self.spec.width+self.config.cost_floor
            self.update_batch(states, actions, costs, successors)
            self.pending.clear()

    def diagnostics(self):
        return dict(algorithm="upstream_tabular_IDAC", information_access="observed_primitive_trajectories_only",
                    actor_updates=self.updates, cost_critic_updates=self.updates, shadow_critic_updates=self.updates,
                    mu_cost=self.mu_cost, mu_entropy=self.mu_entropy,
                    ratio_estimate=self.mu_cost/(self.mu_entropy+self.config.kappa),
                    batch_cost=self.last_batch_cost, batch_entropy=self.last_batch_entropy,
                    gradient_norm=self.last_gradient_norm, pending_steps=len(self.pending),
                    observed_steps=self.observed_steps, analytic_padding_steps=self.padded_steps,
                    rollout_rule="frozen_softmax_policy_until_batch_episodes_complete")


class UCBQ(PrimitiveLearner):
    """Jin et al. 2018, Algorithm 1: greedy Q-learning with UCB-Hoeffding.

    Time is already encoded in every state. Forward replay of an episode is
    exactly the primitive online update: no time-indexed row is visited twice.
    It must not be used for an unaugmented cyclic state representation.
    """
    def __init__(self, spec, rng, episodes, config=None):
        super().__init__(spec)
        if episodes < 1:
            raise ValueError("Positive planned episode budget required")
        self.config = config or UCBQConfig()
        self.mask, self.rng = spec.extended_mask, rng
        self.q = np.full(self.mask.shape, float(spec.horizon))
        self.counts = np.zeros(self.mask.shape, dtype=int)
        self.log_factor = math.log(len(self.mask)*self.mask.shape[1]*spec.horizon*episodes/self.config.delta)
        self.updates = 0
        self.last_bonus = None

    def select(self, index):
        del index
        masked = np.where(self.mask, self.q, -np.inf)
        ties = masked == masked.max(axis=1, keepdims=True)
        # A frozen random tie break is a valid argmax, with no softmax alteration.
        chosen = np.argmax(np.where(ties, self.rng.random(ties.shape), -np.inf), axis=1)
        probabilities = np.zeros_like(self.q)
        probabilities[np.arange(len(probabilities)), chosen] = 1.
        return self._publish(probabilities, "ucb_q")

    def observe(self, episode, policy, prices):
        del prices
        rows = self._check(episode, policy)
        for t, (state, action, reward, successor) in enumerate(rows):
            self.counts[state, action] += 1
            n = self.counts[state, action]
            alpha = (self.spec.horizon+1)/(self.spec.horizon+n)
            bonus = self.config.bonus_constant*math.sqrt(self.spec.horizon**3*self.log_factor/n)
            value = (0. if t+1 == self.spec.horizon else
                     min(self.spec.horizon, self.q[successor, self.mask[successor]].max()))
            normalized_reward = (reward-self.spec.reward_lower)/self.spec.width
            self.q[state, action] = (1-alpha)*self.q[state, action]+alpha*(normalized_reward+value+bonus)
            self.last_bonus = float(bonus)
            self.updates += 1

    def diagnostics(self):
        return dict(algorithm="Jin2018_Algorithm1_UCB_Hoeffding", information_access="observed_primitive_trajectories_only",
                    updates=self.updates, physical_steps=self.observed_steps,
                    visited_state_actions=int(np.count_nonzero(self.counts)),
                    last_bonus=self.last_bonus, log_factor=self.log_factor,
                    bonus_constant=self.config.bonus_constant,
                    action_rule="greedy_argmax_with_random_ties", softmax=False)


@dataclass(frozen=True)
class POMDConfig:
    delta: float = .05
    policy_step: float | None = None

    def __post_init__(self):
        if not 0 < self.delta < 1 or (self.policy_step is not None and
                (not np.isfinite(self.policy_step) or self.policy_step <= 0)):
            raise ValueError("Invalid optimistic POMD configuration")


class OptimisticPOMD(PrimitiveLearner):
    """Shani/Efroni/Rosenberg/Mannor 2020, Algorithm 2, Appendix B bonuses.

    Stochastic bandit feedback; sparse empirical transition model; backward
    evaluation of the current softmax policy; KL mirror-descent improvement.
    Algorithm 2 ingests the current trajectory AFTER evaluation/improvement.
    """
    def __init__(self, spec, episodes, config=None):
        super().__init__(spec)
        if episodes < 1:
            raise ValueError("Positive planned episode budget required")
        self.config = config or POMDConfig()
        self.mask = spec.extended_mask
        self.elapsed = np.r_[spec.elapsed, np.arange(spec.horizon)]
        self.levels = [np.flatnonzero(self.elapsed == t) for t in range(spec.horizon)]
        self.logits = np.zeros(self.mask.shape)
        self.counts = np.zeros(self.mask.shape, dtype=int)
        self.cost_sum = np.zeros(self.mask.shape)
        self.transitions = {}
        self.q = np.zeros(self.mask.shape)
        self.values = np.zeros(len(self.mask)+1)
        S, A, H = len(self.mask), self.mask.shape[1], spec.horizon
        T, delta_prime = episodes*H, self.config.delta/3
        self.cost_log = math.log(2*S*A*H*T/delta_prime)
        self.transition_log = math.log(3*S*A*H*T/delta_prime)
        self.policy_step = (math.sqrt(2*math.log(A)/(H*H*episodes))
                            if self.config.policy_step is None else self.config.policy_step)
        self.updates = 0
        self.last_model_episodes = 0

    @property
    def probabilities(self):
        return softmax(np.where(self.mask, self.logits, -np.inf), axis=1)

    def select(self, index):
        del index
        return self._publish(self.probabilities, "pomd")

    def bonuses(self):
        n = np.maximum(1, self.counts)
        cost = np.sqrt(2*self.cost_log/n)
        transition = self.spec.horizon*np.sqrt(4*len(self.mask)*self.transition_log/n)
        return cost, transition

    def evaluate_and_improve(self):
        """Algorithm 2 policy evaluation/improvement using prior-episode data."""
        n = np.maximum(1, self.counts)
        cost_bonus, transition_bonus = self.bonuses()
        optimistic_cost = self.cost_sum/n-cost_bonus-transition_bonus
        policy = self.probabilities
        self.q.fill(0.)
        self.values.fill(0.)
        # Sparse counts only: unobserved transition rows have zero mass, as in
        # the paper's empirical-model definition with denominator max(1,N).
        if self.transitions:
            edges = np.asarray(list(self.transitions), dtype=int)
            mass = np.asarray(list(self.transitions.values()), dtype=float)
            sources, actions, destinations = edges.T
            edge_time = self.elapsed[sources]
        else:
            sources = actions = destinations = edge_time = np.empty(0, dtype=int)
            mass = np.empty(0)
        for t in range(self.spec.horizon-1, -1, -1):
            states = self.levels[t]
            selected = edge_time == t
            continuation = np.bincount(
                sources[selected]*self.mask.shape[1]+actions[selected],
                weights=mass[selected]*self.values[destinations[selected]],
                minlength=self.q.size).reshape(self.q.shape)/n
            self.q[states] = np.maximum(0., optimistic_cost[states]+continuation[states])
            self.q[states] *= self.mask[states]
            self.values[states] = np.sum(policy[states]*self.q[states], axis=1)
        self.logits -= self.policy_step*self.q
        # A state-wise additive logit constant leaves the policy unchanged.
        self.logits -= np.max(np.where(self.mask, self.logits, -np.inf), axis=1, keepdims=True)
        self.updates += 1

    def observe(self, episode, policy, prices):
        del prices
        rows = self._check(episode, policy)
        self.last_model_episodes = self.observed_episodes-1
        self.evaluate_and_improve()
        # Preserve the explicit one-episode lag in the published Algorithm 2.
        for t, (state, action, reward, successor) in enumerate(rows):
            self.counts[state, action] += 1
            self.cost_sum[state, action] += (self.spec.reward_upper-reward)/self.spec.width
            destination = len(self.mask) if t+1 == self.spec.horizon else successor
            edge = state, action, destination
            self.transitions[edge] = self.transitions.get(edge, 0)+1

    def diagnostics(self):
        return dict(algorithm="Shani2020_Algorithm2_optimistic_POMD",
                    information_access="observed_primitive_trajectories_only",
                    actor_updates=self.updates, policy_step=self.policy_step,
                    model_episodes_used=self.last_model_episodes,
                    observed_steps=self.observed_steps, learned_transition_edges=len(self.transitions),
                    visited_state_actions=int(np.count_nonzero(self.counts)),
                    optimistic_q_nonzero=int(np.count_nonzero(self.q[self.mask])),
                    all_optimistic_q_zero=bool(np.all(self.q[self.mask] == 0)),
                    initial_optimistic_cost=float(self.values[self.spec.initial]),
                    action_rule="softmax_KL_mirror_descent", softmax=True,
                    bonuses="Appendix_B_cost_and_transition_constants", delta=self.config.delta)
