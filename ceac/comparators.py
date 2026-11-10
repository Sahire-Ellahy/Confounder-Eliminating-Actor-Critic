"""Finite-policy witnesses for historical benchmark audits; no comparator learners.

The custom policy-bank UCB, confidence, coverage and OIR agents were retired.
Their exact original source remains in version history and study source bundles.
Active trajectory-only controls live in primitives.py.
"""
from __future__ import annotations
import numpy as np
from .design import PolicyBank, bounded_design


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
