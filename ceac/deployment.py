"""Reward-only deployment response, distinct from the pricing-policy proposer.

Appendix D defines a deployment response; Section 4 and Appendix G require its
reward-only certificate. The Section 3 selector does not spell out this bridge.
The maintained finite-model lane makes it explicit and retains actor-only
execution as a named ablation. No true-model input is accepted here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .contracts import Policy


@dataclass(frozen=True)
class DeploymentCertificate:
    policy: Policy
    references: tuple[int, ...]
    weights: tuple[float, ...]
    gaps: tuple[float, ...]
    tolerance: float

    @property
    def residual(self):
        return max(self.gaps)

    @property
    def certified(self):
        return self.residual <= self.tolerance

    def valid_for(self, plausible):
        return self.certified and bool(plausible) and set(plausible).issubset(self.references)

    def record(self):
        return {"policy_id": self.policy.identity, "references": self.references,
                "weights": self.weights, "model_gaps": self.gaps, "residual": self.residual,
                "tolerance": self.tolerance, "certified": self.certified,
                "scope": "reward_optimal_for_every_plausible_model"}


def certify_deployment(model, evidence, tolerance=1e-10):
    """Propose under the confidence-restricted estimate, verify each model.

The certificate itself does not depend on the choice of proposal weights:
max_{m in C} (J*_m - J_m(pi)) <= tolerance implies the same true-model bound
on the simultaneous model-confidence event. If C re-expands, recheck it.
"""
    refs = tuple(int(m) for m in evidence.plausible)
    if not refs or tolerance < 0:
        raise ValueError("Deployment needs a nonempty model set and nonnegative tolerance")
    weights = np.zeros(model.num_models)
    weights[list(refs)] = evidence.mixture[list(refs)]
    weights /= weights.sum()
    proposal, _ = model.response(weights, np.zeros((2, 2)), "certified-deployment")
    returns, _ = model.evaluate(proposal)
    gaps = tuple(float(max(0., model.optimal_returns[m]-returns[m])) for m in refs)
    return DeploymentCertificate(proposal, refs, tuple(float(x) for x in weights), gaps, tolerance)
