"""IDS with information about the optimal action (Russo & Van Roy), not the full model.

`ceac_hardness.ids.BayesianIDS` uses I(theta; Z), which coincides with I(A*; Z)
only when every model has a distinct optimal policy. In product classes with
decision-irrelevant nuisance this is a different algorithm, so this control
uses I(A*; Z) where A* is the optimal-policy class. The stage-wise formula
I(C; Y) = sum_c p_c KL(P_c || P_mix) is exact here because every informative
(state, action) pair is at the initial state (checked).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.special import kl_div, logsumexp
from ceac_hardness.ids import BayesianIDS, best_pair


class ActionInformationIDS(BayesianIDS):
    def __init__(self, model, rng, prior):
        super().__init__(model, rng, prior)
        if any(s != model.initial for s, _ in model._informative_pairs):
            raise ValueError("Optimal-action IDS stage formula requires root-only information")
        groups = defaultdict(list)
        for m, policy in enumerate(model.optimal_policies):
            groups[policy.identity].append(m)
        self.classes = [np.array(v) for v in groups.values()]
        self._laws = {(s, a): np.array([[law[k] for k in model.laws[s, a][0]] for law in model.laws[s, a]])
                      for s, a in model._informative_pairs}

    def action_information_stage(self, posterior):
        stage = np.zeros(self.model.mask.shape)
        masses = np.array([posterior[c].sum() for c in self.classes])
        for (s, a), probs in self._laws.items():
            mix = posterior @ probs
            cls = np.array([posterior[c] @ probs[c] / max(w, 1e-300) for c, w in zip(self.classes, masses)])
            stage[s, a] = max(0.0, float(masses @ kl_div(cls, mix[None, :]).sum(axis=1)))
        return stage

    def distribution(self):
        posterior = self.posterior
        key = tuple(self.log_weights)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        stage_mi = self.action_information_stage(posterior)
        stage_return = np.einsum("m,ms,msa->sa", posterior, self.model.density, self.model.reward)
        for _ in range(128):
            returns, _ = self.bank.arrays()
            regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - returns)
            information = np.maximum(0.0, np.array([self.model.evaluate_stage(p, stage_mi) for p in self.bank.policies]))
            if regret.min() <= self.regret_tolerance or not np.any(information > 0):
                distribution = np.zeros(len(regret)); distribution[int(np.argmin(regret))] = 1.0
                d, g, residual = float(distribution @ regret), float(distribution @ information), 0.0
                break
            distribution = best_pair(regret, information)
            d, g = float(distribution @ regret), float(distribution @ information)
            if d == 0:
                residual = 0.0
                break
            price = d / (2 * g)
            proposal, _ = self.model.response_stage(stage_return + price * stage_mi, "ids-action-separator")
            new_returns, _ = self.model.evaluate(proposal)
            new_d = float(posterior @ np.maximum(0.0, self.model.optimal_returns - new_returns))
            new_g = max(0.0, self.model.evaluate_stage(proposal, stage_mi))
            residual = max(0.0, d - new_d + price * (new_g - g))
            if residual <= 1e-7 * d:
                break
            if not self.bank.add(proposal):
                raise RuntimeError("Optimal-action IDS certificate did not close")
        else:
            raise RuntimeError("Optimal-action IDS separation exceeded insertion limit")
        self.record = {"regret": d, "information": g, "ratio": 0.0 if d == 0 else (d * d / g if g > 0 else None),
                       "column_residual": residual, "policy_count": len(self.bank.policies),
                       "information_target": "optimal_action_class", "classes": len(self.classes)}
        self._cache = key, distribution
        return distribution
