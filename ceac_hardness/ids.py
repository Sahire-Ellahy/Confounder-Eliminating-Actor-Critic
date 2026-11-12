# Dimension-general port of ceac/ids.py; see docs/hardness_implementation.md.
"""Bayesian IDS with exact pair minimization and a full-policy certificate."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .design import PolicyBank


def best_pair(regret, information):
    """Minimize (p.d)^2/(p.g) over a finite bank, without a mixture grid.

Independent rescaling prevents tiny posterior tails from being rounded to a
zero-information branch. A zero-regret/zero-information policy is assigned
ratio zero, the explicit IDS convention used in the support-hole control.
"""
    d, g = np.asarray(regret, dtype=float), np.asarray(information, dtype=float)
    if d.shape != g.shape or not len(d) or not np.isfinite(d).all() or not np.isfinite(g).all() or (d < 0).any() or (g < 0).any():
        raise ValueError("Invalid IDS objective")
    zero = np.flatnonzero(d == 0)
    if len(zero):
        p = np.zeros(len(d)); p[zero[0]] = 1.
        return p
    if not np.any(g > 0):
        raise ValueError("All policies have positive regret and zero information")
    ds, gs = d / d.max(), g / g.max()
    best, distribution = float("inf"), None
    for i in range(len(d)):
        for j in range(i, len(d)):
            a, b, c, slope = ds[i], ds[j]-ds[i], gs[i], gs[j]-gs[i]
            candidates = [0., 1.]
            if b != 0 and slope != 0:
                root = (slope*a - 2*b*c)/(b*slope)
                if 0 < root < 1:
                    candidates.append(root)
            for x in candidates:
                den = c + slope*x
                score = (a+b*x)**2/den if den > 0 else float("inf")
                if score < best:
                    best = score; distribution = np.zeros(len(d))
                    distribution[i] += 1-x; distribution[j] += x
    if distribution is None:
        raise RuntimeError("IDS optimization failed")
    return distribution


class BayesianIDS:
    def __init__(self, model, rng, prior=.5):
        self.model, self.rng = model, rng
        p = (np.array([1-prior,prior],dtype=float) if np.ndim(prior)==0 else np.asarray(prior,dtype=float))
        if p.shape!=(model.num_models,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):
            raise ValueError("Prior must be a normalized model-class vector")
        self.regret_tolerance=1e-12
        self.posterior_underflow_entries=0
        self.log_weights = np.full(model.num_models, -np.inf)
        self.log_weights[p > 0] = np.log(p[p > 0])
        self.initial_support = p > 0
        self.bank = PolicyBank(model, model.seed_policies() + model.optimal_policies)
        self.record = {}
        self._cache = None

    @property
    def posterior(self):
        p = np.exp(self.log_weights-logsumexp(self.log_weights))
        # Log weights retain all initially supported models. Underflow here is
        # numerical evaluation only, never an absorbing support update.
        self.posterior_underflow_entries=int(np.sum((p==0)&np.isfinite(self.log_weights)))
        return p

    def distribution(self):
        posterior = self.posterior
        key = tuple(self.log_weights)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        stage_mi = self.model.mutual_information_stage(posterior)
        stage_return = np.einsum("m,ms,msa->sa", posterior, self.model.density, self.model.reward)
        for insertion in range(128):
            returns, _ = self.bank.arrays()
            gaps = np.maximum(0., self.model.optimal_returns[:, None]-returns)
            regret = posterior @ gaps
            information = np.array([self.model.evaluate_stage(p, stage_mi) for p in self.bank.policies])
            information = np.maximum(0., information)
            if regret.min() <= self.regret_tolerance:
                distribution=np.zeros(len(regret));distribution[int(np.argmin(regret))]=1.
                d=float(distribution@regret);g=float(distribution@information);residual=0.
                break
            distribution = best_pair(regret, information)
            d, g = float(distribution @ regret), float(distribution @ information)
            if d == 0:
                residual = 0.
                break
            if g <= 0:
                raise RuntimeError("Positive-regret IDS solution has no information")
            price = d/(2*g)
            proposal, _ = self.model.response_stage(stage_return + price*stage_mi, "ids-separator")
            new_returns, _ = self.model.evaluate(proposal)
            new_d = float(posterior @ np.maximum(0., self.model.optimal_returns-new_returns))
            new_g = max(0., self.model.evaluate_stage(proposal, stage_mi))
            residual = max(0., d-new_d+price*(new_g-g))
            if residual <= 1e-7*d:
                break
            if not self.bank.add(proposal):
                raise RuntimeError("IDS full-policy certificate did not close at posterior scale")
        else:
            raise RuntimeError("IDS separation exceeded insertion limit")
        self.record = {"posterior": posterior.tolist(), "regret": d, "information": g,
                       "ratio": 0. if d == 0 else (d*d/g if g>0 else None), "column_residual": residual,
                       "policy_count": len(self.bank.policies), "prior_support": self.initial_support.tolist(),
                       "expected_regret_tolerance":self.regret_tolerance,
                       "posterior_underflow_entries":self.posterior_underflow_entries,
                       "solver_status":"expected_regret_below_tolerance" if d<=self.regret_tolerance else "ratio_certified"}
        self._cache = key, distribution
        return distribution

    def select(self, index):
        del index
        distribution = self.distribution()
        p = self.bank.policies[int(self.rng.choice(len(distribution), p=distribution))]
        return p, "ids", np.zeros((self.model.num_models,self.model.num_models))

    def observe(self, episode, policy, prices):
        del policy, prices
        likelihood = episode.log_likelihood
        if np.all(likelihood==likelihood[0]):
            return  # Common likelihood factors cannot change the posterior.
        self.log_weights += likelihood
        self.log_weights -= logsumexp(self.log_weights)

    def diagnostics(self):
        return dict(self.record, posterior=self.posterior.tolist(), log_posterior=self.log_weights.tolist())
