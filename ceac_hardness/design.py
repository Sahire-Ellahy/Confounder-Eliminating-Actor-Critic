# Dimension-general port of ceac/design.py; see docs/hardness_implementation.md.
"""Bounded Graves--Lai and finite target/calibration LPs with separation.

All solves fail closed on infeasibility, missing certificates, or an insertion
limit. No fallback silently substitutes the old fixed-bank design.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


class PolicyBank:
    def __init__(self, model, policies):
        self.model, self.policies, self._ids = model, [], {}
        for p in policies:
            self.add(p)

    def add(self, policy):
        identity = policy.identity
        if identity in self._ids:
            return False
        self.model.validate_policy(policy)
        self._ids[identity] = len(self.policies)
        self.policies.append(policy)
        return True

    def arrays(self):
        values = [self.model.evaluate(p) for p in self.policies]
        return np.stack([v[0] for v in values], axis=1), np.stack([v[1] for v in values], axis=2)


@dataclass
class Design:
    policy_ids: tuple[str, ...]
    eta: np.ndarray
    beta: np.ndarray
    mass_price: float
    cost: float
    row_minimum: float
    primal_dual_gap: float
    column_residual: float
    inserted: int
    references: tuple[int, ...]
    constant: float
    reward_weights: np.ndarray

    def record(self):
        return {"policy_ids": self.policy_ids, "eta": self.eta.tolist(), "beta": self.beta.tolist(),
                "mass_price": self.mass_price, "cost": self.cost, "row_minimum": self.row_minimum,
                "primal_dual_gap": self.primal_dual_gap, "column_residual": self.column_residual,
                "inserted": self.inserted, "references": self.references,
                "estimation": "exact_declared_model_evaluation"}


def bounded_design(model, bank, references, mass_bound=20., max_insertions=128):
    refs = tuple(sorted(references))
    if not refs or mass_bound <= 0:
        raise ValueError("A nonempty reference set and positive mass bound are required")
    rows = [r for r in model.alternatives if r[0] in refs]
    rw = np.zeros(model.num_models); rw[list(refs)] = 1.
    # U*=max_m J*_m and L(pi)=min_m J*_m-sum_m Delta_m(pi).
    # Thus U*-L is a linear, conservative truth-gap upper bound on the
    # simultaneous plausible-model event; it is exact for a singleton.
    opt = model.optimal_returns[list(refs)]
    constant = float(opt.max() - opt.min() + opt.sum())
    for insertion in range(max_insertions + 1):
        returns, information = bank.arrays()
        costs = np.maximum(0., constant - rw @ returns)
        k = np.array([information[m, j] for m, j in rows]).reshape(len(rows), -1)
        n = len(costs)
        a = np.vstack([-k, np.ones(n)])
        b = np.r_[-np.ones(len(rows)), mass_bound]
        res = linprog(costs, A_ub=a, b_ub=b, bounds=(0, None), method="highs")
        if not res.success:
            raise RuntimeError("Bounded master is infeasible; seed coverage/mass bound must be repaired: " + res.message)
        prices, mass_price = -res.ineqlin.marginals[:-1], float(-res.ineqlin.marginals[-1])
        beta = np.zeros((model.num_models, model.num_models))
        for row, price in zip(rows, prices):
            beta[row] = price
        policy, response = model.response(rw, beta, "gl-separator")
        residual = max(0., response - constant - mass_price)
        dual = float(prices.sum() - mass_bound * mass_price)
        gap = abs(float(res.fun) - dual)
        row_min = float(np.min(k @ res.x)) if rows else 1.
        if gap > 1e-7 or row_min < 1 - 1e-7 or res.x.sum() > mass_bound + 1e-7:
            raise RuntimeError("LP certificate failed")
        if residual <= TOL:
            return Design(tuple(p.identity for p in bank.policies), res.x, beta, mass_price,
                          float(res.fun), row_min, gap, residual, insertion, refs, constant, rw)
        if insertion == max_insertions or not bank.add(policy):
            raise RuntimeError("Policy separation did not close")
    raise AssertionError("unreachable")


def _aec_restricted(model, bank, xi, good, coefficients, epsilon, n_max, phase_one=False):
    """One epigraph LP. Phase I certifies infeasibility over omitted columns too."""
    n, g = len(bank.policies), len(good)
    returns, info = bank.arrays()
    excluded = tuple(m for m in range(model.num_models) if m not in good)
    v_index, width = 2 * n + g, 2 * n + g + 1
    rows, t_specs, c_specs = [], [], []
    tolerance = epsilon / 6.
    for pos, m in enumerate(good):
        a = np.zeros(width); a[:n] = model.optimal_returns[m] - returns[m]
        a[2*n+pos] = -(1+tolerance) * coefficients[m]
        rows.append(a)
        rw = np.zeros(model.num_models); rw[m] = -1.
        t_specs.append((rw, np.zeros((model.num_models, model.num_models)))); c_specs.append((np.zeros(model.num_models), np.zeros((model.num_models, model.num_models))))
        for ref, alt in model.alternatives:
            if ref != m:
                continue
            a = np.zeros(width); a[:n] = -info[ref, alt]; a[2*n+pos] = 1-tolerance
            rows.append(a)
            iw = np.zeros((model.num_models, model.num_models)); iw[ref, alt] = -1.
            t_specs.append((np.zeros(model.num_models), iw)); c_specs.append((np.zeros(model.num_models), np.zeros((model.num_models, model.num_models))))
    for m in excluded:
        a = np.zeros(width); a[n:2*n] = -np.einsum("j,ji->i", xi, info[:, m])
        a[v_index] = 1.
        rows.append(a)
        iw = np.zeros((model.num_models, model.num_models)); iw[:, m] = -xi
        t_specs.append((np.zeros(model.num_models), np.zeros((model.num_models, model.num_models)))); c_specs.append((np.zeros(model.num_models), iw))
    a = np.array(rows).reshape(-1, width)
    eq = np.zeros((2, width)); eq[0, :n] = 1.; eq[1, n:2*n] = 1.
    cost = np.zeros(width); cost[v_index] = -1. if excluded else 0.
    bounds = [(0., None)] * (2*n) + [(1./n_max, None)] * g + ([(0., None)] if excluded else [(0., 0.)])
    if phase_one:
        cost = np.r_[np.zeros(width), np.ones(len(rows))]
        a = np.hstack([a, -np.eye(len(rows))])
        eq = np.hstack([eq, np.zeros((2, len(rows)))])
        bounds += [(0., None)] * len(rows)
    result = linprog(cost, A_ub=a, b_ub=np.zeros(len(rows)), A_eq=eq,
                     b_eq=np.ones(2), bounds=bounds, method="highs")
    return result, t_specs, c_specs, excluded


def _aec_candidate(model, bank, xi, good, coefficients, epsilon, n_max):
    residuals = [0., 0.]
    for _ in range(128):
        res, target_specs, calibration_specs, excluded = _aec_restricted(
            model, bank, xi, good, coefficients, epsilon, n_max)
        phase_one = not res.success
        if phase_one:
            if res.status != 2:
                raise RuntimeError("AEC solver failed: " + res.message)
            res, target_specs, calibration_specs, excluded = _aec_restricted(
                model, bank, xi, good, coefficients, epsilon, n_max, True)
            if not res.success:
                raise RuntimeError("AEC phase I failed: " + res.message)
        inserted = False
        for side, specs in enumerate((target_specs, calibration_specs)):
            rw = sum((y * spec[0] for y, spec in zip(res.ineqlin.marginals, specs)), np.zeros(model.num_models))
            iw = sum((y * spec[1] for y, spec in zip(res.ineqlin.marginals, specs)), np.zeros((model.num_models, model.num_models)))
            proposal, score = model.response(rw, iw, "aec-target" if side == 0 else "aec-calibration")
            returns, info = bank.arrays()
            best_bank = float(np.max(rw @ returns + np.einsum("mj,mji->i", iw, info)))
            residuals[side] = max(0., score - best_bank)
            if residuals[side] > TOL:
                if not bank.add(proposal):
                    raise RuntimeError("Duplicate AEC separator with a positive residual")
                inserted = True
                break
        if inserted:
            continue
        if phase_one:
            if res.fun <= 1e-8:
                raise RuntimeError("AEC feasibility disagreement at numerical tolerance")
            return {"good": good, "feasible": False, "phase_one_slack": float(res.fun),
                    "target_gap": residuals[0], "calibration_gap": residuals[1]}
        n = len(bank.policies)
        return {"good": good, "feasible": True, "value": float(-res.fun),
                "target": res.x[:n], "calibration": res.x[n:2*n],
                "target_gap": residuals[0], "calibration_gap": residuals[1],
                "policy_ids": tuple(p.identity for p in bank.policies), "all_models_acceptable": not excluded}
    raise RuntimeError("AEC separation exceeded its insertion limit")


def target_calibration(model, bank, xi, coefficients, epsilon=.25, n_max=20.):
    if not 0 < epsilon <= .5 or n_max <= 0:
        raise ValueError("Invalid finite-wrapper normalization")
    xi = np.asarray(xi)
    if xi.shape != (model.num_models,) or (xi < 0).any() or not np.isclose(xi.sum(), 1.):
        raise ValueError("Invalid randomized model estimate")
    # Restart all candidate solves after insertions so every reported vector
    # and infeasibility certificate refers to the same final bank.
    for _ in range(32):
        before = len(bank.policies)
        candidates = [_aec_candidate(model, bank, xi, good, coefficients, epsilon, n_max)
                      for good in feasible_good_sets(model, bank, coefficients, epsilon, n_max)]
        if len(bank.policies) == before:
            break
    else:
        raise RuntimeError("AEC common-bank closure failed")
    feasible = [c for c in candidates if c["feasible"]]
    if not feasible:
        raise RuntimeError("No feasible target/calibration set")
    all_good = [c for c in feasible if c["all_models_acceptable"]]
    chosen = all_good[0] if all_good else max(feasible, key=lambda c: (c["value"], len(c["good"])))
    positive = np.asarray(coefficients)[np.asarray(coefficients) > 1e-12]
    g_min = float(positive.min()) if len(positive) else 0.
    q = (4*n_max + epsilon*g_min) / (4*n_max + 2*epsilon*g_min)
    chosen = dict(chosen)
    chosen["q"] = q
    chosen["mixture"] = q*chosen["target"] + (1-q)*chosen["calibration"]
    chosen["candidates"] = [{k: v for k, v in c.items() if k not in {"target", "calibration", "policy_ids"}} for c in candidates]
    return chosen


def feasible_good_sets(model, bank, coefficients, epsilon, n_max):
    """Exact maximal feasible target sets, with certified subset pruning.

    Target feasibility is independent of xi: calibration is feasible at v=0.
    A feasible superset dominates its subsets because the calibration objective
    is independent of the target mixture and has fewer excluded-model rows.
    Infeasibility is upward closed. No heuristic task grouping is used.
    Binary models keep the historical exhaustive ordering for parity.
    """
    count=model.num_models
    if count==2:
        return [(0,1),(0,),(1,),()]
    key=(tuple(float(c) for c in coefficients),float(epsilon),float(n_max))
    cache=getattr(model,"_good_sets_cache",{})
    if key not in cache:
        scratch=PolicyBank(model,model.seed_policies())
        xi=np.ones(count)/count; bad=[]; good=[]; certificates=[]
        def solve(indices):
            record=_aec_candidate(model,scratch,xi,tuple(indices),coefficients,epsilon,n_max)
            certificates.append({k:v for k,v in record.items() if k not in {"target","calibration","policy_ids"}})
            return record["feasible"]
        whole=tuple(range(count))
        if solve(whole):
            good=[set(whole)]
        else:
            # Cheap incompatible pairs prune most of the product-family lattice.
            for pair in combinations(range(count),2):
                if not solve(pair):bad.append(set(pair))
            for size in range(count-1,-1,-1):
                for indices in combinations(range(count),size):
                    candidate=set(indices)
                    if any(b<=candidate for b in bad) or any(candidate<=g for g in good):continue
                    if solve(indices):good.append(candidate)
                    else:bad.append(candidate)
        if not good:raise RuntimeError("No feasible AEC target set")
        cache[key]=(tuple(tuple(sorted(g)) for g in good),tuple(scratch.policies),certificates)
        model._good_sets_cache=cache
    groups,policies,certificates=cache[key]
    for policy in policies:bank.add(policy)
    model.good_set_certificates=certificates
    return groups
