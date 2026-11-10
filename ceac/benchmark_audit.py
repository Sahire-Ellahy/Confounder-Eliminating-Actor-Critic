"""Pre-training scientific checks on public benchmark kernels and objectives."""
from __future__ import annotations
import hashlib
import numpy as np
from .comparators import public_comparison_bank, trajectory_return_range
from .design import PolicyBank, bounded_design


def normalized_occupancy(model, policy, mode):
    """E[episode state counts / realized episode length], exactly.

    E[counts]/E[length] is different for variable-length episodes. The backward
    factor here is E[1/T | visit s], paired with the forward visitation mass.
    The finite-horizon state includes elapsed time and cannot repeat.
    """
    model.validate_policy(policy)
    source = model.edge // model.actions
    edge_weight = policy.probabilities.ravel()[model.edge] * model.prob[mode]
    forward = np.zeros(model.n+1); forward[model.initial] = 1.
    elapsed = np.empty(model.n, dtype=int)
    for t, states in enumerate(model.levels):
        elapsed[states] = t
    for t in range(len(model.levels)):
        selected = elapsed[source] == t
        forward += np.bincount(model.destination[selected],
                               weights=forward[source[selected]]*edge_weight[selected],
                               minlength=model.n+1)
    terminal = model.destination == model.n
    terminal_probability = np.bincount(model.edge[terminal], weights=model.prob[mode,terminal],
                                      minlength=model.n*model.actions).reshape(model.mask.shape)
    reciprocal = np.zeros(model.n+1)
    for t in range(len(model.levels)-1, -1, -1):
        states = model.levels[t]
        q = model._continuation(reciprocal, mode)[states] + terminal_probability[states]/(t+1)
        reciprocal[states] = np.sum(policy.probabilities[states]*q, axis=1)
    occupancy = forward[:model.n]*reciprocal[:model.n]
    if not np.isclose(occupancy.sum(), 1., atol=1e-10):
        raise ValueError("Episode-normalized occupancy does not sum to one")
    return occupancy


def entropy(distribution):
    p = distribution[distribution > 0]
    return float(-p @ np.log(p))


def audit_model(model, mass_bound=None):
    if mass_bound is None:
        mass_bound = 20. if model.benchmark == "legacy" else 128.
    bank = public_comparison_bank(model)
    low, high = trajectory_return_range(model)
    if high >= 1.25:
        raise ValueError("OIR reference cost does not bound the episodic return")
    rows = []
    for policy in bank.policies:
        returns, information = model.evaluate(policy)
        h = [entropy(normalized_occupancy(model, policy, m)) for m in (0,1)]
        rows.append({"label": policy.label, "policy_id": policy.identity,
                     "expected_return": returns.tolist(),
                     "task_kl": [float(information[m,1-m]) for m in (0,1)],
                     "occupancy_entropy": h,
                     "oir_objective": [(1.25-float(returns[m]))/(.25+h[m]) for m in (0,1)]})
    optimum_objectives = []
    for m in (0,1):
        h = entropy(normalized_occupancy(model, model.optimal_policies[m], m))
        optimum_objectives.append((1.25-float(model.optimal_returns[m]))/(.25+h))
    safe_label = "exploit" if model.kind == "deepsea" else "inner-bridge"
    risky_label = "challenge" if model.kind == "deepsea" else "outer-causeway"
    nuisance_label = "sonar-sweep" if model.kind == "deepsea" and model.benchmark != "legacy" else ("tv" if model.kind == "deepsea" else "reed-scan")
    seed = {p.label:p for p in model.seed_policies()}
    safe, nuisance = seed[safe_label], seed[nuisance_label]
    if not np.allclose(model.evaluate(safe)[1], 0., atol=1e-12):
        raise ValueError("Deployed safe route is not confounded")
    if not np.allclose(model.evaluate(nuisance)[1], 0., atol=1e-12):
        raise ValueError("Nuisance route leaks task information")
    if np.any(model.optimal_returns-model.evaluate(nuisance)[0] <= 0):
        raise ValueError("Nuisance route is not strictly costly")
    design_bank = PolicyBank(model, model.seed_policies())
    designs = [bounded_design(model, design_bank, (m,), mass_bound) for m in (0,1)]
    risky_j, risky_k = model.evaluate(seed[risky_label])
    risky_cost_per_nat = (model.optimal_returns[0]-risky_j[0])/risky_k[0,1]
    dominating = [r["label"] for r in rows if max(r["task_kl"]) < 1e-12 and
                  r["expected_return"][0] < model.optimal_returns[0]-1e-10 and
                  r["oir_objective"][0] < optimum_objectives[0]-1e-10]
    if model.benchmark != "legacy":
        gap = model.specification["configuration_overrides"]
        signal = (.5-gap["challenge_prob_mode0"] if model.kind == "deepsea" else .502-gap["outer_success_normal"])
        if not np.isclose(model.optimal_returns[0]-risky_j[0], signal, atol=1e-10):
            raise ValueError("Risky-route regret differs from benchmark specification")
        if not designs[0].cost < risky_cost_per_nat:
            raise ValueError("Targeted diagnosis is not cheaper than optimistic route trials")
    return {"benchmark": model.benchmark, "environment": model.kind, "context": model.context,
            "specification": model.specification, "reachable_states": model.n,
            "state_domain_sha256": hashlib.sha256(model.state_ids.tobytes()).hexdigest(),
            "action_mask_sha256": hashlib.sha256(model.mask.tobytes()).hexdigest(),
            "design_mass_bound": mass_bound, "seed_policies": len(seed), "public_bank_size": len(bank.policies),
            "return_range": [low,high], "optimal_returns": model.optimal_returns.tolist(),
            "safe_route_kl_zero": True, "nuisance_route_kl_zero": True,
            "optimal_policy_oir_objective": optimum_objectives,
            "zero_information_policies_beating_reward_optimum_in_oir_mode0": dominating,
            "risky_route_regret_per_nat_mode0": float(risky_cost_per_nat),
            "oracle_design_costs": [d.cost for d in designs],
            "risky_to_optimal_information_cost_ratio_mode0": float(risky_cost_per_nat/designs[0].cost),
            "design_certificates": [d.record() for d in designs], "policies": rows}


def audit_strength_family(audits):
    """Reject a strength sweep that also changes useful signals or the domain."""
    from .benchmarks import STRENGTH_PROFILES
    from collections import defaultdict
    groups = defaultdict(list)
    for row in audits:
        if row["benchmark"] in STRENGTH_PROFILES:
            groups[row["environment"], row["context"]].append(row)
    checked = []
    for (kind, context), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: STRENGTH_PROFILES[r["benchmark"]])
        reference = rows[0]
        useful = (("exploit", "challenge", "probe", "probe-adaptive") if kind == "deepsea" else
                  ("shore", "inner-bridge", "outer-causeway", "deep-survey"))
        def fingerprint(row):
            override = dict(row["specification"]["configuration_overrides"])
            override.pop("nuisance_probability")
            return (override, row["reachable_states"], row["seed_policies"],
                    row["state_domain_sha256"], row["action_mask_sha256"])
        ref_policies = {p["label"]: p for p in reference["policies"]}
        for row in rows:
            if fingerprint(row) != fingerprint(reference):
                raise ValueError("Physical strength comparison changes another setting or state/action domain")
            np.testing.assert_allclose(row["optimal_returns"], reference["optimal_returns"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(row["return_range"], reference["return_range"], atol=1e-10, rtol=0)
            policies = {p["label"]: p for p in row["policies"]}
            for label in useful:
                for metric in ("expected_return", "task_kl"):
                    np.testing.assert_allclose(policies[label][metric], ref_policies[label][metric], atol=1e-10, rtol=0)
        checked.append(dict(environment=kind, context=context,
                            strengths=[STRENGTH_PROFILES[r["benchmark"]] for r in rows],
                            fixed_domain=True, fixed_useful_route_values_and_KL=True,
                            fixed_optimum=True, fixed_return_range=True))
    return checked
