#!/usr/bin/env python3
"""Read-only checks of the released v90 experiments and their declared MDPs.

Runs exact finite-horizon policy separation using the environment's own step
function. Writes only an explicitly requested audit JSON, never release data.
Requires NumPy and SciPy; no GPU, training sweep or network access is used.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np


class OutcomeRNG:
    """Select one branch of the benchmark's single-event transition kernel."""
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def random(self):
        self.calls += 1
        return 0.0 if self.outcome == 1 else np.nextafter(1.0, 0.0)

    def integers(self, upper):
        self.calls += 1
        assert 0 <= self.outcome < upper
        return self.outcome


class ExactKernel:
    """Enumerate (next observation, reward, done) from actual step() calls.

The experiments expose privileged evaluation metadata in info. This oracle
uses their declared observation/reward process, excluding that metadata.
"""
    def __init__(self, module, kind):
        self.module, self.kind = module, kind
        self.cfg = module.DeepSeaConfig() if kind == "deepsea" else module.FrozenShoreConfig()
        factory = module.ShiftingDeepSeaEnv if kind == "deepsea" else module.ShiftingShoreFrozenLakeEnv
        self.envs = [factory(replace(self.cfg, true_modes=(m,) * self.cfg.n_contexts)) for m in (0, 1)]
        self.initial = self.envs[0].reset(seed=0, options={"episode_index": 0})[0]
        self.envs[1].reset(seed=0, options={"episode_index": 0})

    def branch(self, observation, action, mode, outcome):
        env = self.envs[mode]
        decoded = env.decode(observation)
        for key, value in vars(decoded).items():
            setattr(env, key, value)
        env.episode_index = 0
        env._terminated = False
        if self.kind == "deepsea":
            env._episode_return = 0.0
        else:
            env.mode = mode
        env.np_random = OutcomeRNG(outcome)
        nxt, reward, term, trunc, info = env.step(action)
        assert env.np_random.calls <= 1, "Kernel now has multiple events; update enumeration"
        return (int(nxt), float(reward), bool(term or trunc)), info

    @lru_cache(None)
    def law(self, observation, action, mode):
        first, info = self.branch(observation, action, mode, 0)
        event = info["diagnostic_event"]
        if event in ("tv", "reeds"):
            count = self.cfg.tv_bins if event == "tv" else self.cfg.reed_bins
            outcomes = [(n, 1.0 / count) for n in range(count)]
        else:
            probabilities = ({"probe": self.cfg.probe_probability, "challenge": self.cfg.challenge_probability}
                             if self.kind == "deepsea" else
                             {"deep": self.cfg.deep_signal_probability, "inner": self.cfg.inner_success,
                              "outer": self.cfg.outer_success, "shallow": lambda _: self.cfg.shallow_signal_probability})
            if event not in probabilities:
                return {first: 1.0}
            p = probabilities[event](mode)
            outcomes = [(0, 1 - p), (1, p)]
        law = {}
        for outcome, weight in outcomes:
            key, _ = self.branch(observation, action, mode, outcome)
            law[key] = law.get(key, 0.0) + weight
        assert abs(sum(law.values()) - 1) < 1e-12
        return law

    @lru_cache(None)
    def transition(self, observation, action, mode):
        p = self.law(observation, action, mode)
        q = self.law(observation, action, 1 - mode)
        assert set(p) == set(q), "Declared common support no longer holds"
        kl = sum(weight * math.log(weight / q[key]) for key, weight in p.items() if weight)
        assert kl >= -1e-12
        return p, max(kl, 0.0)

    def response(self, mode, price):
        """Exact Bellman max of J + price*KL over mask-respecting policies."""
        @lru_cache(None)
        def value(observation):
            actions = np.flatnonzero(self.envs[mode].valid_action_mask(observation))
            best = None
            for action in actions:
                law, kl = self.transition(observation, int(action), mode)
                ret, information = 0.0, kl
                for (nxt, reward, done), probability in law.items():
                    j, i = (0.0, 0.0) if done else value(nxt)
                    ret += probability * (reward + j)
                    information += probability * i
                score = ret + price * information
                if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and information > best[2]):
                    best = score, ret, information
            assert best is not None
            return best[1], best[2]
        ret, information = value(self.initial)
        return {"return": ret, "information": information, "states": value.cache_info().currsize}

    def evaluate_fixed(self, mode, policy):
        @lru_cache(None)
        def value(observation):
            state = self.envs[mode].decode(observation)
            action = (self.module.policy_action(policy, state, self.cfg) if self.kind == "deepsea"
                      else self.module.policy_action(policy, state))
            assert self.envs[mode].valid_action_mask(observation)[action]
            law, information = self.transition(observation, action, mode)
            ret = 0.0
            for (nxt, reward, done), probability in law.items():
                j, i = (0.0, 0.0) if done else value(nxt)
                ret += probability * (reward + j)
                information += probability * i
            return ret, information
        return value(self.initial)

    def certificate(self, mode, library_coefficient):
        optimum = self.response(mode, 0)["return"]
        price = library_coefficient
        iterations = []
        witness = None
        for _ in range(64):
            response = self.response(mode, price)
            gap = max(0.0, optimum - response["return"])
            residual = max(0.0, response["return"] + price * response["information"] - optimum)
            iterations.append({"price": price, "global_reduced_cost_violation": residual, **response})
            if residual <= 1e-10:
                break
            assert response["information"] > 0
            witness = {**response, "gap": gap}
            price = gap / response["information"]
        else:
            raise AssertionError("Oracle separation did not close")
        if witness is None:
            witness = {**response, "gap": gap}
        return {"optimal_return": optimum, "coefficient": price, "iterations": iterations,
                "last_improving_witness": witness,
                "scope": "declared binary models, full mask-respecting finite-horizon Markov policy class"}


def audit(root):
    release = root / "doc/ceac_v90_accessibility_release"
    experiments = release / "experiments"
    shore = experiments / "shifting_shore_full_slate"
    sys.path[:0] = [str(experiments), str(shore)]
    import shifting_deepsea_gym as dg
    import shifting_deepsea_agents as da
    import shifting_shore_frozen_lake_gym as fg
    import shifting_shore_agents as fa

    result = {"reference": "ceac_v90_accessibility_release", "source_hashes": {}}
    sources = [*experiments.glob("*.py"), *shore.glob("*.py"), release / "sections/03_algorithms.tex",
               release / "appendices/D_online_tracking.tex", release / "appendices/F_experiment_details.tex"]
    for path in sorted(sources):
        result["source_hashes"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Exact oracle comparison, including a full policy-separation check.
    for kind, module in (("deepsea", dg), ("frozenlake", fg)):
        kernel = ExactKernel(module, kind)
        names = da.DESIGN_POLICIES if kind == "deepsea" else fg.POLICY_NAMES
        rows = []
        for mode in (0, 1):
            information = [module.policy_information(kernel.cfg, p, mode) if kind == "deepsea" else
                           module.policy_information(kernel.cfg, p, mode, 1 - mode) for p in names]
            gaps = [module.policy_gap(kernel.cfg, p, mode) for p in names]
            checks = []
            for policy, kl in zip(names, information):
                ret, measured_kl = kernel.evaluate_fixed(mode, policy)
                return_error = abs(ret - module.expected_policy_return(kernel.cfg, policy, mode))
                kl_error = abs(kl - measured_kl)
                assert max(return_error, kl_error) < 1e-10, (kind, mode, policy, return_error, kl_error)
                checks.append({"policy": policy, "return_error": return_error, "kl_error": kl_error})
            coefficient = min(g / i for g, i in zip(gaps, information) if i > 0)
            cert = kernel.certificate(mode, coefficient)
            assert abs(cert["optimal_return"] - module.optimal_expected_return(kernel.cfg, mode)) < 1e-10
            row = {"mode": mode, "library_coefficient": coefficient, "full_policy_certificate": cert,
                   "fixed_policy_kernel_checks": checks}
            if kind == "frozenlake":
                eta, beta, matrix, costs = fa.solve_glc_design(kernel.cfg, mode, [1 - mode])
                row["library_lp"] = {"primal": float(costs @ eta), "dual": float(beta.sum()),
                                     "row_surplus": float(np.min(matrix @ eta - 1)), "eta": eta.tolist()}
                assert abs(float(costs @ eta) - coefficient) < 1e-10
            rows.append(row)
        result[kind + "_oracles"] = rows
        print(kind, "exact policy separation finished", flush=True)

    # An executable missing column, independent of the dynamic-programming oracle.
    cfg = fg.FrozenShoreConfig()
    actions = (fg.DOWN,) + (fg.RIGHT,) * 4 + (fg.UP,) + (fg.RIGHT,) * 2 + (fg.UP,) * 3 + (fg.RIGHT,) * 2
    env = fg.ShiftingShoreFrozenLakeEnv(cfg)
    env.reset(seed=17, options={"episode_index": 0})
    value, invalid = 0.0, 0
    events = []
    for action in actions:
        _, reward, term, trunc, info = env.step(action)
        invalid += not bool(info["action_mask_before"][action])
        events.append(info["diagnostic_event"])
        value += reward
        if info["diagnostic_event"] == "inner":
            value += cfg.inner_reward * (cfg.inner_success(0) - info["diagnostic_outcome"])
    assert term and not trunc and not invalid and info["position"] == fg.INNER_DOCK
    gap = fg.optimal_expected_return(cfg, 0) - value
    kl = fg.policy_information(cfg, "deep-survey", 0, 1)
    result["shore_missing_column"] = {"actions": list(actions), "steps": len(actions), "expected_return": value,
                                      "information": kl, "gap": gap, "coefficient": gap / kl, "events": events}

    # The claimed lower bound exceeds the true row on a likely first observation.
    cfg = dg.DeepSeaConfig()
    ceac = da.TabularCEACAgent(dg.ShiftingDeepSeaEnv(cfg), 0)
    context = ceac.contexts[0]
    context.buffers["probe"].append((dg.event_log_probability(cfg, 0, "probe", 0),
                                    dg.event_log_probability(cfg, 1, "probe", 0)))
    result["deepsea_information_lcb"] = {"reported_lcb": ceac._estimated_information(context, "probe", 0),
                                        "true_kl": dg.policy_information(cfg, "probe", 0),
                                        "probability_of_this_violation": 1 - cfg.probe_probability(0)}

    # Score-function derivative: the implementation omits the inverse temperature.
    logits = np.array([-.8, .2, 1.1]); temperature = ceac.actor_temperature
    probabilities = da._softmax(logits, temperature=temperature)
    implemented = -probabilities.copy(); implemented[1] += 1
    numerical = []
    for j in range(len(logits)):
        dx = np.eye(len(logits))[j] * 1e-6
        numerical.append((np.log(da._softmax(logits + dx, temperature=temperature)[1]) -
                          np.log(da._softmax(logits - dx, temperature=temperature)[1])) / 2e-6)
    assert np.allclose(numerical, implemented / temperature, atol=1e-8)
    result["deepsea_actor_score"] = {"temperature": temperature, "implemented": implemented.tolist(),
                                     "finite_difference": numerical, "missing_factor": 1 / temperature}

    # A zero evidence price should eliminate the evidence advantage. A stale
    # scalar evidence table currently keeps changing the actor at beta=0.
    env = dg.ShiftingDeepSeaEnv(cfg)
    agent = da.TabularCEACAgent(env, 0)
    observation, info = env.reset(seed=0)
    for _ in range(cfg.size - 1):
        observation, _, _, _, info = env.step(dg.RIGHT)
    mask = info["action_mask"].copy()
    nxt, reward, term, trunc, info = env.step(dg.RIGHT)
    info["action_mask_before"] = mask
    agent.actor_logits[observation] = np.zeros(env.action_space.n)
    agent.reward_value[observation] = reward  # zero reward TD error at terminal transition
    agent.evidence_value[observation] = 1.0   # residual from a previous nonzero price
    agent.episode_beta = 0.0
    trace = da.EpisodeTrace(context=0, mode=0, behavior_label="actor", transitions=[
        da.Transition(observation, dg.RIGHT, reward, nxt, term, trunc, info)])
    agent._actor_and_critic_updates(trace)
    result["deepsea_zero_price_critic"] = {
        "beta": 0.0, "reward_td_error": 0.0,
        "max_actor_logit_change": float(np.max(np.abs(agent.actor_logits[observation]))),
        "expected_change_for_beta_conditioned_zero_critic": 0.0}

    # Grid approximation compared with the analytic two-policy mixture search.
    result["deepsea_ids_mixture"] = []
    for prior in (0.5, .001, .00001):
        regrets = np.array([(1-prior) * dg.policy_gap(cfg, p, 0) + prior * dg.policy_gap(cfg, p, 1)
                            for p in da.DESIGN_POLICIES])
        information = np.array([dg.mutual_information_binary(prior, dg.policy_signature_distribution(cfg, p, 0),
                                                            dg.policy_signature_distribution(cfg, p, 1))
                                for p in da.DESIGN_POLICIES])
        q_grid, q_closed = da._best_pair_distribution(regrets, information), fa.best_pair_distribution(regrets, information)
        result["deepsea_ids_mixture"].append({"prior": prior, "grid": q_grid.tolist(), "analytic": q_closed.tolist(),
                                             "grid_ratio": float((q_grid @ regrets)**2 / (q_grid @ information)),
                                             "analytic_ratio": float((q_closed @ regrets)**2 / (q_closed @ information))})

    # Hard-zero prior: existing likelihoods cannot restore omitted model mass.
    cfg = fg.FrozenShoreConfig(context_schedule=(0,), block_length=100, true_modes=(1, 0, 0))
    env = fg.ShiftingShoreFrozenLakeEnv(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # log(0) implements exact zero support.
        ids = fa.BayesianIDSAgent(env, seed=0, prior_shifted=0.0, name="zero-support diagnostic")
    policies = []; info_count = 0
    for episode in range(100):
        observation, info = env.reset(seed=episode, options={"episode_index": episode})
        label = ids.start_episode(observation, info); policies.append(label)
        trace = fa.EpisodeTrace(context=0, mode=1, behavior_label=label)
        while True:
            action = ids.act(observation, info)
            nxt, reward, term, trunc, info = env.step(action)
            trace.transitions.append(fa.Transition(observation, action, reward, nxt, term, trunc, info))
            info_count += info["diagnostic_event"] in ("deep", "outer")
            observation = nxt
            if term or trunc:
                break
        ids.end_episode(trace)
    assert set(policies) == {"inner-bridge"} and info_count == 0
    result["zero_support_diagnostic"] = {"episodes": 100, "selected_policies": sorted(set(policies)),
        "informative_visits": info_count, "final_shifted_posterior": ids.diagnostics(0)["posterior_shifted"],
        "pseudo_regret": sum(fg.policy_gap(cfg, p, 1) for p in policies),
        "scope": "explicitly misspecified zero-prior control, not a new nonstationary CEAC experiment"}

    # Information available to ordinary agents, regardless of whether it is used.
    result["public_info_privilege"] = {}
    for kind, factory in (("deepsea", dg.ShiftingDeepSeaEnv), ("frozenlake", fg.ShiftingShoreFrozenLakeEnv)):
        _, info = factory().reset(seed=0)
        result["public_info_privilege"][kind] = sorted(set(info) & {"mode", "optimal_policy", "shore_label"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.ids_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Audit diagnostics:", args.output)


if __name__ == "__main__":
    main()
