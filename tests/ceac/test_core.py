"""Scientific regression tests: numerical oracles, update invariants and evidence."""
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ceac.agents import CEAC, CEACConfig
from ceac.contracts import Policy, Episode, Transition
from ceac.design import PolicyBank, bounded_design, target_calibration
from ceac.evidence import manifest, verify_run
from ceac.ids import BayesianIDS, best_pair
from ceac.learning import ActorCritic, DebtLedger, LikelihoodEvidence, evidence_scale
from ceac.model import FiniteModel, environment
from ceac.run import collect_episode, execute, stable_seed


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {kind: FiniteModel(kind) for kind in ("deepsea", "frozenlake")}

    def test_all_released_fixed_policy_returns_and_information(self):
        checks = 0
        for kind, model in self.models.items():
            for policy in model.seed_policies():
                returns, information = model.evaluate(policy)
                for m in (0, 1):
                    expected = model.module.expected_policy_return(model.cfg, policy.label, m)
                    kl = (model.module.policy_information(model.cfg, policy.label, m) if kind == "deepsea" else
                          model.module.policy_information(model.cfg, policy.label, m, 1-m))
                    self.assertAlmostEqual(returns[m], expected, places=11)
                    self.assertAlmostEqual(information[m, 1-m], kl, places=11)
                    checks += 1
        self.assertEqual(checks, 22)

    def test_full_oracle_coefficients_and_missing_frozenlake_column(self):
        expected = {"deepsea": .012353602208420736, "frozenlake": .03459008618357806}
        for kind, model in self.models.items():
            bank = PolicyBank(model, model.seed_policies())
            design = bounded_design(model, bank, (0,))
            self.assertAlmostEqual(design.cost, expected[kind], places=11)
            self.assertLess(design.primal_dual_gap, 1e-10)
            self.assertLess(design.column_residual, 1e-10)
            self.assertGreaterEqual(design.row_minimum, 1-1e-10)
            if kind == "frozenlake":
                self.assertGreater(design.inserted, 0)
                selected = bank.policies[int(np.argmax(design.eta))]
                self.assertAlmostEqual(model.evaluate(selected)[0][0], .877, places=11)
            shifted = bounded_design(model, bank, (1,))
            self.assertAlmostEqual(shifted.cost, 0., places=11)

    def test_global_separator_bounds_randomized_policies(self):
        rng = np.random.default_rng(4)
        for model in self.models.values():
            for _ in range(3):
                rw = rng.normal(size=2); iw = rng.normal(size=(2, 2))
                policy, optimum = model.response(rw, iw)
                j, k = model.evaluate(policy)
                self.assertAlmostEqual(optimum, float(rw @ j + np.sum(iw*k)), places=9)
                p = rng.random(model.mask.shape)*model.mask
                p /= p.sum(axis=1, keepdims=True)
                jr, kr = model.evaluate(Policy(p))
                self.assertLessEqual(float(rw @ jr + np.sum(iw*kr)), optimum+1e-9)

    def test_robust_costs_cover_each_plausible_model(self):
        for model in self.models.values():
            bank = PolicyBank(model, model.seed_policies())
            design = bounded_design(model, bank, (0, 1))
            j, _ = bank.arrays()
            upper = design.constant-design.reward_weights@j
            self.assertTrue(np.all(upper >= model.optimal_returns[:, None]-j-1e-10))
            self.assertEqual(design.references, (0, 1))

    def test_target_calibration_constraints_and_global_gaps(self):
        for model in self.models.values():
            bank = PolicyBank(model, model.seed_policies())
            c = [bounded_design(model, bank, (m,)).cost for m in (0, 1)]
            a = target_calibration(model, bank, np.array([.4, .6]), c)
            self.assertLess(a["target_gap"], 1e-9)
            self.assertLess(a["calibration_gap"], 1e-9)
            self.assertAlmostEqual(sum(a["target"]), 1.)
            self.assertAlmostEqual(sum(a["calibration"]), 1.)
            np.testing.assert_allclose(a["mixture"], a["q"]*a["target"]+(1-a["q"])*a["calibration"])
            returns, info = bank.arrays()
            for m in a["good"]:
                gap = float((model.optimal_returns[m]-returns[m]) @ a["target"])
                evidence = float(info[m, 1-m] @ a["target"])
                lower_t = max(1/20, gap/((1+.25/6)*c[m])) if c[m] else 1/20
                self.assertLessEqual(lower_t, evidence/(1-.25/6)+1e-9)
                if c[m] == 0:
                    self.assertLess(abs(gap), 1e-9)
            rejected = [r for r in a["candidates"] if not r["feasible"]]
            self.assertTrue(rejected)
            for row in rejected:
                self.assertGreater(row["phase_one_slack"], 0)
                self.assertLess(max(row["target_gap"], row["calibration_gap"]), 1e-9)

    def test_invalid_design_bound_fails_closed(self):
        model = self.models["deepsea"]
        with self.assertRaises(RuntimeError):
            bounded_design(model, PolicyBank(model, model.seed_policies()), (0,), .001)

    def test_policy_is_an_immutable_snapshot(self):
        p = self.models["deepsea"].seed_policies()[0]
        with self.assertRaises(ValueError):
            p.probabilities[0, 0] = 1
        actor = ActorCritic(self.models["deepsea"])
        snapshot = actor.snapshot(); identity = snapshot.identity
        actor.logits[0, 0] += 1
        self.assertEqual(snapshot.identity, identity)
        self.assertNotEqual(actor.snapshot().identity, identity)

    def test_softmax_score_includes_temperature(self):
        actor = ActorCritic(self.models["deepsea"], temperature=.3)
        actor.logits[0] = np.array([-.2, .1, -.5, .7, .3])
        policy = actor.snapshot(); action = int(np.flatnonzero(actor.model.mask[0])[0])
        analytic = -policy.probabilities[0].copy(); analytic[action] += 1; analytic /= .3
        numeric = []
        for a in range(actor.model.actions):
            actor.logits[0, a] += 1e-6; high = math.log(actor.snapshot().probabilities[0, action])
            actor.logits[0, a] -= 2e-6; low = math.log(actor.snapshot().probabilities[0, action])
            actor.logits[0, a] += 1e-6; numeric.append((high-low)/2e-6)
        np.testing.assert_allclose(numeric, analytic, atol=1e-8)

    def test_zero_price_removes_stale_evidence(self):
        model = self.models["deepsea"]; actor = ActorCritic(model)
        policy = actor.snapshot(); s = model.initial; a = int(np.flatnonzero(model.mask[s])[0])
        actor.row_values[0, 1, s] = 10.
        episode = Episode(0, 0, policy.identity, "actor", (Transition(s, a, 0., -1, True, (-2., -.1)),))
        before = actor.logits.copy()
        actor.update(episode, policy, np.zeros((2, 2)))
        np.testing.assert_array_equal(before, actor.logits)

    def test_frozen_diagnostic_does_not_train_actor_or_critics(self):
        model = self.models["deepsea"]; actor = ActorCritic(model); policy = actor.snapshot()
        s=model.initial; a=int(np.flatnonzero(model.mask[s])[0])
        episode=Episode(0, 0, policy.identity, "design", (Transition(s,a,1.,-1,True,(-1.,-2.)),))
        before=(actor.logits.copy(),actor.reward_value.copy(),actor.row_values.copy())
        self.assertFalse(actor.update(episode,policy,np.ones((2,2))))
        for old,new in zip(before,(actor.logits,actor.reward_value,actor.row_values)):
            np.testing.assert_array_equal(old,new)

    def test_off_policy_actor_update_rejected(self):
        model=self.models["deepsea"]; actor=ActorCritic(model); policy=actor.snapshot()
        actor.logits[0,0] += 1.
        episode=Episode(0,0,policy.identity,"actor",())
        with self.assertRaises(ValueError):actor.update(episode,policy,np.zeros((2,2)))

    def test_likelihood_confidence_anytime_finite_horizon_coverage(self):
        # Enumerate first-crossing probabilities, not a Monte Carlo coverage test.
        for true in (0,1):
            delta=.05; crossing=0.; frontier=[(0.,1.)]
            p=(.15,.85)[true]
            for _ in range(12):
                next_frontier=[]
                for logratio,weight in frontier:
                    for outcome,prob in ((0,1-p),(1,p)):
                        pt=p if outcome else 1-p
                        other=(.15,.85)[1-true]; po=other if outcome else 1-other
                        lr=logratio+math.log(po/pt); w=weight*prob
                        if lr>math.log(1/delta):crossing+=w
                        else:next_frontier.append((lr,w))
                frontier=next_frontier
            self.assertLessEqual(crossing,delta+1e-12)

    def test_replay_labels_returns_and_no_double_credit(self):
        evidence=LikelihoodEvidence(); ledger=DebtLedger()
        e=Episode(0,0,"a","actor",(Transition(0,0,.5,-1,True,(-2.,-.2)),))
        evidence.append(e);ledger.credit(e)
        self.assertEqual(evidence.stratum_statistics("a",0,1),{"count":1,"return_mean":.5,"llr_mean":-1.8})
        with self.assertRaises(ValueError):evidence.append(e)
        with self.assertRaises(ValueError):ledger.credit(e)

    def test_cumulative_quota_changes_and_single_credit(self):
        model=self.models["deepsea"]; bank=PolicyBank(model,model.seed_policies())
        old=bounded_design(model,bank,(0,));new=bounded_design(model,bank,(1,));ledger=DebtLedger()
        ledger.add_design(old,2.); before=dict(ledger.quotas)
        ledger.add_design(new,3.)
        for p,q in before.items():self.assertGreaterEqual(ledger.quotas[p],q)
        owed=next(p for p,q in ledger.quotas.items() if q>0)
        ledger.require_seed(owed);ledger.request_fresh(owed)
        for i in range(100):
            identity,behavior=ledger.select()
            if identity is None:break
            ledger.credit(Episode(i,0,identity,behavior,()))
        self.assertIsNone(ledger.select()[0])
        for p in set(ledger.quotas)|set(ledger.statistical_required):
            target=max(ledger.quotas[p],ledger.statistical_required[p])
            self.assertGreaterEqual(ledger.counts[p],target)
            self.assertLess(ledger.counts[p],target+1)
        with self.assertRaises(ValueError):ledger.add_design(new,1.)

    def test_evidence_scale_is_monotone(self):
        scales=[evidence_scale(t) for t in range(1,10000)]
        self.assertTrue(np.all(np.diff(scales)>=0))

    def test_ids_tiny_prior_analytic_mixture(self):
        model=self.models["deepsea"]
        for prior in (1e-8,1e-5,.001):
            agent=BayesianIDS(model,np.random.default_rng(0),prior)
            p=agent.distribution()
            probe=next(i for i,x in enumerate(agent.bank.policies) if x.label=="probe")
            self.assertAlmostEqual(p[probe]/prior,8/3,places=6)
            self.assertLessEqual(agent.record["column_residual"],1e-7*agent.record["regret"])

    def test_ids_mutual_information_matches_trajectory_enumeration(self):
        # Independent explicit tree expansion for complete policy signatures.
        for model in self.models.values():
            prior=np.array([.37,.63]);stage=model.mutual_information_stage(prior)
            for policy in model.seed_policies():
                def visit(s, weights):
                    a=int(np.argmax(policy.probabilities[s]));pair=model.laws[s,a]
                    total=0.
                    for key in pair[0]:
                        new=weights*np.array([pair[m][key] for m in (0,1)])
                        if key[2]:
                            marginal=float(prior@new)
                            total+=sum(prior[m]*new[m]*math.log(new[m]/marginal) for m in (0,1))
                        else:total+=visit(model.index[key[0]],new)
                    return total
                exact=visit(model.initial,np.ones(2))
                self.assertAlmostEqual(model.evaluate_stage(policy,stage),exact,places=10)

    def test_zero_support_is_absorbing_and_confounded(self):
        for kind,model in self.models.items():
            agent=BayesianIDS(model,np.random.default_rng(0),0.)
            env=environment(kind,(1,1,1));rng=np.random.default_rng(1)
            regret=0.
            for i in range(20):
                p,b,prices=agent.select(i);e,_=collect_episode(env,model,p,i,b,rng,i)
                agent.observe(e,p,prices)
                regret+=model.optimal_returns[1]-model.evaluate(p)[0][1]
                self.assertEqual(agent.posterior[1],0.)
                self.assertAlmostEqual(model.evaluate(p)[1][0,1],0.)
            self.assertAlmostEqual(regret,20*({"deepsea":.04,"frozenlake":.208}[kind]),places=10)

    def test_shared_ceac_executes_actor_and_keeps_public_replay(self):
        for kind,model in self.models.items():
            agent=CEAC(model,np.random.default_rng(0),CEACConfig(deployment_rule="actor_only"));env=environment(kind,(1,1,1));rng=np.random.default_rng(4)
            for i in range(24):
                p,b,prices=agent.select(i);e,_=collect_episode(env,model,p,i,b,rng,i)
                agent.observe(e,p,prices)
                self.assertNotIn("mode",e.__dict__)
                for t in e.transitions:self.assertNotIn("info",t.__dict__)
            self.assertGreater(agent.actor.updates,0)
            self.assertEqual(sum(len(v) for v in agent.replay.strata.values()),24)
            self.assertEqual(sum(agent.ledger.counts.values()),24)
            self.assertTrue(any(e["type"]=="actor_snapshot" for e in agent.events))

    def test_visible_context_changes_kernel_without_revealing_mode(self):
        model=FiniteModel("deepsea",2)
        self.assertEqual(model._models[0].decode(model.initial_observation).context,2)
        self.assertAlmostEqual(model.optimal_returns[0],self.models["deepsea"].optimal_returns[0])
        self.assertEqual(model.cfg.true_modes,(0,0,0)) # candidate-zero specification only

    def test_method_seed_is_independent_of_method_order(self):
        methods=["ceac","ids","oracle"]
        first={m:stable_seed(3,"deepsea",m,0,"learner") for m in methods}
        second={m:stable_seed(3,"deepsea",m,0,"learner") for m in reversed(methods)}
        self.assertEqual(first,second);self.assertEqual(len(set(first.values())),3)

    def test_evidence_roundtrip_and_incompatible_resume(self):
        model=self.models["deepsea"]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"run"
            result=execute(model,root,"ceac",0,1,8,CEACConfig())
            verified=verify_run(root)
            self.assertEqual(verified["status"]["episodes"],8)
            self.assertEqual(result,execute(model,root,"ceac",0,1,8,CEACConfig(),resume=True))
            with self.assertRaises(ValueError):execute(model,root,"ceac",0,0,8,CEACConfig(),resume=True)
            with self.assertRaises(FileExistsError):execute(model,root,"ceac",0,1,8,CEACConfig())
            with (root/"metrics.jsonl").open("a") as out:out.write("{}\n")
            with self.assertRaises(ValueError):verify_run(root)


if __name__ == "__main__":
    unittest.main()
