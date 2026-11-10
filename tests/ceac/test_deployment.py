"""Deployment and comparator regression tests, independent of learning curves."""
import inspect
import unittest
import numpy as np
from ceac.agents import CEAC, CEACConfig, LaiGravesOracle
from ceac.comparators import public_comparison_bank, trajectory_return_range
from ceac.deployment import certify_deployment
from ceac.learning import LikelihoodEvidence
from ceac.model import FiniteModel, environment
from ceac.run import collect_episode


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={k:FiniteModel(k) for k in ('deepsea','frozenlake')}

    def test_reward_certificate_requires_all_plausible_models(self):
        self.assertNotIn('mode',inspect.signature(certify_deployment).parameters)
        for model in self.models.values():
            evidence=LikelihoodEvidence()
            c=certify_deployment(model,evidence)
            self.assertFalse(c.certified)
            for m in (0,1):
                evidence.scores[:]=0.;evidence.scores[m]=10.
                c=certify_deployment(model,evidence)
                self.assertTrue(c.valid_for(evidence.plausible))
                self.assertAlmostEqual(model.evaluate(c.policy)[0][m],model.optimal_returns[m],places=10)
                self.assertLess(c.residual,1e-10)
                self.assertFalse(c.valid_for((0,1)))
                self.assertFalse(c.valid_for((1-m,)))

    def test_certified_deployment_closes_live_gap_without_actor_replay(self):
        for kind,model in self.models.items():
            agent=CEAC(model,np.random.default_rng(0));env=environment(kind,(1,1,1));rng=np.random.default_rng(4)
            deployed=0
            for i in range(64):
                p,b,prices=agent.select(i)
                before=agent.actor.snapshot().identity;updates=agent.actor.updates
                episode,_=collect_episode(env,model,p,i,b,rng,i)
                if b=='certified_deploy':
                    self.assertTrue(agent.deployment.valid_for(agent.replay.plausible))
                    self.assertAlmostEqual(model.evaluate(p)[0][1],model.optimal_returns[1],places=10)
                    deployed+=1
                agent.observe(episode,p,prices)
                if b=='certified_deploy':
                    self.assertEqual(agent.actor.snapshot().identity,before)
                    self.assertEqual(agent.actor.updates,updates)
            self.assertGreater(deployed,32)
            self.assertEqual(sum(agent.ledger.counts.values()),64)
            self.assertEqual(sum(map(len,agent.replay.strata.values())),64)

    def test_actor_only_ablation_still_uses_actor(self):
        model=self.models['deepsea']
        agent=CEAC(model,np.random.default_rng(2),CEACConfig(deployment_rule='actor_only'))
        agent.replay.scores[:]=[0.,10.];agent.refresh()
        for identity in set(agent.ledger.quotas)|set(agent.ledger.statistical_required):
            agent.ledger.counts[identity]=100;agent.ledger.statistical_assigned[identity]=100
        p,b,_=agent.select(0)
        self.assertEqual(b,'actor');self.assertEqual(p.identity,agent.actor.snapshot().identity)
        self.assertTrue(agent.deployment.certified)

    def test_shared_public_bank_includes_optima_and_cheaper_shore_survey(self):
        for kind,model in self.models.items():
            banks=[public_comparison_bank(model) for _ in range(2)]
            self.assertEqual([p.identity for p in banks[0].policies],[p.identity for p in banks[1].policies])
            values=[model.evaluate(p) for p in banks[0].policies]
            for mode in (0,1):
                self.assertAlmostEqual(max(v[0][mode] for v in values),model.optimal_returns[mode],places=10)
            if kind=='frozenlake':
                self.assertTrue(any(abs(v[0][0]-.877)<1e-10 and abs(v[1][0,1]-1.2142207387716744)<1e-10 for v in values))

    def test_return_confidence_bounds_cover_all_reachable_outcomes(self):
        for model in self.models.values():
            low,high=trajectory_return_range(model)
            self.assertLess(low,high)
            for p in public_comparison_bank(model).policies:
                values,_=model.evaluate(p)
                self.assertTrue(np.all(values>=low));self.assertTrue(np.all(values<=high))

    def test_offline_oracle_allocation_is_fixed(self):
        model=self.models['deepsea'];oracle=LaiGravesOracle(model,0)
        eta,beta=oracle.design.eta.copy(),oracle.design.beta.copy()
        env=environment('deepsea',(0,0,0));rng=np.random.default_rng(3)
        for i in range(32):
            p,b,prices=oracle.select(i);e,_=collect_episode(env,model,p,i,b,rng,i)
            oracle.observe(e,p,prices)
            np.testing.assert_array_equal(oracle.design.eta,eta)
            np.testing.assert_array_equal(oracle.design.beta,beta)


if __name__=='__main__':unittest.main()
