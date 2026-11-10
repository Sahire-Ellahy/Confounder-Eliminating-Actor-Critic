"""Scientific checks for controlled nuisance and weak-route pathologies."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from ceac.benchmarks import specification, cell_folder
from ceac.benchmark_audit import normalized_occupancy, entropy, audit_model
from ceac.contracts import Policy
from ceac.comparators import public_comparison_bank
from ceac.model import FiniteModel, environment
from ceac.run import collect_episode


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={(kind,b):FiniteModel(kind,benchmark=b) for kind in ('deepsea','frozenlake')
                    for b in ('weak_signal_m4','weak_signal_m16')}

    def test_mode_independent_public_specification_and_simulator_agreement(self):
        for (kind,b),model in self.models.items():
            public=specification(b,kind)
            self.assertNotIn('true_modes',public['configuration_overrides'])
            for mode in (0,1):
                env=environment(kind,(mode,mode,mode),benchmark=b)
                for field,value in public['configuration_overrides'].items():
                    self.assertEqual(getattr(env.config,field),value)
                rng=np.random.default_rng(5)
                for i,policy in enumerate(model.seed_policies()):
                    episode,metrics=collect_episode(env,model,policy,i,'audit',rng,i)
                    self.assertTrue(np.isfinite(episode.log_likelihood).all())
                    if policy.label in ('sonar-sweep','reed-scan'):
                        self.assertEqual(metrics['events']['diagnostic'],0)
                        self.assertGreater(metrics['events']['nuisance'],0)
                        self.assertAlmostEqual(*episode.log_likelihood)

    def test_alphabet_scaling_keeps_task_values_information_and_seed_count_fixed(self):
        for kind in ('deepsea','frozenlake'):
            low=self.models[kind,'weak_signal_m4'];high=self.models[kind,'weak_signal_m16']
            np.testing.assert_allclose(low.optimal_returns,high.optimal_returns,atol=1e-12)
            left={p.label:p for p in low.seed_policies()};right={p.label:p for p in high.seed_policies()}
            self.assertEqual(set(left),set(right))
            for name in left:
                lv,li=low.evaluate(left[name]);rv,ri=high.evaluate(right[name])
                np.testing.assert_allclose(lv,rv,atol=1e-12)
                np.testing.assert_allclose(li,ri,atol=1e-12)
            self.assertGreater(high.n,low.n)

    def test_sonar_entropy_matches_closed_form_and_loses_task_reward(self):
        for b,M in (('weak_signal_m4',4),('weak_signal_m16',16)):
            model=self.models['deepsea',b]
            policy=next(p for p in model.seed_policies() if p.label=='sonar-sweep')
            occupancy=normalized_occupancy(model,policy,0)
            self.assertAlmostEqual(entropy(occupancy),math.log(8)+7/8*math.log(M),places=10)
            returns,information=model.evaluate(policy)
            np.testing.assert_allclose(information,0.)
            self.assertTrue(np.all(returns<model.optimal_returns))

    def test_weak_route_has_cheaper_diagnostic_and_no_artificial_oracle_cost(self):
        for kind in ('deepsea','frozenlake'):
            audit=audit_model(self.models[kind,'weak_signal_m16'])
            self.assertGreater(audit['risky_to_optimal_information_cost_ratio_mode0'],100.)
            self.assertAlmostEqual(audit['oracle_design_costs'][1],0.,places=10)
            self.assertTrue(audit['zero_information_policies_beating_reward_optimum_in_oir_mode0'])
            self.assertEqual(audit['design_mass_bound'],128.)

    def test_normalized_occupancy_handles_variable_episode_lengths(self):
        # Probability 1/2 of a length-one episode; otherwise length three.
        source=np.array([0,1,2,3,4,5]);dest=np.array([3,1,2,2,3,3])
        model=SimpleNamespace(n=3,actions=2,initial=0,edge=source,destination=dest,
                              prob=np.ones((2,6)),mask=np.ones((3,2),dtype=bool),
                              levels=[np.array([0]),np.array([1]),np.array([2])])
        model.validate_policy=lambda p:None
        model._continuation=lambda v,m:np.bincount(source,weights=v[dest],minlength=6).reshape(3,2)
        policy=Policy(np.full((3,2),.5))
        np.testing.assert_allclose(normalized_occupancy(model,policy,0),[2/3,1/6,1/6],atol=1e-12)

    def test_information_enrichment_ties_do_not_add_costly_nuisance_detours(self):
        model=FiniteModel('frozenlake',benchmark='weak_signal_m64')
        chosen=next(p for p in public_comparison_bank(model).policies if p.label=='maximum-model-information')
        base=self.models['frozenlake','weak_signal_m4']
        reference=next(p for p in public_comparison_bank(base).policies if p.label=='maximum-model-information')
        actual_j,actual_k=model.evaluate(chosen);reference_j,reference_k=base.evaluate(reference)
        np.testing.assert_allclose(actual_j,reference_j,atol=1e-11)
        np.testing.assert_allclose(actual_k,reference_k,atol=1e-11)
        primary,_=model.response(np.zeros(2),np.ones((2,2)))
        self.assertAlmostEqual(model.evaluate(primary)[1].sum(),actual_k.sum(),places=10)

    def test_cells_cannot_overwrite_other_benchmarks_or_legacy(self):
        root=Path('/tmp/example_ceac_study')
        paths=[cell_folder(root,b,'deepsea',0,0,'ceac',0) for b in ('legacy','weak_signal_m4','weak_signal_m16')]
        self.assertEqual(len(set(paths)),3)
        self.assertEqual(paths[0],root/'deepsea/context0_mode0/ceac_seed0')
        self.assertTrue(all(p.is_relative_to(root) for p in paths))


if __name__=='__main__':unittest.main()
