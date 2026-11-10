"""Prior support is a declared assumption, not an IDS numerical failure."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ceac.agents import CEACConfig
from ceac.benchmarks import specification, cell_folder
from ceac.benchmark_audit import audit_model
from ceac.evidence import atomic_json, manifest, verify_run
from ceac.ids import BayesianIDS
from ceac.model import FiniteModel, environment
from ceac.report import validate_reference, load_study, audit_prior_support
from ceac.run import collect_episode, execute


class PriorSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={(k,b):FiniteModel(k,benchmark=b) for k in ('deepsea','frozenlake')
                    for b in ('strong_signal_m4','weak_signal_m4')}

    def test_zero_support_has_exact_linear_regret_in_both_signal_regimes(self):
        for (kind,b),model in self.models.items():
            agent=BayesianIDS(model,np.random.default_rng(0),0.)
            env=environment(kind,(1,1,1),benchmark=b);rng=np.random.default_rng(8)
            gap=.25 if b=='strong_signal_m4' else .05
            regret=0.
            for i in range(16):
                policy,behavior,prices=agent.select(i)
                episode,events=collect_episode(env,model,policy,i,behavior,rng,i)
                agent.observe(episode,policy,prices)
                values,information=model.evaluate(policy)
                regret+=model.optimal_returns[1]-values[1]
                self.assertEqual(agent.posterior[1],0.)
                self.assertEqual(agent.diagnostics()['regret'],0.)
                self.assertEqual(agent.diagnostics()['information'],0.)
                self.assertEqual(events['events']['diagnostic'],0)
                np.testing.assert_allclose(information,0.,atol=1e-12)
            self.assertAlmostEqual(regret,16*gap,places=10)

    def test_even_informative_observations_do_not_resurrect_zero_support(self):
        for (kind,b),model in self.models.items():
            label='probe' if kind=='deepsea' else 'deep-survey'
            policy=next(p for p in model.seed_policies() if p.label==label)
            env=environment(kind,(1,1,1),benchmark=b)
            episode,_=collect_episode(env,model,policy,0,'forced_diagnostic_test',np.random.default_rng(1),2)
            llr=episode.log_likelihood[1]-episode.log_likelihood[0]
            self.assertGreater(abs(llr),1.)
            zero=BayesianIDS(model,np.random.default_rng(1),0.)
            positive=BayesianIDS(model,np.random.default_rng(1),.001)
            for agent in (zero,positive):agent.observe(episode,policy,np.zeros((2,2)))
            self.assertEqual(zero.posterior[1],0.)
            posterior=positive.posterior[1]
            self.assertGreater(posterior,0.)
            self.assertAlmostEqual(np.log(posterior/(1-posterior))-np.log(.001/.999),llr,places=10)

    def test_rare_positive_prior_keeps_a_positive_information_opportunity(self):
        for model in self.models.values():
            agent=BayesianIDS(model,np.random.default_rng(0),.001)
            agent.distribution()
            self.assertTrue(agent.initial_support.all())
            self.assertGreater(agent.record['information'],0.)
            self.assertGreater(agent.record['regret'],0.)
            self.assertLessEqual(agent.record['column_residual'],1e-7*agent.record['regret'])

    def studies(self):
        config=dict(benchmarks=['weak_signal_m4'],environments=['deepsea'],contexts=[0],
                    modes=[0,1],seeds=[0],episodes=16,methods=['ids_rare','ids_zero'])
        body={'sources':{'runtime.py':'fixed'},'runtime':{'python':'test'},
              'configuration':{'study':config,'ceac':asdict(CEACConfig(mass_bound=128,normalization=128)),
                               'benchmark_specifications':[specification('weak_signal_m4','deepsea')]}}
        reference=deepcopy(body);reference['configuration']['study']['methods']=['ceac','oracle','ids']
        return body,reference

    def test_matching_reference_is_accepted_without_training(self):
        primary,reference=self.studies()
        validate_reference(primary,reference,('ceac','oracle','ids'))

    def test_incompatible_reference_is_rejected(self):
        primary,reference=self.studies()
        mutations=[lambda r:r['sources'].update({'runtime.py':'changed'}),
                   lambda r:r['runtime'].update({'python':'different'}),
                   lambda r:r['configuration']['ceac'].update({'mass_bound':20}),
                   lambda r:r['configuration']['study'].update({'episodes':32}),
                   lambda r:r['configuration']['study'].update({'modes':[0]}),
                   lambda r:r['configuration']['study'].update({'methods':['ids']}),
                   lambda r:r['configuration'].update({'benchmark_specifications':[]})]
        for mutation in mutations:
            candidate=deepcopy(reference);mutation(candidate)
            with self.assertRaises(ValueError):validate_reference(primary,candidate,('ceac','oracle','ids'))

    def make_saved_study(self,root):
        model=self.models['deepsea','weak_signal_m4'];config=CEACConfig(mass_bound=128,normalization=128)
        settings=dict(benchmarks=['weak_signal_m4'],environments=['deepsea'],contexts=[0],modes=[1],
                      seeds=[0],episodes=8,methods=['ids_zero'])
        identity=manifest({'study':settings,'ceac':asdict(config),
                           'benchmark_specifications':[model.specification]})
        root.mkdir();atomic_json(root/'study.json',identity)
        folder=cell_folder(root,'weak_signal_m4','deepsea',0,1,'ids_zero',0)
        result=execute(model,folder,'ids_zero',0,1,8,config)
        atomic_json(root/'results.json',[result]);atomic_json(root/'status.json',{'status':'complete','completed_cells':1})
        return model,folder

    def test_study_loader_rejects_corrupted_result_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'study';self.make_saved_study(root)
            _,runs,_,_=load_study(root);self.assertEqual(len(runs),1)
            atomic_json(root/'results.json',[])
            with self.assertRaises(ValueError):load_study(root)

    def test_support_audit_rejects_a_counterfeit_posterior(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'study';model,folder=self.make_saved_study(root)
            run=verify_run(folder);audits=[audit_model(model)]
            rows=audit_prior_support([run],audits)
            self.assertTrue(rows[0]['support_exclusion_verified'])
            self.assertAlmostEqual(rows[0]['zero_support_prediction'],.4)
            tampered=deepcopy(run);tampered['metrics'][0]['diagnostics']['posterior'][1]=.001
            with self.assertRaises(ValueError):audit_prior_support([tampered],audits)
            mislabeled=deepcopy(run);mislabeled["result"]["method"]="ids"
            with self.assertRaisesRegex(ValueError,"method label disagrees"):
                audit_prior_support([mislabeled],audits)


if __name__=='__main__':unittest.main()
