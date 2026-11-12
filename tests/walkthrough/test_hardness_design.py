"""Test the scientific distinction between intervention, hardness and support."""
import unittest
import math
import numpy as np
from ceac_walkthrough.hardness import support_shift_design, bernoulli_kl, summarize_current_sweep


class HardnessDesignTests(unittest.TestCase):
    def test_expected_bayesian_drift_matches_enumerated_observations(self):
        design=support_shift_design()
        for row in design['rows']:
            q=row['true_cue_probability']
            true=np.array([1-q,q])
            likelihood0=np.array([.9,.1]);likelihood1=np.array([.1,.9])
            drift=float(true@np.log(likelihood1/likelihood0))
            self.assertAlmostEqual(drift,row['fixed_prior_expected_log_odds_per_cue'])
        drift=[r['fixed_prior_expected_log_odds_per_cue'] for r in design['rows']]
        self.assertGreater(drift[0],0);self.assertEqual(drift[2],0);self.assertLess(drift[-1],0)

    def test_joint_support_differs_from_task_and_observation_support(self):
        design=support_shift_design()
        self.assertEqual({a['task'] for a in design['frozen_joint_prior']},{0,1})
        self.assertTrue(all(a['mass']>0 for a in design['frozen_joint_prior']))
        for row in design['rows']:
            q=row['true_cue_probability'];truth=np.outer([1-q,q],[.15,.85]).ravel()
            candidates=[]
            for atom in design['frozen_joint_prior']:
                p=atom['q'];v=.85 if atom['task'] else .15
                law=np.outer([1-p,p],[1-v,v]).ravel()
                self.assertTrue(np.all(law>0))
                candidates.append(float(truth@np.log(truth/law)))
            self.assertAlmostEqual(min(candidates),row['closest_fixed_prior_joint_witness_kl'])
            self.assertEqual(row['fixed_prior_true_joint_supported'],row['shift']==0)
            if row['shift']>0:self.assertGreater(min(candidates),0)

    def test_nuisance_can_match_an_opposite_task_while_probe_distinguishes(self):
        for row in support_shift_design()['rows']:
            q=row['true_cue_probability']
            self.assertAlmostEqual(bernoulli_kl(q,q),0)
            self.assertAlmostEqual(row['reliable_diagnostic_kl'],.7*math.log(17/3))
        self.assertEqual(support_shift_design()['product_class_models'],10)

    def test_bad_mild_case_is_not_called_an_easy_to_hard_transition(self):
        rows=[]
        for method in ('idac','ids','ceac'):
            for rho in (.05,.75):
                for seed in (0,1):
                    gap=.5 if method=='idac' else 0.
                    rows.append(dict(environment='frozenlake',context=0,mode=1,method=method,
                        rho=rho,seed=seed,episodes=100,regret=gap*100,last_gap=gap,pomd_zero_fraction=0.))
        bundle={'pathology_statistics':rows}
        summaries={r['method']:r for r in summarize_current_sweep(bundle)}
        self.assertEqual(summaries['idac']['classification'],'fails_both_ends')
        self.assertEqual(summaries['ids']['classification'],'passes_both_ends')
        for r in rows:
            if r['method']=='ids' and r['rho']==.75:r['last_gap']=.2
        changed={r['method']:r for r in summarize_current_sweep(bundle)}
        self.assertEqual(changed['ids']['classification'],'observed_pass_to_fail')
        self.assertEqual(summarize_current_sweep(bundle),summarize_current_sweep({'pathology_statistics':list(reversed(rows))}))

    def test_invalid_channel_designs_are_rejected(self):
        for kwargs in ({'levels':[]},{'levels':[0,0]},{'levels':[-.1,.5]},
                       {'cue_accuracy':1},{'diagnostic_accuracy':.5}):
            with self.assertRaises(ValueError):support_shift_design(**kwargs)


if __name__=='__main__':unittest.main()
