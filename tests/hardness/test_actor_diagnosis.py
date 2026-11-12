"""Exact conditional moments used to distinguish noise from gradient bias."""
import unittest
import numpy as np
from scipy.special import softmax
from ceac_walkthrough.actor_diagnosis import action_moments,conditional_gradient
from ceac_hardness.environment import JunctionConfig
from ceac_hardness.model import JunctionModel


class ActorDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.model=JunctionModel(JunctionConfig(protocol='prior_shift'))
        self.beta=np.zeros((10,10));self.beta[0,6]=.5
        self.moments=action_moments(self.model,0,self.beta,1)
        self.p=np.array([.1,.8,.1])

    def test_gradient_mean_is_baseline_invariant_and_matches_finite_difference(self):
        q=np.array([m['augmented'] for m in self.moments]);z=np.log(self.p)
        for baseline in (-2.,0.,.9,5.):
            stats=conditional_gradient(self.p,self.moments,baseline)
            epsilon=1e-6
            finite=np.array([(.4*(softmax(z+epsilon*np.eye(3)[j])@q-softmax(z-epsilon*np.eye(3)[j])@q)/(2*epsilon)) for j in range(3)])
            np.testing.assert_allclose(stats['vector'],finite,atol=1e-10)

    def test_projected_mean_and_variance_equal_full_outcome_enumeration(self):
        baseline=.8;values=[];probabilities=[]
        for action in range(3):
            laws=self.model.laws[1,action]
            for outcome,probability in laws[0].items():
                ll=np.log([law[outcome] for law in laws])
                e=float(np.sum(self.beta*(ll[:,None]-ll[None,:])))
                score=(int(action==2)-int(action==1))-(self.p[2]-self.p[1])
                values.append(.4*(outcome[1]+e-baseline)*score)
                probabilities.append(self.p[action]*probability)
        probabilities=np.array(probabilities);values=np.array(values)
        mean=probabilities@values;var=probabilities@((values-mean)**2)
        result=conditional_gradient(self.p,self.moments,baseline)
        self.assertAlmostEqual(result['contrast_mean'],mean)
        self.assertAlmostEqual(result['contrast_variance'],var)
        best=result['variance_minimizing_baseline']
        reduced=conditional_gradient(self.p,self.moments,best)
        self.assertAlmostEqual(result['contrast_variance']-reduced['contrast_variance'],result['baseline_excess_variance'])

    def test_terminal_evidence_is_dense_and_probe_beats_risky_at_recorded_prices(self):
        self.assertEqual(self.moments[0]['nonzero_evidence'],0.)
        self.assertEqual(self.moments[1]['nonzero_evidence'],1.)
        self.assertEqual(self.moments[2]['nonzero_evidence'],1.)
        for price in (.4,.58):
            beta=self.beta*(price/.5)
            m=action_moments(self.model,0,beta,1)
            self.assertGreater(m[2]['augmented'],m[1]['augmented'])
            self.assertGreater(m[2]['augmented'],m[0]['augmented'])
        concentrated=np.array([.00018,.99948,.00034])
        g1=conditional_gradient(concentrated,self.moments,.8)['vector']
        g2=conditional_gradient(np.ones(3)/3,self.moments,.8)['vector']
        self.assertLess(np.linalg.norm(g1),.01*np.linalg.norm(g2))

if __name__=='__main__':unittest.main()
