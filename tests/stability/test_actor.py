import unittest,json,tempfile,gzip,copy
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from ceac.contracts import Episode,Transition
from ceac_hardness.environment import JunctionConfig
from ceac_hardness.model import JunctionModel
from ceac_pseudocode.learning import TwoCriticActor
from ceac_stability.actor import StabilityActor,project_kl,guarded_proposal
from ceac_stability.run import run_cell,Evaluator,ROOT
from ceac_stability.experiment import selection

CFG=json.loads((ROOT/'experiments/ceac_actor_stability_20261112.json').read_text())

def candidate(name):return next(c for c in CFG['candidates'] if c['id']==name)

class StabilityTests(unittest.TestCase):
    def test_projection_satisfies_floor_and_kl_kkt(self):
        for logs in (np.log([.00001,.1,.89999]),np.array([-10000.,-300.,0.]),np.array([1.,2.,3.,4.])):
            q=project_kl(logs,.06);mu=.06/len(logs)
            self.assertAlmostEqual(q.sum(),1.)
            self.assertGreaterEqual(q.min(),mu-1e-14)
            free=q>mu+1e-12
            residual=np.log(q[free])-logs[free]
            np.testing.assert_allclose(residual,residual[0],atol=1e-12)
        logs=np.log([.0001,.03,.9699]);q=project_kl(logs,.15)
        opt=minimize(lambda p:float(np.sum(xlogy(p,p)-p*logs)),np.ones(3)/3,
            bounds=[(.05,1.)]*3,constraints={'type':'eq','fun':lambda p:p.sum()-1},method='SLSQP',options={'ftol':1e-13})
        self.assertTrue(opt.success);np.testing.assert_allclose(q,opt.x,atol=1e-6)

    def test_guard_checks_post_projection_kl_and_contracts(self):
        p=np.array([.003,.994,.003]);direction=np.array([0.,0.,300.])
        q,kl,backs,eta=guarded_proposal(p,direction,.1,.006,.001)
        self.assertGreater(backs,0);self.assertLessEqual(kl,.001+1e-14)
        self.assertGreaterEqual(q.min(),.002-1e-14);self.assertLess(eta,.1)

    def test_original_actor_is_bit_identical_over_updates_and_replay(self):
        m=JunctionModel(JunctionConfig(protocol='prior_shift'));base=TwoCriticActor(m)
        new=StabilityActor(m,candidate('original_reference'),CFG['schedule']);rng=np.random.default_rng(20)
        for t in range(100):
            p=base.snapshot();self.assertEqual(p.identity,new.snapshot().identity)
            s=1+t%2;a=int(rng.choice(3,p=p.probabilities[s]));beta=np.zeros((10,10));beta[0,6]=.4+.001*t
            e=Episode(t,0,p.identity,'actor' if t%5 else 'statistical',(
                Transition(0,0,0.,s,False,tuple(rng.normal(size=10))),
                Transition(s,a,float(rng.random()),-1,True,tuple(rng.normal(size=10)))))
            base.update(e,p,beta);new.update(e,p,beta)
            for x,y in ((base.logits,new.logits),(base.reward_value,new.reward_value),(base.evidence_parameters,new.evidence_parameters)):
                np.testing.assert_array_equal(x,y)
        self.assertEqual(base.updates,new.updates)

    def test_scheduled_errors_are_frozen_and_effective_evidence_step_is_scaled(self):
        m=JunctionModel();actor=StabilityActor(m,candidate('scheduled_score_004'),CFG['schedule'])
        actor.reward_value[:]=[.2,.1];actor.evidence_parameters[1,0]=[.3,.2]
        beta=np.array([[0.,0.],[.5,0.]]);p=actor.snapshot();vr=actor.reward_value.copy();z=actor.evidence_parameters.copy();vi=actor.evidence_value(beta)
        e=Episode(0,0,p.identity,'actor',(Transition(0,1,1.,1,False,(-2.,-.1)),Transition(1,0,0.,-1,True,(0.,0.))))
        dr=1-vr[0]+vr[1];di=.5*1.9-vi[0]+vi[1]
        score=-p.probabilities[0].copy();score[1]+=1
        actor.update(e,p,beta)
        np.testing.assert_allclose(actor.logits[0],.04*(dr+di)*score)
        self.assertAlmostEqual(actor.reward_value[0],vr[0]+.24*dr)
        np.testing.assert_allclose(actor.evidence_parameters[:,:,0],z[:,:,0]+.24/(.01+.25)*di*beta)
        self.assertEqual(actor.visits.tolist(),[1,1])

    def test_natural_small_step_has_inverse_propensity_direction(self):
        m=JunctionModel();c=copy.deepcopy(candidate('guarded_natural_001'));c['actor_initial']=1e-7;c['total_floor']=0.
        a=StabilityActor(m,c,CFG['schedule']);a.logits[0]=np.log([.2,.3,.5]);p=a.snapshot();old=a.logits.copy()
        e=Episode(0,0,p.identity,'actor',(Transition(0,1,1.,1,False,(0.,0.)),Transition(1,0,0.,-1,True,(0.,0.))))
        a.update(e,p,np.zeros((2,2)))
        self.assertAlmostEqual((a.logits[0,1]-a.logits[0,2])-(old[0,1]-old[0,2]),1e-7/.3,places=12)

    def test_masked_actions_and_diagnostic_replay_remain_unchanged(self):
        m=JunctionModel(JunctionConfig(burden=4));a=StabilityActor(m,candidate('guarded_natural_003'),CFG['schedule']);p=a.snapshot()
        e=Episode(0,0,p.identity,'statistical',(Transition(0,0,.5,1,False,(0.,0.)),Transition(1,0,0.,-1,True,(0.,0.))))
        before=(a.logits.copy(),a.reward_value.copy(),a.evidence_parameters.copy())
        self.assertFalse(a.update(e,p,np.ones((2,2))))
        for x,y in zip(before,(a.logits,a.reward_value,a.evidence_parameters)):np.testing.assert_array_equal(x,y)
        self.assertEqual(a.visits.sum(),0)
        e=Episode(1,0,p.identity,'actor',e.transitions);a.update(e,p,np.ones((2,2)))
        self.assertTrue(np.all(a.snapshot().probabilities[~m.mask]==0.))

    def test_evaluator_matches_existing_model_values_and_floor_response(self):
        m=JunctionModel(JunctionConfig(protocol='prior_shift'));truth=0;ev=Evaluator(m,truth);beta=np.zeros((10,10));beta[0,6]=.5
        a=StabilityActor(m,candidate('guarded_natural_003'),CFG['schedule']);p=a.snapshot();ev.prices(beta)
        j,k=m.evaluate(p);self.assertAlmostEqual(ev.policy_value(p),j[truth]+np.sum(beta*k))
        floors=np.array([0.,.02,.02]);value,q=ev.response(floors)
        self.assertLess(value,ev.optimum)
        self.assertLess(ev.actor_metrics(a,beta)['final_actor_constrained_augmented_gap'],ev.actor_metrics(a,beta)['final_actor_augmented_gap'])

    def test_complete_run_reference_parity(self):
        from ceac_pseudocode.run import run_cell as original
        config=JunctionConfig(protocol='prior_shift')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);x=original(config,'ceac',2001,0,0.,2048,root/'original')
            y=run_cell(config,candidate('original_reference'),CFG['schedule'],2001,0,0.,2048,root/'new')
            self.assertGreater(x['actor_updates'],0)
            for key in ('actor_updates','priced_actor_updates','late_gap','cumulative_regret','events'):self.assertEqual(x[key],y[key])
            with gzip.open(root/'original/episodes.jsonl.gz','rb') as f, gzip.open(root/'new/episodes.jsonl.gz','rb') as g:self.assertEqual(f.read(),g.read())
            self.assertEqual(json.loads((root/'original/design_history.json').read_text()),json.loads((root/'new/design_history.json').read_text()))

    def test_gate_rejects_incomplete_grid_and_never_relaxes_thresholds(self):
        with self.assertRaises(ValueError):selection(CFG,[])
        rows=[dict(candidate=c['id'],task=t,seed=s,final_actor_augmented_gap=.006,late_gap=.001,
            actor_updates=100,priced_actor_updates=100,cumulative_regret=10.)
            for c in CFG['candidates'] for t in (0,1) for s in CFG['calibration']['seeds']]
        self.assertIsNone(selection(CFG,rows)['selected'])
        for row in rows:
            if row['candidate']=='guarded_natural_001':row['final_actor_augmented_gap']=.001
        self.assertEqual(selection(CFG,rows)['selected'],'guarded_natural_001')

if __name__=='__main__':unittest.main()
