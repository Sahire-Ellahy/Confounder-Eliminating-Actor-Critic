"""Equation-level CEAC fidelity, independent of environment performance."""
import unittest
import numpy as np
from ceac.contracts import Episode,Transition
from ceac_hardness.environment import JunctionConfig,JunctionEnv
from ceac_hardness.model import JunctionModel
from ceac_pseudocode.learning import TwoCriticActor,PolicyReplay
from ceac_pseudocode.design import solve_replay_master
from ceac_pseudocode.agent import CEAC,Config

class FidelityTests(unittest.TestCase):
    def test_two_critic_update_matches_price_conditioned_gradient(self):
        m=JunctionModel();actor=TwoCriticActor(m,actor_rate=.07,critic_rate=.13,temperature=.4)
        actor.reward_value[:]=[.2,.1];actor.evidence_parameters[0,1]=[.3,.2]
        actor.evidence_parameters[1,0]=[-.1,.4]
        beta=np.array([[0.,.7],[.2,0.]]);p=actor.snapshot()
        transitions=(Transition(0,2,.48,1,False,(-2.,-.1)),Transition(1,0,0.,-1,True,(0.,0.)))
        episode=Episode(0,0,p.identity,'actor',transitions)
        old_w=actor.reward_value.copy();old_z=actor.evidence_parameters.copy();old_theta=actor.logits.copy()
        v=np.einsum('ij,ijs->s',beta,old_z);dt=np.zeros_like(old_theta);dw=np.zeros_like(old_w);dz=np.zeros_like(old_z)
        # Direct transcription of the displayed equations, including grad_z V=beta.
        for step in transitions:
            ll=np.asarray(step.log_likelihood);e=sum(beta[i,j]*(ll[i]-ll[j]) for i in range(2) for j in range(2))
            dr=step.reward-old_w[step.state]+(0 if step.done else old_w[step.next_state])
            di=e-v[step.state]+(0 if step.done else v[step.next_state])
            score=(np.eye(m.actions)[step.action]-p.probabilities[step.state])/.4
            dt[step.state]+=.07*(dr+di)*score;dw[step.state]+=.13*dr;dz[:,:,step.state]+=.13*di*beta
        actor.update(episode,p,beta)
        np.testing.assert_allclose(actor.logits,old_theta+dt)
        np.testing.assert_allclose(actor.reward_value,old_w+dw)
        np.testing.assert_allclose(actor.evidence_parameters,old_z+dz)
        self.assertEqual(actor.priced_updates,1)

    def test_zero_price_recovers_reward_only_even_after_evidence_learning(self):
        m=JunctionModel();x=TwoCriticActor(m);y=TwoCriticActor(m);x.evidence_parameters[:]=123.
        p=x.snapshot();e=Episode(0,0,p.identity,'actor',(Transition(0,1,1.,1,False,(-3.,-.1)),Transition(1,0,0.,-1,True,(0.,0.))))
        for actor in (x,y):actor.update(e,p,np.zeros((2,2)))
        np.testing.assert_array_equal(x.logits,y.logits);np.testing.assert_array_equal(x.reward_value,y.reward_value)
        self.assertTrue(np.all(x.evidence_parameters==123.))

    def test_frozen_replay_never_updates_actor_or_critics(self):
        actor=TwoCriticActor(JunctionModel());p=actor.snapshot()
        e=Episode(0,0,p.identity,'statistical',(Transition(0,0,.5,1,False,(0.,0.)),Transition(1,0,0.,-1,True,(0.,0.))))
        before=(actor.logits.copy(),actor.reward_value.copy(),actor.evidence_parameters.copy())
        self.assertFalse(actor.update(e,p,np.ones((2,2))))
        for x,y in zip(before,(actor.logits,actor.reward_value,actor.evidence_parameters)):np.testing.assert_array_equal(x,y)

    def make_replay(self,reward,ll):
        m=JunctionModel();replay=PolicyReplay(m);p=m.seed_policies()[2]
        for i in range(2048):
            e=Episode(i,0,p.identity,'statistical',(Transition(0,2,reward,1,False,tuple(ll)),Transition(1,0,0.,-1,True,(0.,0.))))
            replay.append(e)
        return replay,p

    def test_master_changes_when_observed_replay_changes(self):
        first,p=self.make_replay(.48,[-.1,-2.]);second,q=self.make_replay(.1,[-.1,-1.])
        a=solve_replay_master(first,[p],(0,),.5);b=solve_replay_master(second,[q],(0,),.5)
        self.assertIsNotNone(a);self.assertIsNotNone(b)
        self.assertNotAlmostEqual(a.beta[0,1],b.beta[0,1]);self.assertNotAlmostEqual(a.cost,b.cost)
        self.assertEqual(a.record()['estimation'],'observed_policy_labeled_replay')
        self.assertAlmostEqual(a.data['return_mean'][0],.48)
        self.assertAlmostEqual(a.data['information_mean'][0,0],1.9)

    def test_replay_prices_rescore_data_without_mutating_or_recrediting(self):
        replay,p=self.make_replay(.48,[-.1,-2.]);n=len(replay.seen)
        low=replay.priced_scores([p],((0,1),),np.zeros((2,2)))
        high=replay.priced_scores([p],((0,1),),np.array([[0.,2.],[0.,0.]]))
        self.assertAlmostEqual(high[0]-low[0],3.8);self.assertEqual(len(replay.seen),n)
        with self.assertRaises(ValueError):replay.append(replay.strata[p.identity][0])

    def test_infeasible_replay_is_not_replaced_by_model_expectations(self):
        m=JunctionModel();replay=PolicyReplay(m)
        self.assertIsNone(solve_replay_master(replay,m.seed_policies(),(0,),.5))
        a=CEAC(m,np.random.default_rng(1));p,behavior,prices=a.select(0)
        self.assertEqual(behavior,'statistical');self.assertEqual(a.actor.updates,0)
        self.assertFalse(a.certified)

    def test_no_debt_means_actor_not_an_exact_deployment_policy(self):
        m=JunctionModel();a=CEAC(m,np.random.default_rng(1));a.certified=True
        a.last_refs=a.replay.plausible
        for pid in a.ledger.statistical_required:a.ledger.statistical_assigned[pid]=a.ledger.statistical_required[pid]
        p,behavior,_=a.select(0)
        self.assertEqual(behavior,'actor');self.assertEqual(p.identity,a.actor.snapshot().identity)
        self.assertIn(p.identity,a.replay.slots);self.assertEqual(a.replay.statistics[p.identity][0],0)

if __name__=='__main__':unittest.main()
