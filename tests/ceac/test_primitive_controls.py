"""Source parity, information isolation and finite-horizon adapter contracts."""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from ceac.contracts import Episode, Transition
from ceac.primitives import PrimitiveSpec, IDAC, IDACConfig, UCBQ, UCBQConfig, OptimisticPOMD, POMDConfig
from ceac.run import METHODS, execute


class SourceParityTests(unittest.TestCase):
    def test_idac_matches_unmodified_upstream_update_over_multiple_batches(self):
        fixture = json.loads((Path(__file__).parents[1]/'fixtures/idac_upstream.json').read_text())
        namespace = dict(np=np, defaultdict=defaultdict, entropy=entropy)
        # These two class bodies are verbatim from the pinned source, not a
        # second reimplementation of the local update formula.
        exec(fixture['measure_class'], namespace)
        exec(fixture['agent_class'], namespace)
        reference = object.__new__(namespace['InfoRatioACBASE'])
        spec = PrimitiveSpec(np.ones((3,2),bool),np.arange(3),3,0,-1.,1.)
        cfg = IDACConfig(policy_lr=.017, value_lr=.12, tau=.2, kappa=.3)
        agent = IDAC(spec,np.random.default_rng(2),cfg)
        reference.policy_lr, reference.value_lr = cfg.policy_lr,cfg.value_lr
        reference.mu_c,reference.mu_r=agent.mu_cost,agent.mu_entropy
        reference.tau,reference.kappa,reference.entropy_eps=cfg.tau,cfg.kappa,cfg.entropy_eps
        class Values:
            def __init__(self,values):self.params=values.copy()
            def __call__(self,s):return self.params[s]
            def gradient(self,s):return np.eye(len(self.params))[s]
        class Actor:
            def __init__(self,logits):self.params=logits.copy();self.actions=[]
            def sample(self,s):return self.actions.pop(0)
            def grad_log_policy(self,s,a):
                gradient=np.zeros_like(self.params)
                gradient[s]=-softmax(self.params[s]);gradient[s,a]+=1.
                return gradient
        reference.pi=Actor(agent.logits)
        reference.v_c=Values(agent.cost_value)
        reference.v_h=Values(agent.shadow_value)
        for states, actions, costs, next_states in (
                ([0,1,0,2,1,0],[1,0,0,1,1,0],[.2,.9,.4,.3,.7,.5],[1,0,2,1,0,2]),
                ([2,0,1,2,2,0],[0,1,1,0,1,1],[.8,.2,.1,.6,.9,.3],[0,1,2,2,0,1])):
            class Replay:
                state=states[0]
                cursor=0
                def step(self,action):
                    i=self.cursor;self.cursor+=1
                    assert action==actions[i]
                    return next_states[i],costs[i],False,{}
                def reset(self):return states[0]
            reference.pi.actions=list(actions)
            reference.update(Replay(),len(states))
            agent.update_batch(states,actions,costs,next_states)
            np.testing.assert_allclose(agent.logits,reference.pi.params,rtol=0,atol=2e-14)
            np.testing.assert_allclose(agent.cost_value,reference.v_c.params,rtol=0,atol=2e-14)
            np.testing.assert_allclose(agent.shadow_value,reference.v_h.params,rtol=0,atol=2e-14)
            self.assertAlmostEqual(agent.mu_cost,reference.mu_c,places=14)
            self.assertAlmostEqual(agent.mu_entropy,reference.mu_r,places=14)

    def test_ucb_q_matches_published_first_visit_target_and_forward_order(self):
        spec=PrimitiveSpec(np.ones((2,2),bool),np.array([0,1]),2,0,-1.,1.)
        cfg=UCBQConfig(bonus_constant=.5)
        agent=UCBQ(spec,np.random.default_rng(3),episodes=10,config=cfg)
        policy,_,prices=agent.select(0)
        actions=np.argmax(policy.probabilities,axis=1)
        episode=Episode(0,0,policy.identity,'ucb_q',(
            Transition(0,int(actions[0]),0.,1,False,(0.,0.)),
            Transition(1,int(actions[1]),1.,-1,True,(0.,0.))))
        agent.observe(episode,policy,prices)
        bonus=.5*np.sqrt(8*agent.log_factor)
        # alpha_1=1; next value is its pre-episode H (forward online order).
        self.assertAlmostEqual(agent.q[0,actions[0]],.5+2+bonus)
        self.assertAlmostEqual(agent.q[1,actions[1]],1+bonus)
        self.assertEqual(agent.counts.sum(),2)
        self.assertTrue(np.all((policy.probabilities==0)|(policy.probabilities==1)))

    def test_pomd_matches_backward_policy_evaluation_and_softmax_mirror_step(self):
        spec=PrimitiveSpec(np.ones((2,2),bool),np.array([0,1]),2,0,0.,1.)
        agent=OptimisticPOMD(spec,100,POMDConfig(policy_step=.3))
        agent.logits[0]=np.log([.6,.4]);agent.logits[1]=np.log([.25,.75])
        before=agent.probabilities.copy();N=10**8
        agent.counts[:2]=N
        means=np.array([[.2,.8],[.3,.9]])
        agent.cost_sum[:2]=N*means
        terminal=len(agent.mask)
        for action in (0,1):
            agent.transitions[0,action,1]=N
            agent.transitions[1,action,terminal]=N
        bc,bp=agent.bonuses()
        q1=means[1]-bc[1]-bp[1]
        v1=float(before[1]@q1)
        q0=means[0]-bc[0]-bp[0]+v1
        agent.evaluate_and_improve()
        np.testing.assert_allclose(agent.q[:2],[q0,q1],atol=1e-14)
        for state,q in enumerate((q0,q1)):
            expected=before[state]*np.exp(-.3*q);expected/=expected.sum()
            np.testing.assert_allclose(agent.probabilities[state],expected,atol=1e-14)
        # Evaluate under pi_k, not a greedy next-action minimizer.
        self.assertAlmostEqual(agent.values[1],v1)
        self.assertGreater(agent.probabilities[0,0],before[0,0])

    def test_pomd_preserves_published_model_update_order_and_bonus_constants(self):
        spec=PrimitiveSpec(np.ones((1,2),bool),np.array([0]),1,0,0.,1.)
        agent=OptimisticPOMD(spec,10)
        p,_,prices=agent.select(0)
        e=Episode(0,0,p.identity,'pomd',(Transition(0,0,0.,-1,True,(0.,0.)),))
        agent.observe(e,p,prices)
        np.testing.assert_array_equal(agent.q,0.)
        self.assertEqual(agent.last_model_episodes,0)
        self.assertEqual(agent.counts[0,0],1)
        self.assertEqual(agent.cost_sum[0,0],1.)
        self.assertAlmostEqual(agent.policy_step,np.sqrt(2*np.log(2)/10))
        bc,bp=agent.bonuses();S=len(agent.mask);d=.05/3
        self.assertAlmostEqual(bc[0,0],np.sqrt(2*np.log(2*S*2*1*10/d)))
        self.assertAlmostEqual(bp[0,0],np.sqrt(4*S*np.log(3*S*2*1*10/d)))


class PrimitiveContractTests(unittest.TestCase):
    def spec(self):
        return PrimitiveSpec(np.array([[1,1],[1,0]],bool),np.array([0,1]),3,0,-1.,1.)

    def episode(self,policy,index=0):
        return Episode(index,0,policy.identity,'test',(
            Transition(0,0,.5,1,False,(.1,.2)),
            Transition(1,0,-.1,-1,True,(.1,.2))))

    def test_padding_preserves_affine_episode_return_and_reset_boundary(self):
        spec=self.spec();agent=IDAC(spec,np.random.default_rng(0))
        p,_,_=agent.select(0);e=self.episode(p);rows=spec.padded_episode(e)
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[-1],(spec.n+2,0,0.,spec.initial))
        self.assertEqual(rows[1][-1],spec.n+2)
        normalized=sum((r-spec.reward_lower)/spec.width for _,_,r,_ in rows)
        self.assertAlmostEqual(normalized,(e.total_return-3*spec.reward_lower)/spec.width)

    def test_idac_freezes_policy_and_updates_both_critics_at_batch_boundary(self):
        agent=IDAC(self.spec(),np.random.default_rng(1),IDACConfig(batch_episodes=2))
        p,_,z=agent.select(0);before=agent.cost_value.copy();shadow=agent.shadow_value.copy()
        agent.observe(self.episode(p),p,z)
        next_p,_,_=agent.select(1)
        self.assertEqual(next_p.identity,p.identity)
        np.testing.assert_array_equal(before,agent.cost_value)
        agent.observe(self.episode(next_p,1),next_p,z)
        self.assertEqual(agent.updates,1)
        self.assertGreater(np.linalg.norm(before-agent.cost_value),0.)
        self.assertGreater(np.linalg.norm(shadow-agent.shadow_value),0.)
        self.assertNotEqual(agent.select(2)[0].identity,p.identity)
        self.assertTrue(np.all(agent.probabilities[~agent.mask]==0))

    def test_primitive_learners_ignore_evaluator_likelihood_and_prices(self):
        for factory in (lambda:IDAC(self.spec(),np.random.default_rng(3),IDACConfig(batch_episodes=1)),
                        lambda:UCBQ(self.spec(),np.random.default_rng(3),10),
                        lambda:OptimisticPOMD(self.spec(),10)):
            a,b=factory(),factory()
            for i in range(3):
                pa,_,za=a.select(i);pb,_,zb=b.select(i)
                self.assertEqual(pa.identity,pb.identity)
                e=self.episode(pa,i)
                fake=replace(e,transitions=tuple(replace(t,log_likelihood=(999.,-999.)) for t in e.transitions))
                a.observe(e,pa,za);b.observe(fake,pb,np.full((2,2),999.))
                self.assertEqual(a.diagnostics(),b.diagnostics())
            self.assertFalse(hasattr(a,'model'))

    def test_retired_methods_cannot_launch_even_through_python_api(self):
        for name in ('policy_ucb','confidence','coverage','oir_idac'):
            self.assertNotIn(name,METHODS)
            with self.assertRaisesRegex(ValueError,'retired'):
                execute(None,Path('/must-not-be-created'),name,0,0,1,None)

    def test_noncausal_or_out_of_range_episodes_fail_closed(self):
        agent=IDAC(self.spec(),np.random.default_rng(1));p,_,_=agent.select(0)
        good=self.episode(p)
        for step in (replace(good.transitions[0],reward=2.),
                     replace(good.transitions[0],next_state=0),
                     replace(good.transitions[0],done=True)):
            with self.assertRaises(ValueError):
                self.spec().padded_episode(replace(good,transitions=(step,good.transitions[1])))


if __name__=='__main__':unittest.main()
