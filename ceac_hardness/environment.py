"""Two-step controlled junctions, separate from the historical grid worlds.

Every action is a simulator step. Undertow adds reachable dead-end branches
(one action per branch); False Beacon widens a random, zero-reward chamber
behind a single action. The cue-shift family instead has an obligatory beacon
followed by a route decision. Useful task laws never change with burden/shift.
"""
from dataclasses import dataclass,asdict
from types import SimpleNamespace
import numpy as np

@dataclass(frozen=True)
class JunctionConfig:
    kind: str = 'false_beacon_junction'
    protocol: str = 'burden'
    burden: int = 0
    safe_reward: float = .5
    probe_reward: float = .48
    risky_probabilities: tuple = (.1,.9)
    probe_probabilities: tuple = (.15,.85)
    max_steps: int = 2

    def __post_init__(self):
        if self.kind not in {'false_beacon_junction','undertow_junction'}:raise ValueError('Unknown junction')
        if self.protocol not in {'burden','prior_shift'} or self.burden<0 or int(self.burden)!=self.burden:
            raise ValueError('Invalid protocol/burden')
        if self.protocol=='prior_shift' and self.burden:raise ValueError('Separate the two interventions')
        if self.max_steps!=2:raise ValueError('This registered junction has two physical steps')
        if not 0<self.probe_reward<self.safe_reward<1:raise ValueError('Invalid route rewards')
        if any(not 0<p<1 for p in self.risky_probabilities+self.probe_probabilities):raise ValueError('Common support required')

class JunctionEnv:
    def __init__(self, config=None, task=0, q=.5):
        self.config=config or JunctionConfig()
        if task not in (0,1) or not 0<q<1:raise ValueError('Invalid realized model')
        self.task,self.q=int(task),float(q)
        c=self.config
        actions=(3+c.burden if c.kind=='undertow_junction' else 3+int(c.burden>0)) if c.protocol=='burden' else 3
        self.action_space=SimpleNamespace(n=actions)
        self.states=tuple(range(3 if c.protocol=='prior_shift' else 2+c.burden))
        self.np_random=np.random.default_rng();self.observation=0;self.terminated=False

    def valid_action_mask(self, observation):
        if observation not in self.states:raise ValueError('Not an active state')
        mask=np.zeros(self.action_space.n,dtype=bool)
        decision=(observation>0) if self.config.protocol=='prior_shift' else (observation==0)
        if decision:mask[:]=True
        else:mask[0]=True
        return mask

    def elapsed(self, observation):return int(observation!=0)

    def law(self, observation, action):
        """The sampler and public candidate compiler use this same kernel."""
        if not self.valid_action_mask(observation)[action]:raise ValueError('Masked action')
        c=self.config
        if c.protocol=='burden':
            if observation>0:return {(100,0.,True):1.}
            if action==0:return {(1,c.safe_reward,False):1.}
            if action==1:
                p=c.risky_probabilities[self.task]
                return {(1,0.,False):1-p,(1,1.,False):p}
            if action==2:
                p=c.probe_probabilities[self.task]
                # Same mean payoff in both tasks; central vs extreme readings
                # identify the task without adding occupancy-entropy states.
                return {(1,c.probe_reward-.02,False):(1-p)/2,
                        (1,c.probe_reward,False):p,
                        (1,c.probe_reward+.02,False):(1-p)/2}
            if c.kind=='undertow_junction':return {(action-1,0.,False):1.}
            return {(2+i,0.,False):1/c.burden for i in range(c.burden)}
        if observation==0:return {(1,0.,False):1-self.q,(2,0.,False):self.q}
        route=action if c.protocol=='prior_shift' else observation-1
        if route==0:return {(100,c.safe_reward,True):1.}
        if route==1:
            p=c.risky_probabilities[self.task]
            return {(101,0.,True):1-p,(102,1.,True):p}
        if route==2:
            p=c.probe_probabilities[self.task]
            return {(103,c.probe_reward,True):1-p,(104,c.probe_reward,True):p}
        return {(105,0.,True):1.}

    def reset(self,seed=None,options=None):
        if seed is not None:self.np_random=np.random.default_rng(seed)
        self.observation=0;self.terminated=False
        return 0,{}

    def step(self,action):
        if self.terminated:raise RuntimeError('Reset before stepping a terminal episode')
        previous=self.observation
        law=self.law(previous,int(action));outcomes=list(law)
        chosen=outcomes[int(self.np_random.choice(len(outcomes),p=list(law.values())))]
        nxt,reward,done=chosen;self.observation=nxt;self.terminated=done
        c=self.config
        if c.protocol=='burden':
            event=('safe','risky','probe')[action] if previous==0 and action<3 else ('nuisance' if previous==0 else 'finish')
        else:
            event='cue' if previous==0 else ('safe','risky','probe')[int(action)]
        return nxt,reward,done,False,{'event':event,'physical_steps':1}
