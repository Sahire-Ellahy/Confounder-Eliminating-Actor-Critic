"""Finite product-model compiler with checked shared-policy response geometry."""
from types import SimpleNamespace
from dataclasses import asdict
import math
import numpy as np
from scipy.special import kl_div
from ceac.model import FiniteModel
from ceac.contracts import Policy
from .environment import JunctionEnv,JunctionConfig

class JunctionModel(FiniteModel):
    def __init__(self,config=None,q_values=None):
        self.cfg=config or JunctionConfig();self.kind=self.cfg.kind;self.context=0
        qs=(.1,.3,.5,.7,.9) if self.cfg.protocol=='prior_shift' else (.5,)
        self.q_values=tuple(qs if q_values is None else q_values)
        if len(set(self.q_values))!=len(self.q_values):raise ValueError('Duplicate nuisance law')
        self.hypotheses=tuple((task,float(q)) for task in (0,1) for q in self.q_values)
        self._models=[JunctionEnv(self.cfg,task,q) for task,q in self.hypotheses]
        self.num_models=len(self.hypotheses);self.actions=self._models[0].action_space.n
        self.initial_observation=0;self.benchmark='junction_'+self.cfg.protocol
        self.specification=asdict(self.cfg)
        self._compile_product()
        self.optimal_policies=[self.response(np.eye(self.num_models)[m],np.zeros((self.num_models,self.num_models)))[0]
                               for m in range(self.num_models)]
        self.optimal_returns=np.array([self.evaluate(p)[0][m] for m,p in enumerate(self.optimal_policies)])
        self.alternatives=tuple((m,j) for m in range(self.num_models) for j in range(self.num_models) if m!=j
            and self.optimal_returns[j]-self.evaluate(self.optimal_policies[m])[0][j]>1e-10)
        self._mi_cache=None

    def _compile_product(self):
        count=self.num_models;env=self._models[0]
        self.state_ids=np.array(env.states,dtype=int);self.n=len(self.state_ids)
        self.index={int(s):i for i,s in enumerate(self.state_ids)};self.initial=self.index[0]
        self.mask=np.array([env.valid_action_mask(s) for s in self.state_ids])
        self.levels=[np.array([i for i,s in enumerate(self.state_ids) if env.elapsed(s)==t]) for t in range(2)]
        self.reward=np.zeros((count,self.n,self.actions));self.kl=np.zeros((count,count,self.n,self.actions))
        self.density=np.full((count,self.n),np.nan);self.density[:,self.initial]=1.
        self.laws={};edge=[];destination=[];prob=[]
        for s,obs in enumerate(self.state_ids):
            for action in np.flatnonzero(self.mask[s]):
                laws=[e.law(int(obs),int(action)) for e in self._models]
                if any(law.keys()!=laws[0].keys() or any(p<=0 for p in law.values()) or not np.isclose(sum(law.values()),1) for law in laws):
                    raise ValueError('Candidate outcome support must be common, positive and normalized')
                self.laws[s,int(action)]=laws
                for m,law in enumerate(laws):
                    self.reward[m,s,action]=sum(p*k[1] for k,p in law.items())
                    for j,other in enumerate(laws):
                        self.kl[m,j,s,action]=max(0.,sum(p*math.log(p/other[k]) for k,p in law.items()))
                grouped={}
                for key in laws[0]:
                    nxt,reward,done=key;dest=self.n if done else self.index[nxt]
                    edge.append(s*self.actions+action);destination.append(dest);prob.append([law[key] for law in laws])
                    if not done:
                        if env.elapsed(nxt)<=env.elapsed(int(obs)):raise ValueError('Non-DAG transition')
                        grouped.setdefault(dest,np.zeros(count));grouped[dest]+=[law[key] for law in laws]
                for dest,mass in grouped.items():
                    ratio=self.density[:,s]*mass/mass[0]
                    if np.isnan(self.density[0,dest]):self.density[:,dest]=ratio
                    elif not np.allclose(self.density[:,dest],ratio,atol=1e-12,rtol=1e-10):
                        raise ValueError('Policy-dependent occupancy ratio: unsupported response oracle')
        self.edge=np.array(edge,dtype=int);self.destination=np.array(destination,dtype=int);self.prob=np.array(prob).T
        if not np.isfinite(self.density).all():raise ValueError('Unreachable public state')
        future=np.zeros(self.n+1,dtype=bool)
        for s in reversed(range(self.n)):
            for a in np.flatnonzero(self.mask[s]):
                future[s]|=self.kl[:,:,s,a].max()>1e-12
                future[s]|=any(future[self.index[k[0]]] for k in self.laws[s,int(a)][0] if not k[2])
        for (s,a),laws in self.laws.items():
            for key in laws[0]:
                if key[2]:continue
                dest=self.index[key[0]]
                ratios=self.density[:,s]*np.array([law[key] for law in laws])/laws[0][key]
                if future[dest] and not np.allclose(ratios,self.density[:,dest],atol=1e-12,rtol=1e-10):
                    raise ValueError('Forgotten evidence affects future predictive beliefs')
        self._informative_pairs=tuple((s,a) for s,a in self.laws if self.kl[:,:,s,a].max()>1e-12)
        self._evaluations={}

    def returns(self,policy):
        self.validate_policy(policy)
        immediate=np.sum(policy.probabilities[None,:,:]*self.reward,axis=2)
        edges=np.flatnonzero(self.edge//self.actions==self.initial)
        values=np.c_[immediate,np.zeros(self.num_models)]
        continuation=np.sum(self.prob[:,edges]*values[:,self.destination[edges]]*
            policy.probabilities[self.initial,self.edge[edges]%self.actions],axis=1)
        return immediate[:,self.initial]+continuation

    def evaluate(self,policy):
        self.validate_policy(policy);key=policy.identity
        if key in self._evaluations:return self._evaluations[key]
        returns=self.returns(policy);information=np.zeros((self.num_models,self.num_models))
        for m in range(self.num_models):
            for j in range(self.num_models):
                if m==j:continue
                values=np.zeros(self.n+1)
                for states in reversed(self.levels):
                    q=self.kl[m,j,states]+self._continuation(values,m)[states]
                    values[states]=np.sum(policy.probabilities[states]*q,axis=1)
                information[m,j]=values[self.initial]
        if len(self._evaluations)>512:self._evaluations.clear()
        self._evaluations[key]=(returns,information)
        return returns,information

    def mutual_information_stage(self,posterior):
        posterior=np.asarray(posterior,dtype=float)
        if posterior.shape!=(self.num_models,) or (posterior<0).any() or not np.isclose(posterior.sum(),1):
            raise ValueError('Invalid posterior')
        key=tuple(posterior)
        if self._mi_cache is not None and self._mi_cache[0]==key:return self._mi_cache[1]
        stage=np.zeros(self.mask.shape);masses=posterior[:,None]*self.density
        for s,a in self._informative_pairs:
            laws=self.laws[s,a];weights=masses[:,s];belief=weights/weights.sum()
            probabilities=np.array([[law[k] for k in laws[0]] for law in laws])
            mixture=belief@probabilities
            stage[s,a]=max(0.,float(weights@kl_div(probabilities,mixture[None,:]).sum(axis=1)))
        self._mi_cache=(key,stage)
        return stage

    def seed_policies(self):
        decisions=[0] if self.cfg.protocol=='burden' else [1,2]
        policies=[]
        for a in range(3):
            p=np.zeros(self.mask.shape);p[:,0]=1.;p[decisions]=0.;p[decisions,a]=1.
            policies.append(Policy(p,('incumbent','challenge','diagnostic')[a]))
        return policies

    def prior(self,treatment):
        if treatment=='product':return np.ones(self.num_models)/self.num_models
        if treatment!='correlated':raise ValueError(treatment)
        p=np.array([.5 if ((task==0 and np.isclose(q,.1)) or (task==1 and np.isclose(q,.9))) else 0.
                    for task,q in self.hypotheses])
        if not np.isclose(p.sum(),1):raise ValueError('Frozen prior atoms not in public model class')
        return p

    def true_index(self,task,q=.5):
        hits=[m for m,(t,p) in enumerate(self.hypotheses) if t==task and abs(p-q)<1e-12]
        if len(hits)!=1:raise ValueError('True law absent from the public product class')
        return hits[0]
