"""Predictable schedules and guarded geometry; no model-law or evaluator access."""
from copy import deepcopy
import math
import numpy as np
from scipy.special import logsumexp,xlogy
from ceac_pseudocode.learning import TwoCriticActor


def project_kl(log_weights,total_floor):
    """argmin_q KL(q || normalized weights), q_i >= total_floor/K."""
    w=np.asarray(log_weights,dtype=float)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all():raise ValueError('Finite valid-action log weights required')
    if not 0<=total_floor<1:raise ValueError('Floor mass must lie in [0,1)')
    mu=total_floor/len(w);free=np.ones(len(w),dtype=bool);q=np.zeros(len(w))
    for _ in range(len(w)+1):
        mass=1-mu*int((~free).sum())
        q[free]=mass*np.exp(w[free]-logsumexp(w[free]));q[~free]=mu
        bad=free & (q<mu)
        if not bad.any():return q
        free[bad]=False
        if not free.any():raise RuntimeError('Interior-simplex projection exhausted free mass')
    raise RuntimeError('KL projection failed')


def guarded_proposal(old,direction,eta,total_floor,kl_limit):
    old=np.asarray(old,dtype=float)
    if (old<=0).any() or not np.isfinite(direction).all():raise FloatingPointError('Nonfinite or zero-propensity guarded update')
    if old.min()<total_floor/len(old)-1e-12:raise ValueError('Old policy must satisfy the new floor')
    logs=np.log(old)
    for contractions in range(61):
        q=project_kl(logs+eta*direction,total_floor)
        kl=float(np.sum(xlogy(q,q)-q*logs))
        if kl<=kl_limit+1e-14:return q,max(0.,kl),contractions,eta
        eta*=.5
    raise FloatingPointError('Trust backtracking failed; no silent unguarded step')


class StabilityActor(TwoCriticActor):
    def __init__(self,model,candidate,schedule):
        self.candidate=deepcopy(candidate);self.schedule=deepcopy(schedule)
        self.reference=candidate['id']=='original_reference'
        super().__init__(model,actor_rate=candidate['actor_initial'],
                         critic_rate=candidate.get('critic_initial',schedule['critic_initial']))
        self.visits=np.zeros(model.n,dtype=int)
        self.totals=dict(decisions=0,backtracks=0,contracted_decisions=0,floor_mass_sum=0.,
            reward_td_sq=0.,evidence_td_sq=0.,raw_gradient_sq=0.,effective_evidence_step_sum=0.,max_policy_kl=0.)
        self.last_geometry=[]

    def rates(self,state):
        if self.reference:return self.actor_rate,self.critic_rate,0.,float('inf')
        cfg=self.schedule;h=1+int(self.visits[state])/cfg['offset']
        eta=self.candidate['actor_initial']*h**(-cfg['actor_exponent'])
        critic=cfg['critic_initial']*h**(-cfg['critic_exponent'])
        floor=self.candidate['total_floor']*h**(-cfg['total_floor_exponent'])
        bound=cfg['trust_limit_initial']*h**(-cfg['trust_limit_exponent'])
        return eta,critic,floor,bound

    def floors(self):
        return np.array([0. if self.model.mask[s].sum()==1 else self.rates(s)[2] for s in range(self.model.n)])

    def _record(self,policy,beta,errors,geometry):
        norm2=float(np.sum(beta**2));self.last_geometry=geometry
        for s,dr,di,_ in errors:
            eta,c,floor,_=self.rates(s)
            effective=self.critic_rate*norm2 if self.reference else c*norm2/(self.schedule['evidence_step_ridge']+norm2)
            self.totals['reward_td_sq']+=dr**2;self.totals['evidence_td_sq']+=di**2
            self.totals['effective_evidence_step_sum']+=effective
        for row in geometry:
            self.totals['decisions']+=1;self.totals['backtracks']+=row['backtracks']
            self.totals['contracted_decisions']+=int(row['backtracks']>0)
            self.totals['floor_mass_sum']+=row['floor_mass']
            self.totals['raw_gradient_sq']+=row['raw_gradient_sq']
            self.totals['max_policy_kl']=max(self.totals['max_policy_kl'],row['policy_kl'])
        for s,*_ in errors:self.visits[s]+=1

    def update(self,episode,policy,beta):
        if episode.behavior!='actor':return False
        if self.reference:
            result=super().update(episode,policy,beta)
            new=self.snapshot().probabilities;geometry=[]
            for step,(s,dr,di,_) in zip(episode.transitions,self.last_update['errors']):
                valid=self.model.mask[s]
                if valid.sum()==1:continue
                p=policy.probabilities[s,valid];q=new[s,valid]
                kl=float(np.sum(xlogy(q,q)-xlogy(q,p)))
                score=-policy.probabilities[s].copy();score[step.action]+=1
                geometry.append(dict(state=s,policy_kl=max(0.,kl),backtracks=0,floor_mass=0.,
                    actor_step=self.actor_rate,raw_gradient_sq=float(np.sum(((dr+di)*score)**2))))
            self._record(policy,np.asarray(beta),self.last_update['errors'],geometry)
            return result
        if episode.policy_id!=policy.identity or policy.identity!=self.snapshot().identity:
            raise ValueError('Only the frozen current-actor episode can update this optimizer')
        beta=np.asarray(beta,dtype=float)
        if beta.shape!=self.evidence_parameters.shape[:2] or not np.isfinite(beta).all() or (beta<0).any():raise ValueError('Invalid evidence prices')
        vi=self.evidence_value(beta);errors=[];directions={};geometry=[];before=self.logits.copy()
        for step in episode.transitions:
            s=step.state;ll=np.asarray(step.log_likelihood)
            evidence=float(np.sum(beta*(ll[:,None]-ll[None,:])))
            dr=step.reward-self.reward_value[s]+(0 if step.done else self.reward_value[step.next_state])
            di=evidence-vi[s]+(0 if step.done else vi[step.next_state])
            errors.append((s,float(dr),float(di),evidence))
            valid=np.flatnonzero(self.model.mask[s]);p=policy.probabilities[s,valid]
            if len(valid)==1:continue
            a=int(np.flatnonzero(valid==step.action)[0])
            if self.candidate['geometry']=='natural_mirror':
                if p[a]<=0:raise FloatingPointError('Sampled action has zero propensity')
                direction=np.zeros(len(valid));direction[a]=(dr+di)/p[a]
                direction-=direction.mean() # Gauge only; identical policy to the mirror step.
            else:
                direction=-p.copy();direction[a]+=1;direction*=dr+di
            directions[s]=directions.get(s,np.zeros_like(direction))+direction
        # All actor errors and directions were frozen before either critic update.
        for s,direction in directions.items():
            valid=np.flatnonzero(self.model.mask[s]);p=policy.probabilities[s,valid]
            eta,_,floor,bound=self.rates(s)
            if self.candidate['geometry']=='ordinary':
                self.logits[s,valid]+=eta*direction
                z=self.logits[s,valid];q=np.exp(z-logsumexp(z))
                kl=max(0.,float(np.sum(xlogy(q,q)-xlogy(q,p))));contracted=0
            else:
                q,kl,contracted,eta=guarded_proposal(p,direction,eta,floor,bound)
                self.logits[s,valid]=np.log(q)
            geometry.append(dict(state=s,policy_kl=kl,backtracks=contracted,floor_mass=floor,
                actor_step=eta,raw_gradient_sq=float(direction@direction)))
        norm2=float(np.sum(beta**2))
        for s,dr,di,_ in errors:
            _,c,_,_=self.rates(s)
            self.reward_value[s]+=c*dr
            self.evidence_parameters[:,:,s]+=c/(self.schedule['evidence_step_ridge']+norm2)*di*beta
        self.updates+=1;self.priced_updates+=int(np.any(beta))
        self.last_update=dict(errors=errors,beta=beta.tolist(),actor_step_norm=float(np.linalg.norm(self.logits-before)))
        self._record(policy,beta,errors,geometry)
        if not all(np.isfinite(x).all() for x in (self.logits,self.reward_value,self.evidence_parameters)):
            raise FloatingPointError('Nonfinite stability actor/critics; no silent clipping')
        return True

    def diagnostics(self):
        p=self.snapshot().probabilities;valid=self.model.mask
        entropies=-np.sum(xlogy(p,p),axis=1)
        # Full eigensolve only at logging cadence, never in the learning update.
        conditions=[]
        for s in np.flatnonzero(valid.sum(axis=1)>1):
            row=p[s,valid[s]];eigen=np.linalg.eigvalsh(np.diag(row)-np.outer(row,row))
            positive=eigen[eigen>1e-14]
            conditions.append(float(positive.max()/positive.min()) if len(positive) else None)
        return dict(totals=dict(self.totals),state_visits=self.visits.tolist(),floor_mass=self.floors().tolist(),
            minimum_probability=float(p[valid].min()),state_entropy=entropies.tolist(),
            fisher_positive_eigenvalue_condition=conditions,last_geometry=self.last_geometry)
