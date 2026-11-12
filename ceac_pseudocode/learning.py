"""Section 3 equations: evidence score, actor/critic steps and labeled replay."""
from collections import defaultdict
import math
import numpy as np
from scipy.special import logsumexp
from ceac.contracts import Policy
from ceac_hardness.learning import LikelihoodEvidence

class TwoCriticActor:
    """V_I(s; beta)=<z(s), beta>; update z with delta_I * grad_z V_I.

    This is ONE price-conditioned evidence critic, alongside the reward critic.
    Independent unpriced row TD updates are deliberately not used.
    """
    def __init__(self,model,actor_rate=.4,critic_rate=.12,temperature=1.):
        self.model=model;self.actor_rate=actor_rate;self.critic_rate=critic_rate;self.temperature=temperature
        self.logits=np.zeros(model.mask.shape)
        self.reward_value=np.zeros(model.n)
        self.evidence_parameters=np.zeros((model.num_models,model.num_models,model.n))
        self.updates=0;self.priced_updates=0;self.last_update=None

    def snapshot(self):
        x=np.where(self.model.mask,self.logits/self.temperature,-np.inf)
        return Policy(np.exp(x-logsumexp(x,axis=1,keepdims=True)),'ceac-actor')

    def evidence_value(self,beta):return np.einsum('ij,ijs->s',beta,self.evidence_parameters)

    def update(self,episode,policy,beta):
        if episode.behavior!='actor':return False
        if episode.policy_id!=policy.identity or policy.identity!=self.snapshot().identity:
            raise ValueError('Only the frozen current-actor episode can update actor and critics')
        beta=np.asarray(beta,dtype=float)
        if beta.shape!=self.evidence_parameters.shape[:2] or not np.isfinite(beta).all() or (beta<0).any():
            raise ValueError('Invalid frozen evidence price')
        vi=self.evidence_value(beta);delta_theta=np.zeros_like(self.logits);errors=[]
        for step in episode.transitions:
            s=step.state;ll=np.asarray(step.log_likelihood)
            evidence=float(np.sum(beta*(ll[:,None]-ll[None,:])))
            dr=step.reward-self.reward_value[s];di=evidence-vi[s]
            if not step.done:dr+=self.reward_value[step.next_state];di+=vi[step.next_state]
            score=-policy.probabilities[s].copy();score[step.action]+=1.;score/=self.temperature
            delta_theta[s]+=(dr+di)*score;errors.append((s,float(dr),float(di),evidence))
        # Equations ceac-actor-update then ceac-critic-updates, all errors frozen.
        self.logits+=self.actor_rate*delta_theta
        for s,dr,di,evidence in errors:
            self.reward_value[s]+=self.critic_rate*dr
            self.evidence_parameters[:,:,s]+=self.critic_rate*di*beta
        self.updates+=1;self.priced_updates+=int(np.any(beta))
        self.last_update=dict(errors=errors,beta=beta.tolist(),actor_step_norm=float(np.linalg.norm(delta_theta*self.actor_rate)))
        if not all(np.isfinite(x).all() for x in (self.logits,self.reward_value,self.evidence_parameters)):
            raise FloatingPointError('Nonfinite CEAC actor or critic; no silent clipping')
        return True

class PolicyReplay:
    """All episodes remain labeled; only executed-policy strata estimate columns."""
    def __init__(self,model,delta=.025):
        self.model=model;self.delta=delta
        self.evidence=LikelihoodEvidence(model.num_models,delta)
        self.strata=defaultdict(list);self.seen=set();self.slots={};self.statistics={}
        # Known outcome ranges, not expected rewards or KL coefficients.
        self.return_width=1.0 # The registered junctions have total return in [0,1].
        low=np.zeros((model.num_models,model.num_models));high=low.copy()
        for level in model.levels:
            entries=[]
            for s in level:
                for a in np.flatnonzero(model.mask[s]):
                    laws=model.laws[int(s),int(a)]
                    for key in laws[0]:
                        ll=np.log([law[key] for law in laws]);entries.append(ll[:,None]-ll[None,:])
            low+=np.min(entries,axis=0);high+=np.max(entries,axis=0)
        self.row_width=high-low

    @property
    def plausible(self):return self.evidence.plausible
    @property
    def mixture(self):return self.evidence.mixture

    def register(self,policy):
        pid=policy.identity
        if pid not in self.slots:
            self.slots[pid]=len(self.slots)
            episodes=self.strata[pid]
            self.statistics[pid]=[len(episodes),sum(e.total_return for e in episodes),
                sum((e.log_likelihood for e in episodes),np.zeros(self.model.num_models))]

    def append(self,episode):
        key=(episode.context,episode.index)
        if key in self.seen:raise ValueError('Replay cannot count an episode twice')
        if not 0<=episode.total_return<=1+1e-12:raise ValueError('Return outside registered junction range')
        self.seen.add(key);self.strata[episode.policy_id].append(episode);self.evidence.append(episode)
        if episode.policy_id in self.statistics:
            stat=self.statistics[episode.policy_id];stat[0]+=1;stat[1]+=episode.total_return;stat[2]+=episode.log_likelihood

    def arrays(self,policies,rows):
        counts=[];means=[];info=[];rj=[];ri=[]
        for policy in policies:
            self.register(policy);n,total,ll=self.statistics[policy.identity];counts.append(n)
            if not n:
                means.append(0.);info.append(np.zeros(len(rows)));rj.append(1.);ri.append(np.array([self.row_width[r] for r in rows]));continue
            means.append(total/n);averages=ll/n
            info.append(np.array([averages[m]-averages[j] for m,j in rows]))
            # Summable allocation across predictable policy slots, sample counts,
            # return and every ordered likelihood row; valid under adaptive counts.
            slot=self.slots[policy.identity]+1;statistics=self.model.num_models*(self.model.num_models-1)+1
            alpha=self.delta*36/(math.pi**4*slot**2*n**2*statistics)
            unit=math.sqrt(math.log(2/alpha)/(2*n))
            rj.append(min(1.,self.return_width*unit));ri.append(np.array([self.row_width[r]*unit for r in rows]))
        mean=np.asarray(means);information=np.array(info).T
        reward_radius=np.asarray(rj);row_radius=np.array(ri).T
        return dict(counts=np.array(counts),return_mean=mean,information_mean=information,
            return_radius=reward_radius,information_radius=row_radius,
            return_lower=np.maximum(0.,mean-reward_radius),information_lower=np.maximum(0.,information-row_radius))

    def priced_scores(self,policies,rows,beta):
        data=self.arrays(policies,rows);prices=np.array([beta[r] for r in rows])
        return data['return_mean']+prices@data['information_mean']
