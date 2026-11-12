"""Tabular CEAC episode selector and observed-replay design loop (Section 3)."""
from dataclasses import dataclass
from itertools import product
import math
import numpy as np
from ceac.contracts import Policy
from ceac_hardness.design import PolicyBank,bounded_design,target_calibration
from ceac_hardness.learning import DebtLedger,evidence_scale
from .learning import TwoCriticActor,PolicyReplay
from .design import solve_replay_master

@dataclass(frozen=True)
class Config:
    actor_rate:float=.4
    critic_rate:float=.12
    temperature:float=1.
    delta:float=.05
    refresh_interval:int=128
    snapshot_interval:int=8192
    replay_floor_constant:float=64.
    mass_bound:float=32.
    normalization:float=32.
    column_tolerance:float=1e-9

class CEAC:
    def __init__(self,model,rng,config=None):
        self.model=model;self.rng=rng;self.config=config or Config();cfg=self.config
        self.actor=TwoCriticActor(model,cfg.actor_rate,cfg.critic_rate,cfg.temperature)
        self.replay=PolicyReplay(model,cfg.delta/2);self.ledger=DebtLedger()
        self.bank=PolicyBank(model,model.seed_policies());self.seed_ids=tuple(p.identity for p in self.bank.policies)
        for p in self.bank.policies:self.replay.register(p);self.ledger.require_seed(p.identity)
        coefficient_bank=PolicyBank(model,model.seed_policies())
        self.coefficients=np.array([bounded_design(model,coefficient_bank,(m,),cfg.mass_bound).cost for m in range(model.num_models)])
        count=int(np.prod(model.mask.sum(axis=1),dtype=object))
        if count>100000:raise ValueError('Exact junction policy enumeration too large; implement a certified structured oracle')
        self.full_policies=[]
        for choices in product(*[np.flatnonzero(mask) for mask in model.mask]):
            probabilities=np.zeros(model.mask.shape);probabilities[np.arange(model.n),choices]=1.
            self.full_policies.append(Policy(probabilities,'exact-pricing-candidate'))
        self.full_values=[model.evaluate(p) for p in self.full_policies]
        self.episodes=0;self.design=None;self.prices=np.zeros((model.num_models,model.num_models))
        self.last_refs=();self.certified=False;self.refreshes=0;self.admissions=0;self.exact_admissions=0
        self.events=[];self.snapshots=[];self.candidates={};self.frozen={};self.last_replay_scores=[];self.last_reason='initialization'
        self.refresh()

    def floor(self):
        t=max(self.episodes,2)
        return max(1,math.ceil(self.config.replay_floor_constant*math.sqrt(math.log(t))*math.log(math.log(max(t,math.exp(math.e))))))

    def request_count(self,pid,n):
        self.ledger.statistical_required[pid]=max(self.ledger.statistical_required[pid],int(n))

    def bootstrap(self,reason):
        self.certified=False;self.prices.fill(0.);self.last_reason=reason
        # Complete the weakest sampled seed certificate. The next refresh
        # re-evaluates feasibility; no modeled array replaces missing samples.
        pid=min(self.seed_ids,key=lambda x:self.ledger.counts[x])
        self.request_count(pid,self.ledger.counts[pid]+self.config.refresh_interval)

    def full_policy_check(self,design,reference):
        """Exact enumeration of conservative counterfactual policy columns.

        Bank columns use replay. Unexecuted columns use finite-model response
        values with the current uniform replay-error envelope, as permitted by
        Appendix E. No such policy replaces the actor's behavior branch.
        For stochastic policies the score is convex in occupancy (positive
        parts of linear KL terms); a maximizing deterministic policy suffices.
        """
        rows=design.rows;weights=np.array([design.beta[r] for r in rows])
        reward_error=2*float(design.data['return_radius'].max())
        evidence_error=2*np.max(design.data['information_radius'],axis=1)
        reduced=[]
        for returns,information in self.full_values:
            gap=max(0.,float(self.model.optimal_returns[reference]-returns[reference]))
            conservative=np.maximum(0.,np.array([information[r] for r in rows])-evidence_error)
            reduced.append(gap+reward_error+design.mass_price-float(weights@conservative))
        index=int(np.argmin(reduced))
        # For bank columns the replay LP already establishes nonnegative rc.
        # Uniform counterfactual values only bound the omitted-policy search.
        return max(0.,-float(reduced[index])),self.full_policies[index],dict(
            return_error=reward_error,information_error=evidence_error.tolist(),
            policy_count=len(self.full_policies),minimum_counterfactual_reduced_cost=float(reduced[index]))

    def refresh(self):
        cfg=self.config;refs=self.replay.plausible;self.last_refs=refs;self.refreshes+=1
        floor=self.floor()
        if self.design is not None:
            for pid,(candidate,target) in list(self.candidates.items()):
                if self.ledger.counts[pid]<target:continue
                data=self.replay.arrays([candidate],self.design.rows)
                weights=np.array([self.design.beta[r] for r in self.design.rows])
                upper=max(self.model.optimal_returns[list(self.design.references)])
                reduced=float(upper-data['return_lower'][0]+self.design.mass_price-weights@data['information_lower'][:,0])
                admitted=reduced < -cfg.column_tolerance
                if admitted and self.bank.add(candidate):self.admissions+=1
                self.events.append(dict(episode=self.episodes,snapshot=pid,upper_reduced_cost=reduced,
                    certification_count=int(data['counts'][0]),admitted=admitted))
                del self.candidates[pid]
        for p in self.bank.policies:self.replay.register(p);self.request_count(p.identity,floor)
        # All likelihood and design rows remain public. Finishing the reference
        # certificate avoids treating E_true log(p_ref/p_alt) as KL_ref while
        # ref is unresolved; this is statistical debt, not oracle truth access.
        if len(refs)!=1:
            self.bootstrap('reference_certificate_pending');return
        if any(self.ledger.counts[p.identity]<floor for p in self.bank.policies):
            self.certified=False;self.prices.fill(0.);self.last_reason='replay_floor_pending';return
        reference=refs[0]
        aec=target_calibration(self.model,self.bank,self.replay.mixture,self.coefficients,n_max=cfg.normalization)
        for p in self.bank.policies:
            self.replay.register(p);self.request_count(p.identity,floor)
        if any(self.ledger.counts[p.identity]<floor for p in self.bank.policies):
            self.certified=False;self.prices.fill(0.);self.last_reason='new_response_column_pending';return
        upper=float(max(self.model.optimal_returns[list(refs)]))
        design=solve_replay_master(self.replay,self.bank.policies,refs,upper,cfg.mass_bound)
        if design is None:self.bootstrap('replay_master_infeasible');return
        residual,proposal,certificate=self.full_policy_check(design,reference);design.column_residual=residual
        if residual>cfg.column_tolerance:
            if self.bank.add(proposal):
                self.exact_admissions+=1;self.replay.register(proposal);self.request_count(proposal.identity,floor)
                self.certified=False;self.prices.fill(0.);self.last_reason='exact_separator_certificate_pending';return
            # A failed uniform bound at an already present column calls for
            # more data, never a fabricated zero residual or fallback policy.
            self.request_count(proposal.identity,self.ledger.counts[proposal.identity]+cfg.refresh_interval)
            self.certified=False;self.prices.fill(0.);self.last_reason='uniform_bound_pending';return
        self.design=design;self.prices=design.beta.copy();self.certified=True;self.last_reason='replay_design_certified'
        self.ledger.add_design(design,evidence_scale(max(1,self.episodes)))
        if not set(refs).issubset(aec['good']):
            selected=int(self.rng.choice(len(aec['mixture']),p=aec['mixture']));self.ledger.request_fresh(aec['policy_ids'][selected])
        scores=self.replay.priced_scores(self.bank.policies,design.rows,self.prices)
        self.last_replay_scores=scores.tolist()
        self.events.append(dict(episode=self.episodes,design=design.record(),priced_replay_scores=self.last_replay_scores,
            global_certificate=certificate,actor_updates=self.actor.updates,ledger=self.ledger.record()))

    def select(self,index):
        del index
        if (self.episodes and self.episodes%self.config.refresh_interval==0) or self.replay.plausible!=self.last_refs:self.refresh()
        pid,behavior=self.ledger.select()
        if pid is None and not self.certified:
            self.refresh();pid,behavior=self.ledger.select()
        if pid is not None:
            policy=self.bank.policies[self.bank._ids[pid]] if pid in self.bank._ids else self.frozen[pid]
            return policy,behavior,self.prices.copy()
        if not self.certified:raise RuntimeError('Unfinished CEAC certificate has no statistical request')
        policy=self.actor.snapshot()
        self.replay.register(policy) # Allocate its confidence budget before observing this policy’s data.
        return policy,'actor',self.prices.copy()

    def observe(self,episode,policy,prices):
        self.actor.update(episode,policy,prices)
        self.replay.append(episode);self.ledger.credit(episode);self.episodes+=1
        if self.episodes%self.config.snapshot_interval==0 and self.design is not None and not self.candidates:
            candidate=self.actor.snapshot();self.snapshots.append(candidate.identity)
            ref=self.design.references[0];j,k=self.model.evaluate(candidate)
            exact_rc=float(self.model.optimal_returns[ref]-j[ref]+self.design.mass_price-np.sum(self.design.beta*k))
            if exact_rc < -self.config.column_tolerance:
                self.replay.register(candidate);target=self.floor()
                self.frozen[candidate.identity]=candidate;self.candidates[candidate.identity]=(candidate,target)
                self.request_count(candidate.identity,target)
                self.events.append(dict(episode=self.episodes,snapshot=candidate.identity,
                    status='awaiting_independent_policy_labeled_replay',required=target))

    def diagnostics(self):
        return dict(algorithm='CEAC_Section3',actor_updates=self.actor.updates,priced_actor_updates=self.actor.priced_updates,
            replay_episodes=len(self.replay.seen),policy_strata=len(self.replay.strata),bank_policies=len(self.bank.policies),
            refreshes=self.refreshes,certified=self.certified,reason=self.last_reason,
            priced_replay_scores=self.last_replay_scores,design=self.design.record() if self.design is not None else None)
