"""Matched-seed runs with frozen CEAC replay and separately versioned optimizers."""
from pathlib import Path
from dataclasses import asdict
from collections import deque
import gzip,json,time,hashlib
import numpy as np
from ceac.contracts import Episode,Transition
from ceac_hardness.environment import JunctionConfig,JunctionEnv
from ceac_hardness.model import JunctionModel
from ceac_pseudocode.run import ROOT,atomic_json,digest,encoded,seed_for,source_identity as original_sources
from .agent import StabilityCEAC


def source_identity():
    sources=original_sources()
    for path in sorted((ROOT/'ceac_stability').glob('*.py')):
        if path.name not in ('report.py','experiment.py'):sources[str(path.relative_to(ROOT))]=digest(path)
    return sources


class Evaluator:
    """Truth is used solely for metrics, never passed back into the learner."""
    def __init__(self,model,truth):
        self.model=model;self.truth=truth;self.last_beta=None
        self.expected_log=np.zeros((model.num_models,model.n,model.actions))
        for (s,a),laws in model.laws.items():
            for outcome,prob in laws[truth].items():
                self.expected_log[:,s,a]+=prob*np.log([law[outcome] for law in laws])

    def prices(self,beta):
        if self.last_beta is not None and np.array_equal(beta,self.last_beta):return
        self.last_beta=beta.copy()
        coefficients=beta.sum(axis=1)-beta.sum(axis=0)
        self.stage=self.model.reward[self.truth]+np.einsum('i,isa->sa',coefficients,self.expected_log)
        self.optimum,self.q=self.response(np.zeros(self.model.n))

    def response(self,floors):
        m=self.model;v=np.zeros(m.n+1);q=np.zeros(m.mask.shape)
        for states in reversed(m.levels):
            q[states]=self.stage[states]+m._continuation(v,self.truth)[states]
            for s in states:
                row=q[s,m.mask[s]];epsilon=floors[s]
                v[s]=(1-epsilon)*row.max()+epsilon*row.mean()
        return float(v[m.initial]),q

    def policy_value(self,policy):
        m=self.model;v=np.zeros(m.n+1)
        for states in reversed(m.levels):
            q=self.stage[states]+m._continuation(v,self.truth)[states]
            v[states]=np.sum(q*policy.probabilities[states],axis=1)
        return float(v[m.initial])

    def actor_metrics(self,actor,beta):
        self.prices(beta);p=actor.snapshot();value=self.policy_value(p)
        constrained,_=self.response(actor.floors())
        gap=self.optimum-value;cgap=constrained-value
        if min(gap,cgap)<-1e-9:raise AssertionError('Actor exceeds evaluated feasible optimum')
        return dict(final_actor_augmented_gap=max(0.,gap),final_actor_constrained_augmented_gap=max(0.,cgap),
            final_actor_augmented_value=value,augmented_optimum=self.optimum,constrained_augmented_optimum=constrained,
            final_actor_reward_gap=max(0.,float(self.model.optimal_returns[self.truth]-self.model.returns(p)[self.truth])),
            actor_probabilities=p.probabilities.tolist())


def verify_cell(folder,expected_manifest=None):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if expected_manifest is not None and manifest!=json.loads(encoded(expected_manifest)):raise ValueError('Cell identity/settings changed')
    state=json.loads((folder/'status.json').read_text())
    if state['status']!='complete':raise ValueError(f'Incomplete existing cell: {folder}')
    for name,expected in state['artifacts'].items():
        if Path(name).name!=name or digest(folder/name)!=expected:raise ValueError(f'Cell artifact mismatch: {folder/name}')
    summary=json.loads((folder/'summary.json').read_text())
    if manifest['identity']!=summary['identity']:raise ValueError('Manifest/summary identity mismatch')
    return summary


def run_cell(config,candidate,schedule,seed,task,dose,episodes,output,log_every=128):
    config=JunctionConfig(**config) if isinstance(config,dict) else config
    q=(.9-.8*dose if task else .1+.8*dose) if config.protocol=='prior_shift' else .5
    model=JunctionModel(config);truth=model.true_index(task,q);output=Path(output)
    settings=dict(environment=asdict(config),candidate=candidate,schedule=schedule,seed=seed,
        task_evaluator_only=task,dose=dose,episodes=episodes,physical_budget=2*episodes,log_every=log_every,
        seed_coupling='All optimizer variants use original CEAC learner/action seed streams; environment stream is shared.')
    manifest=dict(schema=1,settings=settings,sources=source_identity())
    manifest['identity']=hashlib.sha256(encoded(manifest).encode()).hexdigest()
    if output.exists():return verify_cell(output,manifest)
    output.mkdir(parents=True);atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'status.json',dict(status='running',episodes=0))
    rng=np.random.default_rng(seed_for(seed,config.kind,task,'ceac','learner'))
    actions=np.random.default_rng(seed_for(seed,config.kind,task,'ceac','action'))
    started=time.monotonic();learner=StabilityCEAC(model,rng,candidate,schedule)
    evaluator=Evaluator(model,truth)
    env=JunctionEnv(config,task,q);env.reset(seed=seed_for(seed,config.kind,task,'shared_environment'))
    cumulative=0.;window=deque(maxlen=256);events=dict(nuisance=0,probe=0,risky=0,cue=0)
    last_policy=None;expected=None;best_visits=0;actor_decisions=0
    try:
        with gzip.open(output/'episodes.jsonl.gz','wt') as trajectories,gzip.open(output/'trace.jsonl.gz','wt') as traces:
            for t in range(episodes):
                policy,behavior,beta=learner.select(t)
                if policy.identity!=last_policy:
                    expected=float(model.returns(policy)[truth]);last_policy=policy.identity
                gap=max(0.,float(model.optimal_returns[truth]-expected));evaluator.prices(beta)
                observation,_=env.reset();steps=[]
                for _ in range(2):
                    s=model.index[observation];action=int(actions.choice(model.actions,p=policy.probabilities[s]))
                    nxt,reward,done,_,info=env.step(action)
                    ll=model.log_likelihood(s,action,nxt,reward,done)
                    steps.append(Transition(s,action,reward,-1 if done else model.index[nxt],done,ll))
                    if info['event'] in events:events[info['event']]+=1
                    if behavior=='actor' and model.mask[s].sum()>1:
                        actor_decisions+=1
                        best_visits+=int(evaluator.q[s,action]>=evaluator.q[s,model.mask[s]].max()-1e-10)
                    observation=nxt
                episode=Episode(t,0,policy.identity,behavior,tuple(steps))
                learner.observe(episode,policy,beta)
                trajectories.write(json.dumps(dict(episode=t,policy_id=policy.identity,behavior=behavior,
                    transitions=[dict(state=x.state,action=x.action,reward=x.reward,next_state=x.next_state,done=x.done,
                        log_likelihood=list(x.log_likelihood)) for x in steps]),allow_nan=False)+'\n')
                cumulative+=gap;window.append(gap)
                if t==0 or (t+1)%log_every==0 or t+1==episodes:
                    metrics=evaluator.actor_metrics(learner.actor,learner.prices)
                    row=dict(episode=t+1,interactions=2*(t+1),window_gap=float(np.mean(window)),
                        cumulative_regret=cumulative,actor_updates=learner.actor.updates,
                        priced_actor_updates=learner.actor.priced_updates,events=dict(events),
                        best_priced_action_visits=best_visits,actor_decisions=actor_decisions,
                        price_sum=float(learner.prices.sum()),certified=learner.certified,
                        optimizer=learner.actor.diagnostics(),**metrics)
                    traces.write(json.dumps(row,allow_nan=False)+'\n')
                if (t+1)%8192==0:atomic_json(output/'status.json',dict(status='running',episodes=t+1))
        summary=dict(identity=manifest['identity'],settings=settings,late_gap=float(np.mean(window)),
            cumulative_regret=cumulative,actor_updates=learner.actor.updates,priced_actor_updates=learner.actor.priced_updates,
            actor_decisions=actor_decisions,best_priced_action_visits=best_visits,events=events,
            seconds=time.monotonic()-started,ceac_diagnostics=learner.diagnostics(),
            **evaluator.actor_metrics(learner.actor,learner.prices))
        atomic_json(output/'summary.json',summary);atomic_json(output/'design_history.json',learner.events)
        names=('manifest.json','summary.json','design_history.json','episodes.jsonl.gz','trace.jsonl.gz')
        atomic_json(output/'status.json',dict(status='complete',artifacts={name:digest(output/name) for name in names}))
        return summary
    except Exception as exc:
        atomic_json(output/'status.json',dict(status='failed',error=repr(exc)));raise
