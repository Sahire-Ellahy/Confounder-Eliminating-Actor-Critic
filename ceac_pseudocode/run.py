"""Reproducible junction calibration and sweeps with fixed public model classes."""
from dataclasses import asdict,replace
from pathlib import Path
from collections import deque
import hashlib,json,time,gzip,math
import numpy as np
from ceac.contracts import Episode,Transition
from ceac.primitives import PrimitiveSpec,IDAC,IDACConfig,UCBQ,UCBQConfig,OptimisticPOMD,POMDConfig
from ceac_hardness.environment import JunctionConfig,JunctionEnv
from ceac_hardness.model import JunctionModel
from .agent import CEAC,Config


ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encoded(value):return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'

def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(encoded(value));temporary.replace(path)

def source_identity():
    paths=list((ROOT/'ceac_hardness').glob('*.py'))+list((ROOT/'ceac_pseudocode').glob('*.py'))
    paths += [ROOT/'ceac'/name for name in ('model.py','contracts.py','primitives.py')]
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths) if p.name not in {'report.py','experiment.py'}}

def seed_for(*parts):return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8],'little')

def primitive_spec(model):
    elapsed=np.zeros(model.n,dtype=int)
    for t,states in enumerate(model.levels):elapsed[states]=t
    return PrimitiveSpec(model.mask,elapsed,2,model.initial,0.,1.)

def construct(model,method,rng,episodes,ceac_config=None,primitive_configs=None):
    if method!='ceac':raise ValueError('This runner executes the requested CEAC loop only')
    return CEAC(model,rng,ceac_config or Config())

def run_cell(config,method,seed,task,dose,episodes,output,log_every=128,primitive_configs=None):
    config=JunctionConfig(**config) if isinstance(config,dict) else config
    q=(.9-.8*dose if task==1 else .1+.8*dose) if config.protocol=='prior_shift' else .5
    model=JunctionModel(config);truth=model.true_index(task,q)
    settings=dict(environment=asdict(config),method=method,seed=seed,task_evaluator_only=task,
        dose=dose,episodes=episodes,physical_budget=2*episodes,log_every=log_every,
        hypotheses=model.hypotheses,primitive_configs=primitive_configs or {},
        ceac_config=asdict(Config()),
        numerical_ids_expected_regret_tolerance=1e-12)
    manifest=dict(schema=1,settings=settings,sources=source_identity())
    manifest['identity']=hashlib.sha256(encoded(manifest).encode()).hexdigest()
    output=Path(output)
    if output.exists():
        old=json.loads((output/'manifest.json').read_text())
        if old!=json.loads(encoded(manifest)):raise ValueError('Existing cell has a different identity')
        status=json.loads((output/'status.json').read_text())
        if status['status']!='complete':raise ValueError('Incomplete cell: use a fresh directory')
        for name,checksum in status['artifacts'].items():
            if digest(output/name)!=checksum:raise ValueError('Cell artifact mismatch')
        return json.loads((output/'summary.json').read_text())
    output.mkdir(parents=True);atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'status.json',dict(status='running'))
    rng=np.random.default_rng(seed_for(seed,config.kind,task,method,'learner'))
    action_rng=np.random.default_rng(seed_for(seed,config.kind,task,method,'action'))
    cfg=Config(**settings['ceac_config'])
    started=time.monotonic()
    learner=construct(model,method,rng,episodes,cfg,primitive_configs)
    setup=time.monotonic()-started
    env=JunctionEnv(config,task,q);env.reset(seed=seed_for(seed,config.kind,task,'shared_environment'))
    cumulative=0.;window=deque(maxlen=256);events=dict(nuisance=0,probe=0,risky=0,cue=0)
    saturation=0;actor_updates=0;priced_updates=0;last_policy=None;expected=None
    product_support=bool(model.prior(method.removeprefix('ids_'))[truth]>0) if method.startswith('ids_') else True
    try:
        with gzip.open(output/'trace.jsonl.gz','wt') as target, gzip.open(output/'episodes.jsonl.gz','wt') as trajectory:
            for t in range(episodes):
                policy,behavior,prices=learner.select(t)
                # Frozen IDAC policies and repeated deployment policies reuse exact evaluation.
                if policy.identity!=last_policy:
                    expected=float(model.returns(policy)[truth]);last_policy=policy.identity
                gap=max(0.,float(model.optimal_returns[truth]-expected))
                observation,_=env.reset();transitions=[]
                for step in range(2):
                    s=model.index[observation]
                    action=int(action_rng.choice(model.actions,p=policy.probabilities[s]))
                    nxt,reward,done,_,info=env.step(action)
                    ll=model.log_likelihood(s,action,nxt,reward,done)
                    transitions.append(Transition(s,action,reward,-1 if done else model.index[nxt],done,ll))
                    if info['event'] in events:events[info['event']]+=1
                    observation=nxt
                episode=Episode(t,0,policy.identity,behavior,tuple(transitions))
                learner.observe(episode,policy,prices)
                trajectory.write(json.dumps(dict(episode=t,policy_id=policy.identity,behavior=behavior,
                    transitions=[dict(state=x.state,action=x.action,reward=x.reward,next_state=x.next_state,
                        done=x.done,log_likelihood=list(x.log_likelihood)) for x in transitions]),allow_nan=False)+'\n')
                cumulative+=gap;window.append(gap)
                if method=='pomd':saturation+=int(np.all(learner.q[learner.mask]==0))
                if behavior=='actor':
                    actor_updates+=1;priced_updates+=int(np.any(prices>0))
                if t==0 or (t+1)%log_every==0 or t+1==episodes:
                    row=dict(episode=t+1,interactions=2*(t+1),gap=gap,window_gap=float(np.mean(window)),
                             cumulative_regret=cumulative,events=dict(events),actor_updates=actor_updates,
                             priced_actor_updates=priced_updates,pomd_zero_fraction=saturation/(t+1))
                    if method.startswith('ids_'):
                        posterior=learner.posterior
                        row.update(task1_posterior=float(sum(p for p,(theta,_) in zip(posterior,model.hypotheses) if theta==1)),
                            true_model_log_posterior=(float(learner.log_weights[truth]) if product_support else None),
                            prior_contains_true_joint_model=product_support,
                            posterior_underflow_entries=learner.posterior_underflow_entries)
                    target.write(json.dumps(row,allow_nan=False)+'\n')
        summary=dict(settings=settings,identity=manifest['identity'],cumulative_regret=cumulative,
            late_gap=float(np.mean(window)),events=events,actor_updates=actor_updates,
            priced_actor_updates=priced_updates,pomd_zero_fraction=saturation/episodes,
            seconds=time.monotonic()-started,setup_seconds=setup,
            prior_contains_true_joint_model=product_support,
            public_class_contains_true_model=True,reachable_states=model.n,actions=model.actions,
            optimum=float(model.optimal_returns[truth]))
        if method.startswith('ids_'):
            summary['task1_posterior']=float(sum(p for p,(theta,_) in zip(learner.posterior,model.hypotheses) if theta==1))
            summary['fixed_prior']=model.prior(method.removeprefix('ids_')).tolist()
        if hasattr(learner,'actor'):summary['actor_admissions']=learner.admissions
        summary['ceac_diagnostics']=learner.diagnostics()
        atomic_json(output/'design_history.json',learner.events)
        atomic_json(output/'summary.json',summary)
        atomic_json(output/'status.json',dict(status='complete',artifacts={name:digest(output/name) for name in ('manifest.json','summary.json','trace.jsonl.gz','episodes.jsonl.gz','design_history.json')}))
        return summary
    except Exception as exc:
        atomic_json(output/'status.json',dict(status='failed',error=repr(exc)))
        raise
