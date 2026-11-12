"""Conditional-gradient audit of immutable CEAC trajectories; no new rollouts.

For the prior junction the only choice is terminal. A state-value critic is
therefore an action-independent baseline: it changes variance, not the mean
policy gradient. Exact conditional moments let us distinguish target mismatch,
noise and softmax saturation without changing the algorithm.
"""
from pathlib import Path
from collections import Counter
import gzip
import json
import numpy as np
from ceac.contracts import Episode, Transition
from ceac_hardness.environment import JunctionConfig
from ceac_hardness.model import JunctionModel
from ceac_hardness.run import ROOT, digest
from ceac_pseudocode.learning import TwoCriticActor


def action_moments(model, truth, beta, state):
    """Exact immediate r+e moments under the realized law (diagnosis only)."""
    values=[]
    for action in range(model.actions):
        laws=model.laws[state,action]
        outcomes=[]
        for key, probability in laws[truth].items():
            ll=np.log([law[key] for law in laws])
            evidence=float(np.sum(beta*(ll[:,None]-ll[None,:])))
            outcomes.append((probability,key[1],evidence))
        outcomes=np.array(outcomes)
        p,r,e=outcomes.T
        q=float(p@(r+e))
        values.append(dict(reward=float(p@r),evidence=float(p@e),augmented=q,
            variance=float(p@((r+e-q)**2)),nonzero_evidence=float(p@(np.abs(e)>1e-12))))
    return values


def conditional_gradient(probabilities, moments, baseline, rate=.4, temperature=1.):
    """Probe-minus-risky logit-step moments and full expected logit step."""
    p=np.asarray(probabilities);q=np.array([m['augmented'] for m in moments])
    variance=np.array([m['variance'] for m in moments])
    scale=rate/temperature
    gradient=scale*p*(q-p@q)
    contrast=np.array([0.,-1.,1.])
    scores=contrast-p@contrast
    mean=float(contrast@gradient)
    second=float(scale**2*np.sum(p*((q-baseline)**2+variance)*scores**2))
    denom=float(p@(scores**2))
    best_baseline=float((p*(scores**2))@q/denom) if denom else float(p@q)
    return dict(vector=gradient,contrast_mean=mean,contrast_variance=max(0.,second-mean**2),
        variance_minimizing_baseline=best_baseline,
        baseline_excess_variance=scale**2*denom*(baseline-best_baseline)**2)


def diagnose(folder):
    folder=Path(folder)
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='complete':raise ValueError('Incomplete cell')
    for name,expected in status['artifacts'].items():
        if digest(folder/name)!=expected:raise ValueError('Raw artifact mismatch')
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,expected in manifest['sources'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Numerical source mismatch')
    summary=json.loads((folder/'summary.json').read_text());settings=summary['settings']
    config=JunctionConfig(**settings['environment'])
    if config.protocol!='prior_shift':raise ValueError('This audit requires the terminal-decision cue junction')
    model=JunctionModel(config);cfg=settings['ceac_config']
    task=settings['task_evaluator_only'];dose=settings['dose'];q=.9-.8*dose if task else .1+.8*dose
    truth=model.true_index(task,q);weights=np.array([1-q,q])
    actor=TwoCriticActor(model,cfg['actor_rate'],cfg['critic_rate'],cfg['temperature'])
    designs=[x for x in json.loads((folder/'design_history.json').read_text()) if 'design' in x]
    for e in designs:
        if e['design']['references']!=[truth]:raise ValueError('A design used an incorrect reference')
    cumulative={name:np.zeros((3,3)) for name in ('reward','evidence','expected')}
    variance=np.zeros(3);visits=np.zeros((3,3),dtype=int)
    all_beta=[];gaps=[];trace=[];crossings=[];cursor=-1;moments=None;beta=None
    bands={name:dict(count=0,nonzero=0,abs_evidence=0.,reward_error_sq=0.,evidence_error_sq=0.,
        expected_step=0.,step_variance=0.,baseline_excess_variance=0.,visits=np.zeros((3,3),dtype=int))
        for name in ('first_512_actor','last_4096_episodes','all_actor')}
    def capture(count):
        p=actor.snapshot().probabilities
        states=[]
        for s in (1,2):
            m=moments[s];qvalues=np.array([x['augmented'] for x in m])
            cond=conditional_gradient(p[s],m,actor.reward_value[s]+actor.evidence_value(beta)[s],cfg['actor_rate'],cfg['temperature'])
            states.append(dict(state=s,occupancy=float(weights[s-1]),probabilities=p[s].tolist(),
                q_augmented=qvalues.tolist(),q_reward=[x['reward'] for x in m],q_evidence=[x['evidence'] for x in m],
                augmented_gap=float(qvalues.max()-p[s]@qvalues),
                reward_value=float(actor.reward_value[s]),evidence_value=float(actor.evidence_value(beta)[s]),
                expected_step=cond['vector'].tolist(),expected_contrast_step=cond['contrast_mean'],
                conditional_step_sd=float(np.sqrt(cond['contrast_variance'])),visits=visits[s].tolist(),
                reward_contrast=float(cumulative['reward'][s,2]-cumulative['reward'][s,1]),
                evidence_contrast=float(cumulative['evidence'][s,2]-cumulative['evidence'][s,1]),
                actual_contrast=float(actor.logits[s,2]-actor.logits[s,1]),
                expected_contrast=float(cumulative['expected'][s,2]-cumulative['expected'][s,1]),
                martingale_variance=float(variance[s])))
        trace.append(dict(episode=count,actor_updates=actor.updates,price_sum=float(beta.sum()),states=states))
    with gzip.open(folder/'episodes.jsonl.gz','rt') as handle:
        for count,line in enumerate(handle,1):
            raw=json.loads(line);i=raw['episode']
            if i!=count-1:raise ValueError('Episode sequence mismatch')
            while cursor+1<len(designs) and designs[cursor+1]['episode']<=i:
                cursor+=1;beta=np.array(designs[cursor]['design']['beta'])
                moments={s:action_moments(model,truth,beta,s) for s in (1,2)}
                all_beta.append(float(beta.sum()))
                gaps.append(moments[1][2]['augmented']-moments[1][1]['augmented'])
            if raw['behavior']=='actor':
                if beta is None:raise ValueError('Actor precedes design')
                p=actor.snapshot()
                if p.identity!=raw['policy_id']:raise ValueError(f'Policy reconstruction differs at {i}')
                episode=Episode(i,0,p.identity,'actor',tuple(Transition(
                    t['state'],t['action'],t['reward'],t['next_state'],t['done'],tuple(t['log_likelihood'])) for t in raw['transitions']))
                choice=episode.transitions[-1];s=choice.state;a=choice.action
                if not choice.done:raise ValueError('Nonterminal actor decision')
                if actor.updates==0:capture(i)
                vi=actor.evidence_value(beta);vr=actor.reward_value[s]
                conditional=conditional_gradient(p.probabilities[s],moments[s],vr+vi[s],cfg['actor_rate'],cfg['temperature'])
                cumulative['expected'][s]+=conditional['vector'];variance[s]+=conditional['contrast_variance']
                ll=np.array(choice.log_likelihood);ev=float(np.sum(beta*(ll[:,None]-ll[None,:])))
                vr_error=vr-float(p.probabilities[s]@np.array([x['reward'] for x in moments[s]]))
                vi_error=vi[s]-float(p.probabilities[s]@np.array([x['evidence'] for x in moments[s]]))
                for name,b in bands.items():
                    if name=='first_512_actor' and actor.updates>=512:continue
                    if name=='last_4096_episodes' and count<=settings['episodes']-4096:continue
                    b['count']+=1;b['nonzero']+=int(abs(ev)>1e-12);b['abs_evidence']+=abs(ev)
                    b['reward_error_sq']+=vr_error**2;b['evidence_error_sq']+=vi_error**2
                    b['expected_step']+=conditional['contrast_mean'];b['step_variance']+=conditional['contrast_variance']
                    b['baseline_excess_variance']+=conditional['baseline_excess_variance'];b['visits'][s,a]+=1
                old=actor.logits.copy();actor.update(episode,p,beta)
                for step,(_,dr,di,_) in zip(episode.transitions,actor.last_update['errors']):
                    score=-p.probabilities[step.state].copy();score[step.action]+=1
                    score*=cfg['actor_rate']/cfg['temperature']
                    cumulative['reward'][step.state]+=dr*score;cumulative['evidence'][step.state]+=di*score
                visits[s,a]+=1
                np.testing.assert_allclose(cumulative['reward']+cumulative['evidence'],actor.logits,atol=1e-10)
                updated=actor.snapshot().probabilities
                for state in (1,2):
                    if updated[state,1]>.99 and not any(x['state']==state for x in crossings):
                        crossings.append(dict(state=state,episode=count,actor_updates=actor.updates,
                            probabilities=updated[state].tolist(),probe_advantage=moments[state][2]['augmented']-moments[state][1]['augmented'],
                            actual_contrast=float(actor.logits[state,2]-actor.logits[state,1]),
                            expected_contrast=float(cumulative['expected'][state,2]-cumulative['expected'][state,1]),
                            noise_sd=float(np.sqrt(variance[state]))))
                if actor.updates%64==0 or count==settings['episodes']:capture(count)
    if actor.updates!=summary['actor_updates'] or count!=settings['episodes']:raise ValueError('Wrong audited budget')
    if trace[-1]['episode']!=count:capture(count)
    for b in bands.values():
        n=b['count'];b['nonzero_fraction']=b['nonzero']/n
        b['mean_abs_evidence']=b.pop('abs_evidence')/n
        b['reward_critic_rmse']=float(np.sqrt(b.pop('reward_error_sq')/n))
        b['evidence_critic_rmse']=float(np.sqrt(b.pop('evidence_error_sq')/n))
        b['visits']=b['visits'].tolist()
    last=designs[-1]['design'];arrays=last['arrays']
    from ceac_hardness.design import PolicyBank,bounded_design
    exact=bounded_design(model,PolicyBank(model,model.seed_policies()),(truth,),cfg['mass_bound'])
    final=trace[-1]
    aug_gap=float(weights@np.array([s['augmented_gap'] for s in final['states']]))
    final_p=actor.snapshot();final_return=float(model.returns(final_p)[truth])
    exact_price=float(exact.beta.sum())
    return dict(schema=1,identity=summary['identity'],folder=str(folder.relative_to(ROOT)),
        input_hashes=status['artifacts'],seed=settings['seed'],task=task,dose=dose,cue_probability=q,
        checked_updates=actor.updates,price_range=[min(all_beta),max(all_beta)],
        probe_minus_risky_augmented_range=[min(gaps),max(gaps)],
        final_actor_return=final_return,final_actor_reward_gap=float(model.optimal_returns[truth]-final_return),
        reported_late_behavior_gap=summary['late_gap'],final_augmented_gap=aug_gap,
        crossings=crossings,bands=bands,trace=trace,
        final_replay=dict(beta=last['beta'],rows=last['rows'],arrays=arrays,counts=arrays['counts'],
            policy_ids=last['policy_ids'],exact_kernel_price_sum=exact_price,
            replay_price_sum=float(np.sum(last['beta'])),
            exact_price_interpretation='Evaluator-only known-kernel Graves-Lai solve; not a deployed policy or a retrained learner.'))
