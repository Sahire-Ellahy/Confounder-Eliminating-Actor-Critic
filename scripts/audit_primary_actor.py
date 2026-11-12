"""Reconstruct the registered task-0 anomaly from saved episodes; no new training."""
from pathlib import Path
import sys,json,gzip,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ceac_pseudocode.agent import CEAC
from ceac_pseudocode.run import seed_for,atomic_json,digest
from ceac_hardness.environment import JunctionConfig
from ceac_hardness.model import JunctionModel
from ceac.contracts import Episode,Transition
rows=[]
for seed in (0,1):
 p=ROOT/f'runs/ceac_primary_prior_20261111/cells/false_beacon_junction/dose_0/task_0/ceac_s{seed}'
 m=JunctionModel(JunctionConfig(protocol='prior_shift'));a=CEAC(m,np.random.default_rng(seed_for(seed,'false_beacon_junction',0,'ceac','learner')))
 maximum_step=0.;maximum_critic=0.
 with gzip.open(p/'episodes.jsonl.gz','rt') as f:
  for line in f:
   raw=json.loads(line);i=raw['episode'];policy,behavior,prices=a.select(i)
   if policy.identity!=raw['policy_id'] or behavior!=raw['behavior']:raise ValueError(f'Trajectory diverged at {i}')
   e=Episode(i,0,policy.identity,behavior,tuple(Transition(x['state'],x['action'],x['reward'],x['next_state'],x['done'],tuple(x['log_likelihood'])) for x in raw['transitions']))
   a.observe(e,policy,prices)
   if a.actor.last_update:maximum_step=max(maximum_step,a.actor.last_update['actor_step_norm'])
   maximum_critic=max(maximum_critic,float(np.max(np.abs(a.actor.reward_value))),float(np.max(np.abs(a.actor.evidence_value(prices)))))
 final=a.actor.snapshot();j,k=m.evaluate(final)
 shaped=float(j[0]+np.sum(k*a.prices));best=max(float(jj[0]+np.sum(kk*a.prices)) for jj,kk in a.full_values)
 row=dict(seed=seed,raw_sha256=digest(p/'episodes.jsonl.gz'),matched_episodes=a.episodes,
    reference=[int(x) for x in a.replay.plausible],true_model=0,actor_probabilities=final.probabilities.tolist(),
    max_actor_step_norm=maximum_step,max_abs_critic_value=maximum_critic,final_actor_gap=float(m.optimal_returns[0]-j[0]),
    final_augmented_return=shaped,best_augmented_return=best,augmented_optimization_gap=best-shaped,
    actor_updates=a.actor.updates,actor_admissions=a.admissions,exact_admissions=a.exact_admissions,
    interpretation='Observed trajectories reproduce exactly; correct likelihood reference; finite-budget actor optimization residual persists.' if seed==0 else 'Matched easy-case successful seed under the same fixed settings.')
 rows.append(row);print(json.dumps(row),flush=True)
atomic_json(ROOT/'audits/ceac_primary_actor_replay_20261111.json',dict(schema=1,training_launched=False,algorithm_modified=False,cells=rows))
