#!/usr/bin/env python3
"""Evaluate local actor geometry at a saved failure; never update a learner."""
from pathlib import Path
import json,gzip,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ceac_walkthrough.actor_diagnosis import action_moments,conditional_gradient
from ceac_hardness.environment import JunctionConfig
from ceac_hardness.model import JunctionModel
from ceac_hardness.run import digest,atomic_json


def main():
    folder=ROOT/'audits/ceac_actor_failure_20261112'
    manifest=json.loads((folder/'manifest.json').read_text())
    if digest(folder/'diagnostics.json.gz')!=manifest['files']['diagnostics.json.gz']:
        raise ValueError('Actor-diagnosis hash mismatch')
    with gzip.open(folder/'diagnostics.json.gz','rt') as f:data=json.load(f)
    run=next(r for r in data['cells'] if r['task']==0 and r['dose']==0 and r['seed']==0)
    state=run['trace'][-1]['states'][0];p=np.array(state['probabilities']);q=np.array(state['q_augmented'])
    beta=np.array(run['final_replay']['beta'])
    model=JunctionModel(JunctionConfig(protocol='prior_shift'))
    moments=action_moments(model,0,beta,1)
    baseline=state['reward_value']+state['evidence_value']
    fisher=np.diag(p)-np.outer(p,p);ordinary=fisher@q
    natural=np.linalg.pinv(fisher,rcond=1e-12)@ordinary
    np.testing.assert_allclose(fisher@natural,ordinary,atol=1e-12)
    np.testing.assert_allclose(natural-natural.mean(),q-q.mean(),atol=1e-10)
    expected=conditional_gradient(p,moments,baseline,rate=1.)
    # Inverse-Fisher sampled contrast: delta * (1[a=probe]/p_probe - 1[a=risky]/p_risky).
    values=[];weights=[]
    for a in range(3):
        laws=model.laws[1,a]
        for outcome,prob in laws[0].items():
            ll=np.log([law[outcome] for law in laws])
            evidence=float(np.sum(beta*(ll[:,None]-ll[None,:])))
            td=outcome[1]+evidence-baseline
            values.append(td*((a==2)/p[2]-(a==1)/p[1]));weights.append(p[a]*prob)
    values=np.array(values);weights=np.array(weights)
    mean=float(weights@values);var=float(weights@((values-mean)**2))
    np.testing.assert_allclose(mean,natural[2]-natural[1],atol=1e-10)
    specification=json.loads((ROOT/'experiments/ceac_actor_stability_20261112.json').read_text())
    schedule=specification['schedule'];a=schedule['actor_exponent'];b=schedule['total_floor_exponent'];c=schedule['critic_exponent']
    assert .5<c<a<=1 and 2*a-b>1 and 0<b<=1
    result=dict(schema=1,training_launched=False,algorithm_modified=False,
        diagnosis_sha256=digest(folder/'diagnostics.json.gz'),analysis_source_sha256=digest(__file__),
        protocol_sha256=digest(ROOT/'experiments/ceac_actor_stability_20261112.json'),
        snapshot=dict(task=0,dose=0,seed=0,state=1,probabilities=p.tolist(),augmented_q=q.tolist()),
        ordinary_unit_step=dict(expected_probe_minus_risky_logit_increment=expected['contrast_mean'],
            sampled_increment_sd=float(np.sqrt(expected['contrast_variance']))),
        natural_unit_step=dict(expected_probe_minus_risky_logit_increment=mean,
            sampled_increment_sd=float(np.sqrt(var))),
        mean_direction_ratio=float(mean/expected['contrast_mean']),
        schedule_check=dict(actor_exponent=a,critic_exponent=c,floor_exponent=b,noise_summability_exponent=2*a-b,
            scope='Bounded conditional TD second moments at a repeatedly visited fixed state; does not prove the full adaptive CEAC theorem.'),
        interpretation='Local expectations and variances at the same saved policy, per unit optimizer step. Not a new trajectory, optimizer selection, or performance result.')
    out=ROOT/'audits/ceac_actor_geometry_20261112.json';atomic_json(out,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
