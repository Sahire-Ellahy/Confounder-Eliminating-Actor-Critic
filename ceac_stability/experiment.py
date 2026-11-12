"""Frozen-grid calibration and prospective confirmation of complete CEAC."""
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import argparse,json,tarfile,platform
import numpy as np
from .run import ROOT,atomic_json,digest,encoded,source_identity,run_cell,verify_cell
from ceac_hardness.environment import JunctionConfig

PROTOCOL=ROOT/'experiments/ceac_actor_stability_20261112.json'


def job_result(job):
    s=run_cell(**job)
    c=job['config']
    return dict(folder=str(Path(job['output']).relative_to(ROOT)),candidate=job['candidate']['id'],
        protocol=c['protocol'],kind=c['kind'],burden=c['burden'],dose=job['dose'],task=job['task'],seed=job['seed'],
        late_gap=s['late_gap'],cumulative_regret=s['cumulative_regret'],
        final_actor_augmented_gap=s['final_actor_augmented_gap'],final_actor_constrained_augmented_gap=s['final_actor_constrained_augmented_gap'],
        final_actor_reward_gap=s['final_actor_reward_gap'],actor_updates=s['actor_updates'],priced_actor_updates=s['priced_actor_updates'],
        seconds=s['seconds'],identity=s['identity'])


def selection(cfg,results):
    groups=defaultdict(list)
    for row in results:groups[row['candidate']].append(row)
    gate=cfg['calibration']['gate'];rows=[]
    for candidate in cfg['candidates']:
        group=groups[candidate['id']]
        expected={(t,s) for t in cfg['calibration']['tasks'] for s in cfg['calibration']['seeds']}
        actual={(r['task'],r['seed']) for r in group}
        if len(group)!=len(expected) or actual!=expected:raise ValueError('Incomplete/duplicate calibration grid')
        worst=max(r['final_actor_augmented_gap'] for r in group)
        worst_external=max(r['late_gap'] for r in group)
        passed=(worst<=gate['maximum_final_actor_augmented_gap'] and worst_external<=gate['maximum_late_behavior_reward_gap']
            and all(r['priced_actor_updates']>0 and r['actor_updates']>0 for r in group))
        rows.append(dict(candidate=candidate['id'],geometry=candidate['geometry'],passed=passed,
            worst_augmented_gap=worst,worst_late_reward_gap=worst_external,
            mean_regret=float(np.mean([r['cumulative_regret'] for r in group])),cells=len(group)))
    complexity={'ordinary':0,'projected_ordinary':1,'natural_mirror':2}
    eligible=[r for r in rows if r['passed']]
    eligible.sort(key=lambda r:(r['worst_augmented_gap'],r['mean_regret'],complexity[r['geometry']],r['candidate']))
    return dict(status='selected' if eligible else 'no_candidate_passed',selected=eligible[0]['candidate'] if eligible else None,
        candidates=rows,rule=cfg['calibration']['ranking'],gate=gate)


def load_completed(folder):
    folder=Path(folder);state=json.loads((folder/'status.json').read_text())
    if state['status']!='complete':raise ValueError('Study not complete')
    for name in ('registration','results'):
        if digest(folder/f'{name}.json')!=state[f'{name}_sha256']:raise ValueError('Study hash mismatch')
    registration=json.loads((folder/'registration.json').read_text());results=json.loads((folder/'results.json').read_text())
    if len(results)!=registration['jobs']:raise ValueError('Incomplete study index')
    for row in results:
        s=verify_cell(ROOT/row['folder'])
        if s['identity']!=row['identity']:raise ValueError('Study identity mismatch')
        for key in ('late_gap','cumulative_regret','final_actor_augmented_gap','final_actor_reward_gap'):
            if s[key]!=row[key]:raise ValueError('Study result/cell mismatch')
    return registration,results


def execute_phase(phase,output,workers,calibration=None):
    cfg=json.loads(PROTOCOL.read_text());out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    selected=None;provenance=None
    if phase=='confirmation':
        if calibration is None:raise ValueError('Confirmation needs a completed calibration')
        reg,prior=load_completed(calibration)
        if reg['sources']!=source_identity() or reg['protocol_sha256']!=digest(PROTOCOL) or reg['driver_sha256']!=digest(__file__):raise ValueError('Calibration source/protocol changed')
        choice=selection(cfg,prior)
        stored=json.loads((Path(calibration)/'selection.json').read_text())
        if choice!=stored:raise ValueError('Frozen selection disagrees with recomputed selection')
        if not choice['selected']:raise ValueError('No calibrated candidate; confirmation is prohibited')
        selected=choice['selected'];provenance=dict(folder=str(calibration),selection_sha256=digest(Path(calibration)/'selection.json'),
            registration_sha256=digest(Path(calibration)/'registration.json'),results_sha256=digest(Path(calibration)/'results.json'))
    candidates=cfg['candidates'] if phase=='calibrate' else [c for c in cfg['candidates'] if c['id'] in {'original_reference',selected}]
    jobs=[]
    if phase=='calibrate':
        families=[('prior_shift',cfg['calibration']['episodes'],['false_beacon_junction'],[0],cfg['calibration']['tasks'])]
        seeds=cfg['calibration']['seeds']
    else:
        seeds=cfg['confirmation']['seeds'];families=[]
        for name in ('prior_shift','burden'):
            spec=cfg['confirmation'][name]
            families.append((name,spec['episodes'],spec.get('kinds',['false_beacon_junction']),spec['doses'],spec['tasks']))
    for protocol,budget,kinds,doses,tasks in families:
        for kind in kinds:
            for dose in doses:
                env=JunctionConfig(kind=kind,protocol=protocol,burden=int(dose) if protocol=='burden' else 0)
                for task in tasks:
                    for candidate in candidates:
                        for seed in seeds:
                            folder=out/'cells'/protocol/kind/f'dose_{dose:g}'/f'task_{task}'/f'{candidate["id"]}_s{seed}'
                            jobs.append(dict(config=asdict(env),candidate=candidate,schedule=cfg['schedule'],seed=seed,task=task,
                                dose=float(dose) if protocol=='prior_shift' else 0.,episodes=budget,output=str(folder),
                                log_every=128 if budget<=32768 else 512))
    registration=dict(schema=1,phase=phase,protocol=cfg,protocol_sha256=digest(PROTOCOL),sources=source_identity(),
        driver_sha256=digest(__file__),python=platform.python_version(),jobs=len(jobs),selection_provenance=provenance,
        source_contract='Full CEAC, optimizer varies; evaluator metrics never used in action selection.')
    path=out/'registration.json'
    if path.exists():
        if json.loads(path.read_text())!=json.loads(encoded(registration)):raise ValueError('Registration changed; use a new study directory')
    else:
        atomic_json(path,registration)
        with tarfile.open(out/'source_bundle.tar.gz','w:gz') as archive:
            for name in registration['sources']:archive.add(ROOT/name,arcname=name)
            for p in (Path(__file__),PROTOCOL):archive.add(p,arcname=str(p.relative_to(ROOT)))
    results=[];pending=[]
    for job in jobs:
        if (Path(job['output'])/'status.json').exists():results.append(job_result(job))
        else:pending.append(job)
    atomic_json(out/'status.json',dict(status='running',expected=len(jobs),completed=len(results)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(job_result,j):j for j in pending}
        for future in as_completed(futures):
            row=future.result();results.append(row)
            print(json.dumps(dict(completed=len(results),expected=len(jobs),**row)),flush=True)
            atomic_json(out/'status.json',dict(status='running',expected=len(jobs),completed=len(results)))
    results.sort(key=lambda r:(r['protocol'],r['kind'],r['burden'],r['dose'],r['task'],r['candidate'],r['seed']))
    atomic_json(out/'results.json',results)
    atomic_json(out/'status.json',dict(status='complete',expected=len(jobs),completed=len(results),
        registration_sha256=digest(path),results_sha256=digest(out/'results.json')))
    if phase=='calibrate':
        choice=selection(cfg,results);atomic_json(out/'selection.json',choice);print(json.dumps(choice),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['calibrate','confirmation','pipeline'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.phase=='pipeline':
        calibration=execute_phase('calibrate',args.output/'calibration',args.workers)
        chosen=json.loads((calibration/'selection.json').read_text())
        if chosen['selected']:execute_phase('confirmation',args.output/'confirmation',args.workers,calibration)
        else:print('No candidate passed. Stopping before confirmation; no post-test tuning.',flush=True)
    else:execute_phase(args.phase,args.output,args.workers,args.calibration)

if __name__=='__main__':main()
