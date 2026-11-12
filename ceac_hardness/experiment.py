"""Archived reference driver; primary CEAC studies use ceac_pseudocode.experiment."""
import argparse,json,os,tarfile,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from .environment import JunctionConfig
from .run import ROOT,run_cell,source_identity,atomic_json,digest,encoded

PRIMITIVES=('idac','ucb_q','pomd')


def execute(job):
    config,method,seed,task,dose,budget,folder=job
    result=run_cell(config,method,seed,task,dose,budget,folder,log_every=512 if budget>32768 else 128)
    return dict(folder=str(Path(folder).relative_to(ROOT)),method=method,kind=config['kind'],
        burden=config['burden'],dose=dose,seed=seed,task=task,late_gap=result['late_gap'],
        cumulative_regret=result['cumulative_regret'],seconds=result['seconds'])


def launch(jobs,output,registration,workers):
    output.mkdir(parents=True,exist_ok=True)
    registered=output/'registration.json'
    if registered.exists():
        if json.loads(registered.read_text())!=json.loads(encoded(registration)):
            raise ValueError('Registration differs; use a fresh study directory')
    else:
        atomic_json(registered,registration)
        with tarfile.open(output/'source_bundle.tar.gz','w:gz') as archive:
            for name in registration['sources']:archive.add(ROOT/name,arcname=name)
            archive.add(Path(__file__),arcname='ceac_hardness/experiment.py')
    atomic_json(output/'status.json',dict(status='running',expected=len(jobs),completed=0))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(execute,j) for j in jobs]):
            result=future.result();results.append(result)
            print(json.dumps(result),flush=True)
            atomic_json(output/'status.json',dict(status='running',expected=len(jobs),completed=len(results)))
    atomic_json(output/'results.json',results)
    atomic_json(output/'status.json',dict(status='complete',expected=len(jobs),completed=len(results),
        registration_sha256=digest(registered),results_sha256=digest(output/'results.json')))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archived-reference',action='store_true',help='Explicitly reproduce the superseded reference protocol')
    parser.add_argument('phase',choices=['calibrate','sweep'])
    parser.add_argument('--protocol',choices=['burden','prior_shift'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not args.archived_reference:parser.error('Use python -m ceac_pseudocode.experiment for primary CEAC. This historical protocol requires --archived-reference.')
    cfg=json.loads((ROOT/'experiments/hardness_redesign.json').read_text())
    sources=source_identity();spec=cfg[args.protocol]
    budget=args.episodes if args.phase=='calibrate' else spec['episodes']
    if not budget:parser.error('calibration requires --episodes')
    methods=spec['methods'];seeds=cfg['calibration_seeds'] if args.phase=='calibrate' else cfg['evaluation_seeds']
    tasks=[0,1] if args.phase=='calibrate' else spec['tasks']
    kinds=['false_beacon_junction'] if args.phase=='calibrate' or args.protocol=='prior_shift' else spec['kinds']
    doses=[0] if args.phase=='calibrate' else spec['doses']
    gate=None;excluded=[]
    if args.phase=='sweep':
        if args.calibration is None:parser.error('--calibration gate.json is required')
        gate=json.loads(args.calibration.read_text())
        # Historical exclusion is retained solely to reproduce the archived index.
        # This modified-loop trial is not a conclusion about primary CEAC.
        if args.protocol=='prior_shift':
            actor=[r for r in gate['cells'] if r['method']=='ceac_actor']
            if actor and any(r['late_gap']>cfg['easy_gap_tolerance'] for r in actor):
                excluded=[dict(method='ceac_actor',reason='failed independent easy-case gate',calibration=actor)]
                methods=[m for m in methods if m!='ceac_actor']
        required=PRIMITIVES if args.protocol=='burden' else tuple(methods)
        matching=[r for r in gate['cells'] if r['method'] in required]
        observed={(r['method'],r['seed'],r['task']) for r in matching}
        expected={(m,s,t) for m in required for s in cfg['calibration_seeds'] for t in (0,1)}
        if (observed!=expected or gate['sources']!=sources or any(r['budget']!=budget or
            r['late_gap']>cfg['easy_gap_tolerance'] or r['protocol']!=args.protocol for r in matching)):
            raise ValueError('Independent easy-case calibration has not passed at this frozen budget/source')
    jobs=[]
    for kind in kinds:
        for dose in doses:
            env=JunctionConfig(kind=kind,protocol=args.protocol,burden=int(dose) if args.protocol=='burden' else 0)
            from dataclasses import asdict
            for task in tasks:
                for method in methods:
                    for seed in seeds:
                        folder=args.output/'cells'/kind/f'dose_{dose:g}'/f'task_{task}'/f'{method}_s{seed}'
                        jobs.append((asdict(env),method,seed,task,float(dose) if args.protocol=='prior_shift' else 0.,budget,str(folder.resolve())))
    registration=dict(schema=1,phase=args.phase,protocol=args.protocol,configuration=cfg,sources=sources,
        driver_sha256=digest(__file__),python=platform.python_version(),episodes=budget,jobs=len(jobs),
        excluded_from_sweep=excluded,calibration_gate=gate,calibration_sha256=digest(args.calibration) if args.calibration else None,
        metric='exact pre-trajectory expected policy gap; final 256 episodes; seed is replicate',
        scope='stationary two-step junction variants; historical grid benchmarks unchanged')
    results=launch(jobs,args.output.resolve(),registration,args.workers)
    if args.phase=='calibrate':
        rows=[dict(protocol=args.protocol,method=r['method'],seed=r['seed'],task=r['task'],budget=budget,late_gap=r['late_gap']) for r in results]
        atomic_json(args.output/'gate.json',dict(tolerance=cfg['easy_gap_tolerance'],all_pass=all(r['late_gap']<=cfg['easy_gap_tolerance'] for r in rows),cells=rows,sources=sources))

if __name__=='__main__':main()
