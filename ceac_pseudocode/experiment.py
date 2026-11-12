"""Calibration-gated primary CEAC studies, with verified reuse of controls."""
import argparse,json,tarfile,platform
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from ceac_hardness.environment import JunctionConfig
from ceac_hardness import run as controls
from . import run as primary

ROOT=primary.ROOT

def execute(job):
    config,method,seed,task,dose,budget,folder=job
    runner=primary if method=='ceac' else controls
    result=runner.run_cell(config,method,seed,task,dose,budget,folder,log_every=512 if budget>32768 else 128)
    return dict(folder=str(Path(folder).relative_to(ROOT)),method=method,kind=config['kind'],
        burden=config['burden'],dose=dose,seed=seed,task=task,late_gap=result['late_gap'],
        cumulative_regret=result['cumulative_regret'],seconds=result['seconds'],
        actor_updates=result['actor_updates'],priced_actor_updates=result['priced_actor_updates'])

def validate_gates(paths,protocol,budget,cfg,sources_by_method):
    required=('idac','ucb_q','pomd','ceac') if protocol=='burden' else ('ids_correlated','ids_product','ceac')
    expected={(m,s,t) for m in required for s in cfg['calibration_seeds'] for t in (0,1)}
    found={};gates=[]
    for path in paths:
        gate=json.loads(path.read_text());gates.append(dict(path=str(path),sha256=primary.digest(path),data=gate))
        for row in gate['cells']:
            key=(row['method'],row['seed'],row['task'])
            if key not in expected:continue
            sources=gate.get('sources_by_method',{}).get(row['method'],gate.get('sources'))
            if sources!=sources_by_method[row['method']]:raise ValueError('Calibration numerical sources changed')
            if row['protocol']!=protocol or row['budget']!=budget or row['late_gap']>cfg['easy_gap_tolerance']:
                raise ValueError('Required easy-case gate failed or used a different budget')
            if row['method']=='ceac' and min(row.get('actor_updates',0),row.get('priced_actor_updates',0))<=0:
                raise ValueError('CEAC calibration must exercise both critics and nonzero prices')
            found[key]=row
    if set(found)!=expected:raise ValueError(f'Missing calibration conditions: {expected-set(found)}')
    return gates

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['calibrate','sweep'])
    parser.add_argument('--protocol',choices=['burden','prior_shift'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,nargs='+',default=[])
    parser.add_argument('--reuse-controls',type=Path)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();cfg=json.loads((ROOT/'experiments/ceac_primary.json').read_text())
    spec=cfg[args.protocol];budget=spec['episodes'];methods=spec['methods']
    if args.methods:
        if args.phase!='calibrate' or not set(args.methods)<=set(methods):parser.error('--methods is only a calibration subset')
        methods=args.methods
    sources=primary.source_identity();control_sources=controls.source_identity()
    if any(sources[k]!=v for k,v in control_sources.items()):raise ValueError('Shared numerical environment/control code differs')
    by_method={m:sources if m=='ceac' else control_sources for m in methods}
    gates=validate_gates(args.calibration,args.protocol,budget,cfg,by_method) if args.phase=='sweep' else []
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    reused=[];reuse_record=None
    if args.reuse_controls:
        if args.phase!='sweep' or args.protocol!='burden':parser.error('Control reuse is only for the matched-budget burden sweep')
        from ceac_hardness.report import load_study
        old=load_study(args.reuse_controls)
        if old['registration']['sources']!=control_sources or old['registration']['episodes']!=budget:
            raise ValueError('Reused controls have incompatible numerical sources or budgets')
        reused=[r for r in json.loads((args.reuse_controls/'results.json').read_text()) if r['method'] in methods and r['method']!='ceac']
        reuse_record=dict(folder=str(args.reuse_controls.resolve()),registration_sha256=primary.digest(args.reuse_controls/'registration.json'),
            results_sha256=primary.digest(args.reuse_controls/'results.json'),cells=len(reused))
    seeds=cfg['calibration_seeds'] if args.phase=='calibrate' else cfg['evaluation_seeds']
    tasks=[0,1] if args.phase=='calibrate' else spec['tasks']
    kinds=['false_beacon_junction'] if args.phase=='calibrate' or args.protocol=='prior_shift' else spec['kinds']
    doses=[0] if args.phase=='calibrate' else spec['doses']
    keys={(r['kind'],r['burden'],r['dose'],r['task'],r['method'],r['seed']) for r in reused};jobs=[]
    for kind in kinds:
        for dose in doses:
            env=JunctionConfig(kind=kind,protocol=args.protocol,burden=int(dose) if args.protocol=='burden' else 0)
            shift=float(dose) if args.protocol=='prior_shift' else 0.
            for task in tasks:
                for method in methods:
                    for seed in seeds:
                        if (kind,env.burden,shift,task,method,seed) in keys:continue
                        folder=output/'cells'/kind/f'dose_{dose:g}'/f'task_{task}'/f'{method}_s{seed}'
                        jobs.append((asdict(env),method,seed,task,shift,budget,str(folder)))
    registration=dict(schema=2,phase=args.phase,protocol=args.protocol,configuration=cfg,sources=sources,
        sources_by_method=by_method,driver_sha256=primary.digest(__file__),python=platform.python_version(),
        episodes=budget,jobs=len(jobs)+len(reused),calibration_gates=gates,reused_controls=reuse_record,
        metric='exact pre-trajectory expected policy gap; final 256 episodes; seed is replicate',
        scope='stationary two-step junction variants; replay-driven two-critic CEAC; historical grids separate')
    path=output/'registration.json'
    if path.exists():
        if json.loads(path.read_text())!=json.loads(primary.encoded(registration)):raise ValueError('Registration changed; use a fresh study directory')
    else:
        primary.atomic_json(path,registration)
        with tarfile.open(output/'source_bundle.tar.gz','w:gz') as archive:
            for name in sources:archive.add(ROOT/name,arcname=name)
            for name in ('ceac_pseudocode/experiment.py','experiments/ceac_primary.json'):
                archive.add(ROOT/name,arcname=name)
    results=list(reused);total=len(jobs)+len(reused)
    primary.atomic_json(output/'status.json',dict(status='running',expected=total,completed=len(results)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute,job) for job in jobs]):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
            primary.atomic_json(output/'status.json',dict(status='running',expected=total,completed=len(results)))
    primary.atomic_json(output/'results.json',results)
    primary.atomic_json(output/'status.json',dict(status='complete',expected=total,completed=total,
        registration_sha256=primary.digest(path),results_sha256=primary.digest(output/'results.json')))
    if args.phase=='calibrate':
        cells=[dict(protocol=args.protocol,budget=budget,**{k:r[k] for k in ('method','seed','task','late_gap','actor_updates','priced_actor_updates')}) for r in results]
        primary.atomic_json(output/'gate.json',dict(tolerance=cfg['easy_gap_tolerance'],sources_by_method=by_method,cells=cells,
            all_pass=all(r['late_gap']<=cfg['easy_gap_tolerance'] for r in cells)))
    print(f'Complete: {total}/{total}; reused {len(reused)} unchanged comparator cells.',flush=True)

if __name__=='__main__':main()
