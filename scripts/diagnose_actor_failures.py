#!/usr/bin/env python3
"""Audit every saved primary cue-junction CEAC run; do not train."""
from pathlib import Path
import sys,json,gzip,argparse
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ceac_walkthrough.actor_diagnosis import diagnose
from ceac_hardness.run import digest,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'runs/ceac_primary_prior_20261111')
    parser.add_argument('--output',type=Path,default=ROOT/'audits/ceac_actor_failure_20261112')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();cells=[]
    for entry in json.loads((args.study/'results.json').read_text()):
        folder=ROOT/entry['folder']
        if json.loads((folder/'summary.json').read_text())['settings']['method']=='ceac':cells.append(folder)
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(diagnose,cells):
            results.append(r)
            print(f"Audited {len(results)}/{len(cells)}: task {r['task']}, dose {r['dose']}, seed {r['seed']}, augmented gap {r['final_augmented_gap']:.6f}",flush=True)
    with gzip.open(args.output/'diagnostics.json.gz','wt') as f:
        json.dump(dict(schema=1,training_launched=False,cells=results),f,allow_nan=False,separators=(',',':'))
    atomic_json(args.output/'manifest.json',dict(schema=1,files={'diagnostics.json.gz':digest(args.output/'diagnostics.json.gz')},
        sources={name:digest(ROOT/name) for name in ('ceac_walkthrough/actor_diagnosis.py','scripts/diagnose_actor_failures.py')},
        cells=len(results),actor_updates=sum(r['checked_updates'] for r in results)))

if __name__=='__main__':main()
