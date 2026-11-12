#!/usr/bin/env python3
"""Prospective secondary comparison of all non-reference calibration passers.

The original endpoint-first selection is preserved. This extension was specified
from calibration results before inspecting primary-confirmation outcomes.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ceac_stability.experiment import PROTOCOL, job_result, load_completed, selection
from ceac_stability.run import atomic_json, digest, encoded, source_identity
from ceac_hardness.environment import JunctionConfig


def prepare(study):
    study = Path(study).resolve()
    calibration = study / 'calibration'
    cfg = json.loads(PROTOCOL.read_text())
    reg, results = load_completed(calibration)
    driver = ROOT / 'ceac_stability/experiment.py'
    if (reg['sources'] != source_identity() or reg['protocol_sha256'] != digest(PROTOCOL)
            or reg['driver_sha256'] != digest(driver)):
        raise ValueError('Calibration sources or protocol changed')
    choice = selection(cfg, results)
    if choice != json.loads((calibration / 'selection.json').read_text()):
        raise ValueError('Frozen primary selection changed')
    if choice['selected'] != 'original_reference':
        raise ValueError('This extension is registered for a reference-only primary selection')
    passing = sorted(r['candidate'] for r in choice['candidates']
                     if r['passed'] and r['candidate'] != 'original_reference')
    if not passing:
        raise ValueError('No non-reference calibration passer to confirm')
    candidates = [c for c in cfg['candidates'] if c['id'] in passing]
    out = study / 'confirmation_secondary'
    jobs = []
    for protocol in ('prior_shift', 'burden'):
        spec = cfg['confirmation'][protocol]
        for kind in spec.get('kinds', ['false_beacon_junction']):
            for dose in spec['doses']:
                env = JunctionConfig(kind=kind, protocol=protocol,
                                     burden=int(dose) if protocol == 'burden' else 0)
                for task in spec['tasks']:
                    for candidate in candidates:
                        for seed in cfg['confirmation']['seeds']:
                            folder = out / 'cells' / protocol / kind / f'dose_{dose:g}' / f'task_{task}' / f'{candidate["id"]}_s{seed}'
                            jobs.append(dict(config=asdict(env), candidate=candidate, schedule=cfg['schedule'],
                                seed=seed, task=task, dose=float(dose) if protocol == 'prior_shift' else 0.,
                                episodes=spec['episodes'], output=str(folder),
                                log_every=128 if spec['episodes'] <= 32768 else 512))
    registration = dict(schema=1, phase='confirmation_secondary', protocol=cfg,
        protocol_sha256=digest(PROTOCOL), sources=source_identity(),
        primary_driver_sha256=digest(driver), driver_sha256=digest(__file__),
        python=platform.python_version(), jobs=len(jobs), candidates=passing,
        primary_selected=choice['selected'],
        extension_rule='Confirm every non-reference candidate passing the original calibration gates; no reranking or threshold changes.',
        declaration='Registered after calibration and before inspecting any confirmation outcomes. This is a secondary extension, not the original primary selection.',
        endpoints=['paired cumulative external regret', 'paired late behavior reward gap',
                   'unconstrained and floor-constrained final actor augmented gaps'],
        analysis='Keep both tasks and every dose. Report seed-paired differences separately by family/task/dose and family/task averages with seeds as the unit; mean +/- sample SD, no post-confirmation tuning.',
        reference=str(study / 'confirmation'),
        selection_provenance={name: digest(calibration / f'{name}.json') for name in ('registration', 'results', 'selection')},
        source_contract='Complete CEAC: both critics and evidence-priced replay; no evaluator values used in learning.')
    return out, registration, jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--register-only', action='store_true')
    args = p.parse_args()
    out, registration, jobs = prepare(args.study)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'registration.json'
    if path.exists():
        stored = json.loads(path.read_text())
        registration['registered_utc'] = stored['registered_utc']
        if stored != json.loads(encoded(registration)):
            raise ValueError('Secondary registration changed')
    else:
        registration['registered_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(path, registration)
        with tarfile.open(out / 'source_bundle.tar.gz', 'w:gz') as archive:
            names = set(registration['sources']) | {'ceac_stability/experiment.py',
                str(PROTOCOL.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT))}
            for name in sorted(names):
                archive.add(ROOT / name, arcname=name)
    print(json.dumps(dict(registration=str(path), jobs=len(jobs), candidates=registration['candidates'],
                          registered_utc=registration['registered_utc'])), flush=True)
    if args.register_only:
        return
    results, pending = [], []
    for job in jobs:
        if (Path(job['output']) / 'status.json').exists():
            results.append(job_result(job))
        else:
            pending.append(job)
    atomic_json(out / 'status.json', dict(status='running', expected=len(jobs), completed=len(results)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job_result, job): job for job in pending}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            # Outcomes stay in immutable cells; progress logging is deliberately compact.
            print(json.dumps(dict(completed=len(results), expected=len(jobs), candidate=row['candidate'],
                                  protocol=row['protocol'], task=row['task'], seed=row['seed'])), flush=True)
            atomic_json(out / 'status.json', dict(status='running', expected=len(jobs), completed=len(results)))
    results.sort(key=lambda r: (r['protocol'], r['kind'], r['burden'], r['dose'], r['task'], r['candidate'], r['seed']))
    atomic_json(out / 'results.json', results)
    atomic_json(out / 'status.json', dict(status='complete', expected=len(jobs), completed=len(results),
        registration_sha256=digest(path), results_sha256=digest(out / 'results.json')))


if __name__ == '__main__':
    main()
