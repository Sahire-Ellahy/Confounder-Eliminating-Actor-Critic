"""Calibration-gated benchmark driver for sequential locks and epistatic signalling."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import platform
import tarfile
from .environments import EpistasisConfig, LockConfig
from .run import ROOT, atomic_json, digest, encoded, run_cell, source_identity


def execute(job):
    protocol, cfg_dict, method, seed, task, dose, budget, folder = job
    log_every = 512 if budget >= 65536 else 128
    result = run_cell(protocol, cfg_dict, method, seed, task, dose, budget, folder, log_every=log_every)
    p_folder = Path(folder).resolve()
    rel_folder = str(p_folder.relative_to(ROOT)) if p_folder.is_relative_to(ROOT) else str(p_folder)
    return dict(
        folder=rel_folder,
        protocol=protocol,
        kind=cfg_dict["kind"],
        method=method,
        dose=dose,
        seed=seed,
        task=task,
        late_gap=result["late_gap"],
        cumulative_regret=result["cumulative_regret"],
        actor_updates=result["actor_updates"],
        priced_actor_updates=result["priced_actor_updates"],
        events=result["events"],
        seconds=result["seconds"],
    )


def required_calibration_methods(protocol: str, methods: list[str]) -> tuple[str, ...]:
    # Excluded-support priors (ids_decoy / ids_canonical / ids_additive) fail by design on task 1;
    # all primitive explorers, supported-prior IDS variants, and CEAC must pass easy calibration on both tasks.
    excluded = {"ids_decoy", "ids_canonical", "ids_additive", "oracle"}
    return tuple(m for m in methods if m not in excluded)


def calibration_kinds(spec: dict) -> list[str]:
    """Kinds sharing an identical easy kernel need one calibration; otherwise calibrate all."""
    return list(spec["kinds"]) if spec.get("calibrate_all_kinds") else [spec["kinds"][0]]


def validate_gates(paths: list[Path], protocol: str, budget: int, cfg: dict, sources: dict):
    spec = cfg[protocol]
    required = required_calibration_methods(protocol, spec["methods"])
    kinds = calibration_kinds(spec)
    expected = {(k, m, s, t) for k in kinds for m in required for s in cfg["calibration_seeds"] for t in (0, 1)}
    found = {}
    gates = []
    for path in paths:
        gate = json.loads(path.read_text())
        gates.append(dict(path=str(path), sha256=digest(path), data=gate))
        if gate["sources"] != sources:
            raise ValueError("Calibration numerical sources changed")
        for row in gate["cells"]:
            key = (row.get("kind", kinds[0]), row["method"], row["seed"], row["task"])
            if key not in expected:
                continue
            if row["protocol"] != protocol or row["budget"] != budget or row["late_gap"] > cfg["easy_gap_tolerance"]:
                raise ValueError(f"Required easy-case calibration gate failed: {row}")
            if row["method"].startswith("ceac") and min(row.get("actor_updates", 0), row.get("priced_actor_updates", 0)) <= 0:
                raise ValueError(f"CEAC calibration must exercise both critics and nonzero prices: {row}")
            found[key] = row
    if set(found) != expected:
        raise ValueError(f"Missing calibration conditions: {expected - set(found)}")
    return gates


def make_env_config(protocol: str, kind: str, dose: int) -> dict:
    if protocol == "locks":
        return asdict(LockConfig(kind=kind, depth=int(dose)))
    if protocol == "epistasis":
        return asdict(EpistasisConfig(kind=kind, pleiotropy=int(dose)))
    if protocol == "gi_screen":
        from ceac_gi.environment import build_config
        return json.loads(json.dumps(asdict(build_config(kind, int(dose)))))
    raise ValueError(protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "sweep"])
    parser.add_argument("--protocol", choices=["locks", "epistasis", "gi_screen"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/pathologies_primary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", default=[])
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    spec = cfg[args.protocol]
    budget = int(spec["episodes"])
    methods = list(spec["methods"])
    if args.methods:
        if args.phase != "calibrate" or not set(args.methods) <= set(methods):
            parser.error("--methods is only permitted as a calibration subset")
        methods = args.methods
    elif args.phase == "calibrate":
        methods = list(required_calibration_methods(args.protocol, methods))

    sources = source_identity()
    gates = []
    if args.phase == "sweep":
        if not args.calibration:
            parser.error("--calibration gate.json is required before launching a sweep")
        gates = validate_gates(args.calibration, args.protocol, budget, cfg, sources)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    seeds = cfg["calibration_seeds"] if args.phase == "calibrate" else cfg["evaluation_seeds"]
    tasks = [0, 1] if args.phase == "calibrate" else spec["tasks"]
    kinds = calibration_kinds(spec) if args.phase == "calibrate" else spec["kinds"]
    doses = [spec["doses"][0]] if args.phase == "calibrate" else spec["doses"]

    jobs = []
    for kind in kinds:
        for dose in doses:
            env_dict = make_env_config(args.protocol, kind, int(dose))
            for task in tasks:
                for method in methods:
                    for seed in seeds:
                        folder = output / "cells" / kind / f"dose_{dose}" / f"task_{task}" / f"{method}_s{seed}"
                        jobs.append((args.protocol, env_dict, method, int(seed), int(task), int(dose), budget, str(folder)))

    registration = dict(
        schema=1,
        phase=args.phase,
        protocol=args.protocol,
        configuration=cfg,
        sources=sources,
        driver_sha256=digest(__file__),
        python=platform.python_version(),
        episodes=budget,
        jobs=len(jobs),
        calibration_gates=gates,
        metric="exact pre-trajectory expected policy gap; final 256 episodes; seed is replicate",
        scope="stationary multi-step combination locks and epistatic signalling discovery; replay-driven two-critic CEAC",
    )
    reg_path = output / "registration.json"
    if reg_path.exists():
        if json.loads(reg_path.read_text()) != json.loads(encoded(registration)):
            raise ValueError("Registration changed; use a fresh study directory")
    else:
        atomic_json(reg_path, registration)
        with tarfile.open(output / "source_bundle.tar.gz", "w:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
            archive.add(Path(__file__), arcname="ceac_pathologies/experiment.py")
            if args.config.exists():
                cfg_resolved = args.config.resolve()
                arc = str(cfg_resolved.relative_to(ROOT)) if cfg_resolved.is_relative_to(ROOT) else cfg_resolved.name
                archive.add(cfg_resolved, arcname=arc)

    results = []
    total = len(jobs)
    atomic_json(output / "status.json", dict(status="running", expected=total, completed=0))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute, job) for job in jobs]):
            row = future.result()
            results.append(row)
            print(json.dumps({k: row[k] for k in ("kind", "dose", "task", "method", "seed", "late_gap", "cumulative_regret", "seconds")}), flush=True)
            atomic_json(output / "status.json", dict(status="running", expected=total, completed=len(results)))

    results.sort(key=lambda r: (r["kind"], r["dose"], r["task"], r["method"], r["seed"]))
    atomic_json(output / "results.json", results)
    atomic_json(
        output / "status.json",
        dict(
            status="complete",
            expected=total,
            completed=total,
            registration_sha256=digest(reg_path),
            results_sha256=digest(output / "results.json"),
        ),
    )
    if args.phase == "calibrate":
        cells = [
            dict(
                protocol=args.protocol,
                budget=budget,
                **{k: r[k] for k in ("kind", "method", "seed", "task", "late_gap", "actor_updates", "priced_actor_updates")},
            )
            for r in results
        ]
        all_pass = all(
            r["late_gap"] <= cfg["easy_gap_tolerance"]
            and (not r["method"].startswith("ceac") or min(r["actor_updates"], r["priced_actor_updates"]) > 0)
            for r in cells
        )
        atomic_json(
            output / "gate.json",
            dict(tolerance=cfg["easy_gap_tolerance"], sources=sources, cells=cells, all_pass=all_pass),
        )
        print(f"Calibration complete ({total}/{total}); all_pass={all_pass}", flush=True)
    else:
        print(f"Sweep complete ({total}/{total}).", flush=True)


if __name__ == "__main__":
    main()
