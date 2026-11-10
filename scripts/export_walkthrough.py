#!/usr/bin/env python3
"""Export portable teaching evidence from hash-verified completed studies.

This is a reporting operation. It never imports a training entrypoint or runs
an experiment. Raw trajectories and local absolute paths are not published.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ceac.evidence import atomic_json, digest
from ceac.report import load_study, validate_reference
from ceac.visualization import NUISANCE_PROFILES, RETIRED_METHODS


RESULT_FIELDS = ("benchmark", "environment", "context", "true_mode", "method", "seed", "episodes",
                 "pseudo_regret", "total_return", "last_window_expected_return", "event_counts",
                 "deployment_certified", "unique_observed_states", "primitive_steps", "success_rate")
METRIC_FIELDS = ("episode", "expected_return", "optimal_return", "cumulative_pseudo_regret",
                 "cumulative_return", "cumulative_expected_model_information", "cumulative_observed_llr",
                 "cumulative_diagnostic_visits", "cumulative_nuisance_visits", "cumulative_risky_visits",
                 "terminal_success", "cumulative_diversions", "cumulative_lost_steps")


def export(base, prior, destination):
    destination = Path(destination)
    (destination / "evidence").mkdir(parents=True, exist_ok=True)
    (destination / "assets").mkdir(parents=True, exist_ok=True)
    sources, all_runs, loaded = [], [], []
    for root in (Path(base), Path(prior)):
        study, runs, manifests, _ = load_study(root)
        audit = json.loads((root / "report_audit.json").read_text())
        if digest(root / "report.md") != audit["report_sha256"]:
            raise ValueError("Report digest mismatch")
        for name, expected in audit["artifacts"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != expected:
                raise ValueError("Report artifact mismatch: " + name)
        sources.append(dict(name=root.name, study_identity=study["identity"],
                            study_sha256=digest(root / "study.json"), results_sha256=digest(root / "results.json"),
                            report_audit_sha256=digest(root / "report_audit.json"),
                            run_manifest_hashes=manifests, run_count=len(runs)))
        loaded.append(study)
        all_runs.extend(runs)
    validate_reference(loaded[1], loaded[0], ("ceac", "oracle", "ids"))
    source_runs = len(all_runs)
    retired_runs = [r for r in all_runs if r["result"]["method"] in RETIRED_METHODS]
    all_runs = [r for r in all_runs if r["result"]["method"] not in RETIRED_METHODS]
    curves = []
    for run in all_runs:
        if run["result"].get("benchmark") not in NUISANCE_PROFILES:
            continue
        curves.append({"manifest": {"configuration": run["manifest"]["configuration"]},
                       "result": {key: run["result"][key] for key in RESULT_FIELDS if key in run["result"]},
                       "metrics": [{**{key: m[key] for key in METRIC_FIELDS if key in m},
                                    "diagnostics": {"posterior": m["diagnostics"]["posterior"]}
                                       if "posterior" in m["diagnostics"] else {}}
                                   for m in run["metrics"]]})
    bundle = dict(schema=1, source_studies=sources, curves=curves,
                  endpoints=[{key: r["result"][key] for key in RESULT_FIELDS if key in r["result"]} for r in all_runs],
                  history_audits=[r["history_audit"] for r in all_runs],
                  model_audits=json.loads((Path(base) / "environment_audit.json").read_text())["models"])
    target = destination / "evidence/curves.json.gz"
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    runtime = destination / "evidence/source_bundle.tar.gz"
    shutil.copyfile(Path(base) / "source_bundle.tar.gz", runtime)
    with tarfile.open(runtime, "r:gz") as archive:
        for name, checksum in loaded[0]["sources"].items():
            if hashlib.sha256(archive.extractfile("ceac/"+name).read()).hexdigest() != checksum:
                raise ValueError("Archived training source mismatch: " + name)
    assets = [target, runtime]
    for name in (f"aggregate_nuisance_{kind}_context0_mode{mode}.png"
                 for kind in ("deepsea", "frozenlake") for mode in (0, 1)):
        output = destination / "assets" / name
        shutil.copyfile(Path(base) / "figures" / name, output)
        assets.append(output)
    atomic_json(destination / "evidence/manifest.json", dict(
        schema=2, source_studies=sources, runtime_sources=loaded[0]["sources"],
        runtime_archive="evidence/source_bundle.tar.gz",
        diagnostic_sources={name: checksum for name, checksum in loaded[0]["sources"].items()
                            if name not in {"run.py", "comparators.py"}},
        excluded_method_ids=sorted(RETIRED_METHODS), excluded_runs=len(retired_runs),
        scope="Derived, verified teaching export; full raw trajectories remain in the original studies",
        curves_benchmarks=list(NUISANCE_PROFILES), source_runs=source_runs, retained_runs=len(all_runs), curve_runs=len(curves),
        artifacts={str(p.relative_to(destination)): digest(p) for p in assets}))
    print(f"Exported {len(curves)} seed traces and {len(all_runs)} endpoints to {destination / 'evidence'}")



def export_study(study_root, destination):
    """Publish one complete, audited study; do not pool incompatible runtimes."""
    from ceac.benchmarks import STRENGTH_PROFILES
    from ceac.pathology_report import strength_groups
    study_root, destination = Path(study_root), Path(destination)
    study, runs, manifests, _ = load_study(study_root)
    if not runs or any(r["result"].get("benchmark") not in STRENGTH_PROFILES for r in runs):
        raise ValueError("The physical-strength export requires a physical-strength study")
    strength_groups(runs)
    report_audit = json.loads((study_root/"report_audit.json").read_text())
    if digest(study_root/"report.md") != report_audit["report_sha256"]:
        raise ValueError("Report Markdown digest mismatch")
    if report_audit["study_identity"] != study["identity"]:
        raise ValueError("Report identity differs from study")
    for name, expected in report_audit["artifacts"].items():
        path = (study_root/name).resolve()
        if not path.is_relative_to(study_root.resolve()) or digest(path) != expected:
            raise ValueError("Report artifact mismatch: "+name)
    source = dict(name=study_root.name, study_identity=study["identity"],
        study_sha256=digest(study_root/"study.json"), results_sha256=digest(study_root/"results.json"),
        report_audit_sha256=digest(study_root/"report_audit.json"),
        run_manifest_hashes=manifests, run_count=len(runs))
    curves = []
    for run in runs:
        curves.append(dict(manifest={"configuration":run["manifest"]["configuration"]},
            result={k:run["result"][k] for k in RESULT_FIELDS if k in run["result"]},
            metrics=[{**{k:m[k] for k in METRIC_FIELDS if k in m},
                      "diagnostics":{k:m["diagnostics"][k] for k in
                          ("posterior","all_optimistic_q_zero","actor_updates","gradient_norm")
                          if k in m["diagnostics"]}} for m in run["metrics"]]))
    environment_audit=json.loads((study_root/"environment_audit.json").read_text())
    statistics=json.loads((study_root/"pathology_statistics.json").read_text())
    if statistics["study_identity"] != study["identity"]:
        raise ValueError("Strength statistics belong to another study")
    bundle=dict(schema=1,source_studies=[source],curves=curves,
        endpoints=[r["result"] for r in curves],history_audits=[r["history_audit"] for r in runs],
        model_audits=environment_audit["models"],strength_checks=environment_audit["strength_checks"],
        pathology_statistics=statistics["rows"])
    (destination/"evidence").mkdir(parents=True,exist_ok=True)
    (destination/"assets").mkdir(parents=True,exist_ok=True)
    target=destination/"evidence/curves.json.gz"
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as compressed:
            compressed.write(json.dumps(bundle,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
    runtime=destination/"evidence/source_bundle.tar.gz"
    shutil.copyfile(study_root/"source_bundle.tar.gz",runtime)
    with tarfile.open(runtime,"r:gz") as archive:
        for name,checksum in study["sources"].items():
            if hashlib.sha256(archive.extractfile("ceac/"+name).read()).hexdigest()!=checksum:
                raise ValueError("Archived training source mismatch: "+name)
    assets=[target,runtime]
    # The report manifest, rather than a filesystem glob, defines published figures.
    for name in report_audit["artifacts"]:
        path=Path(name)
        if path.parent==Path("figures") and path.name.startswith("pathology_") and path.suffix==".png":
            output=destination/"assets"/path.name
            shutil.copyfile(study_root/path,output);assets.append(output)
    atomic_json(destination/"evidence/manifest.json",dict(schema=2,source_studies=[source],
        runtime_sources=study["sources"],runtime_archive="evidence/source_bundle.tar.gz",
        diagnostic_sources={k:v for k,v in study["sources"].items() if k not in {"run.py","comparators.py"}},
        excluded_method_ids=sorted(RETIRED_METHODS),excluded_runs=0,
        scope="Matched physical-diversion sweep; primitive learners and full-support Bayesian IDS kept distinct",
        curves_benchmarks=study["configuration"]["study"]["benchmarks"],
        source_runs=len(runs),retained_runs=len(runs),curve_runs=len(curves),
        artifacts={str(p.relative_to(destination)):digest(p) for p in assets}))
    print(f"Exported {len(curves)} physical-strength seed curves to {destination/'evidence'}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "runs/pathology_strength_20261110")
    parser.add_argument("--base", type=Path, help="Optional historical base study; requires --prior")
    parser.add_argument("--prior", type=Path, help="Historical support extension; requires --base")
    parser.add_argument("--destination", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args()
    if args.base is not None or args.prior is not None:
        if args.base is None or args.prior is None:
            parser.error("Historical export requires both --base and --prior")
        export(args.base, args.prior, args.destination)
    else:
        export_study(args.study, args.destination)


if __name__ == "__main__":
    main()
