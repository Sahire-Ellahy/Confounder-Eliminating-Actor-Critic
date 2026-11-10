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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ceac.evidence import atomic_json, digest
from ceac.report import load_study, validate_reference


RESULT_FIELDS = ("benchmark", "environment", "context", "true_mode", "method", "seed", "episodes",
                 "pseudo_regret", "total_return", "last_window_expected_return", "event_counts",
                 "deployment_certified", "unique_observed_states", "primitive_steps")
METRIC_FIELDS = ("episode", "expected_return", "optimal_return", "cumulative_pseudo_regret",
                 "cumulative_return", "cumulative_expected_model_information", "cumulative_observed_llr",
                 "cumulative_diagnostic_visits", "cumulative_nuisance_visits", "cumulative_risky_visits")


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
    validate_reference(loaded[1], loaded[0], ("ceac", "oracle", "confidence", "policy_ucb", "oir_idac", "coverage", "ids"))
    curves = []
    for run in all_runs:
        if run["result"].get("benchmark") != "weak_signal_m64":
            continue
        curves.append({"result": {key: run["result"][key] for key in RESULT_FIELDS if key in run["result"]},
                       "metrics": [{**{key: m[key] for key in METRIC_FIELDS},
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
    assets = [target]
    for name in ("aggregate_nuisance_context0_mode0.png", "aggregate_nuisance_context0_mode1.png"):
        output = destination / "assets" / name
        shutil.copyfile(Path(base) / "figures" / name, output)
        assets.append(output)
    atomic_json(destination / "evidence/manifest.json", dict(
        schema=1, source_studies=sources, runtime_sources=loaded[0]["sources"],
        scope="Derived, verified teaching export; full raw trajectories remain in the original studies",
        curves_benchmark="weak_signal_m64", source_runs=len(all_runs), curve_runs=len(curves),
        artifacts={str(p.relative_to(destination)): digest(p) for p in assets}))
    print(f"Exported {len(curves)} seed traces and {len(all_runs)} endpoints to {destination / 'evidence'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "runs/ceac_undertow_20261110")
    parser.add_argument("--prior", type=Path, default=ROOT / "runs/ceac_prior_support_20261110")
    parser.add_argument("--destination", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args()
    export(args.base, args.prior, args.destination)


if __name__ == "__main__":
    main()
