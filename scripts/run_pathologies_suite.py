"""Execute calibration, held-out sweeps, and verified report exports for ceac_pathologies."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_step(args_list: list[str]):
    env = dict(os.environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    cmd = [sys.executable] + args_list
    print(f"==> Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20261119_v1")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--protocols", nargs="+", default=["locks", "epistasis", "gi_screen"],
                        choices=["locks", "epistasis", "gi_screen"])
    args = parser.parse_args()

    w = str(args.workers)
    for protocol in args.protocols:
        cal = f"runs/pathologies_{protocol}_cal_{args.tag}"
        sweep = f"runs/pathologies_{protocol}_sweep_{args.tag}"
        out = f"notebooks/evidence/pathologies_{protocol}"
        run_step(["-m", "ceac_pathologies.experiment", "calibrate", "--protocol", protocol, "--output", cal, "--workers", w])
        run_step(["-m", "ceac_pathologies.experiment", "sweep", "--protocol", protocol, "--calibration", f"{cal}/gate.json", "--output", sweep, "--workers", w])
        run_step(["-m", "ceac_pathologies.report", "--study", sweep, "--output", out])

    print("ALL REQUESTED CALIBRATIONS, SWEEPS, AND REPORTS COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
