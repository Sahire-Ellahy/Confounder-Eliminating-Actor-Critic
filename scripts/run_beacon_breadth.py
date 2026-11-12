#!/usr/bin/env python3
"""Plan, preflight, smoke-test, or explicitly launch the registered breadth slate."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load_config(path):
    from ceac.breadth import BREADTH_PROFILES
    from ceac.run import DEFAULT_METHODS
    cfg=json.loads(Path(path).read_text())
    if cfg["schema"]!=1 or cfg["benchmarks"]!=list(BREADTH_PROFILES):
        raise ValueError("Registered plan requires all five ordered breadths, including zero")
    if cfg["environments"]!=["frozenlake"] or cfg["modes"]!=[0,1] or set(cfg["methods"])!=set(DEFAULT_METHODS):
        raise ValueError("Registered plan requires both modes and all seven controls")
    if len(cfg["methods"])!=7 or len(set(cfg["seeds"]))!=len(cfg["seeds"]) or len(cfg["seeds"])<5:
        raise ValueError("Need unique methods and at least five unique seeds")
    if cfg["contexts"]!=[0] or cfg["episodes"]<2048 or cfg["workers"]<1:
        raise ValueError("Invalid registered context, budget or workers")
    smoke=cfg["smoke"]
    if smoke["benchmarks"]!=[cfg["benchmarks"][0],cfg["benchmarks"][-1]] or smoke["episodes"]<8:
        raise ValueError("Smoke must cover both extreme breadths and seed initialization")
    return cfg


def command(cfg, stage, output, pdf=False):
    selected=dict(cfg)
    if stage=="smoke":selected.update(cfg["smoke"])
    argv=[sys.executable,"-m","ceac.run","--output",str(output)]
    for field in ("benchmarks","environments","contexts","modes","seeds","methods"):
        argv.extend(["--"+field,*map(str,selected[field])])
    argv.extend(["--episodes",str(selected["episodes"]),"--workers",str(selected["workers"])])
    if pdf:argv.append("--report-pdf")
    cells=1
    for field in ("benchmarks","environments","contexts","modes","seeds","methods"):
        cells*=len(selected[field])
    return argv,cells,selected["episodes"]


def preflight(cfg, output):
    from ceac.breadth import make_model,audit_breadth_model,audit_breadth_family
    from ceac.evidence import atomic_json,source_identity
    from ceac.breadth_report import draw_breadth_geometry
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=False)
    sources=source_identity();rows=[]
    for benchmark in cfg["benchmarks"]:
        model=make_model("frozenlake",benchmark=benchmark)
        rows.append(audit_breadth_model(model))
        print(f"Verified {benchmark}: {model.n} reachable states; optimum={model.optimal_returns.tolist()}",flush=True)
    checks=audit_breadth_family(rows)
    if source_identity()!=sources:raise ValueError("Source changed during preflight")
    atomic_json(output/"preflight.json",dict(schema=1,status="verified",configuration=cfg,
        sources=sources,checks=checks,models=rows,training_launched=False))
    fig=draw_breadth_geometry(plt=plt)
    for extension in ("png","pdf"):
        fig.savefig(output/f"geometry.{extension}",dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("Prepared design only; no full performance sweep launched.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"experiments/false_beacon_breadth.json")
    p.add_argument("--stage",choices=("plan","preflight","smoke","full"),default="plan")
    p.add_argument("--output",type=Path)
    p.add_argument("--pdf",action="store_true")
    args=p.parse_args();cfg=load_config(args.config)
    if args.stage=="plan":
        for stage in ("smoke","full"):
            argv,cells,episodes=command(cfg,stage,ROOT/f"runs/beacon_breadth_{stage}_NEW",args.pdf)
            print(f"{stage}: {cells} cells x {episodes} episodes = {cells*episodes:,} episodes")
            print(shlex.join(argv))
        return
    if args.output is None:p.error("--output is required; use a fresh directory")
    output=args.output.expanduser().resolve()
    if output.exists():p.error("Output exists; choose a fresh directory")
    if args.stage=="preflight":
        preflight(cfg,output)
    else:
        argv,cells,episodes=command(cfg,args.stage,output,args.pdf)
        print(f"Launching {args.stage}: {cells} cells, {cells*episodes:,} episodes",flush=True)
        subprocess.run(argv,cwd=ROOT,check=True)


if __name__=="__main__":main()
