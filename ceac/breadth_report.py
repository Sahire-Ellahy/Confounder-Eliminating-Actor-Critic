"""Measured breadth responses and a separate, non-empirical geometry diagram."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from .breadth import BREADTH_PROFILES, make_environment
from .pathology_report import METHODS
from .visualization import LABELS, STYLES, seed_summary, band_description


def breadth_statistics(runs):
    groups=defaultdict(dict)
    selected=[r for r in runs if r["result"].get("benchmark") in BREADTH_PROFILES]
    if not selected:
        return []
    levels={BREADTH_PROFILES[r["result"]["benchmark"]] for r in selected}
    if 0 not in levels or len(levels)<2:
        raise ValueError("Breadth comparisons require zero-cell control and at least one positive breadth")
    for run in selected:
        r=run["result"];b=BREADTH_PROFILES[r["benchmark"]]
        key=r["environment"],r["context"],r["true_mode"],r["method"]
        if (b,r["seed"]) in groups[key]:
            raise ValueError("Duplicate breadth seed cell")
        cfg=run["manifest"]["configuration"]
        if cfg["benchmark"]["configuration_overrides"]["beacon_cells"]!=b:
            raise ValueError("Breadth differs from recorded configuration")
        history=run["metrics"]
        if [m["episode"] for m in history]!=list(range(r["episodes"])) or not np.isclose(
                history[-1]["cumulative_pseudo_regret"],r["pseudo_regret"],atol=1e-10,rtol=1e-10):
            raise ValueError("Incomplete or inconsistent breadth trace")
        groups[key][b,r["seed"]]=run
    rows=[]
    identities={}
    for key,grid in sorted(groups.items()):
        seeds={s for b,s in grid}
        if set(grid)!={(b,s) for b in levels for s in seeds}:
            raise ValueError("Breadth comparisons require paired seeds at every dose")
        identities[key]=(seeds,{r["result"]["episodes"] for r in grid.values()})
        for (b,seed),run in sorted(grid.items()):
            base=grid[0,seed]
            def fixed(r):
                cfg=deepcopy(r["manifest"]["configuration"])
                cfg["benchmark"].pop("name")
                cfg["benchmark"]["configuration_overrides"].pop("beacon_cells")
                return cfg
            if fixed(run)!=fixed(base) or run["manifest"].get("sources")!=base["manifest"].get("sources"):
                raise ValueError("A non-breadth setting or runtime changed")
            result,history=run["result"],run["metrics"];last=history[-1]
            row=dict(environment=key[0],context=key[1],mode=key[2],method=key[3],breadth=b,
                seed=seed,episodes=result["episodes"],regret=result["pseudo_regret"],
                added_regret=result["pseudo_regret"]-base["result"]["pseudo_regret"],
                last_gap=float(np.mean([m["optimal_return"]-m["expected_return"] for m in history[-32:]])),
                beacon_steps=last["cumulative_beacon_steps"],unique_beacon_cells=last["unique_beacon_cells"],
                information=last["cumulative_expected_model_information"],
                pomd_zero_fraction=float(np.mean([bool(m["diagnostics"].get("all_optimistic_q_zero",False)) for m in history])))
            if not np.isfinite([row[k] for k in ("regret","added_regret","last_gap","beacon_steps","information")]).all():
                raise ValueError("Nonfinite breadth endpoint")
            if not 0<=row["unique_beacon_cells"]<=b or (b==0 and row["beacon_steps"]!=0):
                raise ValueError("Impossible beacon exposure count")
            rows.append(row)
    for condition in {k[:3] for k in groups}:
        relevant=[k for k in groups if k[:3]==condition]
        if len(identities[relevant[0]][1])!=1 or any(identities[k]!=identities[relevant[0]] for k in relevant):
            raise ValueError("Breadth methods require the same seeds and episode budgets")
    lookup={(r["environment"],r["context"],r["mode"],r["method"],r["breadth"],r["seed"]):r for r in rows}
    for row in rows:
        key=row["environment"],row["context"],row["mode"]
        ceac=lookup.get((*key,"ceac",row["breadth"],row["seed"]))
        if ceac is not None:
            row["regret_saved_by_ceac"]=row["regret"]-ceac["regret"]
            row["additional_ceac_advantage"]=row["added_regret"]-ceac["added_regret"]
    return rows


def draw_breadth_response(rows, context=0, mode=1, plt=None, band="sd"):
    if plt is None:
        import matplotlib.pyplot as plt
    rows=[r for r in rows if (r["context"],r["mode"])==(context,mode)]
    if not rows:
        raise ValueError("No measured breadth rows")
    fig,axes=plt.subplots(2,3,figsize=(18,11))
    fields=("regret","added_regret","last_gap","beacon_steps","unique_beacon_cells","information")
    labels=("Cumulative pseudo-regret (symlog)","Additional regret vs zero cells","Last 32-episode value gap",
            "Cumulative chamber steps","Distinct physical chamber cells visited","Cumulative expected task KL (nats)")
    handles={}
    for method in METHODS:
        subset=[r for r in rows if r["method"]==method]
        if not subset:continue
        levels=sorted({r["breadth"] for r in subset})
        seeds=sorted({r["seed"] for r in subset})
        lookup={(r["breadth"],r["seed"]):r for r in subset}
        for axis,field,label in zip(axes.flat,fields,labels):
            values=np.array([[lookup[b,s][field] for b in levels] for s in seeds])
            stats=seed_summary(values,band);color,line,marker=STYLES[method]
            handles[method],=axis.plot(levels,stats["mean"],color=color,ls=line,marker=marker,
                markerfacecolor="none",lw=3.2 if method=="ceac" else 1.8,label=LABELS[method])
            if stats["lower"] is not None:
                axis.fill_between(levels,stats["lower"],stats["upper"],color=color,alpha=.12,lw=0)
            axis.set(xlabel="Additional beacon chamber cells",ylabel=label,xticks=levels)
            axis.grid(alpha=.2)
    axes[0,0].set_yscale("symlog",linthresh=.1);axes[0,0].set_ylim(bottom=0)
    axes[0,1].axhline(0,color="#555555",lw=.8)
    fig.suptitle(f"False Beacon: physical breadth response · mode {mode} · context {context}",fontsize=20,fontweight="bold",y=.99)
    budgets=sorted({r["episodes"] for r in rows});counts=sorted({len({r['seed'] for r in rows if r['method']==m}) for m in handles})
    fig.text(.5,.945,f"Measured doses; no monotonic fit. Seeds/method: {counts}; episodes/cell: {budgets}. Reachable domain grows.",ha="center",fontsize=11)
    fig.legend(list(handles.values()),[LABELS[m] for m in handles],loc="lower center",bbox_to_anchor=(.5,.045),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.015,band_description(band),ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.17,1,.91),h_pad=2.5,w_pad=2.5)
    return fig


def make_breadth_figures(destination, runs, plt, band="sd"):
    rows=breadth_statistics(runs);artifacts=[]
    for context,mode in sorted({(r["context"],r["mode"]) for r in rows}):
        fig=draw_breadth_response(rows,context,mode,plt,band)
        for extension in ("png","pdf"):
            path=Path(destination)/f"beacon_breadth_context{context}_mode{mode}.{extension}"
            fig.savefig(path,dpi=150,bbox_inches="tight",facecolor="white");artifacts.append(path)
        plt.close(fig)
    return artifacts


def breadth_report_section(rows):
    if not rows:return []
    text=["## Physical False Beacon breadth experiment", "",
        "Breadth is the number of additional traversable cells in a side chamber, reached only through "
        "the uninformative shallow buoy. It is not observation-alphabet size or dwell probability. "
        "The zero-cell control retains the original shoreline and its original nuisance routes. "
        "The horizon, rewards, diagnostic law, six seed routes, optimal values and oracle costs stay fixed. "
        "The reachable state/action domain grows; published optimism bonuses can therefore change with domain size. "
        "This is part of the breadth intervention, not a fixed-domain comparison.", "",
        "Total regret and added regret relative to zero cells are distinct. Subtraction is within seed before "
        "aggregation. Large regret already present at zero cells is not evidence of a breadth effect. "
        "POMD bonus saturation, physical chamber steps and distinct chamber cells are reported separately. "
        "CEAC/IDS know candidate laws; primitive controls do not. Bayesian prior exclusion is a separate factor. "
        "No monotone failure or causal learned-critic benefit is assumed.", ""]
    if min(r["episodes"] for r in rows)<2048:
        text += ["**Short integration run: these results validate execution and reporting, not a performance claim.**", ""]
    text += ["At the widest recorded dose: means ± sample SD over paired training seeds. Positive regret saved favors CEAC. "
             "Bold saved-regret entries favor CEAC in every recorded pair, not a significance test.", "",
        "| Context | Mode | Method | Seeds | Regret | Added vs zero | Saved by CEAC | Extra CEAC advantage vs zero | Chamber steps | Distinct cells | POMD saturation |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for context,mode,method in sorted({(r["context"],r["mode"],r["method"]) for r in rows}):
        group=[r for r in rows if (r["context"],r["mode"],r["method"])==(context,mode,method)]
        group=[r for r in group if r["breadth"]==max(t["breadth"] for t in group)]
        def fmt(field):
            if any(field not in r for r in group):return "n/a"
            values=np.array([r[field] for r in group]);sd=f"{values.std(ddof=1):.3f}" if len(values)>1 else "n/a"
            answer=f"{values.mean():.3f} ± {sd}"
            return f"**{answer}**" if field in {"regret_saved_by_ceac","additional_ceac_advantage"} and np.all(values>1e-10) else answer
        saturation=f"{np.mean([r['pomd_zero_fraction'] for r in group]):.1%}" if method=="pomd" else "n/a"
        text.append(f"| {context} | {mode} | {LABELS[method]} | {len(group)} | {fmt('regret')} | {fmt('added_regret')} | "
                    f"{fmt('regret_saved_by_ceac')} | {fmt('additional_ceac_advantage')} | {fmt('beacon_steps')} | "
                    f"{fmt('unique_beacon_cells')} | {saturation} |")
    return text+[""]


def draw_breadth_geometry(breadths=(0,2,8), plt=None):
    """Specification diagram only: edges are actual legal primitive moves."""
    if plt is None:
        import matplotlib.pyplot as plt
    from .environments import frozenlake as shore
    from .environments.beacon_breadth import GATEWAY, MOVES
    fig,axes=plt.subplots(1,len(breadths),figsize=(5.4*len(breadths),5),squeeze=False)
    for b,axis in zip(breadths,axes.flat):
        env=make_environment("frozenlake",benchmark=f"beacon_breadth_b{b:02d}")
        env.reset(seed=0)
        for row,col in sorted(env.open_cells):
            state=shore.DecodedState(0,row,col,-1,-1,-1,0)
            mask=env.valid_action_mask(env.encode(state))
            for action in np.flatnonzero(mask):
                dr,dc=MOVES[int(action)];other=(row+dr,col+dc)
                if other<(row,col):continue
                field=(row,col) in env.field_cells or other in env.field_cells
                axis.plot([col,col+dc],[row,row+dr],color="#d97a13" if field else "#bec6ce",lw=3,zorder=1)
            axis.scatter([col],[row],s=160,marker="s",color="#f5bd77" if (row,col) in env.field_cells else "#e3e8ee",edgecolor="white",zorder=2)
        for pos,label,color in ((shore.START,"S","#222222"),(shore.INNER_DOCK,"I","#16804a"),
                (shore.OUTER_DOCK,"O","#16804a"),(shore.SAFE_DOCK,"D","#16804a"),
                (shore.DEEP_BUOY,"Q","#0058c7"),(GATEWAY,"q","#b76400")):
            axis.text(pos[1],pos[0],label,ha="center",va="center",color=color,fontweight="bold",fontsize=13,zorder=3)
        axis.set(title=f"{b} additional chamber cells",xlim=(-.5,8.5),ylim=(6.6,-.6),aspect="equal")
        axis.axis("off")
    fig.suptitle("Prepared experiment: widen a misleading chamber without changing the useful routes",fontsize=16,fontweight="bold",y=.98)
    fig.text(.5,.11,"Orange = added physical cells. Lines = legal moves; neighboring cells without a line are separated by a wall.",ha="center",fontsize=11)
    fig.text(.5,.055,"q = uninformative entrance buoy; Q = useful diagnostic. Same 16-step horizon, rewards and six seed routes. No sweep results shown.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.16,1,.94))
    return fig
