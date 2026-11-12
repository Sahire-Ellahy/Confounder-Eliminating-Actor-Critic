"""Matched physical-strength figures and diagnostics, from recorded runs only."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from .benchmarks import STRENGTH_PROFILES
from .visualization import (ENVIRONMENT_LABELS, LABELS, STYLES, seed_summary,
                            trace_values, plot_seed_trace, band_description, plot_summary, pad_plot_axis)

METHODS = ("ucb_q", "pomd", "idac", "ids", "ids_zero", "ceac", "oracle")
DOSE_COLORS = ("#0058c7", "#008747", "#ad5700", "#c90078")
DOSE_LINES = ("-", "--", "-.", ":")
METRICS = ("regret", "last_gap", "success", "nuisance_visits", "lost_steps", "information")
METRIC_LABELS = ("Cumulative pseudo-regret", "Last 32-episode\nexpected value gap",
                 "Observed episode\nsuccess rate", "Cumulative nuisance visits",
                 "Cumulative lost-progress\nor dwell steps", "Cumulative expected\ntask KL (nats)")


def strength_groups(runs):
    """Fail on missing matched cells, changed settings, or inconsistent traces."""
    selected = [r for r in runs if r["result"].get("benchmark") in STRENGTH_PROFILES]
    if not selected:
        return {}
    levels = {STRENGTH_PROFILES[r["result"]["benchmark"]] for r in selected}
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for run in selected:
        r = run["result"]
        rho = STRENGTH_PROFILES[r["benchmark"]]
        key = r["environment"], r["context"], r["true_mode"], r["method"]
        cell = *key, rho, r["seed"]
        if cell in seen:
            raise ValueError("Duplicate physical-strength seed cell")
        seen.add(cell)
        config = run["manifest"]["configuration"]
        if config["benchmark"]["configuration_overrides"]["nuisance_probability"] != rho:
            raise ValueError("Strength differs from its recorded configuration")
        rows = run["metrics"]
        if len(rows) != r["episodes"] or [m["episode"] for m in rows] != list(range(r["episodes"])):
            raise ValueError("Physical-strength trace budget is incomplete")
        if not np.isclose(rows[-1]["cumulative_pseudo_regret"], r["pseudo_regret"], rtol=1e-10, atol=1e-10):
            raise ValueError("Physical-strength endpoint disagrees with its trace")
        groups[key][rho].append(run)
    seed_sets = {}
    for key, doses in groups.items():
        if set(doses) != levels:
            raise ValueError("Missing physical-strength level for a method")
        reference = {r["result"]["seed"]: r for r in doses[min(levels)]}
        seed_sets[key] = set(reference)
        for rho, subset in doses.items():
            if {r["result"]["seed"] for r in subset} != set(reference):
                raise ValueError("Strength comparison requires paired seed identities")
            for run in subset:
                baseline = reference[run["result"]["seed"]]
                def fixed(r):
                    c = deepcopy(r["manifest"]["configuration"])
                    c["benchmark"].pop("name")
                    c["benchmark"]["configuration_overrides"].pop("nuisance_probability")
                    return c
                if fixed(run) != fixed(baseline):
                    raise ValueError("A non-strength setting changed in the dose comparison")
                if run["manifest"].get("sources") != baseline["manifest"].get("sources"):
                    raise ValueError("Runtime source changed across physical-strength cells")
    for condition in {k[:3] for k in groups}:
        relevant = [k for k in groups if k[:3] == condition]
        seed_set = seed_sets[relevant[0]]
        budgets = {r["result"]["episodes"] for k in relevant for rs in groups[k].values() for r in rs}
        if any(seed_sets[k] != seed_set for k in relevant) or len(budgets) != 1:
            raise ValueError("Method overlays require identical seed identities and episode budgets")
    return dict(groups)


def strength_statistics(runs):
    groups = strength_groups(runs)
    rows = []
    for (kind, context, mode, method), doses in sorted(groups.items()):
        baseline = {r["result"]["seed"]: r["result"]["pseudo_regret"] for r in doses[min(doses)]}
        for rho, subset in sorted(doses.items()):
            for run in sorted(subset, key=lambda r: r["result"]["seed"]):
                r, history = run["result"], run["metrics"]
                last = history[-1]
                gap = float(np.mean([m["optimal_return"]-m["expected_return"] for m in history[-32:]]))
                saturation = float(np.mean([bool(m["diagnostics"].get("all_optimistic_q_zero", False)) for m in history]))
                flags = []
                if saturation >= .9:
                    flags.append("POMD_BONUS_SATURATION")
                if method == "ids_zero" and mode == 1:
                    flags.append("PRIOR_SUPPORT_EXCLUDED")
                if last["cumulative_lost_steps"]:
                    flags.append("PHYSICAL_NUISANCE_EXPOSURE")
                if gap > .01:
                    flags.append("FINITE_BUDGET_VALUE_GAP")
                rows.append(dict(environment=kind, context=context, mode=mode, method=method, rho=rho,
                    seed=r["seed"], episodes=r["episodes"], regret=r["pseudo_regret"], last_gap=gap,
                    success=float(np.mean([m["terminal_success"] for m in history])),
                    nuisance_visits=last["cumulative_nuisance_visits"],
                    diversions=last["cumulative_diversions"], lost_steps=last["cumulative_lost_steps"],
                    information=last["cumulative_expected_model_information"],
                    added_regret=r["pseudo_regret"]-baseline[r["seed"]],
                    pomd_zero_fraction=saturation, diagnostic_flags=flags))
    return rows


def _title(fig, kind, context, mode, subtitle):
    fig.suptitle(ENVIRONMENT_LABELS[kind]+f" · mode {mode} · context {context}",
                 fontsize=23, fontweight="bold", y=.99)
    fig.text(.5, .945, subtitle, ha="center", fontsize=13)


def _legend(fig, handles, methods, band, bottom=.06):
    fig.legend([handles[m] for m in methods if m in handles], [LABELS[m] for m in methods if m in handles],
               loc="lower center", bbox_to_anchor=(.5,bottom), ncol=3, frameon=False, fontsize=11)
    fig.text(.5,.018,band_description(band),ha="center",fontsize=10)


def draw_method_overlay(groups, kind, context=0, mode=1, plt=None, band="sd", prior=False):
    """One panel per strength; method color is invariant across all panels."""
    if plt is None:
        import matplotlib.pyplot as plt
    methods = ("ceac","ids","ids_zero") if prior else METHODS
    present = [m for m in methods if (kind,context,mode,m) in groups]
    if not present:
        raise ValueError("No recorded methods for the requested overlay")
    levels = sorted(groups[kind,context,mode,present[0]])
    fig,axes=plt.subplots(2,2,figsize=(17,11),sharex=True,sharey=True,squeeze=False)
    if len(levels)>4:
        raise ValueError("The standard layout supports at most four strength levels")
    handles={}
    for rho,axis in zip(levels,axes.flat):
        endpoints={}
        for method in present:
            values=trace_values(groups[kind,context,mode,method][rho],"cumulative_pseudo_regret")
            handles[method]=plot_seed_trace(axis,values,method,band)
            endpoints[method]=float(values[:,-1].mean())
        if all(m in endpoints for m in ("ucb_q","pomd","idac")):
            axis.text(.98,.42, f'Final R: UCB-Q {endpoints["ucb_q"]:.1f} | POMD {endpoints["pomd"]:.1f}\nIDAC {endpoints["idac"]:.1f}',
                      ha="right",va="center",transform=axis.transAxes,fontsize=10,
                      bbox=dict(fc="white",ec="#cccccc",alpha=.9,boxstyle="round,pad=.4"))
        axis.set(title=fr"Diversion probability $\rho={rho:g}$",
                 xlabel="Training episode",ylabel="Cumulative pseudo-regret (symlog)")
        axis.set_yscale("symlog",linthresh=.01)
        axis.grid(alpha=.23)
        axis.tick_params(labelleft=True,labelbottom=True)
    for axis in list(axes.flat)[len(levels):]:
        axis.set_axis_off()
    pad_plot_axis(axes[0,0])
    subtitle = (r"Bayesian priors: full support $p_0(1)=0.5$ vs excluded mode 1 $p_0(1)=0$" if prior else
                "Optimism vs frequentist IDAC vs Bayesian IDS (full support and misspecified prior)")
    _title(fig,kind,context,mode,subtitle)
    _legend(fig,handles,present,band)
    fig.tight_layout(rect=(0,.18,1,.92),h_pad=2.3,w_pad=1.5)
    return fig


def draw_strength_learning(groups, kind, context=0, mode=1, plt=None, band="sd", incremental=False):
    """STEERING layout: training on x, regret on y, strength as curve color."""
    if plt is None:
        import matplotlib.pyplot as plt
    present=[m for m in METHODS if (kind,context,mode,m) in groups]
    if not present:
        raise ValueError("No recorded methods for the requested strength curves")
    columns = 4 if len(present) > 6 else 3
    rows = (len(present) + columns - 1) // columns
    fig,axes=plt.subplots(rows,columns,figsize=(6*columns+1,5*rows+1),
                          sharex=True,sharey=True,squeeze=False)
    handles={}
    for method,axis in zip(present,axes.flat):
        doses=groups[kind,context,mode,method]
        matrices=[]
        baseline=trace_values(doses[min(doses)],"cumulative_pseudo_regret")
        for i,(rho,subset) in enumerate(sorted(doses.items())):
            values=trace_values(subset,"cumulative_pseudo_regret")
            if incremental:values=values-baseline
            matrices.append(values)
            stats=seed_summary(values,band);x=np.arange(1,values.shape[1]+1)
            color=DOSE_COLORS[i]
            handles[rho]=plot_summary(axis,x,stats,color=color,linestyle=DOSE_LINES[i],
                marker=("o","s","^","D")[i],label=fr"$\rho={rho:g}$",
                lw=2.2,marker_slot=i,marker_slots=len(doses),time_trace=True)
        if len(matrices)>1 and all(np.allclose(matrices[0],v,atol=1e-10,rtol=1e-10) for v in matrices[1:]):
            axis.text(.04,.93,"Strength curves coincide",transform=axis.transAxes,va="top",fontsize=11,
                      bbox=dict(fc="white",ec="none",alpha=.9))
        axis.set(title=LABELS[method].replace(": ", ":\n"),xlabel="Training episode",
                 ylabel="Added pseudo-regret vs mildest\n(symlog)" if incremental else "Cumulative pseudo-regret (symlog)")
        axis.set_yscale("symlog",linthresh=.01)
        if incremental:axis.axhline(0,color="#555555",lw=.8)
        axis.tick_params(labelleft=True,labelbottom=True)
        axis.grid(alpha=.23)
    for axis in list(axes.flat)[len(present):]:axis.set_axis_off()
    pad_plot_axis(axes[0,0])
    _title(fig,kind,context,mode,("Additional cost of stronger nuisance · paired subtraction within each seed" if incremental else
                               "Physical diversion strength · identical domain, useful routes and task signals"))
    fig.legend(list(handles.values()),[fr"$\rho={r:g}$" for r in handles],loc="lower center",
               bbox_to_anchor=(.5,.065),ncol=4,frameon=False,fontsize=14)
    fig.text(.5,.035,"Shared symlog y scale (linear within ±0.01). Markers identify measured episodes; no artificial offsets.",ha="center",fontsize=11)
    fig.text(.5,.012,band_description(band),ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.14,1,.92),h_pad=2.3,w_pad=1.4)
    return fig


def draw_strength_diagnostics(rows, kind, context=0, mode=1, plt=None, band="sd"):
    """Exposure, information and terminal success distinguish observed mechanisms."""
    if plt is None:
        import matplotlib.pyplot as plt
    rows=[r for r in rows if (r["environment"],r["context"],r["mode"])==(kind,context,mode)]
    fig,axes=plt.subplots(2,3,figsize=(19,11))
    handles={}
    for method in METHODS:
        subset=[r for r in rows if r["method"]==method]
        if not subset:continue
        levels=sorted({r["rho"] for r in subset})
        for axis,metric,label in zip(axes.flat,METRICS,METRIC_LABELS):
            matrix=np.array([[r[metric] for r in sorted(subset,key=lambda r:r["seed"]) if r["rho"]==rho] for rho in levels]).T
            stats=seed_summary(matrix,band)
            color,line,marker=STYLES[method]
            handles[method]=plot_summary(axis,levels,stats,color=color,linestyle=line,
                marker=marker,label=LABELS[method],lw=2.6 if method=="ceac" else 1.9)
            axis.set(xlabel=DOSE_LABELS[kind]+r" $\rho$",ylabel=label,xticks=levels)
            axis.grid(alpha=.23)
    for axis in axes.flat:
        pad_plot_axis(axis)
    _title(fig,kind,context,mode,"Why performance differs: value gap, successful arrivals, wasted steps and useful evidence")
    _legend(fig,handles,METHODS,band,bottom=.045)
    fig.tight_layout(rect=(0,.18,1,.92),h_pad=2.2,w_pad=1.7)
    return fig



# These labels describe the recorded intervention, not an inferred mechanism.
DOSE_LABELS = {
    "deepsea": "Undertow: probability of losing progress",
    "frozenlake": "False Beacon: probability of one dwell step",
}


def _endpoint_grid(rows):
    """Validate complete paired endpoints before subtracting methods or doses."""
    groups = defaultdict(dict)
    for row in rows:
        condition = row["environment"], row["context"], row["mode"]
        key = row["method"], row["rho"], row["seed"]
        if key in groups[condition]:
            raise ValueError("Duplicate strength endpoint")
        if row["method"] not in METHODS or not np.isfinite(
                [row[k] for k in ("rho", "regret", "last_gap", "episodes")]).all():
            raise ValueError("Unknown method or nonfinite strength endpoint")
        groups[condition][key] = row
    for grid in groups.values():
        methods = {key[0] for key in grid}
        levels = {key[1] for key in grid}
        seeds = {key[2] for key in grid}
        if len(grid) != len(methods)*len(levels)*len(seeds):
            raise ValueError("Strength endpoints require paired seeds at every method and dose")
        if len({r["episodes"] for r in grid.values()}) != 1:
            raise ValueError("Strength endpoints require equal episode budgets")
    return groups


def ceac_advantage_statistics(rows):
    """Positive = CEAC better; subtract within seed BEFORE computing variation.

    extra_regret_saved is the change in CEAC's advantage relative to the
    mildest dose, not the absolute advantage already present at that dose.
    """
    paired = []
    for (kind, context, mode), grid in sorted(_endpoint_grid(rows).items()):
        if not any(m == "ceac" for m, _, _ in grid):
            raise ValueError("CEAC comparisons require a recorded CEAC reference")
        mild = min(rho for _, rho, _ in grid)
        for (method, rho, seed), row in sorted(grid.items()):
            if method == "ceac":
                continue
            ceac = grid["ceac", rho, seed]
            baseline = grid[method, mild, seed]["regret"] - grid["ceac", mild, seed]["regret"]
            saved = row["regret"] - ceac["regret"]
            paired.append(dict(environment=kind, context=context, mode=mode, method=method,
                rho=rho, seed=seed, episodes=row["episodes"], mildest_rho=mild,
                regret_saved=saved, extra_regret_saved=saved-baseline,
                gap_reduction=row["last_gap"]-ceac["last_gap"]))
    return paired


def summarize_ceac_advantages(paired):
    groups = defaultdict(list)
    for row in paired:
        key = tuple(row[k] for k in ("environment", "context", "mode", "method", "rho"))
        groups[key].append(row)
    summaries = []
    for key, subset in sorted(groups.items()):
        summary = dict(zip(("environment", "context", "mode", "method", "rho"), key))
        summary["n"] = len(subset)
        for metric in ("regret_saved", "extra_regret_saved", "gap_reduction"):
            values = np.array([r[metric] for r in sorted(subset, key=lambda r:r["seed"])])
            summary[metric] = dict(mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)) if len(values)>1 else None,
                minimum=float(values.min()), maximum=float(values.max()),
                ceac_better=int(np.sum(values>1e-10)),
                tied=int(np.sum(np.abs(values)<=1e-10)),
                ceac_worse=int(np.sum(values < -1e-10)))
        summaries.append(summary)
    return summaries


def draw_pathology_response(rows, context=0, mode=1, plt=None, band="sd"):
    """Headline: absolute cost, dose-induced cost, and remaining value gap."""
    if plt is None:
        import matplotlib.pyplot as plt
    grids = _endpoint_grid([r for r in rows if (r["context"], r["mode"]) == (context, mode)])
    if not grids:
        raise ValueError("No recorded strength endpoints")
    fig, axes = plt.subplots(len(grids), 3, figsize=(18, 4.2*len(grids)+2.5), sharey="col", squeeze=False)
    handles = {}
    for i, ((kind, _, _), grid) in enumerate(sorted(grids.items())):
        levels = sorted({rho for _, rho, _ in grid})
        seeds = sorted({seed for _, _, seed in grid})
        for method in METHODS:
            if not any(m == method for m, _, _ in grid):
                continue
            raw = np.array([[grid[method,rho,seed]["regret"] for rho in levels] for seed in seeds])
            gaps = np.array([[grid[method,rho,seed]["last_gap"] for rho in levels] for seed in seeds])
            for axis, matrix in zip(axes[i], (raw, raw-raw[:, :1], gaps)):
                stats = seed_summary(matrix, band)
                color, line, marker = STYLES[method]
                handles[method] = plot_summary(axis,levels,stats,color=color,linestyle=line,
                    marker=marker,label=LABELS[method],lw=2.6 if method=="ceac" else 1.9)
        for axis, title, label in zip(axes[i],
                ("Total exploration cost", "Extra cost as pathology increases", "Remaining policy error"),
                ("Cumulative pseudo-regret (symlog)", "Regret minus mildest dose (linear)",
                 "Last 32-episode expected value gap")):
            axis.set(title=title, xlabel=DOSE_LABELS[kind]+r" $\rho$", ylabel=label, xticks=levels)
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=10, labelleft=True)
            axis.xaxis.label.set_size(11)
            axis.yaxis.label.set_size(12)
            axis.title.set_size(14)
        axes[i,0].set_yscale("symlog", linthresh=.01)
        # Signed added-cost bands are intentionally untruncated.
        axes[i,1].axhline(0, color="#555555", lw=.8)
        axes[i,0].text(0, 1.19, ENVIRONMENT_LABELS[kind], transform=axes[i,0].transAxes,
                       fontsize=16, fontweight="bold")
        axes[i,2].text(.98,.77,f"{len(seeds)} seeds; {next(iter(grid.values()))['episodes']:,} episodes",
                       ha="right",va="top",transform=axes[i,2].transAxes,fontsize=10)
    for axis in axes[0]:
        pad_plot_axis(axis)
    fig.suptitle(f"Increasing pathology: overall failure versus additional harm · mode {mode} · context {context}",
                 fontsize=19, fontweight="bold", y=.995)
    fig.text(.5,.953,"Shared y scales within columns. Lower is better. False Beacon varies activation probability, not breadth.",
             ha="center",fontsize=12)
    _legend(fig, handles, METHODS, band, bottom=.045)
    fig.tight_layout(rect=(0,.17,1,.98), h_pad=3.2, w_pad=2.5)
    return fig


def draw_endpoint_comparison(rows, kind, context=0, mode=1, plt=None, band="sd", rho=None):
    """One row per method makes exact ties visible without moving any values."""
    if plt is None:
        import matplotlib.pyplot as plt
    selected = [r for r in rows if (r["environment"],r["context"],r["mode"]) == (kind,context,mode)]
    grids = _endpoint_grid(selected)
    if not grids:
        raise ValueError("No recorded strength endpoints")
    grid = grids[kind,context,mode]
    levels = sorted({p for _,p,_ in grid})
    rho = levels[-1] if rho is None else rho
    if rho not in levels:
        raise ValueError("Requested strength was not recorded")
    seeds = sorted({seed for _,_,seed in grid})
    methods = [m for m in METHODS if any(a==m for a,_,_ in grid)]
    fig, axes = plt.subplots(1,3,figsize=(20,7),sharey=True)
    metrics = ("regret","added_regret","last_gap")
    for col,(axis,metric,title) in enumerate(zip(axes,metrics,
            ("Total exploration cost", "Extra cost above mildest dose", "Remaining policy error"))):
        for i,method in enumerate(methods):
            values = np.array([grid[method,rho,seed]["regret"]-grid[method,levels[0],seed]["regret"]
                               if metric=="added_regret" else grid[method,rho,seed][metric] for seed in seeds])
            stats = seed_summary(values[:,None],band)
            mean = float(stats["mean"][0])
            spread = None if stats["lower"] is None else np.array(
                [[mean-stats["lower"][0]], [stats["upper"][0]-mean]])
            color,_,marker = STYLES[method]
            axis.errorbar(mean,i,xerr=spread,fmt=marker,color=color,ms=7,mfc="white",
                          capsize=3,lw=1.5,label=LABELS[method],zorder=3)
            # Numeric columns distinguish nearly equal results even on log axes.
            digits = 4 if metric=="last_gap" else 3
            numeric = f"{mean:.{digits}f}"
            if stats["lower"] is not None:
                if band=="sd":
                    numeric += f' ± {stats["sample_sd"][0]:.{digits}f}'
                else:
                    numeric += f' [{stats["lower"][0]:.{digits}f}, {stats["upper"][0]:.{digits}f}]'
            axis.text(1.03,i,numeric,transform=axis.get_yaxis_transform(),va="center",fontsize=10)
            if i%2==0:axis.axhspan(i-.5,i+.5,color="#f1f3f6",zorder=0)
        if col==0:axis.set_xscale("symlog",linthresh=.01)
        axis.set(title=title,xlabel=("Cumulative pseudo-regret (symlog)",
            "Regret minus mildest dose (linear)","Last 32-episode expected value gap")[col])
        axis.axvline(0,color="#777777",lw=.8,zorder=1)
        axis.set_yticks(range(len(methods)),[LABELS[m] for m in methods])
        axis.tick_params(axis="y",length=0,labelsize=11)
        axis.spines[["top","right","left"]].set_visible(False)
        axis.grid(axis="x",alpha=.2)
        pad_plot_axis(axis,coordinate="x",fraction=.09)
    axes[0].set_ylim(len(methods)-.5,-.5)
    fig.suptitle(ENVIRONMENT_LABELS[kind]+f" · mode {mode} · context {context}",
                 fontsize=21,fontweight="bold",y=.99)
    fig.text(.5,.91,f"Separate rows preserve ties · recorded strength ρ={rho:g} · "
             f"{len(seeds)} seeds × {next(iter(grid.values()))['episodes']:,} episodes",
             ha="center",fontsize=12)
    fig.text(.5,.09,"Lower is better. Extra cost uses within-seed subtraction relative to "
             +f"ρ={levels[0]:g}. Points are seed means; no offsets in metric values.",ha="center",fontsize=11)
    fig.text(.5,.045,"Error bars: "+("±1 sample SD" if band=="sd" else "observed seed range")
             +"; descriptive, not confidence intervals. No error bar for a single seed.",ha="center",fontsize=11)
    fig.subplots_adjust(left=.205,right=.88,bottom=.23,top=.81,wspace=1.2)
    return fig


def draw_ceac_advantage(rows, context=0, mode=1, plt=None, band="sd"):
    """All comparator/dose pairs, including CEAC losses and oracle references."""
    if plt is None:
        import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    selected = [r for r in rows if (r["context"],r["mode"]) == (context,mode)]
    summaries = summarize_ceac_advantages(ceac_advantage_statistics(selected))
    if not summaries:
        raise ValueError("No recorded CEAC comparisons")
    kinds = sorted({r["environment"] for r in summaries})
    limit = max(1., max(abs(r["regret_saved"]["mean"]) for r in summaries))
    norm = SymLogNorm(linthresh=1., vmin=-limit, vmax=limit, base=10)
    fig, axes = plt.subplots(1,len(kinds),figsize=(9.5*len(kinds),7),squeeze=False)
    cmap = plt.get_cmap("PuOr")
    for kind, axis in zip(kinds, axes.flat):
        subset = [r for r in summaries if r["environment"]==kind]
        levels = sorted({r["rho"] for r in subset})
        methods = [m for m in METHODS if any(r["method"]==m for r in subset)]
        lookup = {(r["method"],r["rho"]):r for r in subset}
        matrix = np.array([[lookup[m,rho]["regret_saved"]["mean"] for rho in levels] for m in methods])
        axis.imshow(matrix, norm=norm, cmap=cmap, aspect="auto")
        for i, method in enumerate(methods):
            for j, rho in enumerate(levels):
                record = lookup[method,rho]; stats = record["regret_saved"]
                spread = "n/a" if stats["sample_sd"] is None else f'{stats["sample_sd"]:.2f}'
                rgba = cmap(norm(stats["mean"]))
                luminance = sum(w*c for w,c in zip((.2126,.7152,.0722),rgba[:3]))
                axis.text(j,i,f'{stats["mean"]:+.2f}\n± {spread}',ha="center",va="center",fontsize=10,
                    color="white" if luminance<.45 else "#161616",
                    fontweight="bold" if stats["ceac_better"]==record["n"] else "normal")
        axis.set(xticks=range(len(levels)), xticklabels=[f"{p:g}" for p in levels],
                 yticks=range(len(methods)), yticklabels=[LABELS[m] for m in methods],
                 xlabel=DOSE_LABELS[kind]+r" $\rho$", title=ENVIRONMENT_LABELS[kind])
        axis.tick_params(labelsize=11)
        axis.xaxis.label.set_size(11)
        axis.set_xticks(np.arange(-.5,len(levels),1),minor=True)
        axis.set_yticks(np.arange(-.5,len(methods),1),minor=True)
        axis.grid(which="minor",color="white",linewidth=2)
        axis.tick_params(which="minor",bottom=False,left=False)
    fig.suptitle(f"When CEAC saves exploration cost · mode {mode} · context {context}",
                 fontsize=20,fontweight="bold",y=.99)
    counts=sorted({r["n"] for r in summaries})
    fig.text(.5,.93,"Comparator regret − CEAC regret: positive favors CEAC; negative favors comparator. "
             +f"Paired seeds/cell: {', '.join(map(str,counts))}.",
             ha="center",fontsize=12)
    fig.text(.5,.10,"Cells: paired seed mean ± sample SD. Bold: CEAC lower in every recorded pair; not a significance test.",
             ha="center",fontsize=11)
    fig.text(.5,.065,"CEAC/IDS know candidate models; primitive controls learn transitions. Oracle additionally knows the true mode.",
             ha="center",fontsize=10)
    fig.text(.5,.03,"Last-window value ties can coexist with different cumulative costs. Colors use a symmetric log scale.",
             ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.16,1,.88),w_pad=3)
    return fig


def ceac_comparison_section(rows, context=None, mode=None):
    """Data-derived assessment shared by notebook and standard report."""
    selected = [r for r in rows if (context is None or r["context"]==context)
                and (mode is None or r["mode"]==mode)]
    summaries = summarize_ceac_advantages(ceac_advantage_statistics(selected))
    text = ["## Where CEAC performs better", "",
        "Positive saved regret means comparator minus CEAC. The additional advantage subtracts "
        "the same contrast at the mildest dose, within each seed. It separates an advantage "
        "already present at mild pathology from one that grows with pathology. "
        "All uncertainties below are sample SD across paired seeds, not confidence intervals. "
        "Bold entries favor CEAC in every recorded pair; this is descriptive, not a significance test.", "",
        "False Beacon currently varies the probability of one dwell step, not beacon breadth. "
        "The full-support and excluded-support Bayesian treatments remain distinct. "
        "CEAC/IDS have candidate-model access; the primitive controls learn from transitions. "
        "These are finite-budget comparisons, not evidence that a learned evidence critic caused the advantage.", ""]
    def fmt(stats, digits=2):
        spread = "n/a" if stats["sample_sd"] is None else f'{stats["sample_sd"]:.{digits}f}'
        value = f'{stats["mean"]:+.{digits}f} ± {spread}'
        return f"**{value}**" if stats["minimum"]>1e-10 else value
    for (kind, c, m), grid in sorted(_endpoint_grid(selected).items()):
        levels = sorted({rho for _,rho,_ in grid})
        def mean(method, rho, metric):
            return float(np.mean([r[metric] for (a,p,_),r in grid.items() if a==method and p==rho]))
        ceac_low,ceac_high = [mean("ceac",p,"regret") for p in (levels[0],levels[-1])]
        text += [f"### {ENVIRONMENT_LABELS[kind]} · mode {m} · context {c}", "",
            f"CEAC cumulative regret: {ceac_low:.3f} at rho={levels[0]:g} to {ceac_high:.3f} "
            f"at rho={levels[-1]:g}; strongest-dose last-window value gap: "
            f"{mean('ceac',levels[-1],'last_gap'):.4f}.", ""]
        if any(method=="ids" for method,_,_ in grid):
            ids_high = mean("ids",levels[-1],"regret")
            relation = "lower" if ids_high<ceac_high-1e-10 else "higher" if ids_high>ceac_high+1e-10 else "equal"
            text += [f"Full-support IDS has {relation} cumulative regret ({ids_high:.3f}) at the strongest dose; "
                     f"its last-window value gap is {mean('ids',levels[-1],'last_gap'):.4f}.", ""]
        text += [f"Strongest-dose comparison (rho={levels[-1]:g}); positive numbers favor CEAC.", "",
            "| Comparator | Regret saved | Additional advantage over mild dose | Last value-gap reduction | Pairs: CEAC lower / tied / higher regret |",
            "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            match = [r for r in summaries if (r["environment"],r["context"],r["mode"],r["method"],r["rho"])
                     ==(kind,c,m,method,levels[-1])]
            if not match:
                continue
            record=match[0]; saved=record["regret_saved"]
            text.append(f'| {LABELS[method]} | {fmt(saved)} | {fmt(record["extra_regret_saved"])} | '
                        f'{fmt(record["gap_reduction"],4)} | '
                        f'{saved["ceac_better"]} / {saved["tied"]} / {saved["ceac_worse"]} |')
        text += [""]
    return text


def make_strength_figures(destination, runs, plt, band="sd"):
    groups=strength_groups(runs)
    if not groups:return []
    rows=strength_statistics(runs)
    artifacts=[]
    for context,mode in sorted({k[1:3] for k in groups}):
        for family,draw in (("response",draw_pathology_response),("ceac_advantage",draw_ceac_advantage)):
            if family=="ceac_advantage" and not any(k[1:]==(context,mode,"ceac") for k in groups):
                continue
            fig=draw(rows,context=context,mode=mode,plt=plt,band=band)
            for extension in ("png","pdf"):
                path=Path(destination)/f"pathology_{family}_context{context}_mode{mode}.{extension}"
                fig.savefig(path,dpi=150,facecolor="white",bbox_inches="tight")
                artifacts.append(path)
            plt.close(fig)
    for kind,context,mode in sorted({k[:3] for k in groups}):
        for family,draw,data,extra in (
            ("overlay",draw_method_overlay,groups,{}),
            ("learning",draw_strength_learning,groups,{}),
            ("added_regret",draw_strength_learning,groups,{"incremental":True}),
            ("diagnostics",draw_strength_diagnostics,rows,{}),
            ("endpoints",draw_endpoint_comparison,rows,{}),
            ("prior",draw_method_overlay,groups,{"prior":True})):
            if family=="prior" and (kind,context,mode,"ids_zero") not in groups:continue
            fig=draw(data,kind,context,mode,plt,band,**extra)
            for extension in ("png","pdf"):
                path=Path(destination)/f"pathology_{family}_{kind}_context{context}_mode{mode}.{extension}"
                fig.savefig(path,dpi=150,facecolor="white",bbox_inches="tight")
                artifacts.append(path)
            plt.close(fig)
    return artifacts


def report_section(rows):
    if not rows:return []
    text=["", "## Physical pathology strength: matched method overlays", "",
          "The standard figures overlay UCB-Hoeffding Q-learning, optimistic softmax POMD, "
          "IDAC (frequentist occupancy-information actor-critic), Bayesian IDS with full support, "
          "Bayesian IDS with a misspecified prior, CEAC and the offline oracle at each strength. "
          "Both Bayesian arms use the same algorithm: full support is p0=(0.5,0.5); "
          "the restricted prior is p0=(1,0). The latter excludes the true model in mode 1; "
          "mode 0 is its correctly specified negative control. Prior support is held fixed "
          "while physical diversion strength varies. IDAC optimizes an occupancy-entropy ratio; "
          "it is not a frequentist implementation of the same posterior IDS objective.", "",
          "Only the mode-independent diversion probability rho changes (0.05, 0.25, 0.50, 0.75). "
          "Interior probabilities preserve the same reachable state/action domain. Geometry, "
          "horizon, four nuisance symbols, useful-route values, probe KL and reward ranges are "
          "checked across settings before training. A DeepSea nuisance move loses rightward "
          "progress with probability rho; otherwise it advances. On the first reed visit, "
          "False Beacon adds one forced physical dwell step with probability rho. Both branches "
          "have zero task KL. The beacon state records pending/completed dwell; seed routes "
          "resume their intended leg after the wait.", "",
          r"$$P_{m,\rho}(\cdot\mid s,a_N)=(1-\rho)P_{\mathrm{pass}}(\cdot\mid s,a_N)"
          r"+\rho P_{\mathrm{divert}}(\cdot\mid s,a_N).$$", "",
          "The report automatically emits method overlays, STEERING-style strength curves in "
          "separate method panels, paired added-regret traces, and exposure/value/success diagnostics. "
          "Training traces use shared symlog axes, linear within ±0.01, to keep small costs visible. "
          "Markers are drawn above the lines on staggered recorded episodes; no values are shifted. "
          "A strongest-dose endpoint view gives each method its own row, including exact ties. Raw per-seed summaries "
          "and paired added regret are saved in pathology_statistics.json. No monotone deterioration "
          "or algorithm ranking is imposed. Training budgets match in episodes; physical steps are "
          "recorded separately. CEAC/IDS have candidate-model access that the primitive controls lack.", "",
          "| Environment | Mode | Method | Regret: mild / strong | Paired added regret (mean ± sample SD) | Last value gap: mild / strong | POMD all-zero costs, strong |",
          "|---|---:|---|---:|---:|---:|---:|"]
    for kind,context,mode,method in sorted({(r["environment"],r["context"],r["mode"],r["method"]) for r in rows}):
        subset=[r for r in rows if (r["environment"],r["context"],r["mode"],r["method"])==(kind,context,mode,method)]
        lo=min(r["rho"] for r in subset);hi=max(r["rho"] for r in subset)
        low=[r for r in subset if r["rho"]==lo];high=[r for r in subset if r["rho"]==hi]
        mean=lambda rs,k:float(np.mean([r[k] for r in rs]))
        d=[r["added_regret"] for r in high]
        spread=f"{np.std(d,ddof=1):.3f}" if len(d)>1 else "n/a"
        saturation=f'{mean(high,"pomd_zero_fraction"):.1%}' if method=="pomd" else "n/a"
        text.append(f'| {kind} (c{context}) | {mode} | {LABELS[method]} | {mean(low,"regret"):.3f} / {mean(high,"regret"):.3f} | {np.mean(d):.3f} ± {spread} | {mean(low,"last_gap"):.4f} / {mean(high,"last_gap"):.4f} | {saturation} |')
    text += ["", "Diagnostic flags describe observations, not causal proofs: POMD_BONUS_SATURATION "
             "means at least 90% of episodes had all optimistic costs zero; PHYSICAL_NUISANCE_EXPOSURE "
             "records lost progress/dwell; FINITE_BUDGET_VALUE_GAP means a final 32-episode mean "
             "expected gap above 0.01. A plateau already present at mild strength is not evidence "
             "that increasing pathology caused it. PRIOR_SUPPORT_EXCLUDED is the deliberately "
             "misspecified zero-prior treatment, explicitly labeled alongside full-support Bayesian IDS "
             "in every main figure. A dedicated prior comparison also isolates that assumption.", ""]
    return text
