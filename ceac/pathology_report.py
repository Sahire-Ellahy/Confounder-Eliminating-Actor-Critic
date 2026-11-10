"""Matched physical-strength figures and diagnostics, from recorded runs only."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from .benchmarks import STRENGTH_PROFILES
from .visualization import (ENVIRONMENT_LABELS, LABELS, STYLES, seed_summary,
                            trace_values, plot_seed_trace, band_description)

METHODS = ("ucb_q", "pomd", "idac", "ids", "ceac", "oracle")
DOSE_COLORS = ("#0058c7", "#008747", "#df7400", "#c90078")
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
        axis.set_yscale("symlog",linthresh=1)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.23)
        axis.tick_params(labelleft=True,labelbottom=True)
    for axis in list(axes.flat)[len(levels):]:
        axis.set_axis_off()
    subtitle = ("Excluded Bayesian prior support is a separate assumption treatment" if prior else
                "Optimism vs IDAC (frequentist) vs Bayesian IDS · matched finite training budget")
    _title(fig,kind,context,mode,subtitle)
    _legend(fig,handles,present,band)
    fig.tight_layout(rect=(0,.15,1,.92),h_pad=2.3,w_pad=1.5)
    return fig


def draw_strength_learning(groups, kind, context=0, mode=1, plt=None, band="sd", incremental=False):
    """STEERING layout: training on x, regret on y, strength as curve color."""
    if plt is None:
        import matplotlib.pyplot as plt
    present=[m for m in METHODS if (kind,context,mode,m) in groups]
    fig,axes=plt.subplots(2,3,figsize=(19,11),sharex=True,sharey=True,squeeze=False)
    handles={}
    maximum=0.
    minimum=0.
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
            handles[rho],=axis.plot(x,stats["mean"],color=color,ls=DOSE_LINES[i],
                lw=4 if method=="ceac" else 2,marker=("o","s","^","D")[i],ms=4,
                markevery=(i*max(1,len(x)//40),max(1,len(x)//8)),markerfacecolor="white",
                label=fr"$\rho={rho:g}$")
            if stats["lower"] is not None:
                axis.fill_between(x,stats["lower"],stats["upper"],color=color,alpha=.12,lw=0)
            maximum=max(maximum,float(np.max(stats["upper"] if stats["upper"] is not None else stats["mean"])))
            minimum=min(minimum,float(np.min(stats["lower"] if stats["lower"] is not None else stats["mean"])))
        if len(matrices)>1 and all(np.allclose(matrices[0],v,atol=1e-10,rtol=1e-10) for v in matrices[1:]):
            axis.text(.04,.93,"Strength curves coincide",transform=axis.transAxes,va="top",fontsize=11,
                      bbox=dict(fc="white",ec="none",alpha=.9))
        axis.set(title=LABELS[method],xlabel="Training episode",
                 ylabel="Added pseudo-regret\nvs mildest strength" if incremental else "Cumulative pseudo-regret")
        if incremental:axis.axhline(0,color="#555555",lw=.8)
        axis.tick_params(labelleft=True,labelbottom=True)
        axis.grid(alpha=.23)
    for axis in list(axes.flat)[len(present):]:axis.set_axis_off()
    axes[0,0].set_ylim(minimum*1.13 if incremental else 0,max(1,maximum)*1.13)
    _title(fig,kind,context,mode,("Additional cost of stronger nuisance · paired subtraction within each seed" if incremental else
                               "Physical diversion strength · identical domain, useful routes and task signals"))
    fig.legend(list(handles.values()),[fr"$\rho={r:g}$" for r in handles],loc="lower center",
               bbox_to_anchor=(.5,.065),ncol=4,frameon=False,fontsize=14)
    fig.text(.5,.035,"Shared linear y scale. Coincident curves remain coincident; no artificial offsets.",ha="center",fontsize=11)
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
            handles[method],=axis.plot(levels,stats["mean"],color=color,ls=line,marker=marker,
                                      lw=4 if method=="ceac" else 2,ms=7,markerfacecolor="white")
            if stats["lower"] is not None:
                axis.fill_between(levels,stats["lower"],stats["upper"],color=color,alpha=.12,lw=0)
            axis.set(xlabel=r"Physical diversion probability $\rho$",ylabel=label,xticks=levels)
            axis.set_ylim(bottom=0)
            axis.grid(alpha=.23)
            if metric=="success":axis.set_ylim(0,1)
    _title(fig,kind,context,mode,"Why performance differs: value gap, successful arrivals, wasted steps and useful evidence")
    _legend(fig,handles,METHODS,band,bottom=.045)
    fig.tight_layout(rect=(0,.15,1,.92),h_pad=2.2,w_pad=1.7)
    return fig


def make_strength_figures(destination, runs, plt, band="sd"):
    groups=strength_groups(runs)
    if not groups:return []
    rows=strength_statistics(runs)
    artifacts=[]
    for kind,context,mode in sorted({k[:3] for k in groups}):
        for family,draw,data,extra in (
            ("overlay",draw_method_overlay,groups,{}),
            ("learning",draw_strength_learning,groups,{}),
            ("added_regret",draw_strength_learning,groups,{"incremental":True}),
            ("diagnostics",draw_strength_diagnostics,rows,{}),
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
          "IDAC (frequentist occupancy-information actor-critic), full-support Bayesian IDS, CEAC "
          "and the offline oracle at each strength. IDAC optimizes an occupancy-entropy ratio; "
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
          "separate method panels, paired added-regret traces, and exposure/value/success diagnostics. Raw per-seed summaries "
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
             "misspecified zero-prior treatment, shown separately from full-support Bayesian IDS.", ""]
    return text
