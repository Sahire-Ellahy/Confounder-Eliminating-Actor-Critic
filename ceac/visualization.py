"""Scientific figures from verified metric rows; no learner/environment calls."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import numpy as np

ENVIRONMENT_LABELS = {"deepsea": "Undertow DeepSea", "frozenlake": "False Beacon Frozen Lake"}


LABELS = {
    "ceac": "CEAC (exact-model single loop)", "ceac_actor": "CEAC: actor-only ablation",
    "oracle": "Offline Graves–Lai oracle", "ids": "Bayesian IDS: full support",
    "idac": "IDAC (frequentist actor-critic)", "ucb_q": "UCB Q-learning (Jin et al.)",
    "pomd": "Optimistic POMD (softmax)",
    "ids_rare": "IDS: rare prior (0.001)", "ids_zero": "Bayesian IDS: misspecified prior",
}
STYLES = {
    "ceac": ("#cf202f", "-", "o"), "ceac_actor": ("#8b3e2f", ":", "v"),
    "oracle": ("#161616", "--", "*"), "ucb_q": ("#07854d", "-.", "x"),
    "idac": ("#7c2fc5", (0, (5, 1, 1, 1)), "^"), "pomd": ("#ad5700", "--", "s"), "ids": ("#0058c7", "--", "D"),
    "ids_rare": ("#bd319b", "-.", "s"), "ids_zero": ("#c90078", ":", "X"),
}


def rolling_mean(values, width=16):
    values = np.asarray(values, dtype=float)
    total = np.r_[0., np.cumsum(values)]
    end = np.arange(1, len(values)+1)
    start = np.maximum(0, end-width)
    return (total[end]-total[start])/(end-start)



def band_description(band):
    if band == "sd":
        return "Lines: seed means. Bands: mean ±1 sample SD across seeds, not a confidence interval. No band for a single seed."
    if band == "seed_range":
        return "Lines: seed means. Bands: observed seed range, not a confidence interval. No band for a single seed."
    raise ValueError("Unknown descriptive band")


def seed_summary(values, band="sd"):
    """Pointwise descriptive variation; rows are independent training seeds."""
    band_description(band)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty, finite seeds-by-time matrix")
    mean = values.mean(axis=0)
    minimum, maximum = values.min(axis=0), values.max(axis=0)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else None
    lower = upper = None
    if sd is not None:
        lower, upper = (mean-sd, mean+sd) if band == "sd" else (minimum, maximum)
    return dict(n=len(values), mean=mean, sample_sd=sd, minimum=minimum,
                maximum=maximum, lower=lower, upper=upper)


def trace_values(runs, field):
    """Align seed curves; smooth within seed before any aggregation."""
    if not runs:
        raise ValueError("No seed runs")
    if len({r["result"]["seed"] for r in runs}) != len(runs):
        raise ValueError("Duplicate seed in a trace group")
    curves = []
    for run in sorted(runs, key=lambda r: r["result"]["seed"]):
        rows = run["metrics"]
        if [m["episode"] for m in rows] != list(range(len(rows))):
            raise ValueError("Trace episode indices are not contiguous")
        values = ([m["diagnostics"]["posterior"][1] for m in rows]
                  if field == "mode1_posterior" else [m[field] for m in rows])
        curves.append(rolling_mean(values) if field == "expected_return" else values)
    if len({len(v) for v in curves}) != 1:
        raise ValueError("Seed curves have unequal episode budgets")
    return np.asarray(curves, dtype=float)


def plot_summary(axis, x, stats, *, color, linestyle, marker, label, lw=2,
                 marker_slot=0, marker_slots=1, time_trace=False):
    """Draw markers above every line, on measured samples without offsets.

    A separate scatter layer prevents later mean lines from hiding coincident
    curves' markers. For time traces, stagger the sampled episodes by method
    (or dose). Dose-response endpoints stay at their exact measured x values.
    """
    x = np.asarray(x)
    line, = axis.plot(x, stats["mean"], color=color, linestyle=linestyle,
                      lw=lw, marker=marker, markevery=[], markersize=6,
                      markerfacecolor="none", label=label, zorder=3)
    if stats["lower"] is not None:
        axis.fill_between(x, stats["lower"], stats["upper"], color=color,
                          alpha=.10, lw=0, zorder=1)
    indices = np.arange(len(x))
    if time_trace:
        step = max(marker_slots, len(x)//8)
        offset = marker_slot * step // marker_slots
        indices = np.arange(offset, len(x), step)
    if len(indices) == 0:
        indices = np.array([marker_slot % len(x)])
    # Hollow symbols keep interiors transparent when endpoints coincide.
    marker_colors = ({"color":color} if marker in {"x","+"} else
                     {"facecolors":"none", "edgecolors":color})
    axis.scatter(x[indices], stats["mean"][indices], marker=marker, s=38,
                 linewidths=1.4, zorder=10, label="_nolegend_", **marker_colors)
    return line


def pad_plot_axis(axis, coordinate="y", fraction=.05):
    """Pad shared data bounds in display scale; include zero and all SD bands.

    On symlog axes linear padding can create a huge negative empty region.
    Transforming first keeps a small visual margin around zero, without
    clipping genuinely negative changes or the lower descriptive band.
    """
    shared = axis.get_shared_y_axes() if coordinate == "y" else axis.get_shared_x_axes()
    limits = [getattr(a.dataLim, "interval"+coordinate)
              for a in shared.get_siblings(axis) if a.axison]
    finite = np.asarray(limits).ravel()
    finite = finite[np.isfinite(finite)]
    low, high = min(0., finite.min(initial=0.)), max(0., finite.max(initial=0.))
    target = axis.yaxis if coordinate == "y" else axis.xaxis
    transform = target.get_transform()
    edges = np.asarray(transform.transform([low, high]), dtype=float)
    span = edges[1]-edges[0]
    if span == 0:
        span = float(transform.transform([getattr(transform, "linthresh", 1.)])[0])
    lower, upper = transform.inverted().transform(edges + np.array([-1, 1])*span*fraction)
    (axis.set_ylim if coordinate == "y" else axis.set_xlim)(lower, upper)


def plot_seed_trace(axis, values, method, band):
    stats = seed_summary(values, band)
    x = np.arange(1, values.shape[1]+1)
    color, style, marker = STYLES[method]
    return plot_summary(axis, x, stats, color=color, linestyle=style, marker=marker,
                        label=LABELS[method], lw=2.6 if method == "ceac" else 1.9,
                        marker_slot=tuple(STYLES).index(method), marker_slots=len(STYLES),
                        time_trace=True)


def export_trace_statistics(root, runs, band="sd"):
    """Save the exact statistics used for plots without changing raw evidence."""
    band_description(band)
    groups = defaultdict(list)
    for run in runs:
        result = run["result"]
        key = (result.get("benchmark", "legacy"), result["environment"],
               result["context"], result["true_mode"], result["method"])
        groups[key].append(run)
    fields = ("expected_return", "cumulative_return", "cumulative_pseudo_regret",
              "cumulative_expected_model_information", "cumulative_observed_llr",
              "cumulative_diagnostic_visits", "cumulative_nuisance_visits",
              "cumulative_risky_visits", "cumulative_diversions", "cumulative_lost_steps", "mode1_posterior")
    path = Path(root)/"trace_statistics.csv.gz"
    with gzip.open(path, "wt", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("benchmark", "environment", "context", "mode", "method",
                         "metric", "episode", "n_seeds", "mean", "sample_sd",
                         "minimum", "maximum", "band", "lower", "upper",
                         "causal_smoothing_width", "independent_unit"))
        for key, subset in sorted(groups.items()):
            for field in fields:
                first = subset[0]["metrics"][0]
                if field == "mode1_posterior":
                    if "posterior" not in first["diagnostics"]:
                        continue
                elif field not in first:
                    continue
                stats = seed_summary(trace_values(subset, field), band)
                for t in range(len(stats["mean"])):
                    numbers = ["" if stats[k] is None else float(stats[k][t])
                               for k in ("mean", "sample_sd", "minimum", "maximum")]
                    edges = ["" if stats[k] is None else float(stats[k][t])
                             for k in ("lower", "upper")]
                    writer.writerow((*key, field, t+1, stats["n"], *numbers, band,
                                     *edges, 16 if field == "expected_return" else 1,
                                     "training_seed"))
    return path


def make_headline_figures(destination, runs, plt, band):
    """Two environments side by side in scope, with focused comparator subsets."""
    selected = [r for r in runs if r["result"].get("benchmark") == "weak_signal_m64"
                and r["result"]["method"] not in RETIRED_METHODS]
    artifacts = []
    for context in sorted({r["result"]["context"] for r in selected}):
        for mode in sorted({r["result"]["true_mode"] for r in selected}):
            relevant = [r for r in selected if r["result"]["context"] == context
                        and r["result"]["true_mode"] == mode]
            methods = (("ceac", "oracle", "idac", "ucb_q", "pomd", "ids")
                       if mode == 0 else ("ceac", "oracle", "ids", "ids_rare", "ids_zero"))
            kinds = sorted({r["result"]["environment"] for r in relevant})
            fig, axes = plt.subplots(len(kinds), 3, figsize=(20, 5*len(kinds)+1.5), squeeze=False)
            handles = {}
            for row, kind in enumerate(kinds):
                subset = [r for r in relevant if r["result"]["environment"] == kind]
                for method in methods:
                    seeds = [r for r in subset if r["result"]["method"] == method]
                    if not seeds:
                        continue
                    for col, field in enumerate(("expected_return", "cumulative_pseudo_regret", "cumulative_observed_llr")):
                        values = trace_values(seeds, field)
                        handles[method] = plot_seed_trace(axes[row, col], values, method, band)
                optimal = subset[0]["metrics"][0]["optimal_return"]
                optimal_line = axes[row, 0].axhline(optimal, color="#222222", lw=1.4, ls=":", alpha=.8)
                max_regret = max(r["metrics"][-1]["cumulative_pseudo_regret"] for r in subset
                                 if r["result"]["method"] in methods)
                if max_regret > .25:
                    axes[row, 1].set_yscale("symlog", linthresh=.25)
                else:
                    from matplotlib.ticker import MaxNLocator
                    axes[row, 1].yaxis.set_major_locator(MaxNLocator(nbins=5))
                axes[row, 1].set_ylim(bottom=0)
                axes[row, 2].axhline(0, color="#444444", lw=.8, alpha=.6)
                for col, axis in enumerate(axes[row]):
                    axis.set_title(ENVIRONMENT_LABELS[kind]+"\n"+
                                   ("Policy value", "Exploration cost", "Observed evidence")[col], pad=12)
                    axis.set_xlabel("Training episode")
                    axis.set_ylabel(("Expected return\n(causal 16-episode mean)",
                                     "Cumulative pseudo-regret"+("\n(symlog scale)" if max_regret > .25 else ""),
                                     "Cumulative observed log-likelihood ratio\n(true mode / alternative; nats)")[col])
                    axis.set_xlim(1, len(subset[0]["metrics"]))
                    axis.grid(True, color="#dfe3e8", lw=.7, alpha=.8)
                    axis.spines[["top", "right"]].set_visible(False)
            fig.legend([*handles.values(), optimal_line], [*[LABELS[m] for m in handles],
                       "Model-optimal value (evaluator)"], loc="lower center",
                       bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=12)
            fig.suptitle("Weak task signal · M=64 · "+("normal mode (0)" if mode == 0 else "shifted mode (1)")+
                         f" · context {context}", fontsize=23, fontweight="bold", y=.995)
            fig.text(.5, .01, band_description(band), ha="center", fontsize=12, color="#444444")
            fig.tight_layout(rect=(0, .16, 1, .955), h_pad=2.5, w_pad=1.5)
            for extension in ("png", "pdf"):
                path = destination/f"headline_weak_signal_m64_context{context}_mode{mode}.{extension}"
                fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
                artifacts.append(path)
            plt.close(fig)
    return artifacts



NUISANCE_PROFILES = {"weak_signal_m4": 4, "weak_signal_m16": 16, "weak_signal_m64": 64}


def nuisance_sensitivity_rows(runs):
    """Paired endpoint comparisons with all non-nuisance settings held fixed.

    Same-seed subtraction precedes uncertainty aggregation. Missing pairs or
    an accidental second experimental change must not become a dose response.
    """
    groups = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        r = run["result"]
        if r.get("benchmark") not in NUISANCE_PROFILES:
            continue
        key = r["environment"], r["context"], r["true_mode"], r["method"]
        size = NUISANCE_PROFILES[r["benchmark"]]
        if r["seed"] in groups[key][size]:
            raise ValueError("Duplicate seed in nuisance sensitivity")
        groups[key][size][r["seed"]] = run

    def fixed_configuration(run, size):
        config = dict(run["manifest"]["configuration"])
        specification = dict(config["benchmark"])
        overrides = dict(specification["configuration_overrides"])
        nuisance_key = "tv_bins" if config["environment"] == "deepsea" else "reed_bins"
        if overrides.pop(nuisance_key) != size:
            raise ValueError("Nuisance size differs from declared benchmark")
        specification.pop("name")
        specification["configuration_overrides"] = overrides
        config["benchmark"] = specification
        return config

    rows = []
    for (kind, context, mode, method), doses in sorted(groups.items()):
        if len(doses) < 2:
            continue
        if 4 not in doses:
            raise ValueError("Nuisance sensitivity needs its M=4 baseline")
        baseline = doses[4]
        for size, seeds in sorted(doses.items()):
            if set(seeds) != set(baseline):
                raise ValueError("Nuisance sensitivity has unmatched seeds")
            for seed, run in sorted(seeds.items()):
                base = baseline[seed]
                if fixed_configuration(run, size) != fixed_configuration(base, 4):
                    raise ValueError("A non-nuisance setting changed in the sensitivity comparison")
                r, b = run["result"], base["result"]
                rows.append(dict(environment=kind, context=context, mode=mode, method=method,
                                 seed=seed, nuisance_bins=size, baseline_nuisance_bins=4,
                                 benchmark=r["benchmark"], episodes=r["episodes"],
                                 cumulative_regret=r["pseudo_regret"],
                                 added_regret=r["pseudo_regret"]-b["pseudo_regret"],
                                 nuisance_visits=r["event_counts"]["nuisance"]))
    return rows


def summarize_nuisance(rows, band="sd"):
    groups = defaultdict(list)
    for row in rows:
        groups[row["environment"], row["context"], row["mode"], row["method"], row["nuisance_bins"]].append(row)
    summaries = []
    for (kind, context, mode, method, size), group in sorted(groups.items()):
        for field in ("cumulative_regret", "added_regret", "nuisance_visits"):
            stats = seed_summary([[r[field]] for r in group], band)
            summaries.append(dict(environment=kind, context=context, mode=mode, method=method,
                                  nuisance_bins=size, metric=field, n_seeds=stats["n"], band=band,
                                  **{key: None if stats[key] is None else float(stats[key][0])
                                     for key in ("mean", "sample_sd", "minimum", "maximum", "lower", "upper")}))
    return summaries


RETIRED_METHODS = frozenset({"confidence", "policy_ucb", "oir_idac", "coverage"})
NUISANCE_TRACE_METHODS = ("idac", "ucb_q", "pomd", "ids", "ceac", "oracle")
NUISANCE_TRACE_TITLES = {
    "idac": "IDAC\nUpstream two-critic softmax update",
    "ucb_q": "UCB Q-learning\nJin et al. (2018), Algorithm 1",
    "pomd": "Optimistic POMD\nSoftmax policy / learned model",
    "ids": "IDS with full prior support\nFinite-model evidence control",
    "ceac": "CEAC\nExact-model single loop",
    "oracle": "Offline Graves–Lai oracle\nTrue mode known from the outset",
}
NUISANCE_TRACE_STYLES = {4: ("#0058c7", "-", "o"),
                         16: ("#e36b00", "--", "s"),
                         64: ("#b80080", ":", "^")}


def nuisance_trace_groups(runs):
    """Matched learning curves, with the same configuration audit as endpoints.

    Never pool modes, benchmarks, or seeds. A missing dose must fail instead of
    silently changing the comparison. Single-dose studies have no dose response.
    """
    runs = [r for r in runs if r["result"]["method"] not in RETIRED_METHODS]
    paired = nuisance_sensitivity_rows(runs)
    eligible = {(r["environment"], r["context"], r["mode"], r["method"]) for r in paired}
    groups = defaultdict(lambda: defaultdict(list))
    for run in runs:
        r = run["result"]
        key = r["environment"], r["context"], r["true_mode"], r["method"]
        if key not in eligible or r.get("benchmark") not in NUISANCE_PROFILES:
            continue
        size = NUISANCE_PROFILES[r["benchmark"]]
        groups[key][size].append(run)
    values = {}
    for key, doses in sorted(groups.items()):
        if set(doses) != set(NUISANCE_TRACE_STYLES):
            raise ValueError("Learning comparison requires all three nuisance doses")
        values[key] = {}
        for size, subset in sorted(doses.items()):
            matrix = trace_values(subset, "cumulative_pseudo_regret")
            for run in subset:
                r = run["result"]
                if len(run["metrics"]) != r["episodes"]:
                    raise ValueError("Learning curve differs from its episode budget")
                if not np.isclose(run["metrics"][-1]["cumulative_pseudo_regret"],
                                  r["pseudo_regret"], rtol=1e-10, atol=1e-10):
                    raise ValueError("Learning curve differs from its recorded endpoint")
            values[key][size] = matrix
    return values


def draw_nuisance_learning(groups, kind, context, mode, plt, band="sd"):
    """STEERING-style traces: time on x, regret on y, dose in the legend.

    These cached doses are observation diversity, not physical strength. Method
    panels share a linear scale; coincident traces are labeled without jitter.
    """
    selected = {m: groups[kind, context, mode, m] for m in NUISANCE_TRACE_METHODS
                if (kind, context, mode, m) in groups}
    if not selected:
        raise ValueError("No matched nuisance learning curves for this environment/mode")
    matrices = [v for doses in selected.values() for v in doses.values()]
    if len({v.shape for v in matrices}) != 1:
        raise ValueError("Method panels require matched seed counts and episode budgets")
    seeds, budget = matrices[0].shape
    columns = min(3, len(selected))
    rows = (len(selected)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(17, 4.2*rows+2.1),
                             sharex=True, sharey=True, squeeze=False)
    maxima = []
    for method, axis in zip(selected, axes.flat):
        if method not in selected:
            axis.set_axis_off()
            continue
        doses = selected[method]
        x = np.arange(1, budget+1)
        endpoint_text = []
        for offset, (size, matrix) in enumerate(sorted(doses.items())):
            stats = seed_summary(matrix, band)
            color, line_style, marker = NUISANCE_TRACE_STYLES[size]
            axis.plot(x, stats["mean"], color=color, ls=line_style, marker=marker,
                      markevery=(offset*max(1, budget//36), max(1, budget//9)),
                      ms=5, markerfacecolor="white", markeredgewidth=1.2,
                      lw=3 if method == "ceac" else 2, label=f"M = {size}", zorder=3)
            maxima.append(float(np.max(stats["mean"] if stats["upper"] is None else stats["upper"])))
            if stats["lower"] is not None:
                axis.fill_between(x, stats["lower"], stats["upper"], color=color,
                                  alpha=.10, lw=0, zorder=1)
            endpoint_text.append(f"{stats['mean'][-1]:.2f}")
        coincident = all(np.allclose(doses[4], v, rtol=1e-10, atol=1e-10) for v in doses.values())
        note = ("All three M curves coincide\n" if coincident else "")
        note += "Final R: " + " / ".join(endpoint_text) + " (M=4/16/64)"
        axis.text(.03, .96, note, transform=axis.transAxes, va="top", fontsize=10,
                  bbox=dict(fc="white", ec="none", alpha=.85, pad=3))
        axis.set_title(NUISANCE_TRACE_TITLES[method], fontsize=15, pad=12)
        axis.set_xlabel("Training episode", fontsize=13)
        axis.set_ylabel("Cumulative pseudo-regret", fontsize=13)
        axis.tick_params(labelbottom=True, labelleft=True, labelsize=11)
        axis.grid(alpha=.22)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(selected):]:
        axis.set_axis_off()
    axes[0,0].set_xlim(1, budget)
    axes[0,0].set_ylim(0, max(1., max(maxima))*1.2)
    handles, labels = next(ax.get_legend_handles_labels() for ax in axes.flat if ax.has_data())
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.067), ncol=3,
               fontsize=14, frameon=False, title="Cached control: nuisance symbol count (not physical strength)",
               title_fontsize=12)
    fig.suptitle(ENVIRONMENT_LABELS[kind]+f" · mode {mode} · context {context}",
                 fontsize=23, fontweight="bold", y=.99)
    fig.text(.5, .935, "Fixed geometry, useful probe and route gap; color changes nuisance diversity only.",
             ha="center", fontsize=12)
    fig.text(.5,.041,f"{seeds} paired seeds per curve; "+band_description(band), ha="center", fontsize=10)
    fig.text(.5,.018,"Same linear y-scale in every panel. Regret is to the true optimal policy, not the executed oracle allocation.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0,.15,1,.92), h_pad=2.3, w_pad=1.5)
    return fig


def make_aggregate_nuisance_figures(destination, runs, plt, band):
    """Save one method-panel learning comparison per environment and mode."""
    groups = nuisance_trace_groups(runs)
    artifacts = []
    for kind, context, mode in sorted({key[:3] for key in groups}):
        fig = draw_nuisance_learning(groups, kind, context, mode, plt, band)
        for extension in ("png", "pdf"):
            path = Path(destination)/f"aggregate_nuisance_{kind}_context{context}_mode{mode}.{extension}"
            fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
            artifacts.append(path)
        plt.close(fig)
    return artifacts


def make_figures(root, verified_runs, band="sd", sensitivity=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 16,
                               "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
                               "pdf.fonttype": 42, "ps.fonttype": 42})
    from .benchmarks import STRENGTH_PROFILES
    from .pathology_report import make_strength_figures
    from .breadth_report import make_breadth_figures
    from .breadth import BREADTH_PROFILES
    destination = Path(root)/"figures"; destination.mkdir(exist_ok=True)
    groups = sorted({(r["result"].get("benchmark","legacy"),r["result"]["environment"],r["result"]["context"]) for r in verified_runs})
    sets = {
        "comparators": ("ceac","oracle","idac","ucb_q","pomd","ids"),
        "exploration": ("ceac","oracle","idac","ucb_q","pomd","ids"),
        "prior_support": ("ids","ids_rare","ids_zero"),
        "deployment_and_prior": ("ceac","ceac_actor","oracle","ids","ids_rare","ids_zero"),
    }
    if sensitivity is None:
        sensitivity=nuisance_sensitivity_rows(verified_runs)
    groups = [g for g in groups if g[0] not in STRENGTH_PROFILES and g[0] not in BREADTH_PROFILES]
    artifacts=make_strength_figures(destination, verified_runs, plt, band)
    artifacts.extend(make_breadth_figures(destination, verified_runs, plt, band))
    artifacts.extend(make_aggregate_nuisance_figures(destination, verified_runs, plt, band))
    artifacts.extend(make_headline_figures(destination, verified_runs, plt, band))
    for benchmark,kind,context in groups:
        relevant=[r for r in verified_runs if r["result"]["environment"]==kind and r["result"]["context"]==context and r["result"].get("benchmark","legacy")==benchmark]
        modes=sorted({r["result"]["true_mode"] for r in relevant})
        for family,methods in sets.items():
            if family=="exploration" and benchmark=="legacy":continue
            if family=="prior_support" and not any(r["result"]["method"]=="ids_zero" for r in relevant):continue
            present=[m for m in methods if any(r["result"]["method"]==m for r in relevant)]
            if not present or (family=="deployment_and_prior" and not any(m in present for m in ("ceac_actor","ids_rare","ids_zero"))):
                continue
            fig,axes=plt.subplots(len(modes),3,figsize=(18.5,4.5*len(modes)+1.4),squeeze=False)
            fig.patch.set_facecolor("white")
            handles={}
            for row,mode in enumerate(modes):
                subset=[r for r in relevant if r["result"]["true_mode"]==mode]
                for method in present:
                    runs=[r for r in subset if r["result"]["method"]==method]
                    if not runs:continue
                    color,style,marker=STYLES[method]
                    columns=(("cumulative_nuisance_visits","cumulative_diagnostic_visits","cumulative_risky_visits")
                             if family=="exploration" else
                             ("expected_return","cumulative_pseudo_regret","mode1_posterior" if family=="prior_support" else "cumulative_expected_model_information"))
                    for col,field in enumerate(columns):
                        if field != "mode1_posterior" and field not in runs[0]["metrics"][0]:
                            continue
                        values=trace_values(runs, field)
                        handles[method]=plot_seed_trace(axes[row,col], values, method, band)
                if family!="exploration":
                    optimal=subset[0]["metrics"][0]["optimal_return"]
                    axes[row,0].axhline(optimal,color="#222222",lw=.9,ls=":",alpha=.65)
                    max_regret=max(r["metrics"][-1]["cumulative_pseudo_regret"] for r in subset if r["result"]["method"] in present)
                    if max_regret>.25:
                        axes[row,1].set_yscale("symlog",linthresh=.25)
                    else:
                        from matplotlib.ticker import MaxNLocator
                        axes[row,1].yaxis.set_major_locator(MaxNLocator(nbins=5))
                        axes[row,1].ticklabel_format(axis="y",style="plain",useOffset=False)
                    axes[row,1].set_ylim(bottom=0)
                if family=="prior_support":
                    axes[row,2].set_yscale("symlog",linthresh=1e-4,linscale=.6)
                    axes[row,2].set_ylim(-1e-5,1.1)
                    axes[row,2].set_yticks([0,.001,.01,.1,1],["0","0.001","0.01","0.1","1"])
                    axes[row,2].axhline(0,color="#222222",lw=.8,ls=":")
                title="Normal mode (0)" if mode==0 else "Shifted mode (1)"
                for col,axis in enumerate(axes[row]):
                    axis.set_title(title+" · "+(("distraction","diagnosis","route trials") if family=="exploration" else ("policy value","regret","posterior" if family=="prior_support" else "evidence"))[col])
                    axis.set_xlabel("Training episode")
                    axis.set_ylabel((("Cumulative nuisance visits","Cumulative diagnostic visits","Cumulative risky-route trials") if family=="exploration" else ("Expected return · 16-episode mean",("Cumulative pseudo-regret\n(symlog scale)" if max_regret>.25 else "Cumulative pseudo-regret"),("Posterior P(mode 1)\n(symlog; zero shown explicitly)" if family=="prior_support" else "Cumulative expected model KL\n(nats)")))[col])
                    axis.grid(True,color="#dfe3e8",lw=.7,alpha=.8)
                    axis.spines[["top","right"]].set_visible(False)
                    axis.set_xlim(left=1)
            ordered=[m for m in present if m in handles]
            fig.legend([handles[m] for m in ordered],[LABELS[m] for m in ordered],loc="lower center",
                       bbox_to_anchor=(.5,.035),ncol=3 if len(ordered)>4 else 2,frameon=False)
            name=("Shifting DeepSea" if kind=="deepsea" else "Shifting-Shore FrozenLake") if benchmark=="legacy" else ENVIRONMENT_LABELS[kind]
            fig.suptitle(f"{name} · {benchmark} · context {context}",fontsize=21,fontweight="bold",y=.985)
            fig.text(.5,.009,band_description(band),
                     ha="center",fontsize=11,color="#444444")
            fig.tight_layout(rect=(0,.14,1,.95),h_pad=2.,w_pad=1.8)
            stem=f"{kind}_context{context}_{family}" if benchmark=="legacy" else f"{benchmark}_{kind}_context{context}_{family}"
            for extension in ("png","pdf"):
                path=destination/f"{stem}.{extension}"
                fig.savefig(path,dpi=180,facecolor="white",bbox_inches="tight")
                artifacts.append(path)
            plt.close(fig)
    artifacts.extend(make_scaling_figures(destination, verified_runs, plt, band))
    return artifacts


def make_scaling_figures(destination, runs, plt, band="sd"):
    """Matched weak-signal sensitivity; nuisance alphabet is the sole x-axis change."""
    profiles=NUISANCE_PROFILES
    selected=[r for r in runs if r["result"].get("benchmark") in profiles]
    artifacts=[]
    for kind,context in sorted({(r["result"]["environment"],r["result"]["context"]) for r in selected}):
        relevant=[r["result"] for r in selected if r["result"]["environment"]==kind and r["result"]["context"]==context]
        if len({profiles[r["benchmark"]] for r in relevant})<2:continue
        modes=sorted({r["true_mode"] for r in relevant})
        fig,axes=plt.subplots(len(modes),3,figsize=(18.5,4.5*len(modes)+1.4),squeeze=False)
        handles={}
        for row,mode in enumerate(modes):
            for method in LABELS:
                subset=[r for r in relevant if r["method"]==method and r["true_mode"]==mode]
                if not subset:continue
                x=sorted({profiles[r["benchmark"]] for r in subset})
                for col,field in enumerate(("pseudo_regret","unique_observed_states","nuisance")):
                    samples=[[r["event_counts"][field] if field=="nuisance" else r[field] for r in subset if profiles[r["benchmark"]]==m] for m in x]
                    color,style,marker=STYLES[method]
                    line,=axes[row,col].plot(x,[np.mean(v) for v in samples],color=color,ls=style,
                                           marker=marker,lw=4 if method=="ceac" else 2)
                    summaries=[seed_summary(np.asarray(v)[:,None], band) for v in samples]
                    if all(s["lower"] is not None for s in summaries):
                        axes[row,col].fill_between(x, [s["lower"][0] for s in summaries],
                                                  [s["upper"][0] for s in summaries],color=color,alpha=.12)
                    handles[method]=line
            for col,axis in enumerate(axes[row]):
                axis.set_title(f"Mode {mode}")
                axis.set_xscale("log",base=4);axis.set_xticks([4,16,64],["4","16","64"])
                axis.set_xlabel("Nuisance alphabet size M")
                axis.set_ylabel(("Cumulative pseudo-regret","Unique observed states","Cumulative nuisance visits")[col])
                axis.grid(True,alpha=.25);axis.spines[["top","right"]].set_visible(False)
        fig.legend(list(handles.values()),[LABELS[m] for m in handles],loc="lower center",bbox_to_anchor=(.5,.035),ncol=3,frameon=False)
        fig.text(.5,.009,band_description(band),ha="center",fontsize=11,color="#444444")
        fig.suptitle(f"{ENVIRONMENT_LABELS[kind]} · nuisance sensitivity · fixed weak task signal",fontsize=20,fontweight="bold")
        fig.tight_layout(rect=(0,.15,1,.95))
        for extension in ("png","pdf"):
            path=destination/f"{kind}_context{context}_nuisance_scaling.{extension}"
            fig.savefig(path,dpi=180,bbox_inches="tight",facecolor="white");artifacts.append(path)
        plt.close(fig)
    return artifacts
