"""Persistent Layer-4 `Plotter` function classes consuming `StudyBundle` (strict 1x1 / 1x2 layouts)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import numpy as np

from ceac_bench.core.primitives import FrozenFunctionClass, PersistentCallable
from ceac_bench.store.schema import StudyBundle
from .theme import VisualTheme, log_markevery, ordered_methods, style_ceac_legend


@runtime_checkable
class Plotter(PersistentCallable, Protocol):
    """Pure presentation function class: `StudyBundle -> matplotlib.figure.Figure`."""

    def __call__(self, bundle: StudyBundle) -> Any: ...


@dataclass(frozen=True)
class SingleEnvDoseScalingPlot(FrozenFunctionClass):
    """1x2 single-environment scaling figure: Final Policy Gap (left) & Cumulative Regret (right)."""

    variant_id: str
    task: int = 1
    regret_scale: str = "linear"  # "linear" | "symlog"
    gap_ylim: tuple[float, float] = (-0.025, 0.80)
    theme: VisualTheme = field(default_factory=VisualTheme)

    def __call__(self, bundle: StudyBundle) -> Any:
        import matplotlib.pyplot as plt

        self.theme.apply_rcparams()
        sub = bundle.slice(variant_id=self.variant_id, task=self.task)
        rows = sub.summarize_by_dose(task=self.task)
        meta = sub.metadata_for(self.variant_id)
        short = meta.short_title if meta else self.variant_id
        xlab = meta.dose_label if meta else "Intervention dose"
        doses = sorted({r.dose for r in rows})
        methods = ordered_methods(r.method_id for r in rows)
        ranked_present = any(m.startswith("ceac_ranked") for m in methods)

        fig, axes = plt.subplots(1, 2, figsize=(self.theme.figure_width, self.theme.figure_height), squeeze=False)
        reg_lbl = "Cumulative expected regret (linear) ↓" if self.regret_scale == "linear" else "Cumulative pseudo-regret (symlog) ↓"
        reg_sfx = "Cumulative Regret vs. Dose (Linear) ↓" if self.regret_scale == "linear" else "Cumulative Regret vs. Dose ↓"

        specs = [
            ("late_gap_mean", "late_gap_sd", "Final expected policy gap ↓", "Final Policy Gap ↓"),
            ("regret_mean", "regret_sd", reg_lbl, reg_sfx),
        ]
        for c_i, (mean_attr, sd_attr, ylab, title_sfx) in enumerate(specs):
            ax = axes[0, c_i]
            for m in methods:
                grp = sorted([r for r in rows if r.method_id == m], key=lambda x: float(x.dose))
                if not grp:
                    continue
                xs = np.array([float(x.dose) for x in grp], dtype=float)
                ys = np.array([float(getattr(x, mean_attr)) for x in grp], dtype=float)
                sds = np.array([float(getattr(x, sd_attr)) for x in grp], dtype=float)
                st = self.theme.style(m, ranked_present=ranked_present)
                mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if st.marker != "x" else {}
                ax.fill_between(xs, np.maximum(0.0, ys - sds), ys + sds, color=st.color, alpha=st.band_alpha, linewidth=0)
                ax.plot(
                    xs,
                    ys,
                    label=st.label,
                    color=st.color,
                    lw=st.linewidth,
                    linestyle=st.linestyle,
                    marker=st.marker,
                    markersize=st.markersize,
                    zorder=st.zorder,
                    **mk_kw,
                )
            ax.set_xticks(doses)
            if c_i == 0:
                ax.set_ylim(*self.gap_ylim)
            else:
                if self.regret_scale == "symlog":
                    ax.set_yscale("symlog", linthresh=10.0)
                ax.set_ylim(bottom=0.0)
            ax.set_title(f"{short} · {title_sfx}", fontsize=10.6, pad=8)
            ax.set_xlabel(xlab, fontsize=9.4)
            ax.set_ylabel(ylab, fontsize=9.6)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"{short} · Scaling Across Intervention Dose", fontsize=12.3, y=0.985)
        fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=self.theme.legend_cols, frameon=False, fontsize=8.8)
        style_ceac_legend(leg)
        return fig


@dataclass(frozen=True)
class SingleEnvTimeTracePlot(FrozenFunctionClass):
    """1x2 single-environment overlaid time-trace figure: Policy Gap (left) & Cumulative Regret (right)."""

    variant_id: str
    dose: int | float
    task: int = 1
    regret_scale: str = "linear"
    gap_ylim: tuple[float, float] = (-0.025, 0.80)
    theme: VisualTheme = field(default_factory=VisualTheme)

    def __call__(self, bundle: StudyBundle) -> Any:
        import matplotlib.pyplot as plt

        self.theme.apply_rcparams()
        sub = bundle.slice(variant_id=self.variant_id, dose=self.dose, task=self.task)
        meta = sub.metadata_for(self.variant_id)
        short = meta.short_title if meta else self.variant_id
        d_sym = meta.dose_symbol if meta else "dose"
        methods = ordered_methods(sub.methods())
        ranked_present = any(m.startswith("ceac_ranked") for m in methods)

        fig, axes = plt.subplots(1, 2, figsize=(self.theme.figure_width, self.theme.figure_height), squeeze=False)
        is_lin_reg = self.regret_scale == "linear"
        panels = [
            ("window_gap", "Rolling 256-episode policy gap ↓", f"Overlaid Policy Gap ({d_sym} = {self.dose:g}) ↓", True),
            (
                "cumulative_regret",
                "Cumulative expected regret (linear) ↓" if is_lin_reg else "Cumulative expected regret (symlog) ↓",
                f"Overlaid {'Linear ' if is_lin_reg else ''}Regret ({d_sym} = {self.dose:g}) ↓",
                not is_lin_reg,
            ),
        ]

        for c_idx, (field_key, ylab, title_sfx, log_x) in enumerate(panels):
            ax = axes[0, c_idx]
            max_ep = 1.0
            min_ep = 256.0
            for m_i, m in enumerate(methods):
                cells = sorted([c for c in sub.cells if c.method_id == m and field_key in c.trace], key=lambda x: x.condition.seed)
                if not cells:
                    continue
                ep_raw = np.array(cells[0].trace["episode"], dtype=float)
                start = 1 if len(ep_raw) > 1 and ep_raw[0] <= 1.0 and log_x else 0
                xs = ep_raw[start:] if log_x else (ep_raw[start:] / 1000.0)
                if len(xs) == 0:
                    continue
                max_ep = max(max_ep, float(xs[-1]))
                min_ep = min(min_ep, float(xs[0]))
                vals = np.array([c.trace[field_key][start:] for c in cells], dtype=float)
                mean = vals.mean(axis=0)
                sd = vals.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
                st = self.theme.style(m, ranked_present=ranked_present)
                if log_x:
                    mk_idx = log_markevery(len(xs), m_i)
                else:
                    step_mk = max(1, len(xs) // 8)
                    mk_idx = list(range((m_i * 2) % step_mk, len(xs), step_mk)) or [0]
                mk_kw = dict(markeredgecolor="white", markeredgewidth=0.55) if st.marker != "x" else {}
                ax.fill_between(xs, np.maximum(0.0, mean - sd), mean + sd, color=st.color, alpha=st.band_alpha, linewidth=0)
                ax.plot(
                    xs,
                    mean,
                    label=st.label,
                    color=st.color,
                    lw=st.linewidth,
                    linestyle=st.linestyle,
                    marker=st.marker,
                    markersize=st.markersize,
                    markevery=mk_idx,
                    zorder=st.zorder,
                    **mk_kw,
                )
            if log_x:
                ax.set_xscale("log")
                ax.set_xlim(max(1.0, min_ep * 0.92), max_ep * 1.08)
                ax.minorticks_off()
                ax.set_xlabel("Training episodes (log scale)", fontsize=9.6)
                if c_idx == 0:
                    ax.set_ylim(*self.gap_ylim)
                else:
                    ax.set_yscale("symlog", linthresh=10.0)
                    ax.set_ylim(bottom=0.0)
            else:
                ax.set_xlim(0.0, max_ep * 1.02)
                ax.set_ylim(bottom=0.0)
                ax.set_xlabel("Training episodes (thousands)", fontsize=9.6)
            ax.set_title(f"{short} · {title_sfx}", fontsize=10.6, pad=8)
            ax.set_ylabel(ylab, fontsize=9.6)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"{short} ({d_sym} = {self.dose:g}, Task {self.task}) · Overlaid Performance Time Traces", fontsize=12.3, y=0.985)
        fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=self.theme.legend_cols, frameon=False, fontsize=8.8)
        style_ceac_legend(leg)
        return fig


@dataclass(frozen=True)
class SingleEnvCriticBellmanPlot(FrozenFunctionClass):
    """1x2 single-environment Held-Out Reward Critic (left) & Evidence Critic (right) Bellman RMSE traces."""

    variant_id: str
    dose: int | float
    task: int = 1
    theme: VisualTheme = field(default_factory=VisualTheme)

    def __call__(self, bundle: StudyBundle) -> Any:
        import matplotlib.pyplot as plt

        self.theme.apply_rcparams()
        sub = bundle.slice(variant_id=self.variant_id, dose=self.dose, task=self.task)
        meta = sub.metadata_for(self.variant_id)
        short = meta.short_title if meta else self.variant_id
        d_sym = meta.dose_symbol if meta else "dose"
        methods = [m for m in ordered_methods(sub.methods()) if m != "oracle"]
        ranked_present = any(m.startswith("ceac_ranked") for m in methods)

        fig, axes = plt.subplots(1, 2, figsize=(self.theme.figure_width, self.theme.figure_height), squeeze=False)
        panels = [
            ("reward_bellman_error", "Held-out Reward Bellman RMSE (log) ↓", "Reward Critic TD Error ↓"),
            ("evidence_critic_error", "Held-out Evidence Critic RMSE (log) ↓", "Evidence Critic TD / Divergence ↓"),
        ]
        for c_idx, (field_key, ylab, title_sfx) in enumerate(panels):
            ax = axes[0, c_idx]
            max_ep = 32768.0
            for m_i, m in enumerate(methods):
                cells = sorted(
                    [c for c in sub.cells if c.method_id == m and field_key in c.critic_audit and len(c.critic_audit[field_key]) > 0],
                    key=lambda x: x.condition.seed,
                )
                if not cells:
                    continue
                xs = np.array(cells[0].critic_audit["episode"], dtype=float)
                max_ep = max(max_ep, float(xs[-1]))
                vals = np.array([[max(1e-5, v) for v in c.critic_audit[field_key]] for c in cells], dtype=float)
                mean = vals.mean(axis=0)
                sd = vals.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros_like(mean)
                st = self.theme.style(m, ranked_present=ranked_present)
                ax.fill_between(xs, np.maximum(1e-5, mean - sd), mean + sd, color=st.color, alpha=st.band_alpha, linewidth=0)
                ax.plot(
                    xs,
                    mean,
                    label=st.label,
                    color=st.color,
                    lw=st.linewidth,
                    linestyle=st.linestyle,
                    marker=st.marker,
                    markersize=st.markersize,
                    markevery=list(range(m_i % 2, len(xs), 2)) or [0],
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    zorder=st.zorder,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(12.0, max_ep * 1.15)
            ax.minorticks_off()
            ax.set_title(f"{short} · {title_sfx}", fontsize=10.6, pad=8)
            ax.set_xlabel("Training episodes (log scale)", fontsize=9.6)
            ax.set_ylabel(ylab, fontsize=9.6)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            f"{short} ({d_sym} = {self.dose:g}, Task {self.task}) · Overlaid Held-Out Reward & Evidence Critic Error",
            fontsize=12.3,
            y=0.985,
        )
        fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=self.theme.legend_cols, frameon=False, fontsize=8.8)
        style_ceac_legend(leg)
        return fig


@dataclass(frozen=True)
class SingleEnvCriticDosePlot(FrozenFunctionClass):
    """1x2 single-environment late-horizon Reward & Evidence Critic RMSE across dose."""

    variant_id: str
    task: int = 1
    theme: VisualTheme = field(default_factory=VisualTheme)

    def __call__(self, bundle: StudyBundle) -> Any:
        import matplotlib.pyplot as plt

        self.theme.apply_rcparams()
        sub = bundle.slice(variant_id=self.variant_id, task=self.task)
        rows = [r for r in sub.summarize_by_dose(task=self.task) if r.method_id != "oracle" and r.final_evidence_rmse_mean is not None]
        meta = sub.metadata_for(self.variant_id)
        short = meta.short_title if meta else self.variant_id
        xlab = meta.dose_label if meta else "Intervention dose"
        doses = sorted({r.dose for r in rows})
        methods = ordered_methods(r.method_id for r in rows)
        ranked_present = any(m.startswith("ceac_ranked") for m in methods)

        fig, axes = plt.subplots(1, 2, figsize=(self.theme.figure_width, self.theme.figure_height), squeeze=False)
        specs = [
            ("final_reward_rmse_mean", "final_reward_rmse_sd", "Final Reward Bellman RMSE (log) ↓", "Final Reward Critic Error ↓"),
            ("final_evidence_rmse_mean", "final_evidence_rmse_sd", "Final Evidence Critic RMSE (log) ↓", "Final Evidence Critic Error ↓"),
        ]
        for c_i, (m_attr, sd_attr, ylab, sfx) in enumerate(specs):
            ax = axes[0, c_i]
            for m in methods:
                grp = sorted([r for r in rows if r.method_id == m], key=lambda x: float(x.dose))
                if not grp:
                    continue
                xs = np.array([float(x.dose) for x in grp], dtype=float)
                ys = np.array([max(1e-5, float(getattr(x, m_attr) or 1e-5)) for x in grp], dtype=float)
                sds = np.array([float(getattr(x, sd_attr) or 0.0) for x in grp], dtype=float)
                st = self.theme.style(m, ranked_present=ranked_present)
                ax.fill_between(xs, np.maximum(1e-5, ys - sds), ys + sds, color=st.color, alpha=st.band_alpha, linewidth=0)
                ax.plot(
                    xs,
                    ys,
                    label=st.label,
                    color=st.color,
                    lw=st.linewidth,
                    linestyle=st.linestyle,
                    marker=st.marker,
                    markersize=st.markersize,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    zorder=st.zorder,
                )
            ax.set_yscale("log")
            ax.set_xticks(doses)
            ax.set_title(f"{short} · {sfx}", fontsize=10.6, pad=8)
            ax.set_xlabel(xlab, fontsize=9.4)
            ax.set_ylabel(ylab, fontsize=9.6)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"{short} · Late-Horizon Held-Out Critic Error Scaling", fontsize=12.3, y=0.985)
        fig.tight_layout(rect=(0.0, 0.24, 1.0, 0.925))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=self.theme.legend_cols, frameon=False, fontsize=8.8)
        style_ceac_legend(leg)
        return fig
