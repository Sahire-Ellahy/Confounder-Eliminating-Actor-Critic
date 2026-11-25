"""Persistent Layer-4 `TableBuilder` function classes generating Markdown summary & paired-comparison tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ceac_bench.core.primitives import FrozenFunctionClass, PersistentCallable
from ceac_bench.store.schema import StudyBundle
from .theme import ordered_methods, style_for_method


@runtime_checkable
class TableBuilder(PersistentCallable, Protocol):
    """Pure presentation function class: `StudyBundle -> Markdown string`."""

    def __call__(self, bundle: StudyBundle) -> str: ...


@dataclass(frozen=True)
class DoseTransitionMarkdownTable(FrozenFunctionClass):
    """Formats easy-dose -> max-dose Final Gap, Cumulative Regret, and optional Evidence Critic RMSE."""

    task: int = 1
    include_critic_rmse: bool = True

    def __call__(self, bundle: StudyBundle) -> str:
        rows = bundle.summarize_by_dose(task=self.task)
        if not rows:
            return "_No matching cells._\n"

        lines: list[str] = []
        if self.include_critic_rmse:
            lines.append("| Benchmark | Algorithm | Final Gap (min → max) | Regret (min → max) | Evidence Critic RMSE (max) |")
            lines.append("|---|---|---:|---:|---:|")
        else:
            lines.append("| Benchmark | Algorithm | Final Gap (min → max) | Regret (min → max) |")
            lines.append("|---|---|---:|---:|")

        for var in bundle.variants():
            meta = bundle.metadata_for(var)
            v_title = meta.title if meta else var
            v_rows = [r for r in rows if r.variant_id == var]
            if not v_rows:
                continue
            methods = ordered_methods(r.method_id for r in v_rows)
            for m in methods:
                m_rows = sorted([r for r in v_rows if r.method_id == m], key=lambda x: float(x.dose))
                if not m_rows:
                    continue
                lo, hi = m_rows[0], m_rows[-1]
                lbl = style_for_method(m).label
                if self.include_critic_rmse:
                    ev_str = "—" if hi.final_evidence_rmse_mean is None else f"{hi.final_evidence_rmse_mean:,.4f}"
                    lines.append(
                        f"| {v_title} | {lbl} | {lo.late_gap_mean:.4f} → {hi.late_gap_mean:.4f} "
                        f"| {lo.regret_mean:,.1f} → {hi.regret_mean:,.1f} | {ev_str} |"
                    )
                else:
                    lines.append(
                        f"| {v_title} | {lbl} | {lo.late_gap_mean:.4f} → {hi.late_gap_mean:.4f} "
                        f"| {lo.regret_mean:,.1f} → {hi.regret_mean:,.1f} |"
                    )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class PairedAdvantageMarkdownTable(FrozenFunctionClass):
    """Formats seed-paired cumulative-regret savings relative to a reference method (default `ceac_ranked`)."""

    reference_method: str = "ceac_ranked"
    task: int = 1

    def __call__(self, bundle: StudyBundle) -> str:
        paired = bundle.paired_comparisons(reference_method=self.reference_method, task=self.task)
        if not paired:
            return "_No paired comparisons available._\n"
        ref_lbl = style_for_method(self.reference_method).label
        lines = [
            f"| Benchmark | Comparator | Regret Saved by `{self.reference_method}` (mean ± SD) | `{self.reference_method}` Wins All Seeds |",
            "|---|---|---:|---|",
        ]
        by_var_comp = {(p.variant_id, p.comparator_method): p for p in paired}
        for var in bundle.variants():
            meta = bundle.metadata_for(var)
            v_title = meta.title if meta else var
            comps = ordered_methods(p.comparator_method for p in paired if p.variant_id == var)
            for c_m in comps:
                p = by_var_comp[(var, c_m)]
                lines.append(
                    f"| {v_title} | {style_for_method(c_m).label} ({ref_lbl.split('·')[0].strip()} contrast) "
                    f"| {p.regret_saved_mean:+,.1f} ± {p.regret_saved_sd:,.1f} | {p.reference_wins_all_seeds} |"
                )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CriticAuditMarkdownTable(FrozenFunctionClass):
    """Formats held-out Reward Bellman RMSE and Evidence Critic RMSE at maximum (or specified) dose."""

    dose: int | float | None = None
    task: int = 1

    def __call__(self, bundle: StudyBundle) -> str:
        rows = bundle.summarize_by_dose(task=self.task)
        lines = [
            "| Benchmark | Algorithm | Held-Out Reward Bellman RMSE | Held-Out Evidence Critic RMSE |",
            "|---|---|---:|---:|",
        ]
        for var in bundle.variants():
            meta = bundle.metadata_for(var)
            v_title = meta.title if meta else var
            doses_v = bundle.doses(var)
            if not doses_v:
                continue
            target_d = doses_v[-1] if self.dose is None else self.dose
            v_rows = [r for r in rows if r.variant_id == var and float(r.dose) == float(target_d) and r.method_id != "oracle"]
            for m in ordered_methods(r.method_id for r in v_rows):
                match = next((r for r in v_rows if r.method_id == m), None)
                if match is None or match.final_reward_rmse_mean is None or match.final_evidence_rmse_mean is None:
                    continue
                lines.append(
                    f"| {v_title} ({meta.dose_symbol if meta else 'd'} = {target_d:g}) | {style_for_method(m).label} "
                    f"| {match.final_reward_rmse_mean:.4f} | {match.final_evidence_rmse_mean:,.4f} |"
                )
        return "\n".join(lines) + "\n"
