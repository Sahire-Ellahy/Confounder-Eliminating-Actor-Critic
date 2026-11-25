"""Persistent schema v2 data structures for experimental cells, dose summaries, and study bundles."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from ceac_bench.core.contracts import ConditionKey, DomainMetadata


@dataclass(frozen=True)
class CellRecord:
    """Immutable record of a single completed `(condition, method_id)` cell."""

    condition: ConditionKey
    method_id: str
    episodes: int
    late_gap: float
    cumulative_regret: float
    scalars: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, tuple[float, ...]] = field(default_factory=dict)
    critic_audit: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": {
                "suite_id": self.condition.suite_id,
                "family_id": self.condition.family_id,
                "variant_id": self.condition.variant_id,
                "dose": self.condition.dose,
                "task": self.condition.task,
                "seed": self.condition.seed,
            },
            "method_id": self.method_id,
            "episodes": int(self.episodes),
            "late_gap": float(self.late_gap),
            "cumulative_regret": float(self.cumulative_regret),
            "scalars": dict(self.scalars),
            "trace": {k: [float(x) for x in v] for k, v in self.trace.items()},
            "critic_audit": {k: [float(x) for x in v] for k, v in self.critic_audit.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CellRecord":
        c = raw["condition"]
        dose_val = c["dose"]
        if isinstance(dose_val, float) and dose_val.is_integer():
            dose_val = int(dose_val)
        return cls(
            condition=ConditionKey(
                suite_id=str(c["suite_id"]),
                family_id=str(c["family_id"]),
                variant_id=str(c["variant_id"]),
                dose=dose_val,
                task=int(c["task"]),
                seed=int(c["seed"]),
            ),
            method_id=str(raw["method_id"]),
            episodes=int(raw["episodes"]),
            late_gap=float(raw["late_gap"]),
            cumulative_regret=float(raw["cumulative_regret"]),
            scalars=dict(raw.get("scalars", {})),
            trace={k: tuple(float(x) for x in v) for k, v in raw.get("trace", {}).items()},
            critic_audit={k: tuple(float(x) for x in v) for k, v in raw.get("critic_audit", {}).items()},
        )


@dataclass(frozen=True)
class DoseSummaryRow:
    """Seed-aggregated endpoint statistics for one `(variant_id, task, method_id, dose)` group."""

    family_id: str
    variant_id: str
    task: int
    method_id: str
    dose: int | float
    seeds: int
    late_gap_mean: float
    late_gap_sd: float
    regret_mean: float
    regret_sd: float
    final_reward_rmse_mean: float | None = None
    final_reward_rmse_sd: float | None = None
    final_evidence_rmse_mean: float | None = None
    final_evidence_rmse_sd: float | None = None
    scalars_mean: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedComparisonRow:
    """Seed-paired cumulative-regret difference `comparator - reference_method` at a target dose."""

    variant_id: str
    dose: int | float
    task: int
    reference_method: str
    comparator_method: str
    seeds: int
    regret_saved_mean: float
    regret_saved_sd: float
    reference_wins_all_seeds: bool


@dataclass(frozen=True)
class StudyBundle:
    """Schema-v2 persistent study container consumed uniformly by all Layer-4 plotters and tables."""

    suite_id: str
    families: tuple[DomainMetadata, ...]
    cells: tuple[CellRecord, ...]
    calibration_cells: tuple[CellRecord, ...] = ()
    schema_version: int = 2
    manifest_sha256: str = ""

    def metadata_for(self, variant_id: str) -> DomainMetadata | None:
        for meta in self.families:
            if meta.variant_id == variant_id:
                return meta
        return None

    def variants(self) -> tuple[str, ...]:
        seen: list[str] = []
        for c in self.cells:
            if c.condition.variant_id not in seen:
                seen.append(c.condition.variant_id)
        return tuple(seen)

    def methods(self) -> tuple[str, ...]:
        seen: list[str] = []
        for c in self.cells:
            if c.method_id not in seen:
                seen.append(c.method_id)
        return tuple(seen)

    def doses(self, variant_id: str | None = None) -> tuple[int | float, ...]:
        vals = {
            c.condition.dose
            for c in self.cells
            if variant_id is None or c.condition.variant_id == variant_id
        }
        return tuple(sorted(vals))

    def slice(
        self,
        *,
        family_id: str | None = None,
        variant_id: str | None = None,
        dose: int | float | None = None,
        task: int | None = None,
        methods: tuple[str, ...] | list[str] | None = None,
    ) -> "StudyBundle":
        mset = set(methods) if methods is not None else None
        sub = tuple(
            c
            for c in self.cells
            if (family_id is None or c.condition.family_id == family_id)
            and (variant_id is None or c.condition.variant_id == variant_id)
            and (dose is None or float(c.condition.dose) == float(dose))
            and (task is None or c.condition.task == task)
            and (mset is None or c.method_id in mset)
        )
        v_ids = {c.condition.variant_id for c in sub}
        fams = tuple(f for f in self.families if f.variant_id in v_ids) or self.families
        return StudyBundle(
            suite_id=self.suite_id,
            families=fams,
            cells=sub,
            calibration_cells=self.calibration_cells,
            schema_version=self.schema_version,
            manifest_sha256=self.manifest_sha256,
        )

    def summarize_by_dose(self, *, task: int = 1) -> tuple[DoseSummaryRow, ...]:
        groups: dict[tuple[str, str, int, str, int | float], list[CellRecord]] = defaultdict(list)
        for c in self.cells:
            if c.condition.task == task:
                key = (c.condition.family_id, c.condition.variant_id, c.condition.task, c.method_id, c.condition.dose)
                groups[key].append(c)

        rows: list[DoseSummaryRow] = []
        for (fam, var, t_id, m_id, d_val), grp in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], float(x[0][4]), x[0][3])):
            grp_s = sorted(grp, key=lambda r: r.condition.seed)
            gaps = np.array([r.late_gap for r in grp_s], dtype=float)
            regs = np.array([r.cumulative_regret for r in grp_s], dtype=float)

            r_rmse, e_rmse = None, None
            r_sd, e_sd = None, None
            if all("reward_bellman_error" in r.critic_audit and len(r.critic_audit["reward_bellman_error"]) > 0 for r in grp_s):
                rv = np.array([r.critic_audit["reward_bellman_error"][-1] for r in grp_s], dtype=float)
                ev = np.array([r.critic_audit["evidence_critic_error"][-1] for r in grp_s], dtype=float)
                r_rmse = float(rv.mean())
                r_sd = float(rv.std(ddof=1)) if len(rv) > 1 else 0.0
                e_rmse = float(ev.mean())
                e_sd = float(ev.std(ddof=1)) if len(ev) > 1 else 0.0

            scalar_keys = {k for r in grp_s for k, v in r.scalars.items() if isinstance(v, (int, float)) and v is not None}
            scalars_mean: dict[str, float] = {}
            for sk in sorted(scalar_keys):
                sv = [float(r.scalars[sk]) for r in grp_s if isinstance(r.scalars.get(sk), (int, float))]
                if sv:
                    scalars_mean[sk] = float(np.mean(sv))

            rows.append(
                DoseSummaryRow(
                    family_id=fam,
                    variant_id=var,
                    task=t_id,
                    method_id=m_id,
                    dose=d_val,
                    seeds=len(grp_s),
                    late_gap_mean=float(gaps.mean()),
                    late_gap_sd=float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                    regret_mean=float(regs.mean()),
                    regret_sd=float(regs.std(ddof=1)) if len(regs) > 1 else 0.0,
                    final_reward_rmse_mean=r_rmse,
                    final_reward_rmse_sd=r_sd,
                    final_evidence_rmse_mean=e_rmse,
                    final_evidence_rmse_sd=e_sd,
                    scalars_mean=scalars_mean,
                )
            )
        return tuple(rows)

    def paired_comparisons(
        self,
        reference_method: str = "ceac_ranked",
        *,
        dose: int | float | None = None,
        task: int = 1,
    ) -> tuple[PairedComparisonRow, ...]:
        out: list[PairedComparisonRow] = []
        for var in self.variants():
            doses_v = self.doses(var)
            if not doses_v:
                continue
            target_d = doses_v[-1] if dose is None else dose
            sub = [
                c for c in self.cells
                if c.condition.variant_id == var and float(c.condition.dose) == float(target_d) and c.condition.task == task
            ]
            by_ms = {(c.method_id, c.condition.seed): c for c in sub}
            ref_seeds = sorted(s for (m, s) in by_ms if m == reference_method)
            if not ref_seeds:
                continue
            for comp in self.methods():
                if comp == reference_method:
                    continue
                shared = [s for s in ref_seeds if (comp, s) in by_ms]
                if not shared:
                    continue
                diffs = np.array(
                    [by_ms[(comp, s)].cumulative_regret - by_ms[(reference_method, s)].cumulative_regret for s in shared],
                    dtype=float,
                )
                out.append(
                    PairedComparisonRow(
                        variant_id=var,
                        dose=target_d,
                        task=task,
                        reference_method=reference_method,
                        comparator_method=comp,
                        seeds=len(shared),
                        regret_saved_mean=float(diffs.mean()),
                        regret_saved_sd=float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                        reference_wins_all_seeds=bool(np.all(diffs > 0.0)),
                    )
                )
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "suite_id": self.suite_id,
            "families": [
                {
                    "family_id": f.family_id,
                    "variant_id": f.variant_id,
                    "title": f.title,
                    "short_title": f.short_title,
                    "horizon": int(f.horizon),
                    "dose_symbol": f.dose_symbol,
                    "dose_label": f.dose_label,
                    "easy_dose": f.easy_dose,
                    "event_names": list(f.event_names),
                    "reward_bounds": list(f.reward_bounds),
                    "rng_namespace": f.rng_namespace,
                    "restricted_prior_name": f.restricted_prior_name,
                    "reference_prior_name": f.reference_prior_name,
                }
                for f in self.families
            ],
            "cells": [c.to_dict() for c in self.cells],
            "calibration_cells": [c.to_dict() for c in self.calibration_cells],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, manifest_sha256: str = "") -> "StudyBundle":
        fams = tuple(
            DomainMetadata(
                family_id=f["family_id"],
                variant_id=f["variant_id"],
                title=f["title"],
                short_title=f["short_title"],
                horizon=int(f["horizon"]),
                dose_symbol=f["dose_symbol"],
                dose_label=f["dose_label"],
                easy_dose=f.get("easy_dose", 0),
                event_names=tuple(f.get("event_names", ())),
                reward_bounds=tuple(f.get("reward_bounds", (0.0, 1.0))),  # type: ignore[arg-type]
                rng_namespace=f.get("rng_namespace", ""),
                restricted_prior_name=f.get("restricted_prior_name", "null"),
                reference_prior_name=f.get("reference_prior_name", "reference"),
            )
            for f in raw.get("families", [])
        )
        return cls(
            suite_id=str(raw["suite_id"]),
            families=fams,
            cells=tuple(CellRecord.from_dict(c) for c in raw.get("cells", [])),
            calibration_cells=tuple(CellRecord.from_dict(c) for c in raw.get("calibration_cells", [])),
            schema_version=int(raw.get("schema_version", 2)),
            manifest_sha256=manifest_sha256,
        )
