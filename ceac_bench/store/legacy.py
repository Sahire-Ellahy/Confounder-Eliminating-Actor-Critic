"""Zero-mutation read-only adapters lifting all existing v1 evidence exports into `StudyBundle` v2."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from ceac_bench.core.contracts import ConditionKey, DomainMetadata
from ceac_bench.core.primitives import digest_file
from .schema import CellRecord, StudyBundle


def _verify_v1_manifest(folder: Path) -> str:
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    if isinstance(files, dict):
        for name, expected in files.items():
            sha = expected["sha256"] if isinstance(expected, dict) else expected
            if digest_file(folder / name) != sha:
                raise ValueError(f"Checksum mismatch in legacy bundle file {folder / name}")
    return digest_file(manifest_path)


def _index_critic_cells(critic_bundle: dict[str, Any] | None) -> dict[tuple[str, float, int, str, int], dict[str, tuple[float, ...]]]:
    if not critic_bundle:
        return {}
    indexed: dict[tuple[str, float, int, str, int], dict[str, tuple[float, ...]]] = {}
    for c in critic_bundle.get("cells", []):
        kind = str(c["kind"])
        dose = float(c["dose"])
        task = int(c.get("task", c.get("mode", 1)))
        method = str(c["method"])
        seed = int(c["seed"])
        cps = c.get("checkpoints", [])
        indexed[(kind, dose, task, method, seed)] = {
            "episode": tuple(float(pt["episode"]) for pt in cps),
            "reward_bellman_error": tuple(float(pt["reward_bellman_error"]) for pt in cps),
            "evidence_critic_error": tuple(float(pt["evidence_critic_error"]) for pt in cps),
            "stage_epistemic_error": tuple(float(pt.get("stage_epistemic_error", pt["evidence_critic_error"])) for pt in cps),
        }
    return indexed


class LegacyBundleAdapter:
    """Read-only adapter converting verified v1 `evidence.json.gz` directories into `StudyBundle` v2."""

    @staticmethod
    def load_grid_pathdep(folder: Path | str) -> StudyBundle:
        folder = Path(folder)
        m_sha = _verify_v1_manifest(folder)
        with gzip.open(folder / "evidence.json.gz", "rt") as f:
            ev = json.load(f)
        crit = None
        if (folder / "critic_bellman_audit.json.gz").is_file():
            with gzip.open(folder / "critic_bellman_audit.json.gz", "rt") as f:
                crit = json.load(f)
        crit_map = _index_critic_cells(crit)

        families = (
            DomainMetadata(
                family_id="undertow",
                variant_id="undertow_pathdep",
                title="Path-Dependent Undertow DeepSea (Thermocline Lock, H = 3)",
                short_title="Path-Dependent Undertow",
                horizon=3,
                dose_symbol="k",
                dose_label="Path-dependent confounding modules k",
                reward_bounds=(-0.02, 1.02),
            ),
            DomainMetadata(
                family_id="false_beacon",
                variant_id="false_beacon_pathdep",
                title="Path-Dependent False Beacon Shore (Icebreaker Staging, H = 3)",
                short_title="Path-Dependent False Beacon",
                horizon=3,
                dose_symbol="k",
                dose_label="Path-dependent confounding modules k",
                reward_bounds=(-0.02, 1.02),
            ),
        )
        cells: list[CellRecord] = []
        for r in ev["rows"]:
            kind = str(r["kind"])
            fam = "undertow" if "undertow" in kind else "false_beacon"
            dose = int(r["dose"])
            task = int(r["task"])
            seed = int(r["seed"])
            method = str(r["method"])
            tr_pts = r.get("trace", [])
            trace = {
                "episode": tuple(float(p["episode"]) for p in tr_pts),
                "window_gap": tuple(float(p["window_gap"]) for p in tr_pts),
                "cumulative_regret": tuple(float(p["cumulative_regret"]) for p in tr_pts),
            }
            for extra_k in ("target_fraction", "echo_fraction", "vane_fraction"):
                if tr_pts and extra_k in tr_pts[0]:
                    trace[extra_k] = tuple(float(p[extra_k]) for p in tr_pts)
            scalars: dict[str, Any] = {"bank_policies": r.get("bank_policies")}
            for ek, ev_cnt in r.get("events", {}).items():
                scalars[f"event_{ek}"] = float(ev_cnt)
            cells.append(
                CellRecord(
                    condition=ConditionKey("grid_pathdep", fam, kind, dose, task, seed),
                    method_id=method,
                    episodes=int(r["episodes"]),
                    late_gap=float(r["late_gap"]),
                    cumulative_regret=float(r["cumulative_regret"]),
                    scalars=scalars,
                    trace=trace,
                    critic_audit=crit_map.get((kind, float(dose), task, method, seed), {}),
                )
            )
        return StudyBundle(suite_id="grid_pathdep", families=families, cells=tuple(cells), manifest_sha256=m_sha)

    @staticmethod
    def load_bio_pathdep(folder: Path | str) -> StudyBundle:
        folder = Path(folder)
        m_sha = _verify_v1_manifest(folder)
        with gzip.open(folder / "evidence.json.gz", "rt") as f:
            ev = json.load(f)
        crit = None
        if (folder / "critic_bellman_audit.json.gz").is_file():
            with gzip.open(folder / "critic_bellman_audit.json.gz", "rt") as f:
                crit = json.load(f)
        crit_map = _index_critic_cells(crit)

        tr_map = {
            (t["kind"], int(t["dose"]), int(t["task"]), t["method"], int(t["seed"])): t
            for t in ev.get("traces", [])
        }
        families = (
            DomainMetadata(
                family_id="signalling",
                variant_id="signalling_rewiring",
                title="Path-Dependent Signalling Rewiring (EGFR/MEK Priming, H = 3)",
                short_title="Signalling Rewiring (H = 3)",
                horizon=3,
                dose_symbol="k",
                dose_label="Path-dependent confounding modules k",
                reward_bounds=(-0.02, 1.02),
            ),
            DomainMetadata(
                family_id="k562_gi",
                variant_id="canalized_gi",
                title="Canalized Genetic Risk Screen (K562 PPP2R1A Hub, H = 3)",
                short_title="Canalized K562 Genetic Risk (H = 3)",
                horizon=3,
                dose_symbol="k",
                dose_label="Path-dependent confounding modules k",
                reward_bounds=(-0.02, 1.02),
            ),
        )
        episodes = int(ev.get("episodes", 32768))
        cells: list[CellRecord] = []
        for r in ev["sweep_rows"]:
            kind = str(r["kind"])
            fam = "signalling" if kind == "signalling_rewiring" else "k562_gi"
            dose = int(r["dose"])
            task = int(r["task"])
            seed = int(r["seed"])
            method = str(r["method"])
            t_obj = tr_map.get((kind, dose, task, method, seed), {})
            trace: dict[str, tuple[float, ...]] = {}
            for tk in ("episode", "window_gap", "cumulative_regret", "active_hit", "echo_reporter", "passenger_omics", "pleiotropic"):
                if tk in t_obj:
                    trace[tk] = tuple(float(x) for x in t_obj[tk])
            scalars: dict[str, Any] = {"bank_policies": r.get("bank_policies")}
            for ek, ev_cnt in r.get("events", {}).items():
                scalars[f"event_{ek}"] = float(ev_cnt)
            cells.append(
                CellRecord(
                    condition=ConditionKey("bio_pathdep", fam, kind, dose, task, seed),
                    method_id=method,
                    episodes=episodes,
                    late_gap=float(r["late_gap"]),
                    cumulative_regret=float(r["cumulative_regret"]),
                    scalars=scalars,
                    trace=trace,
                    critic_audit=crit_map.get((kind, float(dose), task, method, seed), {}),
                )
            )
        return StudyBundle(suite_id="bio_pathdep", families=families, cells=tuple(cells), manifest_sha256=m_sha)

    @staticmethod
    def load_runs_export(folder: Path | str, *, suite_id: str, family_map: dict[str, DomainMetadata], critic_bundle: dict[str, Any] | None = None) -> StudyBundle:
        """Load `hardness_burden`, `hardness_prior`, `pathologies_locks`, or `pathologies_epistasis` exports."""
        folder = Path(folder)
        m_sha = _verify_v1_manifest(folder)
        with gzip.open(folder / "evidence.json.gz", "rt") as f:
            ev = json.load(f)
        if critic_bundle is None and (folder / "critic_bellman_audit.json.gz").is_file():
            with gzip.open(folder / "critic_bellman_audit.json.gz", "rt") as f:
                critic_bundle = json.load(f)
        crit_map = _index_critic_cells(critic_bundle)

        cells: list[CellRecord] = []
        for r in ev["runs"]:
            kind = str(r["kind"])
            meta = family_map[kind]
            dose_raw = r["dose"]
            dose = int(dose_raw) if isinstance(dose_raw, float) and dose_raw.is_integer() else dose_raw
            task = int(r["task"])
            seed = int(r["seed"])
            method = str(r["method"])
            tr = r.get("trace", {})
            if isinstance(tr, list):
                trace = {
                    "episode": tuple(float(p["episode"]) for p in tr),
                    "window_gap": tuple(float(p["window_gap"]) for p in tr),
                    "cumulative_regret": tuple(float(p["cumulative_regret"]) for p in tr),
                }
            else:
                trace = {k: tuple(float(x) for x in v) for k, v in tr.items() if isinstance(v, list)}
            summ = r.get("summary", r)
            eps = int(summ.get("settings", {}).get("episodes", ev.get("registration", {}).get("episodes", 0)))
            cells.append(
                CellRecord(
                    condition=ConditionKey(suite_id, meta.family_id, kind, dose, task, seed),
                    method_id=method,
                    episodes=eps,
                    late_gap=float(summ["late_gap"]),
                    cumulative_regret=float(summ["cumulative_regret"]),
                    scalars={},
                    trace=trace,
                    critic_audit=crit_map.get((kind, float(dose), task, method, seed), {}),
                )
            )
        return StudyBundle(
            suite_id=suite_id,
            families=tuple(family_map.values()),
            cells=tuple(cells),
            manifest_sha256=m_sha,
        )
