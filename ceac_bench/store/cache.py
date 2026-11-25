"""Content-addressed atomic cell caching and portable StudyBundle v2 persistence."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any

from ceac_bench.core.contracts import ConditionKey
from ceac_bench.core.primitives import atomic_write_json, canonical_json, digest_bytes, digest_file
from .schema import CellRecord, StudyBundle


@dataclass(frozen=True)
class CellCacheStore:
    """Crash-safe per-cell cache keyed by problem, learner, observer, and condition SHA-256 digests."""

    root_dir: Path

    def cell_dir(self, condition: ConditionKey, method_id: str) -> Path:
        return (
            Path(self.root_dir)
            / "cells"
            / condition.variant_id
            / f"dose_{condition.dose}"
            / f"task_{condition.task}"
            / f"{method_id}_s{condition.seed}"
        )

    def try_load(self, condition: ConditionKey, method_id: str, expected_identity: str) -> CellRecord | None:
        folder = self.cell_dir(condition, method_id)
        if not folder.exists():
            return None
        manifest_path = folder / "manifest.json"
        status_path = folder / "status.json"
        if not manifest_path.is_file() or not status_path.is_file():
            raise ValueError(f"Incomplete or corrupted cell cache directory: {folder}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != expected_identity:
            raise ValueError(f"Cached cell identity mismatch in {folder}: expected {expected_identity}, got {manifest.get('identity')}")
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            raise ValueError(f"Incomplete cell in {folder} (status={status.get('status')!r}); use a fresh directory.")
        for name, expected_sha in status.get("artifacts", {}).items():
            if digest_file(folder / name) != expected_sha:
                raise ValueError(f"Cached cell artifact checksum mismatch: {folder / name}")
        with gzip.open(folder / "record.json.gz", "rt") as f:
            return CellRecord.from_dict(json.load(f))

    def save(
        self,
        record: CellRecord,
        *,
        expected_identity: str,
        spec_payload: dict[str, Any],
    ) -> Path:
        folder = self.cell_dir(record.condition, record.method_id)
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": 2, "identity": expected_identity, "spec": spec_payload}
        atomic_write_json(folder / "manifest.json", manifest)
        atomic_write_json(folder / "status.json", {"status": "running"})

        rec_path = folder / "record.json.gz"
        pending = folder / "record.json.gz.pending"
        with gzip.open(pending, "wt") as f:
            f.write(canonical_json(record.to_dict(), indent=None))
        pending.replace(rec_path)

        artifacts = {
            "manifest.json": digest_file(folder / "manifest.json"),
            "record.json.gz": digest_file(rec_path),
        }
        atomic_write_json(folder / "status.json", {"status": "complete", "artifacts": artifacts})
        return folder


class BundleStore:
    """Atomic writer and SHA-256-verified reader for portable `StudyBundle` v2 directories."""

    BUNDLE_FILENAME = "bundle_v2.json.gz"
    MANIFEST_FILENAME = "manifest_v2.json"

    @classmethod
    def save_bundle(
        cls,
        bundle: StudyBundle,
        out_dir: Path | str,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bundle_path = out / cls.BUNDLE_FILENAME
        pending = out / f"{cls.BUNDLE_FILENAME}.pending"
        payload = canonical_json(bundle.to_dict(), indent=None).encode("utf-8")
        with gzip.open(pending, "wb") as f:
            f.write(payload)
        pending.replace(bundle_path)

        manifest = {
            "schema_version": 2,
            "suite_id": bundle.suite_id,
            "cells": len(bundle.cells),
            "calibration_cells": len(bundle.calibration_cells),
            "files": {cls.BUNDLE_FILENAME: digest_file(bundle_path)},
            "metadata": extra_metadata or {},
        }
        manifest_path = out / cls.MANIFEST_FILENAME
        atomic_write_json(manifest_path, manifest)
        return manifest_path

    @classmethod
    def load_bundle(cls, out_dir: Path | str) -> StudyBundle:
        out = Path(out_dir)
        manifest_path = out / cls.MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        for name, expected_sha in manifest["files"].items():
            if digest_file(out / name) != expected_sha:
                raise ValueError(f"StudyBundle checksum mismatch on {out / name}")
        m_sha = digest_bytes(manifest_path.read_bytes())
        with gzip.open(out / cls.BUNDLE_FILENAME, "rt") as f:
            raw = json.load(f)
        return StudyBundle.from_dict(raw, manifest_sha256=m_sha)
