"""Read the portable, hash-checked export; never launch training."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path


def verified_bundle(notebook_root):
    root = Path(notebook_root).resolve()
    manifest = json.loads((root / "evidence/manifest.json").read_text())
    if manifest["schema"] != 1:
        raise ValueError("Unsupported walkthrough evidence schema")
    for name, expected in manifest["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Evidence path leaves notebook directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence digest mismatch: " + name)
    if "evidence/curves.json.gz" not in manifest["artifacts"]:
        raise ValueError("Missing curve digest")
    with gzip.open(root / "evidence/curves.json.gz", "rt") as source:
        bundle = json.load(source)
    if bundle["schema"] != 1 or bundle["source_studies"] != manifest["source_studies"]:
        raise ValueError("Curve provenance differs from manifest")
    from ceac.evidence import source_identity
    if source_identity() != manifest["runtime_sources"]:
        raise ValueError("Training runtime changed: regenerate the evidence export or use its recorded commit")
    return bundle, manifest
