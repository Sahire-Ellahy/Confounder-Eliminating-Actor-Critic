"""Read the portable, hash-checked export; never launch training."""
from __future__ import annotations
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def verified_bundle(notebook_root):
    root = Path(notebook_root).resolve()
    manifest = json.loads((root / "evidence/manifest.json").read_text())
    if manifest["schema"] not in {1, 2}:
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
    current = source_identity()
    if manifest["schema"] == 1:
        if current != manifest["runtime_sources"]:
            raise ValueError("Training runtime changed: use its recorded commit")
    else:
        archive_name = manifest["runtime_archive"]
        if archive_name not in manifest["artifacts"]:
            raise ValueError("Missing archived-source digest")
        with tarfile.open(root / archive_name, "r:gz") as archive:
            for name, expected in manifest["runtime_sources"].items():
                member = archive.getmember("ceac/"+name)
                if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != expected:
                    raise ValueError("Archived training source differs: " + name)
        # Runner/comparator replacement does not rewrite historical experiments.
        # All numerical kernels used by this notebook must still match exactly.
        required = {k: v for k, v in manifest["runtime_sources"].items()
                    if k not in {"run.py", "comparators.py"}}
        if required != manifest["diagnostic_sources"]:
            raise ValueError("Historical diagnostic-source contract changed")
        if any(current.get(k) != v for k, v in required.items()):
            raise ValueError("Current diagnostic kernel differs from historical evidence")
    return bundle, manifest
