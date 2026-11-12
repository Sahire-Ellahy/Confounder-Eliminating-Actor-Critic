#!/usr/bin/env python3
"""Execute the walkthrough using this Python interpreter, then audit outputs."""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Persist executed outputs and execution audit")
    args = parser.parse_args()
    path = ROOT / "notebooks/Undertow_FalseBeacon.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    # Do not accidentally execute with an unrelated globally registered kernel.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    if any(c.execution_count is None for c in code):
        raise AssertionError("Some notebook cells did not execute")
    images = sum("image/png" in output.get("data", {}) for cell in code for output in cell.outputs)
    hardness=notebook.metadata.get('hardness_report_schema')==1
    required_images=(4 if notebook.metadata.get('mechanism_report_schema')==1 else 3)+int(notebook.metadata.get('actor_stability_schema')==2)
    if images < required_images:
        raise AssertionError("Expected all environment and per-algorithm performance figures")
    if any(output.output_type == "error" for cell in code for output in cell.outputs):
        raise AssertionError("Notebook contains an error")
    if args.write:
        # Timing metadata adds noise without contributing scientific evidence.
        for cell in notebook.cells:
            cell.metadata.pop("execution", None)
        nbformat.write(notebook, path)
        maps = {}
        maps_to_save=([('hardness-junction-design','junctions.png')] if hardness else
                      [("walkthrough-03", "undertow.png"), ("walkthrough-05", "false_beacon.png")])
        for cell_id,name in maps_to_save:
            cell = next(c for c in notebook.cells if c.id == cell_id)
            png = next(output["data"]["image/png"] for output in cell.outputs if "image/png" in output.get("data", {}))
            asset = ROOT / "notebooks/assets" / name
            asset.write_bytes(base64.b64decode(png))
            maps[str(asset.relative_to(ROOT / "notebooks"))] = hashlib.sha256(asset.read_bytes()).hexdigest()
        audit = dict(schema=1, status="complete", code_cells=len(code), rendered_figures=images,
                     notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     evidence_manifest_sha256=hashlib.sha256((ROOT / "notebooks/evidence/manifest.json").read_bytes()).hexdigest(),
                     python=platform.python_version(),
                     packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib", "nbclient", "ipykernel")},
                     training_launched=False, explanatory_map_hashes=maps)
        if hardness:
            audit['hardness_evidence_manifests']={name:hashlib.sha256((ROOT/'notebooks/evidence'/name/'manifest.json').read_bytes()).hexdigest()
                for name in (('hardness_burden','hardness_prior','mechanisms') if notebook.metadata.get('mechanism_report_schema')==1 else ('hardness_burden','hardness_prior'))}
        if notebook.metadata.get('actor_stability_schema')==2:
            audit['actor_stability_manifest_sha256']=hashlib.sha256((ROOT/'notebooks/evidence/actor_stability/manifest.json').read_bytes()).hexdigest()
        (ROOT / "notebooks/execution.json").write_text(json.dumps(audit, indent=2, sort_keys=True)+"\n")
    print(f"Notebook passed: {len(code)} code cells, {images} rendered figures, no training launched.")


if __name__ == "__main__":
    main()
