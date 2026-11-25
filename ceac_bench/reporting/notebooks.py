"""Declarative environment-first Jupyter notebook builder & sandbox-safe in-process executor."""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
from typing import Any

from ceac_bench.core.primitives import FrozenFunctionClass, ROOT, atomic_write_json, digest_file


@dataclass(frozen=True)
class MarkdownBlock(FrozenFunctionClass):
    cell_id: str
    markdown: str


@dataclass(frozen=True)
class CodeBlock(FrozenFunctionClass):
    cell_id: str
    source: str


@dataclass(frozen=True)
class VariantSection(FrozenFunctionClass):
    variant_id: str
    heading_markdown: str
    code_blocks: tuple[CodeBlock, ...]


@dataclass(frozen=True)
class FamilySection(FrozenFunctionClass):
    family_id: str
    header_markdown: str
    variants: tuple[VariantSection, ...]


@dataclass(frozen=True)
class NotebookSpec(FrozenFunctionClass):
    """Declarative environment-organized notebook specification."""

    notebook_id: str
    intro: MarkdownBlock
    setup: CodeBlock
    families: tuple[FamilySection, ...]
    closing_blocks: tuple[MarkdownBlock | CodeBlock, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_nbformat(self) -> Any:
        import nbformat as nbf

        cells: list[Any] = []

        def _add_md(cid: str, txt: str) -> None:
            c = nbf.v4.new_markdown_cell(txt.strip() + "\n")
            c.id = cid
            cells.append(c)

        def _add_code(cid: str, src: str) -> None:
            clean = src.strip() + "\n"
            compile(clean, f"<{cid}>", "exec")
            c = nbf.v4.new_code_cell(clean)
            c.id = cid
            cells.append(c)

        _add_md(self.intro.cell_id, self.intro.markdown)
        _add_code(self.setup.cell_id, self.setup.source)
        for fam in self.families:
            _add_md(f"{fam.family_id}-header", fam.header_markdown)
            for var in fam.variants:
                _add_md(f"{var.variant_id}-md", var.heading_markdown)
                for cb in var.code_blocks:
                    _add_code(cb.cell_id, cb.source)
        for blk in self.closing_blocks:
            if isinstance(blk, MarkdownBlock):
                _add_md(blk.cell_id, blk.markdown)
            else:
                _add_code(blk.cell_id, blk.source)

        nb_meta = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            **self.metadata,
        }
        return nbf.v4.new_notebook(cells=cells, metadata=nb_meta)


class NotebookComposer:
    """Builds, executes in-process (zero ZMQ TCP port dependency), validates, and audits notebooks."""

    @staticmethod
    def build_and_execute(
        spec: NotebookSpec,
        output_path: Path | str,
        *,
        audit_path: Path | str | None = None,
        evidence_dirs: tuple[str, ...] = (),
        dpi: int = 160,
    ) -> dict[str, Any]:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from IPython.display import Markdown
        import nbformat as nbf

        out = Path(output_path)
        nb = spec.to_nbformat()
        nbf.validate(nb)

        exec_ns: dict[str, Any] = {"__name__": "__main__"}
        exec_count = 0
        total_figures = 0

        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            exec_count += 1
            cell.execution_count = exec_count
            cell.outputs = []
            pending_figs: list[str] = []
            pending_md: list[str] = []

            orig_show = plt.show

            def _capture_show(*_a: Any, **_kw: Any) -> None:
                for num in list(plt.get_fignums()):
                    fig = plt.figure(num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
                    pending_figs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                    plt.close(fig)

            def _capture_display(*objs: Any, **_kw: Any) -> None:
                for obj in objs:
                    pending_md.append(obj.data if isinstance(obj, Markdown) else str(obj))

            exec_ns["display"] = _capture_display
            plt.show = _capture_show
            stdout_buf = io.StringIO()
            try:
                with redirect_stdout(stdout_buf):
                    exec(compile(cell.source, f"<{cell.id}>", "exec"), exec_ns)
            finally:
                plt.show = orig_show
                plt.close("all")

            txt = stdout_buf.getvalue()
            if txt:
                cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=txt))
            for md_str in pending_md:
                cell.outputs.append(nbf.v4.new_output("display_data", data={"text/markdown": md_str, "text/plain": md_str}, metadata={}))
            for b64 in pending_figs:
                total_figures += 1
                cell.outputs.append(nbf.v4.new_output("display_data", data={"image/png": b64, "text/plain": "<Figure>"}, metadata={}))

        for cell in nb.cells:
            cell.metadata.pop("execution", None)
        nbf.validate(nb)
        out.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, out)

        audit = {
            "schema": 2,
            "status": "complete",
            "notebook": str(out.relative_to(ROOT)) if out.resolve().is_relative_to(ROOT) else str(out),
            "notebook_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "spec_digest": spec.digest(),
            "code_cells": exec_count,
            "rendered_figures": total_figures,
            "training_launched": False,
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "matplotlib", "nbformat")
            },
            "evidence_manifests": {
                d: digest_file(ROOT / "notebooks/evidence" / d / "manifest.json")
                for d in evidence_dirs
                if (ROOT / "notebooks/evidence" / d / "manifest.json").is_file()
            },
        }
        if audit_path is not None:
            atomic_write_json(audit_path, audit)
        return audit
