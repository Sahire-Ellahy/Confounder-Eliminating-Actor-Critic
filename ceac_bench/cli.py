"""Unified CLI entrypoint for `ceac_bench` (`list-suites`, `run`, `inspect-bundle`, `check-boundaries`)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ceac_bench.core.primitives import ROOT
from ceac_bench.reporting import DoseTransitionMarkdownTable
from ceac_bench.store import BundleStore, LegacyBundleAdapter
from ceac_bench.suites import SUITES, get_suite, verify_layer_import_boundaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-suites", help="List registered benchmark suites and their four-layer spec digests")
    sub.add_parser("check-boundaries", help="Run AST import-direction & zero-UI-leak verification across ceac_bench")

    p_run = sub.add_parser("run", help="Execute a registered suite and export its StudyBundle v2 + figures")
    p_run.add_argument("suite_id", choices=sorted(SUITES))
    p_run.add_argument("--output", type=Path, required=True)
    p_run.add_argument("--cache-dir", type=Path, default=None)
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--episodes", type=int, default=None)
    p_run.add_argument("--seeds", type=int, nargs="+", default=None)
    p_run.add_argument("--doses", type=float, nargs="+", default=None)
    p_run.add_argument("--no-figures", action="store_true")

    p_ins = sub.add_parser("inspect-bundle", help="Load a v2 or v1 evidence bundle and print its summary table")
    p_ins.add_argument("path", type=Path)
    p_ins.add_argument("--legacy-kind", choices=["auto", "v2", "grid_pathdep", "bio_pathdep"], default="auto")

    args = parser.parse_args(argv)

    if args.command == "list-suites":
        for sid, spec in sorted(SUITES.items()):
            variants = ", ".join(pb.metadata.variant_id for pb in spec.problem_builders)
            print(f"{sid:18s} | variants=[{variants}] | episodes={spec.episodes:,} | digest={spec.digest()[:12]}")
        return 0

    if args.command == "check-boundaries":
        violations = verify_layer_import_boundaries()
        if violations:
            for v in violations:
                print(f"VIOLATION: {v}", file=sys.stderr)
            return 1
        print("AST layer-boundary verification PASSED: zero upward imports, zero UI leaks in Layers 1–3.")
        return 0

    if args.command == "run":
        spec = get_suite(args.suite_id)
        doses = tuple(int(d) if d.is_integer() else d for d in args.doses) if args.doses is not None else None
        seeds = tuple(args.seeds) if args.seeds is not None else None
        cache_dir = args.cache_dir or (ROOT / "runs" / f"bench_cache_{args.suite_id}")
        bundle = spec.execute_and_save(
            args.output,
            cache_dir=cache_dir,
            workers=args.workers,
            episodes=args.episodes,
            seeds=seeds,
            doses=doses,
            render_figures=not args.no_figures,
        )
        print(f"\nCompleted {bundle.suite_id}: {len(bundle.cells)} cells saved to {args.output}\n")
        print(DoseTransitionMarkdownTable()(bundle))
        return 0

    if args.command == "inspect-bundle":
        folder = args.path
        kind = args.legacy_kind
        if kind == "v2" or (kind == "auto" and (folder / BundleStore.MANIFEST_FILENAME).is_file()):
            bundle = BundleStore.load_bundle(folder)
        elif kind == "grid_pathdep" or (kind == "auto" and "undertow_falsebeacon" in folder.name):
            bundle = LegacyBundleAdapter.load_grid_pathdep(folder)
        else:
            bundle = LegacyBundleAdapter.load_bio_pathdep(folder)
        print(f"Loaded StudyBundle(suite_id={bundle.suite_id!r}, cells={len(bundle.cells)}, sha256={bundle.manifest_sha256[:12]})\n")
        print(DoseTransitionMarkdownTable()(bundle))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
