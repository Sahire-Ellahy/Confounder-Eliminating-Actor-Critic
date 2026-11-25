"""Unified four-layer benchmark engine for Confounder-Eliminating Actor-Critic (CEAC).

Layer hierarchy (strictly unidirectional):
1. ``ceac_bench.core``      — Immutable contracts, RNG streams, and picklable persistent function classes.
2. ``ceac_bench.problems``  — Problem setup details, public vs. oracle-sealed views, and domain builders.
3. ``ceac_bench.runner``    — Organic learner interfaces, single-pass CellRunner, calibration gates, and spawn pools.
4. ``ceac_bench.metrics`` & ``ceac_bench.store`` — Composable MetricObserver function classes and StudyBundle v2 caching.
5. ``ceac_bench.reporting`` — Single-source VisualTheme, Plotter/TableBuilder function classes, and declarative notebooks.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.2.0"
