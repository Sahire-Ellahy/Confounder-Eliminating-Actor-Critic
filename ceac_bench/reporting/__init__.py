"""Layer 4: Pure presentation layer (VisualTheme, Plotter/TableBuilder function classes, declarative notebooks)."""
from __future__ import annotations

from .notebooks import CodeBlock, FamilySection, MarkdownBlock, NotebookComposer, NotebookSpec, VariantSection
from .plotters import (
    Plotter,
    SingleEnvCriticBellmanPlot,
    SingleEnvCriticDosePlot,
    SingleEnvDoseScalingPlot,
    SingleEnvTimeTracePlot,
)
from .tables import (
    CriticAuditMarkdownTable,
    DoseTransitionMarkdownTable,
    PairedAdvantageMarkdownTable,
    TableBuilder,
)
from .theme import (
    CANONICAL_METHOD_ORDER,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    METHOD_STYLES,
    MethodStyle,
    VisualTheme,
    log_markevery,
    ordered_methods,
    style_ceac_legend,
    style_for_method,
)

__all__ = [
    "CANONICAL_METHOD_ORDER",
    "CodeBlock",
    "CriticAuditMarkdownTable",
    "DoseTransitionMarkdownTable",
    "FamilySection",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "METHOD_LINESTYLES",
    "METHOD_MARKERS",
    "METHOD_STYLES",
    "MarkdownBlock",
    "MethodStyle",
    "NotebookComposer",
    "NotebookSpec",
    "PairedAdvantageMarkdownTable",
    "Plotter",
    "SingleEnvCriticBellmanPlot",
    "SingleEnvCriticDosePlot",
    "SingleEnvDoseScalingPlot",
    "SingleEnvTimeTracePlot",
    "TableBuilder",
    "VariantSection",
    "VisualTheme",
    "log_markevery",
    "ordered_methods",
    "style_ceac_legend",
    "style_for_method",
]
