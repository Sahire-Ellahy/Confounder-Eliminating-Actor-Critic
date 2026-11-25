"""Single Source of Truth visual theme for all CEAC, IDS, IDAC, and Optimism plots across the repository."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np

from ceac_bench.core.primitives import FrozenFunctionClass


@dataclass(frozen=True)
class MethodStyle:
    """Visual styling contract for one algorithm family member."""

    method_id: str
    family: str  # "ceac_canonical" | "ceac_ranked" | "ids" | "idac" | "optimism" | "oracle"
    label: str
    color: str
    marker: str
    linestyle: str
    linewidth: float
    markersize: float
    zorder: int
    bold_legend: bool = False
    band_alpha: float = 0.08


_RAW_STYLES: tuple[MethodStyle, ...] = (
    # Occupancy-entropy RL (Emerald Green)
    MethodStyle("idac", "idac", "IDAC · occupancy ratio", "#15803D", "D", "-", 1.75, 5.6, 4, False, 0.08),
    # Optimism / intrinsic-curiosity RL (Vivid Fuchsia & Deep Hot Pink)
    MethodStyle("ucb_q", "optimism", "UCB Q-learning", "#C026D3", ">", "-.", 1.75, 5.6, 4, False, 0.08),
    MethodStyle("pomd", "optimism", "Optimistic POMD", "#FF1493", "h", "-", 1.75, 5.6, 4, False, 0.08),
    MethodStyle("rnd", "optimism", "RND · random network distillation", "#E11D48", "p", ":", 1.75, 5.6, 4, False, 0.08),
    # Restricted / misspecified Bayesian IDS priors (Dark Wine Red)
    MethodStyle("ids_zero", "ids", "IDS · restricted (zero-mode) prior", "#7F1D1D", "<", ":", 1.75, 5.6, 4, False, 0.07),
    MethodStyle("ids_decoy", "ids", "IDS · decoy-lock prior", "#7F1D1D", "<", ":", 1.75, 5.6, 4, False, 0.07),
    MethodStyle("ids_canonical", "ids", "IDS · canonical additive prior", "#7F1D1D", "<", ":", 1.75, 5.6, 4, False, 0.07),
    MethodStyle("ids_additive", "ids", "IDS · no-interaction prior", "#7F1D1D", "<", ":", 1.75, 5.6, 4, False, 0.07),
    MethodStyle("ids_correlated", "ids", "IDS · frozen correlated prior", "#7F1D1D", "<", ":", 1.75, 5.6, 4, False, 0.07),
    MethodStyle("ids_null", "ids", "IDS · null / no-rewiring prior", "#7F1D1D", "<", ":", 1.75, 5.6, 4, False, 0.07),
    # Full-model MI Bayesian IDS (Vivid Scarlet Red)
    MethodStyle("ids", "ids", "IDS · full-support prior", "#DC2626", "^", "--", 1.95, 6.0, 5, False, 0.09),
    MethodStyle("ids_product", "ids", "IDS · full-model MI I(θ,ξ; Z)", "#DC2626", "^", "--", 2.10, 6.2, 6, False, 0.10),
    # Empirical / Policy-Tree Optimal-Action IDS (Coral / Rose Red)
    MethodStyle("ids_empirical", "ids", "IDS · model-MI, empirical prior", "#FB7185", "v", "-.", 2.00, 6.2, 6, False, 0.10),
    MethodStyle("ids_rare", "ids", "IDS · rare-alternative prior", "#FB7185", "v", "-.", 2.00, 6.2, 6, False, 0.10),
    MethodStyle("ids_action_tree", "ids", "IDS · optimal-policy MI I(Π*; Z) (Russo–Van Roy)", "#FB7185", "v", "-.", 2.35, 6.8, 9, False, 0.13),
    # Single-Step & On-Path Quotient Optimal-Action IDS (Ruby / Firebrick Red)
    MethodStyle("ids_action", "ids", "IDS · optimal-action MI (Russo–Van Roy)", "#B91C1C", "s", "-", 2.35, 6.8, 9, False, 0.13),
    MethodStyle("ids_action_onpath", "ids", "IDS · on-path quotient MI I(C*_onpath; Z)", "#B91C1C", "s", "-", 2.35, 6.8, 9, False, 0.13),
    # Canonical CEAC (Cobalt Blue & Royal Cobalt Blue)
    MethodStyle("ceac", "ceac_canonical", "CEAC · reward + evidence critics", "#0047AB", "*", "--", 2.75, 7.6, 8, True, 0.11),
    MethodStyle("ceac_natural", "ceac_canonical", "CEAC · guarded natural", "#1D4ED8", "X", "-", 2.75, 7.4, 8, True, 0.11),
    # Improved / Greedy Price-Ranked CEAC (Periwinkle & Lavender Periwinkle)
    MethodStyle("ceac_ranked", "ceac_ranked", "CEAC · greedy disagreement + price-ranked replay", "#6C7CFA", "o", "-", 3.35, 8.4, 12, True, 0.14),
    MethodStyle("ceac_ranked_natural", "ceac_ranked", "CEAC · greedy + price-ranked (guarded natural)", "#96A2FF", "P", "--", 3.00, 8.0, 11, True, 0.12),
    # Oracle reference (Charcoal Grey)
    MethodStyle("oracle", "oracle", "Oracle Graves–Lai", "#374151", "x", ":", 1.65, 5.5, 3, False, 0.05),
)

METHOD_STYLES: dict[str, MethodStyle] = {s.method_id: s for s in _RAW_STYLES}
CANONICAL_METHOD_ORDER: tuple[str, ...] = tuple(s.method_id for s in _RAW_STYLES)

METHOD_COLORS: dict[str, str] = {k: v.color for k, v in METHOD_STYLES.items()}
METHOD_MARKERS: dict[str, str] = {k: v.marker for k, v in METHOD_STYLES.items()}
METHOD_LINESTYLES: dict[str, str] = {k: v.linestyle for k, v in METHOD_STYLES.items()}
METHOD_LABELS: dict[str, str] = {k: v.label for k, v in METHOD_STYLES.items()}


def style_for_method(method_id: str) -> MethodStyle:
    if method_id in METHOD_STYLES:
        return METHOD_STYLES[method_id]
    if method_id.startswith("ceac_ranked"):
        return MethodStyle(method_id, "ceac_ranked", method_id, "#6C7CFA", "o", "-", 3.2, 8.0, 12, True, 0.13)
    if method_id.startswith("ceac"):
        return MethodStyle(method_id, "ceac_canonical", method_id, "#0047AB", "*", "-", 2.8, 7.6, 9, True, 0.11)
    if method_id.startswith("ids"):
        return MethodStyle(method_id, "ids", method_id, "#DC2626", "^", "-", 2.0, 6.0, 6, False, 0.09)
    return MethodStyle(method_id, "other", method_id, "#4B5563", "o", "-", 1.7, 5.5, 4, False, 0.07)


def ordered_methods(present: Iterable[str]) -> list[str]:
    pset = set(present)
    known = [m for m in CANONICAL_METHOD_ORDER if m in pset]
    extras = sorted(pset - set(known))
    return known + extras


def log_markevery(n: int, series_idx: int = 0, count: int = 8) -> list[int]:
    if n <= count:
        return list(range(n))
    raw = np.geomspace(1, n, num=count) - 1.0
    shifted = np.clip(np.round(raw).astype(int) + (series_idx % 2), 0, n - 1)
    return sorted(set(int(i) for i in shifted))


def style_ceac_legend(legend: Any) -> None:
    if legend is None:
        return
    for txt in legend.get_texts():
        label_str = txt.get_text()
        if "CEAC" in label_str:
            txt.set_fontweight("bold")
            txt.set_color("#4C5FD7" if ("greedy" in label_str or "price-ranked" in label_str) else "#0047AB")


@dataclass(frozen=True)
class VisualTheme(FrozenFunctionClass):
    """Persistent visual theme passed to Layer-4 Plotter function classes."""

    figure_width: float = 13.6
    figure_height: float = 5.1
    dpi: int = 180
    legend_cols: int = 3

    def apply_rcparams(self) -> None:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})

    def style(self, method_id: str, *, ranked_present: bool = True) -> MethodStyle:
        base = style_for_method(method_id)
        if method_id == "ceac" and not ranked_present:
            # When canonical CEAC is the top CEAC variant in a plot, render solid cobalt with top zorder
            return MethodStyle(
                method_id=base.method_id,
                family=base.family,
                label=base.label,
                color=base.color,
                marker=base.marker,
                linestyle="-",
                linewidth=3.05,
                markersize=8.0,
                zorder=11,
                bold_legend=True,
                band_alpha=0.13,
            )
        return base
