"""Non-invasive Held-Out Reward & Evidence Critic TD/Bellman Diagnostics across all suites."""
from .evaluator import (
    SUITE_COLORS,
    SUITE_LABELS,
    SUITE_MARKERS,
    compile_outcome_tables,
    draw_suite_critic_dose_scaling,
    draw_suite_critic_traces,
    evaluate_heldout_critics_fast,
    load_suite_critic_audit,
    save_suite_critic_audit,
)

__all__ = [
    "SUITE_COLORS",
    "SUITE_LABELS",
    "SUITE_MARKERS",
    "compile_outcome_tables",
    "draw_suite_critic_dose_scaling",
    "draw_suite_critic_traces",
    "evaluate_heldout_critics_fast",
    "load_suite_critic_audit",
    "save_suite_critic_audit",
]
