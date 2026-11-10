"""Maintained benchmark dynamics, initially copied verbatim from the v90 release.

Only the runner owns a realized environment. Learners receive FiniteModel, whose
model copies enumerate declared hypotheses and contain no realized-mode input.
"""
