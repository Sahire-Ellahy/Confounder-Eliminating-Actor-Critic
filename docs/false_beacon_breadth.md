# False Beacon physical breadth: prepared experiment

**Calibration prerequisite:** B=0 retains the already difficult shoreline.
The passed integration smoke does not establish an easy baseline. The full
sweep is not yet justified as an easy-to-hard comparison; first satisfy the
[revised scientific design](hardness_redesign.md).


This is a new experiment, separate from the completed one-step activation
sweep. No completed full breadth-sweep result is claimed here.

## Intervention

Breadth B is the number of **additional traversable physical cells** in a
chamber reached through the uninformative shallow buoy: B = 0, 1, 2, 4, 8.
The footprints are nested. Only one edge joins the chamber to the original
shoreline; other adjacent old/new cells are separated by walls. Enlarging
the chamber cannot open a shortcut to a dock or to the useful deep buoy.

B=0 preserves the original weak-signal four-symbol shoreline exactly. It
still contains the original shallow/reed distractions; it is the control
for *adding this chamber*, not a task with all nuisance removed.

The chamber consumes ordinary primitive steps and their existing cost.
It adds neither reward nor task information. It has no forced dwell and
does not change the observation-symbol alphabet. Its geometry creates
more opportunities for irrelevant physical coverage, not more task models.

The preflight checks identical horizon (16), primitive action alphabet (4),
reward range, useful-route values/KL, six seed policies, optimal values and
offline oracle design costs. Every added chamber transition has zero KL
between candidate modes. All added cells must be reachable.

**The reachable state/action domain grows.** This is intentional and is
reported alongside B. Published domain-dependent UCB/POMD bonuses are
recomputed honestly for each domain. Consequently this tests the combined
effect of physical breadth and the methods' prescribed domain dependence;
it does not isolate occupancy entropy while holding bonus magnitude fixed.

The original source files for the shoreline, finite-model solver, and
benchmark definitions remain unchanged. The registered extension is in
ceac/breadth.py and ceac/environments/beacon_breadth.py; the maintained runner
selects it only for the new beacon_breadth identifiers. New studies archive
these sources. The historical notebook keeps its original verified evidence.

## Registered slate

The configuration is [false_beacon_breadth.json](../experiments/false_beacon_breadth.json).
Five breadths, two stationary modes, five seeds, seven methods, and 8,192
training episodes give **350 cells / 2,867,200 environment episodes**.
This longer budget is intended to probe beyond the existing 2,048-episode
plateaus. It is not a guarantee that optimistic POMD leaves bonus saturation.
UCB/POMD know the declared training budget; an earlier prefix of this run
is not equivalent to a separately trained lower-budget run.

All seven source-backed controls are retained: CEAC (exact-model single
loop), offline Graves–Lai, full-support Bayesian IDS, excluded-support
Bayesian IDS, frequentist IDAC, UCB-Hoeffding Q-learning, and optimistic
softmax POMD. Hyperparameters use the maintained runner defaults, without
outcome-dependent tuning. CEAC/IDS know the candidate laws; primitive
learners receive the public domain and observed transitions. The oracle
additionally knows the realized mode.

Bayesian priors remain (0.5,0.5) and (1,0). Mode 1 tests excluded support;
mode 0 is its correctly specified negative control. Adding mode-independent
chamber cells does not by itself invalidate a full-support prior.

## Endpoints and interpretation

- Primary: cumulative evaluator pseudo-regret after 8,192 episodes and
  its paired increase relative to B=0.
- CEAC comparison: comparator-minus-CEAC regret and the change in that
  advantage relative to B=0, subtracting within seed before aggregation.
- Mechanisms: cumulative chamber steps, distinct physical chamber cells
  visited, useful task KL, last-32-episode value gap, and POMD's fraction of
  episodes with all optimistic costs zero.
- Reports retain all methods, both modes, all registered doses, negative
  differences, and seed-level values. Bands are sample SD, not confidence
  intervals. Five seeds remain a small descriptive sample.

A method already failing at B=0 may have little room to get worse. A flat
curve or a negative paired change is a valid outcome. Chamber exposure
alone is not proof of a causal failure mechanism; the intervention and
policy adaptation can both affect the result. Full-support IDS may remain
robust. These runs do not isolate the learned CEAC evidence critic or
replay construction; CEAC still uses exact model planning/certification.

## Run from the repository root

Print the full commands and budgets without launching training:

    .venv/bin/python scripts/run_beacon_breadth.py --stage plan

Verify every registered domain and draw the geometry, without training:

    .venv/bin/python scripts/run_beacon_breadth.py --stage preflight --output audits/beacon_breadth_preflight_NEW

The smoke slate uses B=0 and B=8, both modes, seed 0, all seven methods,
and 32 episodes: **28 cells / 896 episodes**. It checks execution and
artifact schemas and cannot establish a breadth-dependent performance claim.

    .venv/bin/python scripts/run_beacon_breadth.py --stage smoke --output runs/beacon_breadth_smoke_NEW --pdf

Explicitly launch the full registered sweep in a fresh output directory:

    .venv/bin/python scripts/run_beacon_breadth.py --stage full --output runs/beacon_breadth_full_NEW --pdf

The ordinary report command also handles a completed breadth study:

    .venv/bin/python -m ceac.report --input runs/beacon_breadth_full_NEW --pdf

It produces breadth_statistics.json, per-mode breadth-response PNG/PDF
figures and paired CEAC tables automatically. Raw data stay in the new study
directory. Do not send a breadth study to the existing physical-activation
notebook exporter; it deliberately accepts only its original study family.
The prepared geometry can be inspected before any performance sweep.

## Preparation validation (2026-11-11)

The full five-dose exact preflight passed. Reachable state counts are
3,733 / 3,813 / 3,893 / 4,023 / 4,223; optimal returns remain 0.489 and 0.539.
The 28-cell smoke completed and produced a verified PDF report, two per-mode
response figures and the paired endpoint table. All 87 CEAC tests and five
walkthrough tests passed, including historical evidence verification.

The [validation ledger](../audits/beacon_breadth_validation_20261111.json)
records the smoke source identity and run hashes. The full registered sweep
remains unlaunched. These checks establish a runnable experiment, not a
monotonic failure pattern or an algorithm ranking.
