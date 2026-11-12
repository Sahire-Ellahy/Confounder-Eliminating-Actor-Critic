> Design history: the maintained primary implementation and completed study protocol are in [hardness_implementation.md](hardness_implementation.md). The original exact-deployment reference is superseded for primary CEAC claims.

# Replacement hardness experiments: design rationale

The implemented protocol and reproduction commands are in [hardness_implementation.md](hardness_implementation.md). The original rationale below is retained; the new product-class implementation is in `ceac_hardness`.

The existing rho sweep is an activation-cost experiment. It has not established
an easy-to-hard learning transition or environmental induction of prior mismatch.
The compact audit is reproducible with:

    .venv/bin/python scripts/review_pathology_design.py --pdf

This proposal does not authorize interpreting analytic cue calculations as
algorithm performance. No new full experiment is launched by this command.

## A. Irrelevant exploration burden

Start with a small, learnable useful-task skeleton. Add nested dead-end routes
or a branching beacon chamber. Hold the useful route's reward gap, reliable
probe's cost/KL, horizon, and physical interaction budget fixed across doses.
Use physical opportunity count/depth as the intervention, not a cosmetic
observation label or the probability of a harmless one-step dwell.

Zero distractions must be genuinely learnable by the published primitive
controls at the chosen budget. The previously prepared breadth study keeps
the original difficult shoreline even at B=0; its integration smoke does not
meet this calibration requirement. Do not promote it to a definitive hardness
experiment just by increasing B. Domain growth changes prescribed optimism
bonuses and must be reported, not silently normalized away.

Run separate calibration seeds, inspect the complete learning traces and
optimizer activity, then freeze all settings. If an easy control fails, repair
the task scale/budget or report the failure. Do not adjust controls on hard-dose
outcomes, replace published rules, or select seeds that produce a desired rank.

## B. Environmental violation of a frozen joint prior

Let theta be the useful task (0 or 1), and q the nuisance beacon's probability
of emitting 1. The reliable diagnostic has probability 0.15 for theta=0 and
0.85 for theta=1. The incumbent route is identical across theta; an alternative
becomes better in theta=1. Preserve this decision gap at every shift dose.

A Bayesian prior trained in the original cue regime assigns mass 1/2 to each
of (theta=0,q=0.1) and (theta=1,q=0.9). Both task labels have positive marginal
prior mass. Freeze that joint prior and likelihood family for the whole sweep.
For true theta=1, change the environment to q(d)=0.9-0.8d, d in
{0,0.25,0.5,0.75,1}. For theta=0 use the reflected q(d)=0.1+0.8d.
The environment changes; the fixed Bayesian assumption does not.

At d=0 the true joint model is supported. At every nonzero registered d the
true joint law is absent, even though every binary observation still has
positive probability and both task labels retain prior mass. These three
notions of support must be audited separately.

For theta=1, one beacon has expected log posterior-odds increment

    E[log(P_B(X|theta=1)/P_B(X|theta=0))] = (0.8-1.6d) log(9).

At d>0.5 repeated cues point toward the wrong task. This is a channel-level
calculation; actual IDS behavior depends on its choices and must be measured.
Measure wrong-task posterior confidence, useful diagnostic sampling, actual
policy gap, and wasted interactions. Do not infer regret from the drift alone.

Use the same observable channel for all methods. The Bayesian controls are
(1) the frozen correlated joint prior, (2) a product-support prior on both
theta values and every registered q, and optionally (3) small positive
out-of-regime mass to distinguish excluded support from slow learning.
A full-support IDS control is essential and may perform well.

CEAC's public class must contain the product hypotheses before evaluation.
For every (theta,q), the policy-changing alternative (1-theta,q) shares the
beacon law, so nuisance-only observations have zero KL against that row.
The reliable diagnostic still has KL 0.7 log(17/3) in either direction.
The narrow correlated prior and broad CEAC class are different assumptions;
the product-support IDS arm exposes that difference explicitly.

The new simulator must provide predictable cue access without accidentally
creating a shortcut or leaking theta. A cue can be a visible beacon/current
measurement at an obligatory waypoint, with all cue symbols encoded in the
state; test all model/compiler occupancy conditions. No hidden time switch
or oracle reset is assumed: each dose is a separate stationary task.

## Original implementation prerequisites (now addressed in `ceac_hardness`)

The maintained CEAC/IDS compiler, design LPs and actor arrays are specialized
to two models. Extending them to a finite product class requires corresponding
confidence allocation, alternative rows, KL/return arrays, policy separation,
posterior calculations and replay records, plus source-fidelity tests. The separate `ceac_hardness` package now provides this extension with regression tests; the historical two-model kernels remain unchanged.
The response oracle must revalidate its occupancy-ratio and predictive-belief
conditions for the new observation structure. CEAC also needs realizability.

The current exact-model lane made no learned actor/critic updates in the
reported sweep. A learned-evidence/replay claim additionally requires an active
actor/critic loop matching the pseudocode. Exact-model design results should remain a
separate planning reference, not stand in for that attribution experiment.

## Primary presentation

Use one panel per algorithm. Plot training interactions horizontally, excess
policy loss or cumulative regret vertically, and encode low/medium/high
validated intervention with color. A reader should see how that same method
changes as the environment changes. Separate the two mechanisms into two
compact figures; do not combine them into an invented scalar hardness score.

The main notebook should answer three questions: does the easy case learn,
which methods deteriorate as the intervention increases, and which measured
mechanism explains that deterioration? Put complete tables, all modes, priors,
optimizer diagnostics and source audits in the appendix. Report negative
findings and failure to meet the easy-case gate, not just the desired pattern.
