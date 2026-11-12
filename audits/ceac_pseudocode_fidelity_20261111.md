# CEAC fidelity correction

The requested algorithm is Section 3, Algorithm Tabular CEAC, of `doc/ceac_v90_accessibility_release/sections/03_algorithms.tex`.

The earlier implementation introduced three substantive changes that must not define the primary CEAC curve:

1. `ceac_hardness/agents.py:CEAC.select` selected an exact-model deployment response before returning to the actor. Equation `ceac-behavior-selector` instead selects statistical debt, then design debt, then the actor. The name "actor-only ablation" was misleading: that branch still had reward and evidence critics, but removed an added shortcut.
2. `bounded_design` and actor column admission consumed exact expected model return/KL arrays. Equations `ceac-buffer-statistics` and `ceac-conservative-arrays` specify policy-labeled replay means and confidence bounds for the restricted master and priced replay tests. The existing replay principally fed the likelihood scoreboard and debt accounting.
3. The row-wise evidence tables were updated using independent unpriced TD errors. This is not the displayed gradient update of a price-conditioned evidence critic. A faithful price-conditioned parameterization must use the priced TD error and the gradient of that parameterized value.

The manuscript explicitly excludes replay-to-actor off-policy gradients. Every episode enters labeled replay; only fresh actor episodes update the actor and the two critics. Scoring replay with evidence prices is part of design and column testing, not permission to apply uncorrected off-policy actor updates.

Exact finite-model row and policy response oracles remain legitimate in their global-certificate role. They must not silently replace the actor execution branch or observed replay statistics.

Raw results and prior source bundles are preserved. The old exact-model cells are relabeled as a reference; the unrequested modified-loop calibration must not be used to claim that CEAC itself failed. A replacement primary CEAC result requires the corrected loop and equation-level validation.

## Primary repair and verification

The maintained primary implementation is `ceac_pseudocode/learning.py`, `design.py` and `agent.py`, launched by `ceac_pseudocode/experiment.py`. Its seven equation-level regression tests verify the price-conditioned critic gradient, beta-zero reduction, both-critic update gating, replay-dependent master prices, price rescoring without recredit, no exact-data substitution on infeasibility, and return to the actor when debts clear.

The repaired implementation passed independent seeds 1001/1002 on both tasks: burden at 131,072 episodes (maximum late gap 0.020014), prior shift at 32,768 (maximum late gap 0.023225). Every calibration run has nonzero evidence-priced actor updates. These are calibration results, not the held-out effect estimate.

The singleton-reference schedule, conservative omitted-policy certificate extension, fixed learning rates, replay confidence allocation and snapshot admission rule are documented in `docs/hardness_implementation.md`. Global oracle certificates are still model based. The tests establish the stated equations and data flow, not all assumptions of the asymptotic coefficient-one theorem. No actor-only curve is included in the primary experiments or notebook.

## Held-out actor limitation

The primary prior-shift task-0 seed-0 run has a late gap of 0.358108 even at zero shift. `scripts/audit_primary_actor.py` exactly reconstructed all 32,768 saved episodes (including policy identities and selector branches) for it and a successful comparison seed. Both finish with the correct model reference. The anomalous actor assigns probability 0.999481 to the risky action at the common cue state and has final augmented-objective optimization gap 0.117649; the comparison seed’s gap is 0.000107. Critics stay finite. No algorithm settings or held-out results were changed in response. This supports a finite-budget optimization limitation, not a replay bookkeeping or prior-support explanation. See `ceac_primary_actor_replay_20261111.json`.
