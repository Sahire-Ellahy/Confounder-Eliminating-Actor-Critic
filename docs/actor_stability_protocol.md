# CEAC actor stability: step sizes and policy geometry

The optimizer study is **implemented and complete: 56 calibration cells and 260 fresh-seed confirmation cells**. The [configuration](../experiments/ceac_actor_stability_20261112.json) was frozen before execution and is preserved unchanged, including its original prospective wording. The separately versioned `ceac_stability` package implements the settings. The [assessment](../notebooks/evidence/actor_stability/assessment.md) records execution results. The [local geometry calculation](../audits/ceac_actor_geometry_20261112.json) uses one recorded policy for mathematical diagnosis only. No learning uses its evaluator action values.

## What changes can address the measured failure?

The [saved-trajectory diagnosis](../audits/ceac_actor_failure_20261112/assessment.md) distinguishes a large optimizer failure from a smaller external-reward cost caused by conservative evidence prices. Actor geometry addresses the first. It does not by itself remove persistent probing when the priced objective favors the probe.

Keep the reward critic, price-conditioned evidence critic, evidence-priced replay, frozen policy labels and debt selector. Vary the actor optimizer and predictable step schedules. There is no actor-only ablation.

At a terminal state and softmax temperature 1, write `F = diag(pi) - pi*pi^T`. The ordinary population logit gradient is `g = F Q_beta`. On the simplex tangent space, the natural direction is `F^+ g = Q_beta - mean(Q_beta)`. Therefore:

```
ordinary expected log-odds direction for actions i versus j:
    pi_i (Q_i - V) - pi_j (Q_j - V)

natural expected log-odds direction for actions i versus j:
    Q_i - Q_j
```

This removes the extra probability factor that made the better probe difficult to recover. At the failed task-0/d=0/seed-0 endpoint, the probe-minus-risky directions per unit optimizer step are approximately 0.00001243 and 0.130602. The corresponding sampled-update standard deviations, with the recorded frozen critic baseline, are approximately 0.0222 and 32.20. These are local moments at identical policies; they neither compare optimized learning rates nor predict the outcome of running either optimizer.

The sampled statewise natural direction contains an inverse propensity: for the selected action a it is equivalent, up to a constant logit shift, to `delta * e_a / pi_a`. Thus natural-gradient geometry does not guarantee adequate sampling or low variance near the boundary. Rare observations can generate large steps. A trust limit and a controlled interior constraint address different parts of that problem.

## Constrain total exploration mass, not a fixed amount per action

Use the interior simplex

```
Delta(epsilon_s) = {q: sum(q)=1, q_a >= epsilon_s / K_s for every valid action}.
```

Here K_s is the number of valid actions at state s. A fixed per-action floor would force nuisance mass proportional to B in Undertow, reproducing the pathology under study. A fixed **total** floor mass epsilon_s instead bounds the imposed nuisance probability by epsilon_s, independent of how many actions are added. It still imposes a real exploration cost and must decay. In the Undertow junction, the imposed nuisance mass is `B*epsilon/(B+3)`, which stays below epsilon as B grows. In False Beacon with B>0, there is one nuisance entrance among four actions, so the imposed mass is epsilon/4 regardless of chamber breadth. Actual nuisance probability can exceed these imposed minima; the report measures it separately.

After an ordinary or natural/mirror proposal w, use its actual KL projection onto this simplex: `q_i = max(epsilon_s/K_s, c*w_i)`, with c chosen to normalize q. Simply clipping and renormalizing can violate the claimed floor. Evaluate the trust bound on the final projected policy.

For a natural/mirror step, the fresh TD sample supplies an action-vector estimate `Ahat_i = delta * 1[a=i] / pi_i`. The unconstrained proposal is proportional to `pi_i * exp(eta*Ahat_i)`. Combine both reward and evidence TD errors in delta, and keep them frozen before either critic update. Use log-space arithmetic. This is a first-order statewise natural update, not access to the true Q function.

A statewise trust limit bounds `KL(q_new || pi_old)`. Contract the proposed step until the bound is met, recording every contraction. An entropy bonus is a different choice: it modifies the optimized objective. A KL step constraint regulates movement without permanently rewarding broad occupancy. The probability floor also changes the feasible policy class, so report both constrained and unconstrained objective gaps.

## Couple schedules to fresh visits

Let k count fresh actor visits to the particular state, and h(k)=1+k/1024. An illustrative schedule family is:

```
actor step:               eta(k) = eta_0 h(k)^(-0.9)
reward critic step:        c(k)   = 0.24  h(k)^(-0.6)
total probability floor:   eps(k) = 0.02  h(k)^(-0.45)
KL step limit:             rho(k)= 0.01  h(k)^(-1.35)
```

This slows actor adaptation relative to critic tracking and makes the floor proportional to the square root of the nominal actor step. Use state visits so a rarely visited state does not exhaust its step schedule while other states learn. Forced-action states have no actor decision to regularize.

For a repeatedly visited state with bounded conditional TD second moments, a typical inverse-propensity noise bound is `E[||g_nat||^2] = O(1/mu_k)`, where `mu_k=eps(k)/K_s`. The exponents satisfy `2*0.9-0.45=1.35>1`, so the surrogate sum `sum eta_k^2/mu_k` is finite; `sum eta_k` and `sum mu_k` diverge. The KL limit has the same exponent as `eta_k^2/mu_k`, rather than shrinking faster without accounting for inverse propensities.

These are design conditions for a bounded-noise, repeatedly visited-state model. They do **not** prove convergence or the Graves–Lai regret coefficient for adaptive CEAC, do not guarantee state reachability, and do not account automatically for any further step contraction. A floor decaying as k^-0.45 can itself impose an O(T^0.55) exploration cost in this finite-horizon setting. It is a finite-budget stabilization candidate, not an asymptotically free modification.

The evidence critic needs its own scale check. With `V_I=beta·z`, a gradient coefficient `alpha_I` gives an immediate value correction proportional to `alpha_I*||beta||^2`. The registered schedule uses `alpha_I(k,beta)=c(k)/(0.01+||beta||^2)`. Its effective value step is at most c(k), tends to zero when prices vanish, and avoids confusing a raw parameter step with the critic's actual value-tracking rate. This remains a predictable step size in the displayed critic-gradient equation. Log the effective rate and critic errors; it is not a guarantee that a changing-price critic tracks successfully.

## Selection and confirmation

The JSON defines seven full-CEAC configurations: the original reference, two scheduled ordinary-score updates, two projected-score updates, and two guarded natural/mirror updates. Calibration uses both tasks at d=0, seeds 2001–2004, and 32,768 episodes per cell (56 cells). These seven values were fixed before calibration; all results, including failed candidates, are retained.

Require nonzero-price actor updates, final current-actor augmented-objective gap at most 0.005, and late behavior reward gap at most 0.025 in every calibration cell. Among passing candidates, minimize worst-cell augmented gap, then mean cumulative external regret, then prefer simpler geometry. If nothing passes, report that failure and design a newly identified protocol rather than quietly changing this one.

Freeze the selection before confirmation on seeds 3001–3005, both tasks, and all registered burden/shift doses. Compare the selected optimizer to the original complete CEAC on identical seeds and budgets. Keep seeds 0/1/2 as descriptive regression checks rather than selection data. New comparative headline claims also require the external methods on the same confirmation seeds.

The primary diagnostic slate includes action probability, nuisance mass, best-priced-action visits, policy KL, gradient second moments, effective critic rates, price levels, replay counts, and external versus augmented gaps. A lower external gap alone would not explain which mechanism improved.

## Implementation boundary

The manuscript's current actor equation is the ordinary score-function step. Predictable step-size changes instantiate that equation; projection and natural/mirror preconditioning are explicit optimizer extensions and must be labeled accordingly. Neither removes either critic or changes the evidence objective.

The existing numerical packages and raw studies are hash-locked. Implement future variants in a separately versioned optimizer package, reusing the selector and replay contracts, and test the zero-modification configuration against the original actor update. Do not silently change frozen kernels or relabel old results. The original reference is bit-identical to the frozen actor in repeated-update tests and a 2,048-episode end-to-end trajectory/replay parity test.


## Execution and selection record

Calibration completed all **56/56** cells. Two settings passed every gate: the original reference and guarded natural 0.03. The frozen endpoint-first rule selected the original reference (worst priced-objective gap 0.000269 versus 0.002393). Guarded natural 0.03 reduced mean calibration regret from 2,081.5 to 1,056.0, a 49.3% reduction. No threshold or primary ranking was changed.

Because the selected setting is the reference, primary confirmation deduplicates it to **130 unique cells**. Before inspecting any confirmation outcomes, we registered a separate secondary rule: evaluate **every non-reference calibration passer** on the same five seeds and all registered doses and budgets. Only guarded natural 0.03 qualified. Its 130 cells are stored separately in `confirmation_secondary`; `scripts/run_actor_stability_secondary.py` preserves the primary selection and records the extension timestamp, source bundle and calibration hashes. The natural setting is not retroactively labeled the primary winner.

Confirmation completed **260/260** cells with verified artifact hashes. At Undertow B=64/task 0, the guarded natural variant reduces mean regret from 22,550 to 3,467 (84.6%) and worst actor priced-objective error from 0.2471 to 0.0020. Across the cue-shift conditions, four of 25 original task-0 cells exceed the original 0.005 optimization-gap threshold, compared with none for the natural variant. The cue mean-regret benefit is driven by one costly seed sequence; the other four seed averages incur additional cost. Task-1 family-average learning costs increase under the natural update. All five seeds, both tasks, and every dose remain in the report.

The fresh-seed comparison reports cumulative regret, late external-reward gap, and both constrained and unconstrained actor objective gaps. Doses are averaged within each seed before computing the SD of paired differences. All doses remain visible in the supporting figures and endpoint CSVs. The two critic schedules and the actor geometry change together; these runs do not identify which individual change caused any performance difference.

Reproduce or verify completed cells from the canonical repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ceac_stability.experiment pipeline \
  --output runs/ceac_actor_stability_20261112 --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_actor_stability_secondary.py \
  --study runs/ceac_actor_stability_20261112 --workers 6
python -m ceac_stability.report --study runs/ceac_actor_stability_20261112 \
  --output notebooks/evidence/actor_stability
```

Only complete, hash-verified raw cells enter the portable report. Both the standard report renderer and notebook load the compact export without launching training. Original studies and their numerical sources remain intact.
