# CEAC and the controlled hardness studies

The primary algorithm is `ceac_pseudocode`: reward critic, price-conditioned evidence critic, softmax actor, and evidence-priced policy-labeled replay. Its source contract is Section 3, Algorithm Tabular CEAC, in `../doc/ceac_v90_accessibility_release/sections/03_algorithms.tex`. The [fidelity audit](../audits/ceac_pseudocode_fidelity_20261111.md) maps the implementation to those equations.

`ceac_hardness` supplies the environments, finite model compiler, unchanged published controls and reports. Historical `ceac` kernels and archived runs retain their original identities. The earlier exact-deployment design reference does not define primary CEAC, and no actor-only ablation is in the primary slate.

## Two interventions, two scientific claims

These are **two-step junction variants**, not the original grid benchmarks. Every episode makes exactly two simulator calls. Every dose is a separate stationary environment, not an unannounced within-run task switch.

In the physical-burden family, three useful root actions share one waypoint. The incumbent pays 0.5, the risky action pays Bernoulli(0.1 or 0.9), and the diagnostic has mean reward 0.48 in either task. Its three observable rewards are 0.46, 0.48, 0.50, with probabilities ((1-p)/2, p, (1-p)/2), where p is 0.15 or 0.85. The final waypoint action pays zero. Useful occupancy entropy is exactly log(2), even under mixtures, so the easy case does not intrinsically reward spreading over useful routes.

Undertow adds B zero-reward root actions, each reaching its own dead-end waypoint. False Beacon adds one root action that reaches one of B zero-reward chamber cells uniformly. Every distraction has zero task KL. B is 0, 4, 16 or 64. Useful returns, decision gap, diagnostic KL, and the horizon remain fixed. Reachable state counts grow from 2 to 2+B; Undertow has 3+B root actions, whereas False Beacon has four for B>0. This also changes published optimism bonuses; it is part of the intervention, not normalized away. The two B=0 kernels are identical and must not be treated as independent environment evidence.

For IDAC, a deterministic useful route has occupancy entropy log(2). A uniform chamber route has entropy log(2)+0.5 log(B), despite zero reward and zero task information. A policy can further increase entropy by mixing routes. Average normalized cost is 1.01 minus half the expected episode reward. IDAC's population ratio C/(H+0.1), including the implemented kappa regularizer, can therefore prefer genuinely irrelevant occupancy. This objective-level calculation predicts a mechanism, not the measured algorithm ranking; the sweep tests the latter.

The prior-shift family instead emits a mandatory beacon bit before a final route choice. The ten public hypotheses are the Cartesian product of task in {0,1} and q in {0.1,0.3,0.5,0.7,0.9}. The final incumbent/risky payoffs are unchanged. Here the diagnostic pays 0.48 and emits an observable terminal bit with probability 0.15 or 0.85.

Freeze the correlated Bayesian prior at mass one half on each of (0,0.1) and (1,0.9). Change only the realized cue law to q=0.9-0.8d for task 1, or q=0.1+0.8d for task 0, at d in {0,0.25,0.5,0.75,1}. The true joint law leaves this prior's support at every nonzero registered dose. Both task labels and every possible observation retain positive probability. This is an environmental violation of a fixed task–cue assumption, distinct from excluding a task label at every dose.

Product-support IDS is uniform on all ten hypotheses. CEAC receives the same ten-model class. For every true hypothesis, an opposite-task hypothesis with identical q has zero beacon KL, so a beacon alone cannot rule out every policy-changing alternative. The realized task and q are only supplied to the environment, evaluation, and explicitly privileged oracle. The prior-shift junction is shared by both nautical narratives; it is not counted twice as independent evidence.

## The implemented loop

1. The episode selector serves statistical debt, then design debt, then the current actor. It never deploys an exact reward-optimal policy in place of that last branch.
2. Freeze the actor, evidence prices and critic parameters. A transition supplies `e = sum(beta * log(p_reference / p_alternative))`. Both TD errors use pre-update values. Update the actor with their summed advantage, then update both critics. Frozen diagnostic episodes update neither the actor nor its critics.
3. Every episode enters replay under its immutable behavior-policy identity. Return and likelihood-ratio means, with simultaneous confidence radii, supply conservative costs and evidence to the restricted master. The master returns prices and experiment counts. Replay scores are recomputed as `J_hat + beta · I_hat`; changing prices changes the scores without adding observations or credit.
4. Accumulate design quotas using the manuscript’s slowly inflated logarithmic evidence scale. One physical episode can pay compatible statistical and design debt, but is counted once.
5. Freeze periodic actor proposals. Exact model evaluation can reject a proposal that cannot have a negative conservative reduced cost; promising proposals require their own labeled execution and replay check before admission. Missing global separator columns also require replay before entering the master.

The evidence critic is one price-conditioned function, `V_I(s; beta) = sum(beta_ij * z_ij(s))`, with TD gradient `delta_I * beta`. Independent unpriced row-critic updates would implement a different recursion. Setting beta to zero exactly recovers reward-only actor updates, including after prior evidence learning.

### Explicit finite-model choices

These choices instantiate schedules and certificate interfaces; passing the tests is not a proof of the manuscript’s asymptotic theorem.

- Both critics and actor are tabular, the initial actor is uniform, and rates are fixed at actor 0.4 and critic 0.12, with softmax temperature 1. Refresh every 128 episodes; consider a snapshot every 8,192. There is no hand-coded reward-optimal initialization or hidden clipping.
- Half of delta=0.05 is allocated to the likelihood confidence set and half to replay. Pairwise likelihood-ratio martingales give threshold `log((M-1)/0.025)`. Replay uses bounded-return/LLR Hoeffding radii with a summable allocation over predictable policy slots, sample counts and statistics. Policy slots are allocated before observing that policy’s trajectory.
- A singleton reference confidence set is required before issuing the replay design. Pending reference certification is statistical debt. This conservative schedule avoids substituting an observed true-model likelihood ratio for a KL under an unresolved reference. It is a declared scheduling choice, not true-model access; it may spend extra data distinguishing task-irrelevant cue parameters.
- Each bank policy receives a slowly increasing replay floor: `ceil(64 sqrt(log(t)) log(log(max(t,e^e))))`. The mass and target-normalization bounds are both 32. These constants were frozen on independent easy-case calibration and are recorded with every run.
- The restricted master uses **observed replay arrays**. Exact model expectations are used for the public hypothetical target-calibration problem, an upper bound on the optimum, and omitted-policy global checks. The latter enumerate deterministic policies with conservative envelope `gap + 2 max(rJ)` and `max(0, KL - 2 max(rI))`. Bank columns retain their actual replay bounds. This is an explicit counterfactual certificate extension of the kind permitted by Appendix E; it is not an off-policy replay estimate. The envelope and residual are logged. A positive unresolved residual requests data or a missing column; it never fabricates a zero residual.
- The exact deterministic search is valid for these junctions: occupancy measures are convex combinations of deterministic policies, and the shaped maximization involving positive parts of linear KL terms is convex in occupancy. The guard refuses more than 100,000 deterministic policies. This is a finite benchmark implementation, not a scalable neural solver.
- All policy-changing rows are represented. They are not discarded just because a Bayesian posterior makes a task seem unlikely. The same public Cartesian class is fixed before every prior-shift dose.

`design_history.json` records observed master arrays, evidence prices, rescaled replay scores, debts, snapshot tests and global residuals. Per-run summaries count actual actor and nonzero-price updates. This verifies that the requested loop was exercised; it does not by itself identify a causal contribution from each component, and no unrequested ablation is needed to define the algorithm.

## Calibration, budgets and information access

The primary configuration is [ceac_primary.json](../experiments/ceac_primary.json). Calibration seeds are 1001 and 1002, both tasks. Every required calibration cell must have final expected gap at most 0.025 over the last 256 episodes. CEAC must also execute actual actor updates with nonzero evidence prices. Held-out seeds are 0, 1 and 2.

All primitive controls and primary CEAC pass at 131,072 episodes for physical burden. The 144-cell sweep contains two variants, four breadths, three seeds, task 1, and six methods: IDAC, UCB Q-learning, optimistic POMD, product-support IDS, CEAC and offline oracle Graves–Lai. The 120 unchanged comparator cells are reused only after numerical source, kernel, budget and artifact verification; 24 primary CEAC cells replace the exact-deployment reference.

Primary CEAC required 32,768 episodes to pass both-task prior-shift calibration. Both IDS controls are calibrated and rerun at that same budget. The 90-cell prior study contains five doses, two tasks, three seeds and three methods. It does not mix the older 8,192-episode IDS results with longer CEAC runs. Earlier calibration failures remain in local raw results; the lower-budget task-0 CEAC gaps motivated this budget increase before held-out hard-dose runs.

IDAC, UCB and POMD receive primitive trajectories and public state/action masks. IDS and CEAC additionally receive declared candidate likelihoods and exact policy responses. The oracle also knows the realized model, fixes its Graves–Lai design offline, and serves those quotas while otherwise selecting a true-model optimal response. These information assumptions are explicit; this is not a matched model-free architecture comparison. The [comparator source ledger](comparator_provenance.md) gives published equations and parity checks.

For IDS, initially positive atoms retain finite log weights even if exponentiation underflows; later evidence may revive their numerical mass. Initially zero atoms remain excluded. A posterior expected regret at most 1e-12 triggers a documented minimum-regret action rule. This numerical stopping rule is distinct from an exact information-ratio certificate. Existing source-faithful control rules are unchanged.

Three evaluation seeds support a descriptive study. Error shading is sample standard deviation across seeds, not a confidence interval. Final gap and cumulative regret distinguish persistent loss from delayed learning. A surviving full-support IDS control limits any general claim against IDS. Nonzero residual CEAC losses, especially when evidence rewards sustain diagnostic exploration, must be reported as measured.

## Reproduce in the canonical repository

Use its Python environment, with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The driver defaults to three workers. Every cell archives its configuration and numerical source hashes; each study includes a source bundle. Calibration and sweep output folders must be distinct.

```bash
python -m unittest discover -s tests/pseudocode -v
python -m unittest discover -s tests/hardness -v
python -m ceac_pseudocode.experiment calibrate --protocol burden --output runs/my_burden_calibration
python -m ceac_pseudocode.experiment sweep --protocol burden --calibration runs/my_burden_calibration/gate.json --output runs/my_burden
python -m ceac_pseudocode.experiment calibrate --protocol prior_shift --output runs/my_prior_calibration
python -m ceac_pseudocode.experiment sweep --protocol prior_shift --calibration runs/my_prior_calibration/gate.json --output runs/my_prior
python scripts/render_hardness.py --burden runs/my_burden --prior runs/my_prior --execute
```

`--methods` permits calibration subsets; `--calibration` accepts multiple verified gates. The existing primary burden study was assembled with `--reuse-controls runs/hardness_burden_20261111`. This option verifies the archived controls, then keeps their individual source identities and references their raw artifacts without copying them.

The notebook renders compact, hash-verified exports and never launches training. Main figures put pathology on the horizontal axis; reports also contain one learning panel per algorithm, every dose, per-seed endpoints, nuisance/probe exposure and posterior diagnostics. Full raw trajectories remain under ignored `runs/`. The original grid results remain separate historical evidence and are not presented as a replication of the new junction findings.

## Mechanism-first report

The standard renderer now produces `notebooks/evidence/mechanisms/report.md` and `report.pdf`, and rebuilds the main notebook around within-algorithm cumulative-regret fans. Each panel has one method; dose is color, time is x, and bands are sample SD over the three held-out seeds. Full-support IDS/oracle successes and task-0 CEAC exceptions remain explicit.

`scripts/export_critic_updates.py` reconstructs the actor from every saved fresh actor episode and the contemporaneous recorded design prices. It checks every actor policy hash, both update counts, the exact reward/evidence logit decomposition, and every replay score against empirical return plus priced empirical likelihood ratios. It does not resample trajectories, invoke an alternate selector or run an ablation. The portable `updates.json.gz` records all 54 primary CEAC runs, their raw-input identities, reference/design timing and contribution traces. The notebook requires only these exports.

A singleton reference is certified in statistical replay before actor learning begins. The evidence critic consequently cannot be credited with that initial identification. Signed update contributions establish what it did to the recorded actor, not the counterfactual effect of removing it. Task-0/d=0/seed-0 is retained as an example where correct reference identification coexists with a substantial actor optimization residual and evidence updates opposing the reward preference.

## Actor-failure diagnosis (2026-11-12)

The [all-cell audit](../audits/ceac_actor_failure_20261112/assessment.md) reconstructs all 870,000 fresh actor episodes from the 30 primary cue-junction runs. It separates three large priced-objective optimization failures from the small task-0 probing cost under conservative prices. In the failed d=0/seed-0 run, terminal evidence is nonzero on 99.9% of actor decisions; probe is the optimal priced action throughout. Early sampling noise concentrates the softmax on risky, after which corrective probe samples become very rare. This is not a sparse-intrinsic-reward or long-horizon credit-assignment failure.

For the terminal decision, state-value critics are action-independent baselines. Their errors affect variance but cannot bias the conditional mean actor gradient. The exact conditional gradient and variance are checked against finite differences and exhaustive action/outcome enumeration. The audit evaluates the actual saved policy history, not an alternate learner. Reproduce with `python scripts/diagnose_actor_failures.py` then `python scripts/render_actor_diagnosis.py`. Numerical kernels, registered hyperparameters and original results remain unchanged.
