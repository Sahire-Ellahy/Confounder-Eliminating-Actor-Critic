# Comparator provenance and benchmark contract

The active comparison uses CEAC, the offline Graves–Lai oracle, finite-model
Bayesian IDS, **primitive-action IDAC**, **primitive-action UCB Q-learning**, and **optimistic POMD with a softmax policy**.
The custom `policy_ucb`, `confidence`, `coverage`, and `oir_idac` learners have
been removed from the runner and their classes deleted. Their historical runs
remain unchanged and their original implementation is recoverable from Git
and each study's hashed `source_bundle.tar.gz`. Reports and the notebook exclude
those controls. No old result is renamed as a replacement algorithm.

## Source-backed IDAC

The implementation follows the **tabular** `InfoRatioACBASE.update` /
`InfoRatioACTabular` path, not the distinct neural `InfoAC` class:

- Repository: [Sahire-Ellahy/Inforatio-collated](https://github.com/Sahire-Ellahy/Inforatio-collated).
- Pinned revision: [`92b39c31052e90dfb1f7c6e090201895b5491dcf`](https://github.com/Sahire-Ellahy/Inforatio-collated/tree/92b39c31052e90dfb1f7c6e090201895b5491dcf).
- [`inforatio/agents.py`](https://github.com/Sahire-Ellahy/Inforatio-collated/blob/92b39c31052e90dfb1f7c6e090201895b5491dcf/inforatio/agents.py), SHA-256 `7cc486820900b6b76d937b0aebad07f2143cfd192cc6a4d521782e4a42dac48d`.
- [`inforatio/utils.py`](https://github.com/Sahire-Ellahy/Inforatio-collated/blob/92b39c31052e90dfb1f7c6e090201895b5491dcf/inforatio/utils.py), SHA-256 `9065946b90f39dc897b37219921f16de65cc79431211ac6c370cd863b4ab3e80`.
- The provided `inforatio-main.zip` has SHA-256 `e0ed43b5628264c2eb28e84e2d52a97937243ac88b646e50d750f88ec8a31338`. These files and `infoac.py` match the pinned repository byte for byte.
- Method reference: Suttle, Koppel and Liu, [*Occupancy Information Ratio: Infinite-Horizon, Information-Directed, Parameterized Policy Search*](https://arxiv.org/abs/2201.08832), SIAM J. Control Optim. 62(6), 2024. This provenance does not claim to reproduce its neural experimental setup or establish its asymptotic theorem here.

The port in `ceac/primitives.py` has a tabular masked softmax policy, a cost
critic and a shadow-reward critic. For an on-policy block of B transitions,
let empirical occupancy be

$$\widehat d(s)=B^{-1}\sum_{i=1}^B\mathbf 1\{s_i=s\},\qquad
h_i=-\log\widehat d(s_i).$$

It updates moving averages exactly as the source does:

$$\mu_c^+=(1-\tau)\mu_c+\tau\overline c,\qquad
\mu_h^+=(1-\tau)\mu_h+\tau\max(\epsilon_h,\overline h).$$

With pre-update value parameters, the two differential TD errors are

$$\delta_i^c=c_i-\mu_c^++v_c(s_i')-v_c(s_i),\qquad
\delta_i^h=h_i-\mu_h^++v_h(s_i')-v_h(s_i).$$

Writing D = kappa + the updated entropy mean, the actual updates are

$$D=\kappa+\mu_h^+,\qquad
\theta^+=\theta-\frac{\eta_\pi}{BD^2}\sum_i
\left(D\delta_i^c-\mu_c^+\delta_i^h\right)
\nabla_\theta\log\pi_\theta(a_i\mid s_i),$$

$$v_c^+=v_c+\frac{\eta_v}{B}\sum_i\delta_i^c e_{s_i},\qquad
v_h^+=v_h+\frac{\eta_v}{B}\sum_i\delta_i^h e_{s_i}.$$

This is an actor-critic update over primitive actions. It does not choose among
precomputed complete policies, query exact policy values, use mode likelihoods,
or receive the realized mode. Defaults are actor step 0.1, critic step 0.2,
mean step 0.01, entropy offset 0.1, entropy floor 0.0001, and 16 episodes per
on-policy block. Actor and both value tables start from independent N(0,1)
entries, matching the source initialization distribution. These are explicit
initial experimental settings, not tuned performance claims. The source's
constant steps are retained; no two-timescale convergence claim is made.

`tests/fixtures/idac_upstream.json` contains the **unmodified original class
bodies** and their source provenance. The parity test executes those bodies
against controlled trajectories and compares both critics, the actor and both
moving means over multiple batches to numerical precision. It does not compare
the port to another transcription of its own formula.

### Episodic interface adaptation

The source is an average-cost algorithm. The environment adapter is part of the
experimental definition and must not be hidden:

1. An episode is padded analytically to its declared horizon H with zero
   original reward in a one-action absorbing state for each remaining time.
   These rows are bookkeeping, not additional simulator interactions.
2. After time H, the next state is the ordinary initial state. IDAC's
   differential critics bootstrap through this reset, rather than using a
   discounted terminal-zero target. Entropy includes time and padding states.
3. A block contains 16 such reset cycles under one frozen softmax policy.
   Occupancy is pooled across that whole block. This makes repeated-state
   frequencies observable; computing entropy separately on each acyclic
   time-augmented episode would mostly measure its length.
4. If primitive reward lies in [l,u], IDAC uses positive cost
   c = (u-r)/(u-l) + 0.01. For the reset process its average cost is

$$J_c(\pi)=\frac{u}{u-l}+0.01-
\frac{\mathbb E_\pi[R_{\rm episode}]}{H(u-l)}.$$

Thus minimizing **average cost alone** preserves episode-return rankings.
IDAC deliberately divides that cost by occupancy entropy plus kappa, which
can change the preferred policy. The entropy regularization is not silently
annealed away to manufacture success. The reset/time chain can be periodic;
we use occupation frequencies and do not claim all ergodicity assumptions of
the original theorem hold.

The upstream collector calls `env.reset()` after a rollout but appends that
state beyond the zipped transition list. We do not reproduce an unused reset
as a physical step. Reset successors are instead encoded explicitly in the
recorded adapter transitions. The core numerical update matches the source;
collection, masking and fixed-horizon cost semantics are declared adaptations.

## UCB Q-learning

Reference: Jin, Allen-Zhu, Bubeck and Jordan, [*Is Q-learning Provably
Efficient?*](https://arxiv.org/abs/1807.03765), NeurIPS 2018, **Algorithm 1
(UCB-Hoeffding)**. This is not UCB over a policy bank, UCB1, or UCBVI.

The time-augmented state represents (s,h). Initialize Q to H and counts to zero;
use the published greedy argmax rule, with uniform random tie breaking. With
n visits to (s,a,h), T = planned episodes times H, and iota = log(S A T / delta),

$$\alpha_n=\frac{H+1}{H+n},\qquad
b_n=c_b\sqrt{\frac{H^3\iota}{n}},$$

$$Q_h(s,a)\leftarrow(1-\alpha_n)Q_h(s,a)+
\alpha_n\left[\widetilde r+V_{h+1}(s')+b_n\right],\qquad
V_h(s)=\min\{H,\max_a Q_h(s,a)\}.$$

The terminal value at H+1 is zero. Rewards are normalized to [0,1] using
`r_tilde = (r-l)/(u-l)`, including the absorbing padding. Consequently the
normalized total equals `(R_episode-H*l)/(u-l)`: variable termination time
cannot alter reward ordering. Physical metrics stay in original reward units.

Q updates are applied in forward order after an episode. This is equivalent
to primitive online updates here because the state includes time and no row
can recur within an episode; a visited row cannot affect a later action at
another time. The adapter rejects a trajectory that violates that contract.
It must not be reused unchanged on a cyclic unaugmented state representation.

The default bonus multiplier is 1 and delta is 0.05, both recorded in each run.
The paper leaves a sufficiently large universal constant in the theorem;
choosing multiplier 1 is a practical parameter choice, **not a validated
finite-sample coverage guarantee**. Stochastic bounded rewards are permitted
by the paper's stated extension. We retain the H initialization, count-based
step and V clipping rather than replacing them with convenient approximations.

This control has a greedy policy. IDAC and the distinct optimistic POMD
control below use softmax policies.

## Optimistic policy optimization with softmax

`pomd` implements **Algorithm 2 (stochastic optimistic POMD)** from Shani,
Efroni, Rosenberg and Mannor, [*Optimistic Policy Optimization with Bandit
Feedback*](https://arxiv.org/abs/2002.08243v2), ICML 2020, with the explicit
bonus constants in Appendix B (p. 13). This is the bandit-feedback algorithm;
it does not require the full reward-function access of the different OPPO
algorithm by Cai et al.

Initialize the policy uniformly over valid primitive actions. Maintain
empirical costs and transition counts from actual observations. Normalize
costs as `(u-r)/(u-l)` in [0,1], including fixed-horizon padding. This again
preserves episode-return rankings. For K planned episodes, T=KH, and
$\delta'=\delta/3$, the published cost and transition bonuses are

$$b^c_h(s,a)=\sqrt{\frac{2\log(2SAHT/\delta')}{N_h(s,a)\vee1}},\qquad
b^{pv}_h(s,a)=H\sqrt{\frac{4S\log(3SAHT/\delta')}{N_h(s,a)\vee1}}.$$

Backward evaluation is of the **current stochastic policy**, not a greedy
next-action policy:

$$Q_h^k(s,a)=\max\left\{0,\overline c_h^{k-1}(s,a)-b^c_h(s,a)-b^{pv}_h(s,a)
+\overline P_h^{k-1}V_{h+1}^k(s,a)\right\},\qquad
V_h^k(s)=\sum_a\pi_h^k(a\mid s)Q_h^k(s,a),$$

with terminal value zero. The mirror-descent policy step is

$$\pi_h^{k+1}(a\mid s)=\frac{\pi_h^k(a\mid s)\exp(-\eta Q_h^k(s,a))}
{\sum_b\pi_h^k(b\mid s)\exp(-\eta Q_h^k(s,b))},\qquad
\eta=\sqrt{\frac{2\log A}{H^2K}}.$$

It is implemented as an additive logit update with a stable softmax, equivalent
to the published KL mirror step. A user-specified step is permitted and recorded;
the default follows the paper. All historical observations contribute to the
empirical model; no exact model laws are given to the learner. Sparse edge
counts implement the same empirical transition frequencies without allocating
a dense state-squared matrix.

The current episode's data enter the model **after** policy evaluation and
improvement, matching the explicit order of Algorithm 2 and its k-1 estimators.
Tests verify this lag, both Appendix B constants, backward policy-weighted
values, and the multiplicative softmax update on a small numerical instance.
The same tests that alter evaluator likelihoods/prices also cover POMD.

The constants can be very conservative on a short, high-dimensional tabular
run. If every optimistic cost Q is clipped to zero, the policy stays uniform.
`all_optimistic_q_zero` and `optimistic_q_nonzero` are logged explicitly. Such a
short-run plateau is not by itself evidence of an undertow/false-beacon failure
mechanism. We do not shrink the bonuses and present that as the unchanged
published algorithm.

## Information and evidence interfaces

The primitive learners receive only a public state/action mask, elapsed-time
index, horizon, initial state and reward envelope, followed by observed
transitions. State enumeration is compiled by the evaluator and is an explicit
finite-domain privilege; it does not expose transition probabilities or the
hidden mode to the learners. Exact expected returns, model likelihoods and
regret are evaluator diagnostics. Tests change evaluator likelihoods and
prices while requiring learner updates to remain identical.

New method configurations and source hashes enter the existing immutable run
manifest. Old runs cannot resume under changed code. Model-free controls and
model-informed CEAC/IDS have different information access; that distinction
must accompany any performance claim. A short smoke run validates execution,
not convergence or systematic failure.

## Standard overlays and the physical-strength sweep

The standard report and notebook now share one physical-strength renderer.
Method overlays compare optimism, frequentist IDAC, full-support Bayesian IDS
and misspecified-prior Bayesian IDS with CEAC and the oracle. Both IDS arms
use the same algorithm and likelihood family; their priors are `(0.5, 0.5)`
and `(1, 0)`. The restricted prior excludes the truth in mode 1, with mode 0
as a negative control. Both appear in the main figures and default runner slate. STEERING-style panels show training episode versus cumulative
pseudo-regret with strength-colored curves; paired added-regret panels subtract
the mildest strength within each seed before aggregation. Diagnostic panels
separate value gaps, successful arrivals, nuisance visits, lost progress/dwell
and useful task information.

The new physical-strength sweep fixes geometry, state/action domain, horizon,
four nuisance symbols, useful-route values and KL, optimal values and reward
range, and varies rho = 0.05, 0.25, 0.50, 0.75:

$$P_{m,\rho}=(1-\rho)P_{\rm pass}+\rho P_{\rm divert}.$$

Both branches are mode independent. DeepSea diversion loses rightward progress;
False Beacon activation requires one charged physical dwell step. Interior
probabilities keep the state/action domain fixed. The simulator and exact
enumerator use the same joint categorical draw, and preflight checks the
matched task invariants. Details and the reproducible launch command are in
[the sweep protocol](pathology_sweep.md).

This changes only the declared environment settings and evaluator annotations;
the three source-backed comparator updates remain as specified above.
IDAC is a frequentist occupancy-information actor-critic, not the same objective
as Bayesian posterior IDS. The old M=4,16,64 sweep remains an archived
symbol-diversity control and is not relabeled as a physical-strength sweep.

Stronger harm need not defeat every sound exploration method. The reported
failure diagnostics describe observations at the finite recorded budget, not
causal ablations or imposed rankings. A baseline plateau is distinguished from
additional strength-dependent regret. The zero-prior IDS arm remains a separate
information-assumption treatment.

## Completed integration validation

[`audits/comparator_validation.json`](../audits/comparator_validation.json)
records the 48-cell study `primitive_comparator_validation_20261110`: both
environments, both stationary modes, two seeds, six methods, and 128 episodes.
All cells completed with verified run/source/report hashes. Each IDAC cell made
eight actor updates and eight updates to each critic. Each POMD cell made 128
mirror steps, with all optimistic costs zero in all 128 episodes under the
published bonuses; its policy therefore remained uniform. This is a diagnostic
of the short-budget regime, not evidence of a strength-dependent failure.

Validation passed 65 core tests, five walkthrough tests, and all ten notebook
code cells (eight rendered figures). The fresh local PDF is
`runs/primitive_comparator_validation_20261110/report.pdf`. Raw studies remain
local; the compact validation audit is versioned alongside this ledger.
