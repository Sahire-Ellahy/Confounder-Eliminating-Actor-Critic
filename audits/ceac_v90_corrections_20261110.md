# Corrected CEAC finite-model lane and v90 specialization ledger

Date: 2026-11-10. Reference: `doc/ceac_v90_accessibility_release`, Section 3 and
Appendix D. The reference manuscript, released algorithms, historical runs and
figures were not edited. The maintained implementation is `code/ceac`.

This report records an implementation, not a retroactive reinterpretation of the
old plots. Historical FrozenLake's fixed-route agent is still what produced
those plots. New cells identify the corrected source, configuration, model
family, policy class, estimator and prior explicitly.

## Scope and source ownership

Both benchmarks now use one shared episode loop with a softmax actor, reward
critic, row-indexed evidence critics, immutable policy snapshots, full
policy-labeled trajectory replay, exact bounded design solves, the finite
joint target/calibration program, global separation, and cumulative execution
debt. The old environment dynamics were copied verbatim into
`code/ceac/environments` as the starting maintained kernels. No old learner or
runner is imported. The release remains a frozen reference.

This first lane uses the known finite candidate transition/reward laws already
assumed by these benchmarks. It does **not** claim an unknown-dynamics,
simulator-free implementation. Candidate-model planning calls are separated
from real-environment trajectory acquisition, and do not enter the actor's
training buffer as if they were self-play. Runtime is recorded; planning calls
are not counted as physical environment episodes.

## 1. Public observations, model class and policy class

The public class has modes 0 and 1. The realized simulator is owned by the
runner. Its `mode`, `optimal_policy`, `shore_label`, configuration and `info`
dictionaries are not sent to CEAC or IDS. The learner receives only a separately
constructed model-family interface and observations/actions/rewards. Only the
explicit Lai–Graves arm receives the realized mode.

All three methods have the same policy space: mask-respecting finite-horizon
Markov policies on the encoded observation, which includes elapsed time and
observable signals. The banks are computational active sets, not restrictions
on what the corrected oracle or IDS may search. Stochastic actor snapshots are
valid columns in the convex hull of complete deterministic Markov policies.

The exact transition kernel is enumerated from actual `step()` outcomes. It
checks common positive support, normalized laws, one random event per step,
and increasing elapsed time. A changed kernel that violates those assumptions
raises an error instead of silently producing a certificate.

## 2. Full-policy separation without revealing the model

For these two specific structures, the implementation verifies a
policy-independent state-occupancy ratio

$$
 d_m^\pi(s)=w_m(s)d_0^\pi(s), \qquad w_m(s_0)=1.
$$

Every nonterminal edge satisfies

$$
 w_m(s')=w_m(s)\frac{P_m(s'\mid s,a)}{P_0(s'\mid s,a)}
$$

for every predecessor/action reaching that state, marginalizing emitted rewards
for state occupancy. All paths must give the same ratio. For arbitrary
coefficients $c_m,b_{mj}$, a *single shared* policy maximizing

$$
 \sum_m c_mJ_m(\pi)+\sum_{m,j}b_{mj}I^m(\pi;j)
$$

can therefore be found by backward induction under $P_0$, using stage reward

$$
 g(s,a)=\sum_mw_m(s)\left[c_m r_m(s,a)+
                \sum_j b_{mj}\kappa_{mj}(s,a)\right].
$$

This is not a separate policy maximum in each hidden model. The same action is
selected at each public state. The check is benchmark-specific: a general MDP
with path-dependent occupancy ratios needs another structured response oracle.
The implementation fails closed in that case.

For IDS, a further check verifies that any reward evidence absent from the next
encoded state has **no future informative descendant**. This permits state-based
predictive beliefs in the trajectory mutual-information calculation. DeepSea's
challenge reward is the relevant example: its information is counted at that
step, and no later diagnostic can depend on forgotten evidence. This structural
condition is checked rather than assumed for arbitrary environments.

## 3. Likelihood confidence and exact candidate-model arrays

The previous empirical information LCB is removed from the maintained lane.
Replay still stores all returns and per-model transition log likelihoods under
the immutable behavior-policy label. These empirical statistics are diagnostics;
they are not mislabeled confidence bounds.

Candidate-model values $J_m(\pi)$ and trajectory KLs $I^m(\pi;j)$ are instead
computed exactly by finite-horizon evaluation. Sampling error in these arrays is
zero conditional on the declared model family. The unknown quantity remains
which candidate generated the actual trajectories.

For equal initial log scores,

$$
 S_t(m)=\sum_{k<t}\log p_m^{\pi_k}(\tau_k),\quad
 \mathcal C_t=\{m:\max_j S_t(j)-S_t(m)\le
                    \log((|\mathcal M|-1)/\delta)\}.
$$

For any true candidate $m$, each likelihood ratio
$\exp(S_t(j)-S_t(m))$ is a nonnegative martingale under predictable adaptive
policy selection. Ville's inequality and a union bound over $j\ne m$ give

$$
 \Pr_m\{\exists t:m\notin\mathcal C_t\}\le\delta.
$$

The confidence allocation is divided across contexts when more than one is
requested. It is a per-run statement, not a simultaneous statement across all
independent seeds/methods in a study. The uniform likelihood mixture
$\xi_t=\operatorname{softmax}(S_t)$ supplies the randomized model estimate for
the finite target/calibration LP. This uniform initialization does not import
the IDS comparator's deliberately excluded prior support.

Rows are rebuilt for every $(m,j)$ with $m\in\mathcal C_t$ and
$j\in\mathcal M_{\rm alt}(m)$. Rejected references do not impose permanent
rows, while their stored trajectories and previously issued quotas remain.

To make the cost bound conservative before model resolution while retaining a
linear full-policy response, define

$$
 U_t^\star=\max_{m\in\mathcal C_t}J_m^\star,
 \qquad
 L_t(\pi)=\min_{m\in\mathcal C_t}J_m^\star
             -\sum_{m\in\mathcal C_t}\Delta_m(\pi).
$$

On the plausible-model event, $L_t(\pi)\le J_{m_\star}(\pi)$, so
$\overline\Delta_t=U_t^\star-L_t$ is a valid upper bound on true regret.
For a singleton it is the exact regret. This deliberately loose model-robust
bound replaces the paper's sampled return LCB in this exact-evaluation lane.
It is a disclosed specialization, not a new empirical concentration claim.

## 4. Bounded master, global oracle and column admission

The primal and dual are v90's bounded programs:

$$
 \min_{\eta\ge0}\overline\Delta^T\eta
 \quad\text{s.t. }K\eta\ge\mathbf1,\quad\mathbf1^T\eta\le B_{\max},
$$
$$
 \max_{\beta,\lambda\ge0}\mathbf1^T\beta-B_{\max}\lambda
 \quad\text{s.t. }K^T\beta\le\overline\Delta+\lambda\mathbf1.
$$

SciPy/HiGHS solves the LP; a globally maximizing common-policy response checks
all omitted columns. Any negative-reduced-cost separator is inserted and the
master is solved again. The implementation records row coverage, primal/dual
gap, mass price, evidence prices and global column residual. A failed solver or
unclosed certificate raises an error; no fixed-bank fallback is retained.

The seed-mixture coverage bound is checked explicitly. The default
$B_{\max}=20$ contains the displayed certified optimizers. The normal-mode
full-policy coefficients are

- DeepSea: $0.015/1.21422073877=0.0123536022084$.
- FrozenLake: $0.042/1.21422073877=0.0345900861836$.

In shifted mode both coefficients are zero because the optimal policy itself
is informative. FrozenLake's normal-mode separator discovers the 13-step
survey-to-inner-dock route omitted by the released six-route bank.

Actor snapshots are evaluated with the same reduced-cost upper bound before
admission. Because an exact separator already closes the full policy residual,
a later actor proposal need not add another column under unchanged prices.
Column discovery in this lane therefore cannot be attributed solely to the
actor. Exact global response is an explicit part of tabular v90.

The corrected oracle uses expected design counts with the same cumulative
scale as CEAC. It does not use the released DeepSea scheduler's realized-LLR
stopping rule. Its finite-sample execution and asymptotic coefficient are
reported separately.

## 5. Target/calibration program and finite schedule choices

For each candidate good set $G\subseteq\{0,1\}$, the implementation solves
Appendix D's joint LP in $(\lambda,\omega,t,v)$, including the zero-GLC models.
Target acceptability uses the certified per-model coefficients and
$\delta_\epsilon=\epsilon/6$. Calibration uses

$$
 \overline I_t(\pi;m)=\sum_j\xi_t(j)I^j(\pi;m).
$$

Both target and calibration column searches use the shared-policy oracle above.
A restricted infeasible LP is not automatically called globally infeasible:
phase-I slack minimization also receives full-policy separation. Only a
positive globally certified phase-I slack rules out that candidate set.
All candidate solutions are refreshed against the same final bank.

If the all-model good set is feasible, calibration is unnecessary; its otherwise
unconstrained epigraph variable is fixed to zero. Other feasible sets are ranked
by their calibration value, with larger good sets preferred on exact ties. This
makes the unbounded all-good edge case explicit rather than interpreting an
unbounded solver result as failure.

The mixture follows Appendix D:

$$
 p_t=q\lambda_t+(1-q)\omega_t,\quad
 q=\frac{4n_{\max}+\epsilon g_{\mathcal M}}
         {4n_{\max}+2\epsilon g_{\mathcal M}}.
$$

For the correctness slate, $\epsilon=0.25$, $n_{\max}=20$, refresh interval
8 episodes, and snapshot interval 4 episodes are explicit configuration.
The normalization contains the checked finite designs; it is **not** presented
as a derivation of every uniform-regularity constant in the imported AE2 theorem.
At an unresolved refresh, if $\mathcal C_t\not\subseteq G$, one draw from
$p_t$ requests a fresh compatible certificate trajectory. Pending compatible
requests coalesce. The ordinary CEAC selector then executes that request through
the same episode loop as design debt and actor episodes.

The supplied draft does not completely specify how many estimator-oracle calls
are interleaved with each physical episode or how its imported estimator and
stopping rule map to that cadence. The schedule above is consequently an
explicit implementation choice. A coefficient-one theorem for this exact
schedule is **not** claimed. An asymptotic experiment also needs the declared
horizon-dependent tolerance schedule and justified class constants; a fixed
small validation slate cannot establish those claims.

## 6. Actor/critic update and its precise representation change

For $\pi_\theta(a|s)=\operatorname{softmax}(\theta(s,\cdot)/T)_a$, the score is

$$
 \nabla_{\theta(s,\cdot)}\log\pi_\theta(a|s)
      =(\mathbf e_a-\pi_\theta(\cdot|s))/T.
$$

The actor gradient includes this factor. Each episode freezes the behavior
policy and evidence prices. All TD errors are computed before any update; the
actor is updated first, then the critics. Only fresh actor episodes update
either actor or critic. Stored replay and frozen diagnostic episodes never
supply off-policy actor gradients.

Instead of one scalar evidence table reused across prices/references, keep
unpriced row critics $V_{mj}$ with immediate reward
$\ell_{mj}=\log(p_m/p_j)$, and form

$$
 V^I_\beta=\sum_{m,j}\beta_{mj}V_{mj},\qquad
 \delta^I_\beta=\sum_{m,j}\beta_{mj}
       [\ell_{mj}+V_{mj}(s')-V_{mj}(s)].
$$

Each row takes its own on-policy TD update. At a fixed policy the Bellman
operator is linear in reward, so the combined fixed point is exactly the
priced-evidence value. At $\beta=0$ the evidence contribution vanishes
identically even if the unpriced tables contain old values.

This basis-critic update is a representation-level refinement of the draft's
single parameterized evidence-critic update, not identical parameter dynamics.
It avoids contamination from old prices or reversed reference signs. It does
not establish critic convergence from finite data by itself.

The validation initializes the actor from the optimal policy of the **uniform
candidate-model mixture**, with 10% valid-action uniform mixing. This uses
public model knowledge, not the realized mode. `--actor-init uniform` provides
a control. Reward and evidence value tables start at zero. The actor and critic
step sizes decay with actor-update count, with exponents 0.75 and 0.6,
respectively. No hand-coded environment-specific route preference is installed.

## 7. Persistent quotas, statistical debt and evidence

At refresh, use the stated slowly inflated logarithmic scale $b_r$, and add

$$
 Q_{r,i}=Q_{r-1,i}+
       \frac{b_r-b_{r-1}}{\min(1,\min_j(K\eta_r)_j)}\eta_{r,i}.
$$

The denominator is the certified row-shortfall repair. Old quotas are never
replaced by $b_r\eta_r$. Statistical requirements and design quotas have
separate counters; the selector serves statistical debt, then the largest design
debt, then the actor. One actual trajectory increments the immutable policy's
physical count once and the compatible certificate count once. It can satisfy
both debts without being purchased twice. Replaying the same episode ID twice
is an error. The one-episode rounding guarantee is checked after debt is cleared;
it is not asserted during an unfinished debt block.

Every episode records context, immutable policy ID, behavior type, states,
actions, rewards, termination and per-model transition log likelihoods. Frozen
behavior probabilities are persisted even for unadmitted actor versions.
Source/config/runtime/seed/prior identities and artifact hashes prevent
count-only resumption from silently reusing old science. Method RNGs are keyed
by stable identifiers rather than method-list position.

## 8. IDS comparator and support-exclusion controls

The corrected comparator updates exact finite-model likelihood weights and
minimizes the information ratio over policy mixtures using an analytic
pair-support solve. It has no rounded posterior cache or mixture grid. Independent
rescaling of regret and information protects tiny posterior tails from absolute
thresholds; underflow of a still-positive supported probability raises an error.
A common likelihood factor leaves the posterior unchanged exactly.

For $D=a+bx$, $I=c+dx$, the interior stationary mixture satisfies

$$
 x=(da-2bc)/(bd),
$$

when it lies in $(0,1)$; endpoints are also checked. A full-policy conditional
gradient certificate searches for the maximum of
$\overline J(\pi)+(D/2I)\operatorname{MI}(\pi)$. If it improves the restricted
solution, that policy is added and IDS is solved again. The final residual is
scaled to posterior expected regret, not to an arbitrary absolute exploration
floor. The binary models have distinct optimal policies, so model MI and
optimal-policy MI coincide here; a larger family must revisit that identity.

Full-support IDS remains the primary control. `ids_rare` has prior 0.001 on the
shifted mode; `ids_zero` has prior zero and is explicitly labeled misspecified.
With zero shifted support, the confounded old route yields exactly 0.04 regret
per episode in DeepSea and 0.208 in FrozenLake when mode 1 is true. This is a
support-exclusion experiment, not a proof that all Bayesian IDS variants fail.

## 9. Acceptance checks and limits of the fresh slate

The regression suite checks all 22 released policy/model return/KL pairs, both
full-policy oracle coefficients and modes, global weighted-policy responses,
robust costs, target/calibration feasibility and phase-I certificates,
finite-difference actor scores, the zero-price specialization, absence of
frozen-policy actor updates, confidence first-crossing probabilities, immutable
snapshots, single credit, cumulative quotas, tiny IDS priors, exact trajectory
MI, absorbing zero support, context-specific interfaces, stable seeds, and
verified persistence/incompatible resumption.

The short validation slate is stored in
`code/runs/ceac_v90_validation_20261110`, with a separately rendered `report.md`.
It uses 256 episodes, both candidate modes, three seeds, and five methods in each
of the two environments. It is an integration/control slate, not a parameter
sweep. Actor progress and model/design certificates are reported separately.

The current contexts are stationary. A later unannounced within-context shift
will require a declared monitoring/restart or nonstationary likelihood model,
segmentation of historical replay, and matched hazard-aware IDS controls. Old
stationary confidence cannot simply be carried through such a shift. Structural
support-hole constructions must likewise stay inside CEAC's declared candidate
family if its realizability-based guarantees are invoked.

## 10. Earlier actor-only validation and diagnosis (archived)

The completed slate contains **60 runs and 15,360 real-environment episodes**.
All run artifacts verify, and a full resume skips the completed cells by exact
identity. All **22 regression tests** pass. Across **384 CEAC refreshes**, the
maximum recorded primal/dual gap and global column residual are zero at the
reported numerical precision. All 12 CEAC cells finish with the correct
singleton model set, and all 12 actors improve their expected return.
The numerical validation summary is `audits/ceac_v90_validation_20261110.json`.

That does **not** establish CEAC performance superiority. Mean final actor
returns in shifted mode are 0.77984 for DeepSea (optimum 1.024) and 0.89056 for
FrozenLake (optimum 1.127). Full-support IDS reaches the optimal route in the
short slate. Zero-support IDS remains exactly on the confounded old route,
incurring 10.24 and 53.248 cumulative pseudo-regret respectively over 256
episodes. Positive rare-prior IDS sometimes recovers, illustrating why rare
support must not be conflated with missing support.

A post-hoc check of the saved actor tables separates residual random actions
from route preference. Taking the greedy action of each final shifted-mode
actor yields return 0.984 in DeepSea and 1.0538 in FrozenLake, both below their
full optima. Thus the remaining gap is not explained solely by stochastic
execution. These greedified policies are **diagnostics only**; they were not
substituted into the training run or used to inflate its reported performance.

The correct shifted model was first isolated between episodes 3 and 16 in
DeepSea and 6 and 8 in FrozenLake (zero-based indices). The final evidence
prices are zero in that mode, as they should be when the optimal diagnostic
policy has zero regret. Frozen diagnostic trajectories do not train the actor
under the specified on-policy rule. Consequently correct model identification
and a correct experimental design do not instantly teach the actor the optimal
deployment route. The primitive actor still has to learn it through its own
rollouts. Longer-time actor convergence and the coupling between certified
model knowledge and deployment must be checked before using this slate as a
CEAC-recovery claim or launching a larger shift sweep. No planner policy has
been silently substituted for the actor to conceal that gap.


## 11. Deployment correction and expanded comparator slate

The deployment gap above exposed one further interface omission. Appendix D's
`finite-deployment-response` defines an estimator-averaged deployment policy;
Section 4's finite allocation-estimation lemma explicitly requires an exact
deployment response and separately charges pre-certification deployment regret.
Appendix G, Step N4, applies the reward-only certificate at $\beta=0$ to make
deployment optimal without assuming actor SGD has found a global optimum.
Section 3's episode selector, however, says to use the actor whenever no debt is
owed and does not explicitly connect that deployment response to execution.

The previous maintained runner followed that actor-only branch, leaving a
learned model and a certified experimental design disconnected from deployment.
The corrected default makes the connection explicit; `ceac_actor` retains the
actor-only branch as an ablation with the same seeds, design logic and budget.
The v90 manuscript is unchanged.

Let $\xi_t^C$ be the normalized likelihood estimate restricted to the current
plausible model set. Compute a public-model reward response

$$
\widehat\pi_t\in\arg\max_\pi\sum_{m\in\mathcal C_t}\xi_t^C(m)J_m(\pi),
\qquad
\epsilon_t^{\rm deploy}
=\max_{m\in\mathcal C_t}\{J_m^\star-J_m(\widehat\pi_t)\}.
$$

When $\epsilon_t^{\rm deploy}\le10^{-10}$, deployment is certified for every
plausible model. On the anytime model-confidence event, it is therefore equally
accurate for the true model. The proposal uses no realized-mode label. When the
confidence set changes, both design and deployment certificates are refreshed;
a certificate is invalid outside its certified model set.

Execution serves statistical debt, then design debt, then certified deployment
when available, and otherwise the current actor. Frozen certified-deployment
episodes are labeled as such and enter likelihood/replay and execution counts;
they never train the actor off policy. Exact deployment therefore resolves a
policy-execution gap, **not stochastic actor convergence**. In these small
finite-model cases a certificate can precede the first free actor episode, so
some cells have zero actor updates. Those cells validate the planner-assisted
finite implementation, not actor discovery. The report preserves both actor and
deployment performance, counts and policy identities.

Restricting the proposal weights to the confidence set and specifying this
execution priority are explicit implementation choices that resolve the draft's
missing bridge. The all-model reward-gap certificate is independent of how its
proposal is produced. The standing caveats about normalization, estimator
cadence, stationary replay and the coefficient-one theorem still apply.

### Comparators and what is actually matched

The expanded slate contains the corrected CEAC, the actor-only ablation, fixed
offline Lai–Graves, model-confidence optimism, coherent policy-UCB, a policy-level
OIR/IDAC adaptation, uniform coverage, and full/rare/zero-support IDS. Each gets
256 episodes for each environment and mode, with three paired seeds. Only the
offline oracle receives the realized mode; its allocation is solved once and is
never updated using observed rewards or likelihoods. Its count schedule remains
the declared logarithmic cumulative schedule.

The model-confidence baseline optimizes over its anytime likelihood-confidence
set using the same declared model class and full policy responses as CEAC/IDS.
This is a strong control; confidence-driven exploration is not synonymous with
a poorly tuned stepwise bonus.

The three finite-bank controls share one **enriched public bank**, constructed
without knowing the realized mode. It contains both model-optimal policies and
the cheaper survey-to-inner FrozenLake route. Equivalent policies on reachable
trajectories are deduplicated so arbitrary off-path action assignments do not
inflate the number of bandit arms. These controls do not lose because their bank
omits the necessary deployment or diagnostic route. They nevertheless remain
bank-based algorithms and do not share CEAC/IDS's unrestricted response oracle.

Policy-UCB uses the entire attainable episodic return range $R$, found by an
all-outcome finite-horizon recursion. Its radius for arm $i$ after $n$ samples is

$$
 r_n=R\sqrt{\frac{\log(L\pi^2n^2/(3\delta))}{2n}},
$$

where $L$ is the bank size. Hoeffding plus a union bound over arms and counts
spends at most $\delta$. The old hard-coded bonus multiplier 0.18 is not reused
as if it were a calibrated confidence bound.

OIR/IDAC uses the released experiment's finite-policy occupancy-ratio principle,
with the explicit objective

$$
 F(q)=\frac{\sum_iq_i(1.25-\widehat J_i)}
 {0.25+H(\sum_iq_i\widehat d_i)}.
$$

Here $d_i$ is the expected distribution of a uniformly sampled pre-action state
within one episode, including observable survey/noise symbols encoded in that
state. Each observed trajectory contributes unit normalized occupancy mass.
The total occupancy pseudocount is 0.25 per policy, spread across the alphabet;
it is not 0.25 times the number of encoded states. Each policy is sampled once
before ratio updates. One exact softmax-logit gradient step with step size 0.2
follows each episode, using empirical returns and occupancies only. The gradient
is checked against finite differences. The cost reference exceeds every possible
episodic return in both default benchmarks.

This is transparently a **finite episodic policy-level adaptation**, not a claim
to reproduce the original infinite-horizon OIR/IDAC implementation. Uniform
coverage and policy-UCB use the identical public bank. These comparisons do not
establish superiority over every occupancy objective, confidence method or
neural function approximator.

### Reporting and math rendering

The new immutable evidence is in
`runs/ceac_deployment_comparison_20261110`; earlier cells remain untouched.
`ceac.report` renders expected-return, cumulative-pseudo-regret and cumulative
model-KL traces, plus deployment/prior ablations, from verified records. Bands
show the observed seed range and are not mislabeled confidence intervals.
Primitive interaction counts are reported alongside episode budgets. Runtime
source identity excludes the report/figure modules, whose hashes are recorded
separately in the rendering audit. Cosmetic edits therefore do not invalidate
scientific experiment caches. The current deployment renderer refuses to
reinterpret the archived actor-only slate as certified CEAC.

Markdown does not compile LaTeX itself. The previous reports used `\(...\)` and
`\[...\]`, which are not consistently recognized by Markdown math extensions.
The mathematical reports now use `$...$` and `$$...$$` outside code blocks. A
math-aware Pandoc parse verifies recognition, and a compiled PDF is provided as
a viewer-independent alternative. No manuscript TeX sources were modified.


## 12. Completed deployment comparison

The fresh comparison completed **120 runs and 30,720 environment episodes**.
All **31 regression tests** pass, including certificate revocation, live
64-episode gap closure, fixed-oracle allocation, finite-difference OIR gradients,
confidence-range calibration and trajectory-only comparator isolation.
Cached-prefix checks also certify the correct reward response in all 12 earlier
CEAC cells once their model-confidence set resolves.

In every one of the 12 fresh CEAC cells, last-window expected return matches the
paired fixed offline oracle to numerical tolerance. In shifted mode this is
**1.024 for DeepSea and 1.127 for FrozenLake**. Normal-mode last-window means are
0.983531 and 0.917687, including the scheduled diagnostic episodes; the separate
certified deployment responses are optimal at 0.984 and 0.919.

Mean cumulative pseudo-regret after 256 episodes in shifted mode:

| Method | DeepSea | FrozenLake |
|---|---:|---:|
| CEAC with certified deployment | 2.4312 | 2.4368 |
| CEAC actor-only ablation | 72.5510 | 60.3118 |
| Fixed offline Lai–Graves | 0 | 0 |
| Model-confidence optimism | 0 | 0 |
| Policy-UCB on enriched bank | 12.8967 | 40.8458 |
| OIR/IDAC policy adaptation | 47.3577 | 63.1475 |
| Uniform coverage | 77.0267 | 64.0565 |
| IDS with full support | 0.0280 | 0.0732 |
| IDS with rare prior 0.001 | 6.2937 | 30.6189 |
| IDS with zero shifted support | 10.2400 | 53.2480 |

These results close the diagnosed deployment gap and reproduce the intended
support-exclusion failure. They **do not show CEAC beating full-support IDS or
model-confidence optimism** on this simple binary family. Those controls solve
the problem with less startup cost. The plots retain them, alongside the
finite-budget advantages over the tested coverage/ratio and rare/zero-prior
arms. No confidence level, seed, model support or reward was changed to create
a more favorable ranking.

The comparison report includes PNG and vector-PDF figures. The mathematical
ledger and comparison report also compile to PDF, and the run contains a
hash-checked archive of its runtime source and regression tests. No historical
run, published figure or manuscript source was overwritten.
