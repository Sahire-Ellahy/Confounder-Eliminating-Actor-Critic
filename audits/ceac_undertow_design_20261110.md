# Undertow DeepSea and False Beacon Frozen Lake: directed-exploration study

This is a controlled refinement of the maintained CEAC benchmarks, not a rewrite
of the v90 manuscript. Existing studies and the released environments remain
historical evidence. The maintained DeepSea environment has a new optional flag;
its default behavior remains unchanged.

## Scope and slate

The full study uses two environments, two fixed task modes, four public benchmark
settings, seven methods and seeds 0,1,2: **336 cells, each 512 episodes**. The short
16-episode, one-seed slate is an integration test, not performance evidence.

Methods: certified CEAC; fixed offline Lai–Graves; model-confidence optimism;
policy-UCB; finite episodic policy-level OIR/IDAC; uniform public-bank coverage;
and full-support Bayesian IDS. This treatment does not change IDS prior support.
All methods receive identical physical dynamics. The three bank-based controls
share an enriched public bank that includes both candidate-model optima and useful
diagnostic policies. Their online updates use their own returns/occupancies.

| Setting | Nuisance alphabet M | Route probability separation e |
|---|---:|---:|
| strong_signal_m4 | 4 | 0.25 |
| weak_signal_m4 | 4 | 0.05 |
| weak_signal_m16 | 16 | 0.05 |
| weak_signal_m64 | 64 | 0.05 |

Only M changes among the three weak-signal arms. The safe-route value, risky-route
values, diagnostic signals and observation KL remain fixed. The seed-policy count
also remains fixed. M counts observed nuisance outcomes, not extra physical coves
or additional task hypotheses. No linear-in-M route-bank coverage claim is made.

## DeepSea construction

The eight-step rightward route pays terminal reward 0.5 and movement cost 0.002
per step. Its expected return is 0.484 in both modes. At the gate, the challenge
adds +0.5 or -0.5 with positive-outcome probability $0.5-e$ or $0.5+e$.
Consequently the challenge route has values $0.484-e$ and $0.484+e$.

A probe costs 0.02 and observes Bernoulli(0.15) or Bernoulli(0.85), without
preventing terminal arrival. Probe-then-safe therefore costs 0.02 in mode 0 and
provides

$$
k_D = D_{\mathrm{KL}}(\operatorname{Bern}(0.15)\|\operatorname{Bern}(0.85))
    = 0.7\log(17/3) \approx 1.21422074.
$$

The optional distributed nuisance action advances depth without horizontal
progress, charges 0.005, and emits a uniform symbol in an M-element alphabet. Its
law is identical in both task modes. It sacrifices terminal reachability. The
sonar-sweep seed takes this action at every depth; its return is -0.04, its
trajectory task KL is zero, and its normalized pre-action occupancy entropy is

$$
H(d_{\mathrm{sweep}})=\log 8+\frac78\log M.
$$

The existing TV route remains in the public seed set. There is exactly one added
sonar-sweep seed for every refined DeepSea setting, including the low-M control.
CEAC pays its normal one-episode initialization for this seed; that cost is not
removed from reported regret.

## FrozenLake construction

The geometry and allowed moves are retained. The inner route has expected
terminal reward 0.5, 11 movement steps, and value 0.489 in both modes. Outer-dock
success probabilities are $0.502-e$ and $0.502+e$, with unit terminal reward and
13 movement steps, yielding values $0.489-e$ and $0.489+e$. The safe dock pays
0.3. The deep survey fee is 0.02; its signal law stays Bernoulli(0.15/0.85).

Reed symbols are uniform on M outcomes and independent of task mode. The fixed
reed-scan route ends at the safe dock, so both its return and its trajectory KL
are invariant to M. The symbol is retained in the observable state after the
reed visit, increasing occupancy entropy. Full-policy separation supplies the
cheaper survey-and-inner-route policy; we do not restrict CEAC to the original
six routes or omit that useful route from the comparison bank.

## Exact pre-training checkpoints

For every setting and environment, `environment_audit.json` records:

- shared positive transition support, finite-horizon state construction and the
  common-policy response checks performed by the model compiler;
- exact optimal and public-policy values, task KL, normalized occupancy entropy,
  and the OIR objective;
- zero task information from the safe route and nuisance route, with strictly
  positive regret on the nuisance route;
- full-policy Lai–Graves primal/dual and column certificates;
- risky-route regret per expected nat compared with the optimal diagnostic design.

For weak DeepSea, risky-route KL is approximately 0.02006707. Mode-0 regret per
expected nat is $0.05/0.02006707\approx2.49164$, versus
$0.02/1.21422074\approx0.01647$ for the probe: a ratio of approximately **151.27**.
The exact Shores separator yields approximately **137.48**. These are design-cost
ratios, not predicted wall-clock or finite-sample speedups.

The OIR comparator minimizes

$$
F(q)=\frac{1.25-\sum_iq_i\widehat J_i}
 {0.25+H(\sum_iq_i\widehat d_i)}.
$$

A zero-information, strictly suboptimal policy with lower F than the
reward-optimal policy is an explicit objective-mismatch witness. The audit checks
this inequality on the actual environment; it does not infer learning outcomes
from it. Normalized occupancy is $E[N_s/T]$, not $E[N_s]/E[T]$. The exact auditor
uses forward visitation times a backward $E[1/T\mid s]$ factor; a regression test
covers unequal episode lengths.

## Allocation-bound calibration

The old design-mass cap of 20 is below the approximately 49.83 challenge trials
needed for one expected nat under the weak signal. It can therefore produce a
positive bounded-oracle cost even when a reward-optimal route supplies free
information. This slate uses **mass bound 128 and normalization 128**, recorded
in every run. The public-bank enrichment also uses 128 for refined settings.
The exact audit verifies zero mode-1 oracle cost. Confidence delta remains 0.05;
coverage constraints, debt accounting, certified deployment and the actor update
are unchanged. This is a declared numerical design-domain enlargement.

## Logging, comparisons and limits

Each physical trajectory logs realized return, evaluator-only policy value and
regret, task KL, primitive steps, and diagnostic/nuisance/risky-route events.
Coverage is recorded both as full observed states (including time and nuisance
symbols) and physical grid locations. Event labels and realized mode never enter
learner updates. Figures display return/regret/evidence traces, event counts, and
sensitivity to M. Default bands now show mean plus/minus one sample SD across
three paired seeds; the observed seed range remains an optional descriptive
view. Neither is a confidence interval. Episode budgets match; primitive-step and planning costs are reported
separately rather than claimed to be equal.

The runner now accepts public benchmark names and two CPU workers, while retaining
the original launch interface, cache validation and default legacy setting.
Reports are rendered only after all requested cells complete and hashes verify.
The IDS mutual-information calculation skips exactly zero-KL state/action rows;
this changes computation cost, not the information objective. Public-bank maximum-
information enrichment breaks primary-score ties by average candidate-model
reward (numerical tie tolerance 1e-12); an exact regression checks equal primary
information and invariant route cost across M. This prevents an arbitrary tied
route from introducing extra reed fees into only one setting.

The experiment is **paired stationary modes**, not unannounced within-run
nonstationarity. Actual nuisance-only changes, abrupt task switches and repeated
contexts need a separate shared change protocol and valid confidence reset/reuse
rules. These are follow-up experiments, not claims established by this slate.
The exact-model CEAC deployment specialization and finite-policy OIR/IDAC scope
from the previous deviation ledger still apply. Full-support IDS is allowed to
succeed; it is a diagnostic-information control.

## Approved names and stable identifiers

The environment names are **Undertow DeepSea** and **False Beacon Frozen Lake**.
Figures, reports and documentation use these names. The persistent `deepsea` and
`frozenlake` identifiers and existing result paths remain stable.

## Prior-support extension

A separate study, `runs/ceac_prior_support_20261110`, executes **48 new cells**:
strong_signal_m4 and weak_signal_m64; both environments and modes; seeds 0,1,2;
512 episodes; IDS priors 0.001 and zero on mode 1. Its consolidated report reuses
**168 completed controls covering all seven original methods**, for 216 displayed
runs. The initial three-control report has been expanded without new training.
The report checks source/runtime hashes, public environment definitions, CEAC
settings, episode budgets, contexts, modes and seeds before combining curves.
It reads the original trajectories without copying or rerunning them. Reference
study and per-run manifest hashes remain in the report audit.

All IDS arms use the same implementation. Only the mode-1 prior changes:

| Arm | Initial mode-1 probability | Interpretation |
|---|---:|---|
| ids | 0.5 | Well-supported Bayesian control |
| ids_rare | 0.001 | Positive support, potentially long diagnostic waiting time |
| ids_zero | 0 | Explicit prior-support exclusion |

No environment or training-runtime changes were needed: the old route already
has identical trajectory laws under the two modes. All likelihoods on the shared
outcome support are positive, so Bayes' rule gives

$$
p_t(1)=\frac{p_0(1)L_t(1)}{p_0(0)L_t(0)+p_0(1)L_t(1)}=0
\quad\text{whenever }p_0(1)=0.
$$

The implemented IDS convention assigns ratio zero to a policy with zero
subjective regret and zero information. With singleton support on mode 0, IDS
therefore repeatedly chooses the old optimal route. That route gives no evidence
against the supported model, while its mode-1 value gap is $\Delta=e>0$.
Consequently

$$
R_T=T\Delta.
$$

For 512 episodes the exact pseudo-regret is **128** in strong_signal_m4 and
**25.6** in weak_signal_m64, in either environment's mode 1. Mode 0 is a matched
negative control with zero pseudo-regret for this arm. These are predictions for
expected policy regret; realized stochastic returns need not equal their means.
The deployed old route can still reach a terminal location. The failure is
persistent choice of an inferior route, not necessarily zero terminal success.

The report checks the absorbing posterior, zero subjective regret/information,
zero true-model KL along deployment, and the linear-regret identity at every
saved episode. Posterior plots show exact zero explicitly using a symlog axis.
Positive priors must remain numerically positive; they are not rounded down or
reclassified as support exclusion. A dedicated test also supplies a real forced
probe observation: positive prior odds update by its likelihood ratio, while
zero prior mass stays zero.

CEAC and confidence controls retain both models in their candidate family. This
is an explicit information-assumption difference. The comparison does not refute
correctly specified full-support Bayesian IDS, and does not establish that CEAC
can discover a task model outside its own declared family. It is a stationary
out-of-support-mode treatment; unknown change points, Bayesian change models,
posterior resets and model-class expansion require separate experiments.


## Saved-history audit and descriptive time traces

For seed s, let X_{s,t} be a logged cumulative quantity at training episode t.
The displayed center and shaded band are

$$
\overline X_t=\frac{1}{n}\sum_{s=1}^{n}X_{s,t},\qquad
S_t=\sqrt{\frac{1}{n-1}\sum_{s=1}^{n}(X_{s,t}-\overline X_t)^2},\qquad
[\overline X_t-S_t,\overline X_t+S_t].
$$

Here n=3 training seeds per curve. S_t is the sample standard deviation, not the
standard error; the band has no claimed confidence level or simultaneous
coverage. No variance is estimated from a single seed. The optional seed-range
view uses the observed minimum and maximum. Means, SDs, ranges and band edges
are exported without clipping in `trace_statistics.csv.gz`.

Policy-value curves first apply causal smoothing separately within each seed,
with w_t=min(16,t):

$$
\widetilde J_{s,t}=\frac{1}{w_t}\sum_{u=t-w_t+1}^{t}J_{m^\star}(\pi_{s,u}).
$$

These are exact evaluator values of the executed policies, including exploration;
they are not actor-only validation returns. Only the evaluator receives the
realized mode. Cumulative regret and observed evidence are not smoothed.
Independent replication remains the seed, not episodes or overlapping windows.

The report reconstructs observed evidence directly from saved trajectories:

$$
\ell_t(m)=\sum_{u=1}^{t}\sum_{h=0}^{H_u-1}
\log p_m(o_{u,h+1},r_{u,h},d_{u,h}\mid o_{u,h},a_{u,h}),\qquad
Z_t=\ell_t(m^\star)-\ell_t(1-m^\star).
$$

The adaptive behavior policy is common to the candidate-model likelihoods when
conditioning on the recorded history, so its action probabilities cancel from
the likelihood ratio. Z_t is an evaluator-oriented observed statistic and can
be negative. It differs from the separately plotted cumulative expected model
KL, which is computed by exact policy evaluation. For CEAC, the audit verifies
at every episode that the recorded mixture and confidence set equal

$$
q_t(m)=\frac{\exp\ell_t(m)}{\sum_j\exp\ell_t(j)},\qquad
\mathcal C_t=\{m:\max_j\ell_t(j)-\ell_t(m)\le\log(1/\delta)\}.
$$

This is the two-model threshold (in general, log((M-1)/delta)); the mixture q_t
is CEAC's equal-score model estimate, not the misspecified IDS prior treatment.
The audit also verifies each reported immutable-policy credit against the
physical saved episodes, plus actor/deployment episode counts. The ledger report
exposes policies with quotas or statistical requests; an absent ledger key does
not imply that a deployment trajectory was discarded. `history_audit.json`
records the reconstruction residual and counts without modifying training data.

## What the completed slate says about the learned evidence critic

Of the 48 CEAC cells in the 336-run slate, **34 execute zero actor/evidence-critic
updates**. The others execute at most 14; **no learned actor proposal is admitted**
in any cell. The actor and its two TD critics are updated only on fresh episodes
labeled `actor`. Design and certified-deployment trajectories still contribute
to likelihood evidence and immutable-policy counts, but are not replayed as
on-policy actor updates.

The strong performance therefore validates directed finite-model design,
cumulative likelihood accounting and reward-only certified deployment. It does
not identify a causal benefit from learned evidence-value approximation, actor
policy discovery, or replay resampling. Exact declared-model evaluation provides
the design coefficients and separation/deployment responses in this lane.

A mechanism-specific empirical claim would require activating the learned
estimator in those interfaces and comparing matched evidence-pricing and
history-reuse ablations under the same information and interaction budgets.
Those experiments have not been supplied by the cached slate. The present
report keeps that limitation explicit rather than attributing exact-model
planning successes to a critic that was mostly inactive.


## Aggregate nuisance comparison and interpretation

The aggregate figure holds route separation at 0.05 and varies only M=4,16,64.
Each row is one environment; mode 0 and mode 1 remain separate figures.
Three columns report cumulative pseudo-regret, added regret relative to M=4,
and nuisance visits. For each matching seed s, the added-regret observation is

$$
D_s(M)=R_{s,512}(M)-R_{s,512}(4).
$$

The displayed uncertainty is the sample SD of D_s, after pairing, rather than a
sum of marginal variances or an SD over pooled environments. Non-nuisance run
configurations and exact seed sets must match. The source endpoints and derived
statistics are persisted in `nuisance_sensitivity.json` and hashed with the
report. The strong-signal M=4 control is not placed on this axis.

In normal mode, mean OIR/IDAC regret grows from 75.758 to 93.282 in DeepSea and
from 48.319 to 49.110 in Frozen Lake. CEAC stays at 1.245 and 1.070, respectively.
Full-support IDS also stays stable. Confidence, policy-UCB and uniform coverage
are mostly invariant to M, although their baseline costs differ substantially.
The evidence supports a specific nuisance sensitivity, not increasing failure
of every comparator. M is nuisance-symbol diversity, not physical undertow
strength, diagnostic signal amplitude, or a hidden temporal change frequency.

## Meaning of the plotted CEAC and oracle labels

**CEAC (exact-model single loop)** denotes the maintained finite-model
specialization. All real trajectories pass through one episode scheduler, with
actor/two-critic updates restricted to actor episodes. Exact finite-model
optimization supplies diagnostic separation and the reward-only deployment
response; this is not an empirical-only actor/critic policy-discovery result.

For confidence set C_t, a reward-only response is certified when

$$
\max_{m\in\mathcal C_t}\bigl[J_m^\star-J_m(\widehat\pi_t)\bigr]
\le\tau,\qquad \tau=10^{-10}.
$$

On the model-confidence event, this bounds the true-model deployment gap.
The known candidate family supplies the model laws; actual trajectories identify
which laws remain plausible. The real-mode label is never supplied to CEAC.
Statistical requests and design debt retain priority over certified deployment.

The **offline Graves–Lai oracle** is given the real mode before execution. It
solves the normalized true-model allocation once. Its normalized weights stay
fixed while execution quotas increase on the declared evidence scale. Otherwise
it deploys the true-model optimal policy. Its finite-budget curve includes those
scheduled experiment costs and is not itself a finite-time lower-bound curve.

V90's tabular policy separator and Appendix D's deployment response support the
exact-model formulation; connecting the reward-only certificate to free-episode
execution is the documented implementation refinement. One shared episode loop
does not assert that every estimator or timing choice is a literal reproduction
of the manuscript pseudocode, nor that the learned critics caused the observed
performance.
