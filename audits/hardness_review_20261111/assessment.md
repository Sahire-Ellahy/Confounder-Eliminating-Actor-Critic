# Is the current pathology parameter a validated hardness axis?

**No all-seed pass-to-fail learning transition is observed under the declared late-gap criterion.** Rho measures physical diversion probability. It was supplied to the candidate models at every dose; the full-support Bayesian model remained correctly specified. It must not be relabeled as a measured hardness score.

## What the saved runs say

Stationary mode 1; seed means below. A descriptive pass means last-32-episode expected value gap $\le$ 0.01. 0 environment/method pairs pass on every mild-dose seed and fail on every strong-dose seed. The mild endpoint is not called easy merely because its parameter is small.

| Environment | Method | Late gap: mild $\to$ strong | Seeds passing: mild $\to$ strong |
|---|---|---:|---:|
| Undertow DeepSea | UCB Q-learning (Jin et al.) | 0.5276 $\to$ 0.5520 | 0/3 $\to$ 0/3 |
| Undertow DeepSea | Optimistic POMD (softmax) | 0.5310 $\to$ 0.5506 | 0/3 $\to$ 0/3 |
| Undertow DeepSea | IDAC (frequentist actor-critic) | 0.5360 $\to$ 0.5533 | 0/3 $\to$ 0/3 |
| Undertow DeepSea | Bayesian IDS: full support | 0.0000 $\to$ 0.0000 | 3/3 $\to$ 3/3 |
| Undertow DeepSea | Bayesian IDS: misspecified prior | 0.0500 $\to$ 0.0500 | 0/3 $\to$ 0/3 |
| Undertow DeepSea | CEAC (exact-model single loop) | 0.0000 $\to$ 0.0000 | 3/3 $\to$ 3/3 |
| Undertow DeepSea | Offline Graves–Lai oracle | 0.0000 $\to$ 0.0000 | 3/3 $\to$ 3/3 |
| False Beacon Frozen Lake | UCB Q-learning (Jin et al.) | 0.5725 $\to$ 0.5735 | 0/3 $\to$ 0/3 |
| False Beacon Frozen Lake | Optimistic POMD (softmax) | 0.5640 $\to$ 0.5641 | 0/3 $\to$ 0/3 |
| False Beacon Frozen Lake | IDAC (frequentist actor-critic) | 0.5643 $\to$ 0.5643 | 0/3 $\to$ 0/3 |
| False Beacon Frozen Lake | Bayesian IDS: full support | 0.0000 $\to$ 0.0000 | 3/3 $\to$ 3/3 |
| False Beacon Frozen Lake | Bayesian IDS: misspecified prior | 0.0500 $\to$ 0.0500 | 0/3 $\to$ 0/3 |
| False Beacon Frozen Lake | CEAC (exact-model single loop) | 0.0000 $\to$ 0.0000 | 3/3 $\to$ 3/3 |
| False Beacon Frozen Lake | Offline Graves–Lai oracle | 0.0000 $\to$ 0.0000 | 3/3 $\to$ 3/3 |

Every listed primitive control fails the late-gap criterion at the mild endpoint. This is not a demonstrated easy baseline.

The excluded-task prior is a fixed support treatment at every rho. Environmental induction of joint-prior misspecification is a separate, untested mechanism in this sweep.

POMD has all optimistic costs zero in every episode at both endpoints in 2 listed environment/context cases. In those cases its initial uniform policy is unchanged; that budget regime is not a calibrated easy-case learning baseline.

**Check learned-critic attribution:** 36/48 supplied CEAC histories have zero actor/critic updates. These runs use the exact-model planning, likelihood and deployment specialization; they do not isolate benefits of the learned evidence critic or replay estimator.

## Two replacement experiments

1. **Exploration burden:** hold the useful task, reward gap and diagnostic quality fixed; increase the number/depth of irrelevant physical opportunities. First demonstrate that the named controls learn on the zero-distraction task. The prepared chamber study has not passed that calibration: its B=0 case retains the already difficult shoreline.

2. **Joint-prior misspecification:** freeze a Bayesian cue–task model that is correct at shift d=0, then change only the environmental cue relationship. Both task labels keep positive marginal prior mass; the true joint law leaves the prior support. Compare against product-support Bayesian IDS and a CEAC class containing the shifted nuisance laws.

A concrete channel uses prior atoms (task 0,q=0.1) and (task 1,q=0.9), each with mass 1/2. For true task 1, set q(d)=0.9-0.8d. The fixed model interprets every cue as task evidence, with expected log-odds (0.8-1.6d) log(9); the direction reverses beyond d=0.5. This is an analytic design calculation, not a measured IDS learning curve.

CEAC needs a product/composite model class in which an opposite task can have the same q. Those nuisance cues then have zero KL against a relevant matched-nuisance alternative, while an independent reliable diagnostic still distinguishes tasks. **This is an implementation prerequisite:** the current CEAC/IDS planning stack is specialized to two models. CEAC cannot claim robustness to a truth omitted from its own class, and a full-product-support IDS control may remain robust.

## The figure that would answer the question

One panel per algorithm; x = training interactions; y = excess policy loss or cumulative regret; color = low/medium/high validated pathology. Show the same algorithm progressing from successful learning to delayed or failed learning. Use a second compact endpoint-vs-pathology panel only if it adds information. Keep support shifts separate from exploration burden; keep full-support IDS and the privileged oracle visible. Place diagnostics and all additional modes in the linked appendix.

## Before the next full sweep

- Validate an actually learnable zero-distraction case using calibration seeds; freeze budgets and hyperparameters before evaluation seeds.
- Check optimizer activity and bonus saturation; poor learning at every dose does not qualify as an easy-to-hard result.
- Audit true joint-model membership separately from positive observation probabilities and task-label prior mass.
- Keep informative-route value/KL and reward gap fixed; disclose changes in reachable domain and prescribed bonuses.
- For the learned CEAC mechanism, require active actor/critic learning and matched evidence/replay ablations; retain exact-model CEAC as a separate reference.
- Publish flat curves, CEAC losses and successful full-support IDS results as observed. No ranking is guaranteed by the design.
