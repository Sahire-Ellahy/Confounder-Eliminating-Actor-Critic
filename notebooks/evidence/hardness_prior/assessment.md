# Environmental violation of a fixed Bayesian prior

Verified 90 completed cells; 3 held-out seeds per cell; 32,768 two-step episodes. No training is launched by report rendering.

Each panel follows one algorithm as dose changes. Lines are seed means; shading is one sample standard deviation across seeds, not a confidence interval. Coincident lines indicate similar measured behavior. Lower policy gap is better. Endpoint gap averages the final 256 episodes.

These are new short junction variants. They do not establish the same result on the historical DeepSea/Frozen Lake grids. Task 1 is the burden study’s primary scope; both tasks are retained in the prior-shift study. The two burden variants share the same zero-burden kernel.

## Dose response

![Each algorithm as environmental pathology increases](dose_response.png)

## Within-algorithm learning traces

![False Beacon junction: each panel follows one algorithm as dose changes](false_beacon_junction_task1_window_gap.png)

[Task 0 traces](false_beacon_junction_task0_window_gap.png) are retained as the second task, not pooled into the primary figure.

## Measured changes from the weakest to strongest dose (task 1)

| Variant | Algorithm | Final gap: weakest → strongest | Cumulative regret: weakest → strongest |
|---|---|---:|---:|
| False Beacon junction | CEAC · reward + evidence critics | 0.0048 → 0.0049 | 622.7 → 618.1 |
| False Beacon junction | IDS · frozen correlated prior | 0.0000 → 0.4000 | 0.0 → 13107.1 |
| False Beacon junction | IDS · product-support prior | 0.0000 → 0.0000 | 0.0 → 0.0 |

Persistent endpoint loss and larger accumulated learning cost are different findings. A control that eventually recovers must not be described as permanently failing.

## Complete dose table

| Variant | Task | Algorithm | Dose | Final policy gap, mean ± SD | Cumulative regret | Nuisance episodes |
|---|---:|---|---:|---:|---:|---:|
| False Beacon junction | 0 | CEAC · reward + evidence critics | 0 | 0.1347 ± 0.1935 | 4829.0 | 0.0% |
| False Beacon junction | 0 | CEAC · reward + evidence critics | 0.25 | 0.0230 ± 0.0002 | 1666.1 | 0.0% |
| False Beacon junction | 0 | CEAC · reward + evidence critics | 0.5 | 0.0231 ± 0.0002 | 2153.5 | 0.0% |
| False Beacon junction | 0 | CEAC · reward + evidence critics | 0.75 | 0.1099 ± 0.1505 | 3805.5 | 0.0% |
| False Beacon junction | 0 | CEAC · reward + evidence critics | 1 | 0.1346 ± 0.1935 | 4545.0 | 0.0% |
| False Beacon junction | 0 | IDS · frozen correlated prior | 0 | 0.0000 ± 0.0000 | 0.1 | 0.0% |
| False Beacon junction | 0 | IDS · frozen correlated prior | 0.25 | 0.0000 ± 0.0000 | 0.3 | 0.0% |
| False Beacon junction | 0 | IDS · frozen correlated prior | 0.5 | 0.0000 ± 0.0000 | 106.3 | 0.0% |
| False Beacon junction | 0 | IDS · frozen correlated prior | 0.75 | 0.2048 ± 0.0346 | 6579.4 | 0.0% |
| False Beacon junction | 0 | IDS · frozen correlated prior | 1 | 0.4000 ± 0.0000 | 13069.2 | 0.0% |
| False Beacon junction | 0 | IDS · product-support prior | 0 | 0.0000 ± 0.0000 | 0.8 | 0.0% |
| False Beacon junction | 0 | IDS · product-support prior | 0.25 | 0.0000 ± 0.0000 | 0.8 | 0.0% |
| False Beacon junction | 0 | IDS · product-support prior | 0.5 | 0.0000 ± 0.0000 | 0.8 | 0.0% |
| False Beacon junction | 0 | IDS · product-support prior | 0.75 | 0.0000 ± 0.0000 | 0.8 | 0.0% |
| False Beacon junction | 0 | IDS · product-support prior | 1 | 0.0000 ± 0.0000 | 0.8 | 0.0% |
| False Beacon junction | 1 | CEAC · reward + evidence critics | 0 | 0.0048 ± 0.0000 | 622.7 | 0.0% |
| False Beacon junction | 1 | CEAC · reward + evidence critics | 0.25 | 0.0048 ± 0.0000 | 624.0 | 0.0% |
| False Beacon junction | 1 | CEAC · reward + evidence critics | 0.5 | 0.0049 ± 0.0000 | 623.4 | 0.0% |
| False Beacon junction | 1 | CEAC · reward + evidence critics | 0.75 | 0.0049 ± 0.0000 | 621.1 | 0.0% |
| False Beacon junction | 1 | CEAC · reward + evidence critics | 1 | 0.0049 ± 0.0000 | 618.1 | 0.0% |
| False Beacon junction | 1 | IDS · frozen correlated prior | 0 | 0.0000 ± 0.0000 | 0.0 | 0.0% |
| False Beacon junction | 1 | IDS · frozen correlated prior | 0.25 | 0.0000 ± 0.0000 | 0.2 | 0.0% |
| False Beacon junction | 1 | IDS · frozen correlated prior | 0.5 | 0.0000 ± 0.0000 | 10.1 | 0.0% |
| False Beacon junction | 1 | IDS · frozen correlated prior | 0.75 | 0.2667 ± 0.2309 | 8738.2 | 0.0% |
| False Beacon junction | 1 | IDS · frozen correlated prior | 1 | 0.4000 ± 0.0000 | 13107.1 | 0.0% |
| False Beacon junction | 1 | IDS · product-support prior | 0 | 0.0000 ± 0.0000 | 0.0 | 0.0% |
| False Beacon junction | 1 | IDS · product-support prior | 0.25 | 0.0000 ± 0.0000 | 0.0 | 0.0% |
| False Beacon junction | 1 | IDS · product-support prior | 0.5 | 0.0000 ± 0.0000 | 0.0 | 0.0% |
| False Beacon junction | 1 | IDS · product-support prior | 0.75 | 0.0000 ± 0.0000 | 0.0 | 0.0% |
| False Beacon junction | 1 | IDS · product-support prior | 1 | 0.0000 ± 0.0000 | 0.0 | 0.0% |

## What the evidence permits

The primitive controls receive trajectories and public action/state masks. IDS and CEAC receive declared candidate likelihoods and exact model-based policy responses. The oracle additionally receives the realized task/model. These are different information assumptions, not a matched model-free architecture comparison.

Primary CEAC uses the manuscript’s reward critic, price-conditioned evidence critic, and policy-labeled replay master. Statistical debt and design debt precede the actor; there is no exact-deployment branch. Exact model responses serve global certificates and target calibration. Oracle Graves–Lai additionally receives the true model. Historical exact-deployment results are excluded from this primary comparison.

CEAC made 870000 actor updates in total, 870000 with nonzero evidence prices. This does not isolate a learned-critic or replay-buffer causal contribution.

The correlated prior is frozen at (task 0, q=0.1) and (task 1, q=0.9). Both task labels and all observable outcomes remain possible, but the true joint law leaves prior support for d>0. Product-support IDS and CEAC use the same ten-model Cartesian class throughout. A robust product-support IDS result therefore limits any claim that this is an intrinsic failure of information-directed sampling.

| Task | IDS prior | Shift | True joint model supported | Final wrong-task probability |
|---:|---|---:|---|---:|
| 0 | IDS · frozen correlated prior | 0 | True | 0.0000 |
| 0 | IDS · frozen correlated prior | 0.25 | False | 0.0000 |
| 0 | IDS · frozen correlated prior | 0.5 | False | 0.0000 |
| 0 | IDS · frozen correlated prior | 0.75 | False | 0.0707 |
| 0 | IDS · frozen correlated prior | 1 | False | 1.0000 |
| 0 | IDS · product-support prior | 0 | True | 0.0000 |
| 0 | IDS · product-support prior | 0.25 | True | 0.0000 |
| 0 | IDS · product-support prior | 0.5 | True | 0.0000 |
| 0 | IDS · product-support prior | 0.75 | True | 0.0000 |
| 0 | IDS · product-support prior | 1 | True | 0.0000 |
| 1 | IDS · frozen correlated prior | 0 | True | 0.0000 |
| 1 | IDS · frozen correlated prior | 0.25 | False | 0.0000 |
| 1 | IDS · frozen correlated prior | 0.5 | False | 0.0000 |
| 1 | IDS · frozen correlated prior | 0.75 | False | 0.6667 |
| 1 | IDS · frozen correlated prior | 1 | False | 1.0000 |
| 1 | IDS · product-support prior | 0 | True | 0.0000 |
| 1 | IDS · product-support prior | 0.25 | True | 0.0000 |
| 1 | IDS · product-support prior | 0.5 | True | 0.0000 |
| 1 | IDS · product-support prior | 0.75 | True | 0.0000 |
| 1 | IDS · product-support prior | 1 | True | 0.0000 |

**Held-out easy-case exception:** task 0, seed 0: gap 0.3581. Independent calibration did not guarantee success on every held-out seed. These results remain in the means and bands; they prevent a universal CEAC robustness claim.

Full individual-seed endpoints are in `endpoints.json`/`endpoints.csv`; all registered conditions are retained. Three seeds support a descriptive initial study, not a population-level performance guarantee.
