# Sequential Combination & Bidirectional Locks

Verified **192** completed cells (3 held-out seeds per condition, 65,536 episodes per run). Independent easy-case calibration passed before sweep execution.

## Headline Dose Response (Task 1)

![Dose response across pathology levels](dose_response.png)

## Mechanism Diagnostics

![Measured distraction and confounder mechanisms](mechanism_breakdown.png)

## Within-Algorithm Learning Fans

![Bidirectional Combination Lock cumulative regret fans](bidirectional_lock_task1_cumulative_regret.png)

![Bidirectional Combination Lock expected policy gap trajectories](bidirectional_lock_task1_window_gap.png)

![Combination Lock cumulative regret fans](combination_lock_task1_cumulative_regret.png)

![Combination Lock expected policy gap trajectories](combination_lock_task1_window_gap.png)

## Weakest-to-Strongest Pathology Summary (Task 1)

| Environment | Algorithm | Final gap: min → max | Regret: min → max | Primary distraction (max dose) |
|---|---|---:|---:|---:|
| Bidirectional Combination Lock | IDAC · occupancy ratio | 0.0031 → 0.8916 | 1,260.4 → 58,791.9 | sink 99.2% / decoy 0.4% |
| Bidirectional Combination Lock | UCB Q-learning | 0.0021 → 0.6208 | 1,346.7 → 45,688.7 | sink 69.6% / decoy 11.5% |
| Bidirectional Combination Lock | Optimistic POMD | 0.0067 → 0.8951 | 6,533.7 → 58,663.9 | sink 98.8% / decoy 0.4% |
| Bidirectional Combination Lock | IDS · decoy-lock prior | 0.4000 → 0.4000 | 26,214.4 → 26,214.4 | sink 0.0% / decoy 100.0% |
| Bidirectional Combination Lock | IDS · model-MI, uniform prior | 0.0000 → 0.0000 | 0.0 → 0.0 | sink 0.0% / decoy 0.0% |
| Bidirectional Combination Lock | CEAC · reward + evidence critics | 0.0000 → 0.0001 | 443.1 → 1,249.8 | sink 1.4% / decoy 0.8% |
| Bidirectional Combination Lock | CEAC · guarded natural | 0.0014 → 0.0083 | 600.2 → 2,450.9 | sink 3.3% / decoy 0.8% |
| Bidirectional Combination Lock | Oracle Graves–Lai | 0.0000 → 0.0000 | 0.0 → 0.0 | sink 0.0% / decoy 0.0% |
| Combination Lock | IDAC · occupancy ratio | 0.0059 → 0.8306 | 7,340.4 → 57,081.8 | sink 95.3% / decoy 0.0% |
| Combination Lock | UCB Q-learning | 0.0043 → 0.5708 | 1,347.8 → 43,532.3 | sink 67.9% / decoy 0.0% |
| Combination Lock | Optimistic POMD | 0.0062 → 0.8938 | 6,487.2 → 58,577.9 | sink 98.8% / decoy 0.0% |
| Combination Lock | IDS · decoy-lock prior | 0.4000 → 0.4000 | 26,214.4 → 26,214.4 | sink 0.0% / decoy 0.0% |
| Combination Lock | IDS · model-MI, uniform prior | 0.0000 → 0.0000 | 0.0 → 1.5 | sink 0.0% / decoy 0.0% |
| Combination Lock | CEAC · reward + evidence critics | 0.0000 → 0.0001 | 454.6 → 1,090.8 | sink 1.1% / decoy 0.0% |
| Combination Lock | CEAC · guarded natural | 0.0014 → 0.0079 | 603.7 → 2,182.3 | sink 2.9% / decoy 0.0% |
| Combination Lock | Oracle Graves–Lai | 0.0000 → 0.0000 | 0.0 → 0.0 | sink 0.0% / decoy 0.0% |

## Paired CEAC Advantage at Maximum Pathology (Dose = 4, Task 1)

| Environment | Comparator | Regret saved vs CEAC (mean ± SD) | Added advantage over min dose | CEAC lower on all seeds |
|---|---|---:|---:|---|
| Bidirectional Combination Lock | IDAC · occupancy ratio | +57,542.1 ± 220.6 | +56,724.8 ± 492.5 | True |
| Bidirectional Combination Lock | UCB Q-learning | +44,439.0 ± 31.4 | +43,535.4 ± 46.9 | True |
| Bidirectional Combination Lock | Optimistic POMD | +57,414.2 ± 135.6 | +51,323.6 ± 116.5 | True |
| Bidirectional Combination Lock | IDS · decoy-lock prior | +24,964.6 ± 135.6 | -806.7 ± 143.9 | True |
| Bidirectional Combination Lock | IDS · model-MI, uniform prior | -1,249.8 ± 135.6 | -806.7 ± 143.9 | False |
| Bidirectional Combination Lock | CEAC · guarded natural | +1,201.2 ± 268.0 | +1,044.1 ± 269.3 | True |
| Bidirectional Combination Lock | Oracle Graves–Lai | -1,249.8 ± 135.6 | -806.7 ± 143.9 | False |
| Combination Lock | IDAC · occupancy ratio | +55,991.0 ± 2,169.6 | +49,105.2 ± 12,589.5 | True |
| Combination Lock | UCB Q-learning | +42,441.5 ± 337.2 | +41,548.3 ± 392.5 | True |
| Combination Lock | Optimistic POMD | +57,487.1 ± 232.2 | +51,454.5 ± 223.9 | True |
| Combination Lock | IDS · decoy-lock prior | +25,123.6 ± 232.2 | -636.2 ± 236.2 | True |
| Combination Lock | IDS · model-MI, uniform prior | -1,089.3 ± 234.2 | -634.7 ± 237.9 | False |
| Combination Lock | CEAC · guarded natural | +1,091.5 ± 367.8 | +942.4 ± 377.7 | True |
| Combination Lock | Oracle Graves–Lai | -1,090.8 ± 232.2 | -636.2 ± 236.2 | False |

## Complete Endpoint Table

| Environment | Task | Algorithm | Dose | Final gap (mean ± SD) | Cumulative regret (mean ± SD) |
|---|---:|---|---:|---:|---:|
| Bidirectional Combination Lock | 1 | CEAC · reward + evidence critics | 1 | 0.0000 ± 0.0000 | 443.1 ± 8.3 |
| Bidirectional Combination Lock | 1 | CEAC · reward + evidence critics | 2 | 0.0001 ± 0.0000 | 491.4 ± 11.2 |
| Bidirectional Combination Lock | 1 | CEAC · reward + evidence critics | 3 | 0.1334 ± 0.2309 | 9,138.2 ± 14,761.5 |
| Bidirectional Combination Lock | 1 | CEAC · reward + evidence critics | 4 | 0.0001 ± 0.0000 | 1,249.8 ± 135.6 |
| Bidirectional Combination Lock | 1 | CEAC · guarded natural | 1 | 0.0014 ± 0.0000 | 600.2 ± 6.4 |
| Bidirectional Combination Lock | 1 | CEAC · guarded natural | 2 | 0.0037 ± 0.0000 | 936.0 ± 11.2 |
| Bidirectional Combination Lock | 1 | CEAC · guarded natural | 3 | 0.0060 ± 0.0000 | 1,444.5 ± 53.3 |
| Bidirectional Combination Lock | 1 | CEAC · guarded natural | 4 | 0.0083 ± 0.0000 | 2,450.9 ± 146.8 |
| Bidirectional Combination Lock | 1 | IDAC · occupancy ratio | 1 | 0.0031 ± 0.0001 | 1,260.4 ± 706.1 |
| Bidirectional Combination Lock | 1 | IDAC · occupancy ratio | 2 | 0.4705 ± 0.0104 | 36,024.1 ± 2,030.1 |
| Bidirectional Combination Lock | 1 | IDAC · occupancy ratio | 3 | 0.7596 ± 0.1107 | 55,131.4 ± 3,082.2 |
| Bidirectional Combination Lock | 1 | IDAC · occupancy ratio | 4 | 0.8916 ± 0.0100 | 58,791.9 ± 331.8 |
| Bidirectional Combination Lock | 1 | IDS · decoy-lock prior | 1 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Bidirectional Combination Lock | 1 | IDS · decoy-lock prior | 2 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Bidirectional Combination Lock | 1 | IDS · decoy-lock prior | 3 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Bidirectional Combination Lock | 1 | IDS · decoy-lock prior | 4 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Bidirectional Combination Lock | 1 | IDS · model-MI, uniform prior | 1 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | IDS · model-MI, uniform prior | 2 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | IDS · model-MI, uniform prior | 3 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | IDS · model-MI, uniform prior | 4 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | Oracle Graves–Lai | 1 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | Oracle Graves–Lai | 2 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | Oracle Graves–Lai | 3 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | Oracle Graves–Lai | 4 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Bidirectional Combination Lock | 1 | Optimistic POMD | 1 | 0.0067 ± 0.0009 | 6,533.7 ± 42.7 |
| Bidirectional Combination Lock | 1 | Optimistic POMD | 2 | 0.7860 ± 0.0031 | 51,409.9 ± 31.5 |
| Bidirectional Combination Lock | 1 | Optimistic POMD | 3 | 0.8728 ± 0.0000 | 57,196.5 ± 0.0 |
| Bidirectional Combination Lock | 1 | Optimistic POMD | 4 | 0.8951 ± 0.0000 | 58,663.9 ± 0.0 |
| Bidirectional Combination Lock | 1 | UCB Q-learning | 1 | 0.0021 ± 0.0009 | 1,346.7 ± 52.9 |
| Bidirectional Combination Lock | 1 | UCB Q-learning | 2 | 0.1673 ± 0.0060 | 21,082.7 ± 56.0 |
| Bidirectional Combination Lock | 1 | UCB Q-learning | 3 | 0.4449 ± 0.0107 | 37,282.1 ± 135.4 |
| Bidirectional Combination Lock | 1 | UCB Q-learning | 4 | 0.6208 ± 0.0074 | 45,688.7 ± 160.9 |
| Combination Lock | 1 | CEAC · reward + evidence critics | 1 | 0.0000 ± 0.0000 | 454.6 ± 21.7 |
| Combination Lock | 1 | CEAC · reward + evidence critics | 2 | 0.0001 ± 0.0000 | 494.5 ± 13.0 |
| Combination Lock | 1 | CEAC · reward + evidence critics | 3 | 0.0001 ± 0.0000 | 613.2 ± 49.5 |
| Combination Lock | 1 | CEAC · reward + evidence critics | 4 | 0.0001 ± 0.0000 | 1,090.8 ± 232.2 |
| Combination Lock | 1 | CEAC · guarded natural | 1 | 0.0014 ± 0.0000 | 603.7 ± 6.4 |
| Combination Lock | 1 | CEAC · guarded natural | 2 | 0.0035 ± 0.0000 | 905.3 ± 21.6 |
| Combination Lock | 1 | CEAC · guarded natural | 3 | 0.0057 ± 0.0000 | 1,316.4 ± 76.8 |
| Combination Lock | 1 | CEAC · guarded natural | 4 | 0.0079 ± 0.0000 | 2,182.3 ± 151.0 |
| Combination Lock | 1 | IDAC · occupancy ratio | 1 | 0.0059 ± 0.0049 | 7,340.4 ± 10,509.2 |
| Combination Lock | 1 | IDAC · occupancy ratio | 2 | 0.6264 ± 0.1389 | 42,643.1 ± 8,176.1 |
| Combination Lock | 1 | IDAC · occupancy ratio | 3 | 0.6715 ± 0.1655 | 49,985.5 ± 7,468.0 |
| Combination Lock | 1 | IDAC · occupancy ratio | 4 | 0.8306 ± 0.0683 | 57,081.8 ± 2,277.4 |
| Combination Lock | 1 | IDS · decoy-lock prior | 1 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Combination Lock | 1 | IDS · decoy-lock prior | 2 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Combination Lock | 1 | IDS · decoy-lock prior | 3 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Combination Lock | 1 | IDS · decoy-lock prior | 4 | 0.4000 ± 0.0000 | 26,214.4 ± 0.0 |
| Combination Lock | 1 | IDS · model-MI, uniform prior | 1 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Combination Lock | 1 | IDS · model-MI, uniform prior | 2 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Combination Lock | 1 | IDS · model-MI, uniform prior | 3 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Combination Lock | 1 | IDS · model-MI, uniform prior | 4 | 0.0000 ± 0.0000 | 1.5 ± 2.6 |
| Combination Lock | 1 | Oracle Graves–Lai | 1 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Combination Lock | 1 | Oracle Graves–Lai | 2 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Combination Lock | 1 | Oracle Graves–Lai | 3 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Combination Lock | 1 | Oracle Graves–Lai | 4 | 0.0000 ± 0.0000 | 0.0 ± 0.0 |
| Combination Lock | 1 | Optimistic POMD | 1 | 0.0062 ± 0.0004 | 6,487.2 ± 34.5 |
| Combination Lock | 1 | Optimistic POMD | 2 | 0.4830 ± 0.0021 | 47,700.2 ± 80.5 |
| Combination Lock | 1 | Optimistic POMD | 3 | 0.8715 ± 0.0000 | 57,114.6 ± 0.0 |
| Combination Lock | 1 | Optimistic POMD | 4 | 0.8938 ± 0.0000 | 58,577.9 ± 0.0 |
| Combination Lock | 1 | UCB Q-learning | 1 | 0.0043 ± 0.0035 | 1,347.8 ± 101.4 |
| Combination Lock | 1 | UCB Q-learning | 2 | 0.0916 ± 0.0093 | 16,593.2 ± 49.7 |
| Combination Lock | 1 | UCB Q-learning | 3 | 0.3563 ± 0.0251 | 33,801.9 ± 102.6 |
| Combination Lock | 1 | UCB Q-learning | 4 | 0.5708 ± 0.0166 | 43,532.3 ± 266.0 |

## Held-Out Reward & Evidence Critic TD / Bellman Diagnostics

Evaluated all **168** non-oracle `Task 1` sweep cells across log-spaced checkpoints `t ∈ {16 .. 65,536}` on independent held-out rollouts (`512` on-policy + `256` uniform-probe episodes per checkpoint; bit-exact cumulative-regret parity verified `< 1e-6` against `evidence.json.gz`).

![Held-out critic Bellman trajectories](locks_critic_bellman_traces.png)

![Held-out critic dose scaling](locks_critic_dose_scaling.png)

| Environment | Algorithm | Reward Bellman RMSE (L=4, t=65,536) | Evidence Critic RMSE (L=1 → 4) | Active / Stage Critic RMSE (L=1 → 4) |
|---|---|---:|---:|---:|
| Sequential Lock | IDAC · occupancy ratio | 0.0099 | 7.0053 → 3.0554 | 7.0053 → 3.0554 |
| Sequential Lock | UCB Q-learning | 0.5441 | 0.1753 → 0.4648 | 0.1753 → 0.4648 |
| Sequential Lock | Optimistic POMD | 1.0193 | 0.3278 → 2.3685 | 0.3278 → 2.3685 |
| Sequential Lock | IDS · decoy-lock prior | 0.0117 | 49,728.1454 → 3,758.9542 | 0.7588 → 0.0574 |
| Sequential Lock | IDS · model-MI, uniform prior | 0.0034 | 57,599.5099 → 57,453.9727 | 0.0000 → 0.0000 |
| Sequential Lock | CEAC · reward + evidence critics | 0.0274 | 0.0090 → 0.0034 | 0.1667 → 0.2121 |
| Sequential Lock | CEAC · guarded natural | 0.0155 | 0.0103 → 0.0052 | 0.1714 → 0.2112 |
| Bidirectional ('Diabolical') Lock | IDAC · occupancy ratio | 0.0614 | 6.5654 → 2.3111 | 6.5654 → 2.3111 |
| Bidirectional ('Diabolical') Lock | UCB Q-learning | 0.6428 | 0.1820 → 0.6221 | 0.1820 → 0.6221 |
| Bidirectional ('Diabolical') Lock | Optimistic POMD | 1.0198 | 0.3326 → 3.0173 | 0.3326 → 3.0173 |
| Bidirectional ('Diabolical') Lock | IDS · decoy-lock prior | 0.0089 | 50,093.5396 → 3,895.3751 | 0.7644 → 0.0594 |
| Bidirectional ('Diabolical') Lock | IDS · model-MI, uniform prior | 0.0044 | 57,682.2720 → 57,683.7368 | 0.0000 → 0.0000 |
| Bidirectional ('Diabolical') Lock | CEAC · reward + evidence critics | 0.0447 | 0.0046 → 0.0052 | 0.1607 → 0.2015 |
| Bidirectional ('Diabolical') Lock | CEAC · guarded natural | 0.0178 | 0.0111 → 0.0084 | 0.1611 → 0.1975 |
