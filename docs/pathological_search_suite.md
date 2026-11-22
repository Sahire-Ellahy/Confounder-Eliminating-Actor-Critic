# Sequential Combination Locks and Epistatic Signalling Discovery

This extension (`ceac_pathologies`) evaluates Confounder-Eliminating Actor-Critic (CEAC) across two pathological exploration families designed to stress-test horizon bottlenecks, deceptive attractors, and high-entropy biological confounders, without modifying any hash-locked historical package (`ceac`, `ceac_hardness`, `ceac_pseudocode`, `ceac_stability`).

## 1. Sequential Combination Lock & Bidirectional Combination Lock (`locks`)

Both lock environments sweep **lock depth $L \in \{1, 2, 3, 4\}$** (physical horizon $H = L + 1 \in \{2, 3, 4, 5\}$) while holding useful route returns, decision gap, and diagnostic KL strictly invariant across $L$:

- **Shared calibrated easy baseline ($L = 1, H = 2$):** Four root actions share one waypoint, giving identical occupancy entropy $\log 2$ across all deterministic policies:
  - `safe`: deterministic $0.50$.
  - `target`: $\operatorname{Bernoulli}(0.10)$ in task $0$, $\operatorname{Bernoulli}(0.90)$ in task $1$.
  - `probe`: symmetric ternary $(0.46, 0.48, 0.50)$ with probabilities $((1-q)/2, q, (1-q)/2)$, $q \in \{0.15, 0.85\}$, yielding mean $0.48$ and exact KL $0.7 \log(17/3) \approx 1.21422$ nats.
  - `sink`: zero reward.
- **Unidirectional Combination Lock ($L \ge 2$):** Advancing along the live chain requires the alternating key sequence $t \bmod 2$ at steps $t = 0, \dots, L-2$, incurring an anti-shaped step tension $-0.004$ per step (rebated $+(L-1)(0.004)$ at the terminal lock gate so total route returns remain identically $0.50$, $0.10/0.90$, and $0.48$). Any wrong key drops irreversibly into a 3-lane diffusing dead-end sink manifold with zero step cost and zero terminal reward.
- **Bidirectional ("Diabolical") Combination Lock ($L \ge 2$):** From `start`, action $0$ enters the **Left Lock (Decoy Chain)** requiring key $0$ at every step to unlock deterministic $0.50$; action $1$ enters the **Right Lock (Target Chain)** requiring the opposite key $1$ at every step to unlock $\operatorname{Bernoulli}(0.10/0.90)$ (or the $0.48$ gate probe). Wrong keys on either chain drop irreversibly into the diffusing dead sink.

### Predicted and tested failure mechanisms
- **`IDAC` (occupancy-entropy ratio):** Diffusing dead sinks and dual-chain mixing strictly increase state-occupancy entropy $H(d^\pi)$ while avoiding anti-shaped live step costs.
- **`UCB-Q` (Hoeffding optimism):** Horizon-cubed bonuses $\sqrt{H^3 \iota / n}$ and optimistic dead-sink initializations inflate finite-budget exploration cost with $L$.
- **`Optimistic POMD`:** Transition bonuses $H\sqrt{4S\iota/n}$ clip optimistic cost Q-values to zero; uniform softmax reaches depth $L$ with probability $4^{-L}$.
- **`IDS · decoy-lock prior`:** Restricted incumbent support $(1, 0)$ locks onto the $0.50$ decoy chain permanently in task $1$.
- **`CEAC` & `CEAC · guarded natural`:** Certify cross-lock separation via replay-priced Graves–Lai design and propagate $V_R(s) + V_I(s; \beta)$ backward along the $L$-step chain.

---

## 2. Epistatic Signalling Pathway Discovery (`epistasis`)

Models identifying a **rare synthetic-rescue genetic interaction** ($1$ in $G = 128$ genetic subtypes: subtypes $0 \dots 126$ are canalized non-responders where $K_1 \land K_2$ synergy pays $\operatorname{Bernoulli}(0.10)$ and cytostatic care $0.50$ is optimal; subtype $127$ is the epistatically responsive subtype where $K_1 \land K_2$ synergy unlocks $\operatorname{Bernoulli}(0.90)$) amidst **$k \in \{0, 1, 2, 3\}$ off-pathway pleiotropic modules** ($\eta \in \{0, 1\}^k$, total public hypotheses $M = 128 \cdot 2^k \in \{128, 256, 512, 1024\}$):

- **Core causal pathways:** `cytostatic_care` ($0.50$), `epistatic_synergy` ($0.10$ vs $0.90$), and `phospho_probe` (mean $0.48$, $q \in \{0.18, 0.82\}$).
- **$k$ Passenger-Omics Assays ("Omics Siren"):** Administer cytostatic care minus a $0.0001$ sequencing fee (expected return $0.4999$ identically across all $M$ hypotheses via symmetric ternary outcomes) while emitting subtle readouts of passenger bit $\eta_i$ ($0.5 \pm 0.019$). Passenger bits $\eta$ have **zero effect** on optimal therapeutic policy or expected return of any action.
- **$4^k - 1$ Canalized / Pleiotropic Knockouts:** Zero reward and zero KL across all $M$ genotypes, transitioning to $4^k - 1 \in \{0, 3, 15, 63\}$ distinct aberrant transcriptomic states.

### Why CEAC outperforms even full-support Bayesian IDS here
- **Full-support `BayesianIDS` (`ids_product`, uniform over all $128 \cdot 2^k$ models):** Minimizes single-step model information ratio $\Delta(\pi)^2 / I((\theta, \eta); Z^\pi)$. Under rare genetic interaction prevalence ($1/128$), passenger-omics assays yield a strictly smaller model information ratio than causal probing or synergy, trapping exact full-support IDS into profiling irrelevant passenger modules $\eta_1, \dots, \eta_k$ for thousands of episodes before testing causal epistasis.
- **`IDS · canonical additive prior` (`ids_canonical`):** Assigns uniform mass over the $127 \cdot 2^k$ non-epistatic genotypes, incurring linear $0.40 T$ regret in the rare responder.
- **`IDAC`, `UCB-Q`, `Optimistic POMD`:** Degrade sharply across the $4^k - 1$ aberrant pleiotropic states.
- **`CEAC` (same flat $M$-model class, no coordinate hints):** Prunes pointwise-dominated Graves–Lai rows ($\kappa_{m, j'} \ge \kappa_{m, j}$ everywhere) and certifies confounder-equivalent reference sets when no policy-changing pair $(m, j) \in \mathcal{A}$ remains inside $\mathcal{C}_t$. Passenger channels carry zero KL on undominated rows, receiving $\beta = 0$ and $0\%$ passenger-assay visits.

---

## 3. Rare Synthetic-Lethal Pair Discovery on a Real Human GI Map (`gi_screen`)

Grounded in the published K562 CRISPRi gene-level genetic-interaction map of Horlbeck et al. (2018, *Cell* 174:953), Mendeley Data doi:10.17632/rdzk59n6j4.1. `GI_map_treeview.zip` is downloaded once into `data/external/horlbeck2018/` and verified against the published SHA-256 (`ceac_gi/data.py`). Nothing is fetched at run time.

- **Panel (public rule, `select_panel(10, strong=-5, background_floor=-2.5, seed=2018)`):** FDPS–PPP2R1A (GI = −5.24) plus 8 genes whose other 44 pairs all exceed −2.5. In the real map, 9.2% of random 10-gene panels contain a pair at GI ≤ −5, so the interaction is genuinely rare.
- **Hypotheses:** (interacting pair q ∈ {null} ∪ 45 pairs) × (synthetic main-effect bits for k genes), M = 46·2^k. Under q the panel's real GI values hold, except the real SL pair is set to a declared neutral value (0) unless q is that pair, and q carries the real strong value. The true task-1 hypothesis is exactly the published panel. Task 0 is the counterfactual null panel.
- **Actions:** control (0.50); 10 single knockouts (GI-scored reward 0.4999; k of them reveal SYNTHETIC main-effect bits, which never change the optimal action); 45 pair tests with hit probability clip(Φ((−3 − GI)/1.25), 0.01, 0.99) and reward 1 on a hit.
- **Variants:** `gi_arrayed` (primary; sequential arrayed follow-up of pairs, a standard practice) and `gi_multiplexed` (secondary, clearly SYNTHETIC: adds 10 pooled 6-gene block knockouts with a group-test readout, toxic rewards 0.20/0.30; loosely motivated by compressed/multiplexed perturbation designs).
- **Relabelling:** panel gene order is permuted per seed (public seeded permutation) so that no method benefits from the SL pair's index or from deterministic tie-breaking.

### Declared, not estimated
Assay noise σ = 1.25, hit threshold −3, assay floor 0.01, the neutral counterfactual value 0, main-effect profiling halfwidth 0.04 (frozen from the episode-0 IDS allocation check only), and the main-effect nuisance itself (single-gene phenotypes are not in this archive).

### Bayesian controls and a fairness correction
- `ids_additive`: no-interaction prior (excludes the truth; linear-regret control).
- `ids_product`: model-information IDS, uniform prior over all M hypotheses.
- `ids_empirical`: model-information IDS with the data-grounded prior (null mass 0.908).
- `ids_action`: **optimal-action-information IDS** (Russo & Van Roy), data-grounded prior. The legacy `ceac_hardness` IDS measures I(model; Y), which equals I(A*; Y) only when every model has a distinct optimal policy. With decision-irrelevant nuisance these differ: at episode 0, model-information IDS spends 100% of its allocation on main-effect profiling, whereas optimal-action IDS spends 0%. `ids_action` is therefore the strongest fair Bayesian control, and is also added to the epistasis study.

### Registered CEAC limitation and Greedy Price-Ranked Replay extension (`ceac_ranked`)
In the arrayed screen, the only test that separates the null from pair $q$ is the test of $q$ itself, so certified Section-3 CEAC's seed bank must contain all $45$ pair tests plus control (`46` columns $\times 513$ replay-floor episodes $\approx 23,600$ upfront episodes at $T = 65,536$), and unattenuated conservative lower-bound prices $\beta_{0, q}$ cause free actor episodes to continue probing low-reward pairs on `task = 0`.

To address combinatorial pair-screen scaling without modifying the certified Section-3 implementation, `ceac_ranked` (`GreedyPricedCEAC` + `PriceRankedReplay`) implements an explicitly labeled heuristic proxy:
1. **Greedy maximally-disagreeing column selection**: Initializes the active bank with `control` alone (`bank_size = 1`) and resolves $\mathcal{C}_t$ via coordinate-free confounder-projected disagreement $\min_{j' \in C'} I^m(\pi; j')$, assigning strictly zero score to non-causal single-variable main-effect arms and compressing the `Task 1` active bank from `46` columns down to `~2`.
2. **Dynamic evidence-price replay ranking & KKT price attenuation**: Dynamically ranks all buffered strata by $\widehat{J}_t(\pi) + \sum_{r \in \mathcal{R}_t} \beta_{t, r} \tilde{I}_{t, r}(\pi)$ to promote informative replay columns with zero additional environment steps and scales effective actor prices by $\rho_t = \min(1, \text{scale}(t)/(1 + t_{\text{actor}}))$ once `DebtLedger` quotas are satisfied.

---

## 4. Path-Dependent Signalling Rewiring & Canalized K562 Genetic Risk (`ceac_pathdep`, $H = 3$)

Tests multi-step environments where the **optimal intervention is strictly path-dependent** (requiring a Step-1 priming or capacitor-hub de-canalization step with step tension $-0.015$ and zero immediate KL before selecting a Step-2 bypass kinase or K562 risk modifier inside the primed state):

1. **`signalling_rewiring` ($H = 3$)**: Models time-staggered oncogenic signalling rewiring (`EGFR/MEK` Step-1 priming followed by selecting among $R = 8$ primary post-translational bypass co-targets + $k$ peripheral clades; unprimed co-targeting fails at $0.25$ due to intact negative feedback).
2. **`canalized_gi` ($H = 3$, Horlbeck 2018 K562 `PPP2R1A` Hub)**: Models cryptic genetic variation buffered by `PPP2R1A` (`FDPS` real interaction $\text{GI} = -5.2413$ vs. neutral K562 modifiers, unlocked only after Step-1 `PPP2R1A` de-canalization; reverse-order single knockdowns while `PPP2R1A` is intact are buffered at $0.4999$).

Both studies compare `ceac` and `ceac_ranked` (`PathDepGreedyCEAC`) against **two** multi-step generalizations of Russo–Van Roy Optimal-Action IDS (`MultiStepActionIDS`):
- **`ids_action_tree` ($I(\Pi^\star; Z)$)**: Full MDP optimal-policy classes $\Pi^\star(\theta, \xi)$ split by $2^k$ due to counterfactual local rescue rules $\xi \in \{0, 1\}^k$ inside unvisited off-path stress/non-hub branches, trapping policy-tree IDS into profiling non-causal modifier assays.
- **`ids_action_onpath` ($I(C_{\text{on-path}}^\star; Z)$)**: Even after quotienting out all off-path differences via `model.decision_classes()`, scalar mutual information assigns low information ratios to cheap peripheral clade / eQTL expression reporters ($0.49998$ return) that reduce tail entropy on peripheral candidates while providing **zero KL** between `Null` and the realized responder. CEAC's vector-valued pairwise log-likelihood constraints and backward Bellman evidence critic $Q_I(s, a; \beta)$ avoid both traps.

---

## 5. Calibration-Gated Reproduction & Executed Notebook

From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests/pathologies -v

# 1. Calibrate and run Sequential Locks (L = 1, 2, 3, 4)
.venv/bin/python -m ceac_pathologies.experiment calibrate --protocol locks --output runs/pathologies_locks_cal --workers 10
.venv/bin/python -m ceac_pathologies.experiment sweep --protocol locks --calibration runs/pathologies_locks_cal/gate.json --output runs/pathologies_locks_sweep --workers 10
.venv/bin/python -m ceac_pathologies.report --study runs/pathologies_locks_sweep --output notebooks/evidence/pathologies_locks

# 2. Calibrate and run Epistatic Signalling (k = 0, 1, 2, 3)
.venv/bin/python -m ceac_pathologies.experiment calibrate --protocol epistasis --output runs/pathologies_epistasis_cal --workers 10
.venv/bin/python -m ceac_pathologies.experiment sweep --protocol epistasis --calibration runs/pathologies_epistasis_cal/gate.json --output runs/pathologies_epistasis_sweep --workers 10
.venv/bin/python -m ceac_pathologies.report --study runs/pathologies_epistasis_sweep --output notebooks/evidence/pathologies_epistasis

# 3. Run K562 Pairwise Interaction Screen & Greedy Price-Ranked CEAC comparison (k = 0, 1, 2, 3)
.venv/bin/python -m ceac_ranked.study --tag 20261120_ranked_v2 --workers 10

# 4. Build, execute, and SHA-256-audit the Pathological Search Suite Jupyter notebook
.venv/bin/python scripts/render_pathologies_notebook.py --execute
```
