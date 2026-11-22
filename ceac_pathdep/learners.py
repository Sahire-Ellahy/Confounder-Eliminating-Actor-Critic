"""Multi-step Optimal-Action IDS variants and DAG-path-aware GreedyPricedCEAC."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.special import kl_div
from ceac_hardness.design import PolicyBank
from ceac_hardness.ids import BayesianIDS, best_pair
from ceac_hardness.learning import DebtLedger
from ceac_pathologies.agent import (
    NATURAL_CANDIDATE,
    NATURAL_SCHEDULE,
    SparseStabilityActor,
    SparseTwoCriticActor,
)
from ceac_ranked.agent import GreedyPricedCEAC, PriceRankedReplay, RankedCEACConfig


class MultiStepActionIDS(BayesianIDS):
    """Multi-step IDS targeting optimal-action information across DAG horizons H >= 2.

    Supports two explicit definitions of the optimal-action random variable A*:
    - `target="policy_tree"`: Standard Russo & Van Roy MDP optimal-policy class
      Pi*(m) = argmax_pi V_m^pi (grouped by full state-contingent `policy.identity`).
      Exposes the counterfactual off-path optimal-action pathology when off-path
      states carry distinct local rescue rules.
    - `target="onpath_class"`: Quotiented on-path return-equivalence class
      `model.decision_classes()` (models with mutually zero on-path optimal-return
      suboptimality are merged). Exposes the stagewise Markovian mixture inflation
      and scalar peripheral-entropy pathology relative to CEAC's pairwise KL critic.
    """

    def __init__(self, model, rng, prior: np.ndarray, target: str = "policy_tree"):
        super().__init__(model, rng, prior)
        if target not in {"policy_tree", "onpath_class"}:
            raise ValueError(f"Unknown optimal-action target: {target}")
        self.target = target
        if target == "policy_tree":
            groups = defaultdict(list)
            for m, policy in enumerate(model.optimal_policies):
                groups[policy.identity].append(m)
            self.classes = [np.array(v, dtype=int) for v in groups.values()]
        else:
            self.classes = [np.array(v, dtype=int) for v in model.decision_classes()]
        self._laws = {
            (int(s), int(a)): np.array([[law[k] for k in model.laws[int(s), int(a)][0]] for law in model.laws[int(s), int(a)]], dtype=float)
            for s, a in model._informative_pairs
        }

    def action_information_stage(self, posterior: np.ndarray) -> np.ndarray:
        stage = np.zeros(self.model.mask.shape, dtype=float)
        weighted = posterior[:, None] * self.model.density
        for (s, a), probs in self._laws.items():
            w_s = weighted[:, s]
            tot = float(w_s.sum())
            if tot <= 0.0:
                continue
            belief = w_s / tot
            mix = belief @ probs
            c_masses = np.array([belief[c].sum() for c in self.classes], dtype=float)
            active = c_masses > 1e-300
            if np.sum(active) <= 1:
                continue
            cls_laws = np.array(
                [(belief[c] @ probs[c]) / c_masses[idx] for idx, c in enumerate(self.classes) if active[idx]],
                dtype=float,
            )
            mi_sa = float(c_masses[active] @ kl_div(cls_laws, mix[None, :]).sum(axis=1))
            stage[s, a] = max(0.0, tot * mi_sa)
        return stage

    def distribution(self) -> np.ndarray:
        posterior = self.posterior
        key = tuple(self.log_weights)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        stage_mi = self.action_information_stage(posterior)
        stage_return = np.einsum("m,ms,msa->sa", posterior, self.model.density, self.model.reward)
        for _ in range(128):
            returns, _ = self.bank.arrays()
            regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - returns)
            information = np.maximum(0.0, np.array([self.model.evaluate_stage(p, stage_mi) for p in self.bank.policies]))
            if regret.min() <= self.regret_tolerance or not np.any(information > 0):
                distribution = np.zeros(len(regret))
                distribution[int(np.argmin(regret))] = 1.0
                d, g, residual = float(distribution @ regret), float(distribution @ information), 0.0
                break
            distribution = best_pair(regret, information)
            d, g = float(distribution @ regret), float(distribution @ information)
            if d == 0.0:
                residual = 0.0
                break
            price = d / (2.0 * g)
            proposal, _ = self.model.response_stage(stage_return + price * stage_mi, f"ids-{self.target}-sep")
            new_returns, _ = self.model.evaluate(proposal)
            new_d = float(posterior @ np.maximum(0.0, self.model.optimal_returns - new_returns))
            new_g = max(0.0, float(self.model.evaluate_stage(proposal, stage_mi)))
            residual = max(0.0, d - new_d + price * (new_g - g))
            if residual <= 1e-7 * d:
                break
            if not self.bank.add(proposal):
                raise RuntimeError("Multi-step optimal-action IDS certificate did not close")
        else:
            raise RuntimeError("Multi-step optimal-action IDS exceeded insertion limit")
        self.record = {
            "regret": d,
            "information": g,
            "ratio": 0.0 if d == 0 else (d * d / g if g > 0 else None),
            "column_residual": residual,
            "policy_count": len(self.bank.policies),
            "information_target": self.target,
            "classes": len(self.classes),
        }
        self._cache = (key, distribution)
        return distribution


class PathDepGreedyCEAC(GreedyPricedCEAC):
    """GreedyPricedCEAC using `model.canonical_path_policies()` on branching multi-step DAGs."""

    def __init__(self, model, rng, config: RankedCEACConfig | None = None):
        self.model = model
        self.rng = rng
        self.config = config or RankedCEACConfig()
        cfg = self.config
        if cfg.optimizer == "guarded_natural":
            self.actor = SparseStabilityActor(model, NATURAL_CANDIDATE, NATURAL_SCHEDULE)
        else:
            self.actor = SparseTwoCriticActor(model, cfg.actor_rate, cfg.critic_rate, cfg.temperature)
        self.replay = PriceRankedReplay(model, cfg.delta / 2.0, return_lcb_scale=cfg.return_lcb_scale)
        self.ledger = DebtLedger()

        seeds = model.seed_policies()
        self.anchor_policy = seeds[0]
        self.bank = PolicyBank(model, [self.anchor_policy])
        self.replay.register(self.anchor_policy)
        self.request_count(self.anchor_policy.identity, cfg.min_stratum_count)

        self.classes = model.decision_classes()
        self.num_classes = len(self.classes)
        self.class_of = np.zeros(model.num_models, dtype=int)
        for c_idx, members in enumerate(self.classes):
            for m in members:
                self.class_of[m] = c_idx

        self.candidate_policies = list(model.canonical_path_policies())
        num_p = len(self.candidate_policies)
        self.all_rows = tuple(model.alternatives)
        self.row_index = {r: idx for idx, r in enumerate(self.all_rows)}
        self.cand_returns = np.zeros((num_p, model.num_models), dtype=float)
        self.cand_row_kl = np.zeros((num_p, len(self.all_rows)), dtype=float)
        self.cand_class_kl = np.zeros((num_p, model.num_models, self.num_classes), dtype=float)
        class_arrays = [np.asarray(members, dtype=int) for members in self.classes]
        rows_m = np.asarray([r[0] for r in self.all_rows], dtype=int) if self.all_rows else np.zeros(0, dtype=int)
        rows_j = np.asarray([r[1] for r in self.all_rows], dtype=int) if self.all_rows else np.zeros(0, dtype=int)

        for p_idx, pol in enumerate(self.candidate_policies):
            ret, info = model.evaluate(pol)
            self.cand_returns[p_idx] = ret
            if len(self.all_rows):
                self.cand_row_kl[p_idx] = info[rows_m, rows_j]
            for c_idx, c_members in enumerate(class_arrays):
                self.cand_class_kl[p_idx, :, c_idx] = info[:, c_members].min(axis=1)

        self.episodes = 0
        self.actor_episodes_on_class = 0
        self.active_class: int | None = None
        self.warm_stage = 0
        self.design = None
        self.prices = np.zeros((model.num_models, model.num_models))
        self.last_refs: tuple[int, ...] = ()
        self.certified = False
        self.refreshes = 0
        self.admissions = 0
        self.replay_promotions = 0
        self.greedy_admissions = 0
        self.events = []
        self.last_ranked_buffer = []
        self.last_replay_scores = []
        self.last_reason = "initialization"
        self.refresh()
