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


from ceac_critic_diagnostics.run_pathologies_audit import _group_laws_by_k, fast_best_pair


def _base_policy_occ(model, policy) -> np.ndarray:
    """Exact base-model (m=0) state-action occupancy vector for linear `evaluate_stage`."""
    occ = np.zeros(model.n, dtype=float)
    occ[model.initial] = 1.0
    probs = policy.probabilities
    for t in range(len(model.levels) - 1):
        sts = model.levels[t]
        if not len(sts):
            continue
        if getattr(model, "_shared_nonterm", False):
            occ += occ[sts] @ np.einsum("sa,san->sn", probs[sts], model._nonterm_op[sts])
        else:
            occ += occ[sts] @ np.einsum("sa,san->sn", probs[sts], model._nonterm_op[0, sts])
    return (occ[:, None] * probs).ravel()


def _fast_reward_2d(model) -> np.ndarray:
    """Exact `(M, S*A)` density-weighted reward matrix (`reward_2d @ occ == model.returns(p)`)."""
    return (model.density[:, :, None] * model.reward).reshape(model.num_models, -1)


def make_truth_return_vec(model, truth: int) -> np.ndarray:
    """Exact `(S*A,)` density-weighted reward vector for `truth` (`vec @ occ == model.returns(p)[truth]`)."""
    return (model.density[truth, :, None] * model.reward[truth]).ravel()


def _deduplicate_onpath_bank(model, bank: PolicyBank, reward_2d: np.ndarray):
    """Deduplicate policies with identical model returns and base-model state-action occupancy."""
    seen = set()
    unique = []
    rets = []
    occs = []
    for p in bank.policies:
        occ = _base_policy_occ(model, p)
        ret = reward_2d @ occ
        sig = (tuple(np.round(ret, 11)), tuple(np.round(occ, 12)))
        if sig not in seen:
            seen.add(sig)
            unique.append(p)
            rets.append(ret)
            occs.append(occ)
    bank.policies = unique
    bank._ids = {p.identity: idx for idx, p in enumerate(unique)}
    bank_returns = np.column_stack(rets) if rets else np.zeros((model.num_models, 0), dtype=float)
    bank_occ = np.vstack(occs) if occs else np.zeros((0, model.n * model.actions), dtype=float)
    return seen, bank_returns, bank_occ


class FastPathDepBayesianIDS(BayesianIDS):
    """Vectorized BayesianIDS with incremental `(M, B)` returns, `(B, S*A)` occupancies, and fast pair search."""

    def __init__(self, model, rng, prior=0.5):
        super().__init__(model, rng, prior)
        self._reward_2d = _fast_reward_2d(model)
        self._seen_sigs, self._bank_returns, self._bank_occ = _deduplicate_onpath_bank(
            model, self.bank, self._reward_2d
        )
        self._mi_groups = _group_laws_by_k(model, model._informative_pairs)
        self._bank_len = len(self.bank.policies)
        self._zero_prices = np.zeros((model.num_models, model.num_models), dtype=float)
        self._zero_prices.flags.writeable = False

    def _sync_bank_caches(self):
        cur_len = len(self.bank.policies)
        if self._bank_len < cur_len:
            new_pols = self.bank.policies[self._bank_len:]
            new_occs = np.vstack([_base_policy_occ(self.model, p) for p in new_pols])
            new_rets = self._reward_2d @ new_occs.T
            self._bank_occ = np.vstack([self._bank_occ, new_occs])
            self._bank_returns = np.hstack([self._bank_returns, new_rets])
            self._bank_len = cur_len

    def _fast_mi_stage(self, posterior: np.ndarray) -> np.ndarray:
        stage = np.zeros(self.model.mask.shape, dtype=float)
        masses = posterior[:, None] * self.model.density
        for ss, aa, probs_grp in self._mi_groups:
            w_mat = masses[:, ss].T
            totals = w_mat.sum(axis=1)
            pos = totals > 0.0
            if not np.any(pos):
                continue
            belief = np.where(pos[:, None], w_mat / np.maximum(totals[:, None], 1e-300), 0.0)
            mix = np.einsum("pm,pmk->pk", belief, probs_grp)
            kls = kl_div(probs_grp, mix[:, None, :]).sum(axis=2)
            vals = np.maximum(0.0, np.einsum("pm,pm->p", w_mat, kls))
            stage[ss[pos], aa[pos]] = vals[pos]
        return stage

    def distribution(self):
        posterior = self.posterior
        key = tuple(self.log_weights)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        self._sync_bank_caches()
        regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
        if regret.min() <= self.regret_tolerance:
            distribution = np.zeros(len(regret))
            distribution[int(np.argmin(regret))] = 1.0
            self.record = {
                "regret": float(distribution @ regret),
                "information": 0.0,
                "ratio": 0.0,
                "column_residual": 0.0,
                "policy_count": len(self.bank.policies),
                "prior_support": self.initial_support.tolist(),
                "expected_regret_tolerance": self.regret_tolerance,
                "posterior_underflow_entries": self.posterior_underflow_entries,
                "solver_status": "expected_regret_below_tolerance",
            }
            self._cache = (key, distribution)
            return distribution
        stage_mi = self._fast_mi_stage(posterior)
        stage_flat = stage_mi.ravel()
        stage_return = np.einsum("m,ms,msa->sa", posterior, self.model.density, self.model.reward)
        for _ in range(128):
            self._sync_bank_caches()
            regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
            information = np.maximum(0.0, self._bank_occ @ stage_flat)
            if regret.min() <= self.regret_tolerance:
                distribution = np.zeros(len(regret))
                distribution[int(np.argmin(regret))] = 1.0
                d, g, residual = float(distribution @ regret), float(distribution @ information), 0.0
                break
            distribution = fast_best_pair(regret, information)
            d, g = float(distribution @ regret), float(distribution @ information)
            if d == 0.0:
                residual = 0.0
                break
            price = d / (2.0 * g)
            proposal, _ = self.model.response_stage(stage_return + price * stage_mi, "ids-separator")
            prop_occ = _base_policy_occ(self.model, proposal)
            new_returns = self._reward_2d @ prop_occ
            new_d = float(posterior @ np.maximum(0.0, self.model.optimal_returns - new_returns))
            new_g = max(0.0, float(prop_occ @ stage_flat))
            residual = max(0.0, d - new_d + price * (new_g - g))
            if residual <= 1e-7 * d:
                break
            sig = (tuple(np.round(new_returns, 11)), tuple(np.round(prop_occ, 12)))
            if sig in self._seen_sigs or not self.bank.add(proposal):
                break
            self._seen_sigs.add(sig)
            self._bank_occ = np.vstack([self._bank_occ, prop_occ[None, :]])
            self._bank_returns = np.hstack([self._bank_returns, new_returns[:, None]])
            self._bank_len = len(self.bank.policies)
        else:
            raise RuntimeError("IDS separation exceeded insertion limit")
        self.record = {
            "regret": d,
            "information": g,
            "ratio": 0.0 if d == 0 else (d * d / g if g > 0 else None),
            "column_residual": residual,
            "policy_count": len(self.bank.policies),
            "prior_support": self.initial_support.tolist(),
            "expected_regret_tolerance": self.regret_tolerance,
            "posterior_underflow_entries": self.posterior_underflow_entries,
            "solver_status": "expected_regret_below_tolerance" if d <= self.regret_tolerance else "ratio_certified",
        }
        self._cache = (key, distribution)
        return distribution

    def select(self, index):
        del index
        distribution = self.distribution()
        p = self.bank.policies[int(self.rng.choice(len(distribution), p=distribution))]
        return p, "ids", self._zero_prices


class MultiStepActionIDS(BayesianIDS):
    """Vectorized Multi-step IDS targeting optimal-action information across DAG horizons H >= 2."""

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
        self._reward_2d = _fast_reward_2d(model)
        self._seen_sigs, self._bank_returns, self._bank_occ = _deduplicate_onpath_bank(
            model, self.bank, self._reward_2d
        )
        self._mi_groups = _group_laws_by_k(model, model._informative_pairs)
        self._class_mat = np.zeros((len(self.classes), model.num_models), dtype=float)
        for idx_c, cls_m in enumerate(self.classes):
            self._class_mat[idx_c, cls_m] = 1.0
        self._bank_len = len(self.bank.policies)
        self._zero_prices = np.zeros((model.num_models, model.num_models), dtype=float)
        self._zero_prices.flags.writeable = False

    def _sync_bank_caches(self):
        cur_len = len(self.bank.policies)
        if self._bank_len < cur_len:
            new_pols = self.bank.policies[self._bank_len:]
            new_occs = np.vstack([_base_policy_occ(self.model, p) for p in new_pols])
            new_rets = self._reward_2d @ new_occs.T
            self._bank_occ = np.vstack([self._bank_occ, new_occs])
            self._bank_returns = np.hstack([self._bank_returns, new_rets])
            self._bank_len = cur_len

    def action_information_stage(self, posterior: np.ndarray) -> np.ndarray:
        stage = np.zeros(self.model.mask.shape, dtype=float)
        weighted = posterior[:, None] * self.model.density  # (M, S)
        for ss, aa, probs_grp in self._mi_groups:
            w_mat = weighted[:, ss].T  # (P, M)
            tots = w_mat.sum(axis=1)  # (P,)
            pos = tots > 0.0
            if not np.any(pos):
                continue
            belief = np.where(pos[:, None], w_mat / np.maximum(tots[:, None], 1e-300), 0.0)  # (P, M)
            mix = np.einsum("pm,pmk->pk", belief, probs_grp)  # (P, K)
            P_n, M_n, K_n = probs_grp.shape
            c_masses = belief @ self._class_mat.T  # (P, C) via BLAS dgemm
            bp = (belief[:, :, None] * probs_grp).transpose(1, 0, 2).reshape(M_n, P_n * K_n)
            cls_num = (self._class_mat @ bp).reshape(len(self.classes), P_n, K_n).transpose(1, 0, 2)  # (P, C, K) via BLAS dgemm
            cls_laws = np.where(c_masses[:, :, None] > 1e-300, cls_num / np.maximum(c_masses[:, :, None], 1e-300), mix[:, None, :])
            mi_p = np.einsum("pc,pck->p", c_masses, kl_div(cls_laws, mix[:, None, :]))
            stage[ss[pos], aa[pos]] = np.maximum(0.0, tots[pos] * mi_p[pos])
        return stage

    def distribution(self) -> np.ndarray:
        posterior = self.posterior
        key = tuple(self.log_weights)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        self._sync_bank_caches()
        regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
        if regret.min() <= self.regret_tolerance:
            distribution = np.zeros(len(regret))
            distribution[int(np.argmin(regret))] = 1.0
            self.record = {
                "regret": float(distribution @ regret),
                "information": 0.0,
                "ratio": 0.0,
                "column_residual": 0.0,
                "policy_count": len(self.bank.policies),
                "information_target": self.target,
                "classes": len(self.classes),
            }
            self._cache = (key, distribution)
            return distribution
        stage_mi = self.action_information_stage(posterior)
        stage_flat = stage_mi.ravel()
        stage_return = np.einsum("m,ms,msa->sa", posterior, self.model.density, self.model.reward)
        for _ in range(128):
            self._sync_bank_caches()
            regret = posterior @ np.maximum(0.0, self.model.optimal_returns[:, None] - self._bank_returns)
            information = np.maximum(0.0, self._bank_occ @ stage_flat)
            if regret.min() <= self.regret_tolerance or not np.any(information > 0):
                distribution = np.zeros(len(regret))
                distribution[int(np.argmin(regret))] = 1.0
                d, g, residual = float(distribution @ regret), float(distribution @ information), 0.0
                break
            distribution = fast_best_pair(regret, information)
            d, g = float(distribution @ regret), float(distribution @ information)
            if d == 0.0:
                residual = 0.0
                break
            price = d / (2.0 * g)
            proposal, _ = self.model.response_stage(stage_return + price * stage_mi, f"ids-{self.target}-sep")
            prop_occ = _base_policy_occ(self.model, proposal)
            new_returns = self._reward_2d @ prop_occ
            new_d = float(posterior @ np.maximum(0.0, self.model.optimal_returns - new_returns))
            new_g = max(0.0, float(prop_occ @ stage_flat))
            residual = max(0.0, d - new_d + price * (new_g - g))
            if residual <= 1e-7 * d:
                break
            sig = (tuple(np.round(new_returns, 11)), tuple(np.round(prop_occ, 12)))
            if sig in self._seen_sigs or not self.bank.add(proposal):
                break
            self._seen_sigs.add(sig)
            self._bank_occ = np.vstack([self._bank_occ, prop_occ[None, :]])
            self._bank_returns = np.hstack([self._bank_returns, new_returns[:, None]])
            self._bank_len = len(self.bank.policies)
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

    def select(self, index):
        del index
        distribution = self.distribution()
        p = self.bank.policies[int(self.rng.choice(len(distribution), p=distribution))]
        return p, "ids", self._zero_prices


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
