"""Greedy maximally-disagreeing column selection + dynamic evidence-price ranked replay.

Explicitly labeled heuristic proxy for combinatorial action/pair screens where
certified Section-3 CEAC's uniform seed-bank replay floor O(|Pi_seed| * floor(T))
dominates finite-horizon budgets. Preserves public confounder-equivalence
certification, coordinate-free pointwise row dominance, single-credit replay,
and online two-critic updates without oracle access.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import math
import numpy as np
from scipy.optimize import linprog
from ceac.contracts import Policy
from ceac_hardness.design import PolicyBank
from ceac_hardness.learning import DebtLedger, evidence_scale
from ceac_pseudocode.design import ReplayDesign
from ceac_pathologies.agent import (
    NATURAL_CANDIDATE,
    NATURAL_SCHEDULE,
    GeneralPolicyReplay,
    SparseStabilityActor,
    SparseTwoCriticActor,
)
from ceac_stability.actor import project_kl


@dataclass(frozen=True)
class RankedStratum:
    policy_id: str
    count: int
    return_mean: float
    return_lcb: float
    evidence_value: float
    priced_score: float
    lcb_priced_score: float
    uncertainty_priority: float


@dataclass(frozen=True)
class RankedCEACConfig:
    actor_rate: float = 0.4
    critic_rate: float = 0.12
    temperature: float = 1.0
    delta: float = 0.05
    refresh_interval: int = 128
    snapshot_interval: int = 2048
    min_stratum_count: int = 2
    return_lcb_scale: float = 0.5
    info_cover_threshold: float = 1e-3
    mass_bound: float = 64.0
    column_tolerance: float = 1e-6
    warm_start_floor: float = 0.10
    max_replay_promotions_per_refresh: int = 16
    optimizer: str = "reference"

    def __post_init__(self):
        if self.optimizer not in {"reference", "guarded_natural"}:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.min_stratum_count < 1:
            raise ValueError("min_stratum_count must be >= 1")
        if not 0.02 < self.warm_start_floor < 0.5:
            raise ValueError("warm_start_floor must lie in (0.02, 0.5)")


class PriceRankedReplay(GeneralPolicyReplay):
    """Policy-labeled replay dynamically ranked by live evidence prices beta."""

    def __init__(self, model, delta: float = 0.025, return_lcb_scale: float = 0.5):
        super().__init__(model, delta=delta)
        self.return_lcb_scale = float(return_lcb_scale)
        self.policies_by_id: dict[str, Policy] = {}
        self.reusable_ids: set[str] = set()

    def register(self, policy: Policy, reusable: bool = True):
        if reusable or policy.identity not in self.policies_by_id:
            if reusable:
                self.policies_by_id[policy.identity] = policy
                self.reusable_ids.add(policy.identity)
        super().register(policy)

    def empirical_arrays(self, policies: list[Policy], rows: tuple[tuple[int, int], ...]):
        counts, means, lcbs, info_mean, info_proxy, rj = [], [], [], [], [], []
        for policy in policies:
            self.register(policy)
            n, total, ll = self.statistics[policy.identity]
            counts.append(n)
            if not n:
                means.append(self.return_min)
                lcbs.append(self.return_min)
                info_mean.append(np.zeros(len(rows)))
                info_proxy.append(np.zeros(len(rows)))
                rj.append(self.return_width)
                continue
            mu = total / n
            rad = self.return_lcb_scale * self.return_width / math.sqrt(n)
            means.append(mu)
            lcbs.append(max(self.return_min, mu - rad))
            rj.append(rad)
            avg = ll / n
            raw = np.array([avg[m] - avg[j] for m, j in rows], dtype=float) if rows else np.zeros(0)
            shrink = n / (n + 1.0)
            info_mean.append(raw)
            info_proxy.append(np.maximum(0.0, raw * shrink))
        mean_arr = np.asarray(means, dtype=float)
        lcb_arr = np.asarray(lcbs, dtype=float)
        im_arr = np.array(info_mean, dtype=float).T if rows else np.zeros((0, len(policies)))
        ip_arr = np.array(info_proxy, dtype=float).T if rows else np.zeros((0, len(policies)))
        rj_arr = np.asarray(rj, dtype=float)
        return dict(
            counts=np.asarray(counts, dtype=int),
            return_mean=mean_arr,
            return_lower=lcb_arr,
            return_radius=rj_arr,
            information_mean=im_arr,
            information_lower=ip_arr,
            information_radius=np.abs(im_arr - ip_arr),
        )

    def rank_buffer(
        self,
        rows: tuple[tuple[int, int], ...],
        beta: np.ndarray,
        min_count: int = 1,
    ) -> list[RankedStratum]:
        eligible = [p for pid, p in self.policies_by_id.items() if self.statistics[pid][0] >= min_count]
        if not eligible:
            return []
        data = self.empirical_arrays(eligible, rows)
        prices = np.array([beta[r] for r in rows], dtype=float) if rows else np.zeros(0)
        ev = (prices @ data["information_lower"]) if rows else np.zeros(len(eligible))
        ev_unc = (prices @ data["information_radius"]) if rows else np.zeros(len(eligible))
        out = []
        for idx, policy in enumerate(eligible):
            ret_m = float(data["return_mean"][idx])
            ret_l = float(data["return_lower"][idx])
            e_val = float(ev[idx])
            out.append(
                RankedStratum(
                    policy_id=policy.identity,
                    count=int(data["counts"][idx]),
                    return_mean=ret_m,
                    return_lcb=ret_l,
                    evidence_value=e_val,
                    priced_score=ret_m + e_val,
                    lcb_priced_score=ret_l + e_val,
                    uncertainty_priority=float(data["return_radius"][idx] + ev_unc[idx]),
                )
            )
        out.sort(
            key=lambda s: (
                round(s.priced_score, 12),
                round(s.evidence_value, 12),
                round(s.return_mean, 12),
                s.count,
            ),
            reverse=True,
        )
        return out


def solve_ranked_replay_master(
    replay: PriceRankedReplay,
    policies: list[Policy],
    references: tuple[int, ...],
    upper_optimum: float,
    mass_bound: float = 64.0,
) -> ReplayDesign | None:
    """Empirical replay Graves--Lai master LP with lossless row deduplication."""
    model = replay.model
    ref_set = set(references)
    rows = tuple(r for r in model.alternatives if r[0] in ref_set)
    data = replay.empirical_arrays(policies, rows)
    costs = np.maximum(0.0, upper_optimum - data["return_lower"])
    k = data["information_lower"]
    if not len(rows):
        eta = np.zeros(len(policies))
        beta = np.zeros((model.num_models, model.num_models))
        return ReplayDesign(
            tuple(p.identity for p in policies),
            eta,
            beta,
            0.0,
            0.0,
            1.0,
            0.0,
            tuple(references),
            rows,
            data,
        )
    groups = defaultdict(list)
    rep_rows = []
    for idx in range(len(rows)):
        sig = tuple(np.round(k[idx], 13))
        if sig not in groups:
            rep_rows.append(k[idx])
        groups[sig].append(idx)
    k_unique = np.asarray(rep_rows, dtype=float)
    group_lists = list(groups.values())

    res = linprog(
        costs,
        A_ub=np.vstack([-k_unique, np.ones(len(policies))]),
        b_ub=np.r_[-np.ones(len(k_unique)), mass_bound],
        bounds=(0.0, None),
        method="highs",
    )
    if not res.success:
        if res.status == 2:
            return None
        raise RuntimeError("Ranked replay master failed: " + res.message)
    unique_prices = -res.ineqlin.marginals[:-1]
    mass_price = float(-res.ineqlin.marginals[-1])
    beta = np.zeros((model.num_models, model.num_models))
    for u_price, indices in zip(unique_prices, group_lists):
        share = float(u_price) / len(indices)
        for idx in indices:
            beta[rows[idx]] = share
    dual = float(unique_prices.sum() - mass_bound * mass_price)
    minimum = float(np.min(k @ res.x))
    gap = abs(float(res.fun) - dual)
    if gap > 1e-6 or minimum < 1.0 - 1e-6:
        raise RuntimeError("Ranked replay primal/dual certificate failed")
    return ReplayDesign(
        tuple(p.identity for p in policies),
        res.x,
        beta,
        mass_price,
        float(res.fun),
        minimum,
        gap,
        tuple(references),
        rows,
        data,
    )


class GreedyPricedCEAC:
    """Heuristic CEAC with greedy disagreement admission and price-ranked replay."""

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

        # Register ONLY the single safe anchor policy initially, avoiding O(|Pi_seed| * floor) debt.
        seeds = model.seed_policies()
        self.anchor_policy = seeds[0]
        self.bank = PolicyBank(model, [self.anchor_policy])
        self.replay.register(self.anchor_policy)
        self.request_count(self.anchor_policy.identity, cfg.min_stratum_count)

        # Public decision-equivalence classes and compact candidate pool summaries.
        self.classes = model.decision_classes()
        self.num_classes = len(self.classes)
        self.class_of = np.zeros(model.num_models, dtype=int)
        for c_idx, members in enumerate(self.classes):
            for m in members:
                self.class_of[m] = c_idx

        det_count = int(np.prod(model.mask.sum(axis=1), dtype=object))
        if det_count > 100000:
            raise ValueError("Deterministic policy count exceeds 100,000")
        self.candidate_policies: list[Policy] = []
        for choices in product(*[np.flatnonzero(mask) for mask in model.mask]):
            probabilities = np.zeros(model.mask.shape)
            probabilities[np.arange(model.n), choices] = 1.0
            self.candidate_policies.append(Policy(probabilities, "greedy-candidate"))

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
                # Coordinate-free causal projection: min_{j in C'} I^m(pi; j) zeros all within-class confounders.
                self.cand_class_kl[p_idx, :, c_idx] = info[:, c_members].min(axis=1)

        self.episodes = 0
        self.actor_episodes_on_class = 0
        self.active_class: int | None = None
        self.warm_stage = 0
        self.design: ReplayDesign | None = None
        self.prices = np.zeros((model.num_models, model.num_models))
        self.last_refs: tuple[int, ...] = ()
        self.certified = False
        self.refreshes = 0
        self.admissions = 0
        self.replay_promotions = 0
        self.greedy_admissions = 0
        self.events = []
        self.last_ranked_buffer: list[dict] = []
        self.last_replay_scores: list[float] = []
        self.last_reason = "initialization"
        self.refresh()

    def request_count(self, pid: str, n: int):
        self.ledger.statistical_required[pid] = max(self.ledger.statistical_required[pid], int(n))

    def is_decision_certified(self, refs: tuple[int, ...]) -> bool:
        if not refs:
            return False
        if len(refs) == 1:
            return True
        c0 = int(self.class_of[refs[0]])
        if any(int(self.class_of[m]) != c0 for m in refs[1:]):
            return False
        ref_set = set(refs)
        rows_by_ref = defaultdict(list)
        for m, j in self.all_rows:
            if m in ref_set:
                rows_by_ref[m].append(j)
        base_m = refs[0]
        sa_keys = sorted(self.model._informative_pairs)
        base_sigs = {
            tuple(np.round([self.replay._sa_high[sa][base_m, j] - self.replay._sa_low[sa][base_m, j] for sa in sa_keys], 12))
            for j in rows_by_ref[base_m]
        }
        for m in refs[1:]:
            sigs = {
                tuple(np.round([self.replay._sa_high[sa][m, j] - self.replay._sa_low[sa][m, j] for sa in sa_keys], 12))
                for j in rows_by_ref[m]
            }
            if sigs != base_sigs:
                return False
        return True

    def disagreement_scores(self, refs: tuple[int, ...]) -> np.ndarray:
        """Causal disagreement across surviving decision classes; strictly 0 on passenger arms."""
        refs_arr = np.asarray(refs, dtype=int)
        xi_raw = self.replay.mixture[refs_arr]
        xi = xi_raw / xi_raw.sum()
        class_mass = np.zeros(self.num_classes, dtype=float)
        for w, m in zip(xi, refs_arr):
            class_mass[self.class_of[m]] += float(w)
        # Weight Matrix W[local_m_idx, c] = xi[m] * class_mass[c] for c != class_of[m]
        w_mat = np.outer(xi, class_mass)
        for idx, m in enumerate(refs_arr):
            w_mat[idx, self.class_of[m]] = 0.0
        sub_kl = self.cand_class_kl[:, refs_arr, :]  # (P, |refs|, K)
        return np.einsum("mc,pmc->p", w_mat, sub_kl)

    def select_greedy_disagreement(self, refs: tuple[int, ...]) -> Policy:
        scores = self.disagreement_scores(refs)
        refs_arr = np.asarray(refs, dtype=int)
        xi = self.replay.mixture[refs_arr]
        xi = xi / xi.sum()
        exp_ret = self.cand_returns[:, refs_arr] @ xi
        best_idx = max(
            range(len(self.candidate_policies)),
            key=lambda i: (
                round(float(scores[i]), 12),
                round(float(exp_ret[i]), 12),
                -self.ledger.counts[self.candidate_policies[i].identity],
            ),
        )
        return self.candidate_policies[best_idx]

    def effective_price_scale(self) -> float:
        if not self.certified or self.design is None:
            return 0.0
        scale_t = evidence_scale(max(2, self.episodes))
        return float(min(1.0, scale_t / (1.0 + self.actor_episodes_on_class)))

    def effective_prices(self) -> np.ndarray:
        return self.effective_price_scale() * self.prices

    def _warm_align_actor_from_buffer(self, rows: tuple[tuple[int, int], ...]):
        """Reuse the #1 price-ranked replay policy to warm-initialize actor logits."""
        ranked = self.replay.rank_buffer(rows, self.effective_prices(), min_count=self.config.min_stratum_count)
        if not ranked:
            return
        top_pid = ranked[0].policy_id
        top_policy = self.replay.policies_by_id[top_pid]
        eps = self.config.warm_start_floor
        for s in range(self.model.n):
            valid = np.flatnonzero(self.model.mask[s])
            if len(valid) <= 1:
                self.actor.logits[s] = 0.0
                continue
            target_p = (1.0 - eps) * top_policy.probabilities[s, valid] + (eps / len(valid))
            if self.config.optimizer == "guarded_natural":
                _, _, floor, _ = self.actor.rates(s)
                target_p = project_kl(np.log(target_p), max(floor, 1e-6))
                self.actor.logits[s, valid] = np.log(target_p)
            else:
                self.actor.logits[s, valid] = self.config.temperature * np.log(target_p)
        self.actor.reward_value[self.model.initial] = ranked[0].return_mean

    def refresh(self):
        cfg = self.config
        refs = self.replay.plausible
        self.last_refs = refs
        self.refreshes += 1

        # 1. Ensure safe anchor baseline has minimal replay observation.
        if self.ledger.counts[self.anchor_policy.identity] < cfg.min_stratum_count:
            self.certified = False
            self.prices.fill(0.0)
            self.last_reason = "anchor_bootstrap_pending"
            self.request_count(self.anchor_policy.identity, cfg.min_stratum_count)
            return

        # 2. If decision conflict remains in C_t, step the greedy maximally-disagreeing policy.
        if not self.is_decision_certified(refs):
            self.certified = False
            self.prices.fill(0.0)
            self.active_class = None
            self.warm_stage = 0
            self.last_reason = "greedy_disagreement_pending"
            chosen = self.select_greedy_disagreement(refs)
            self.replay.register(chosen)
            self.request_count(chosen.identity, self.ledger.counts[chosen.identity] + 1)
            return

        cur_class = int(self.class_of[refs[0]])
        if self.active_class != cur_class:
            self.active_class = cur_class
            self.actor_episodes_on_class = 0
            self.warm_stage = 0

        ref_set = set(refs)
        rows = tuple(r for r in self.all_rows if r[0] in ref_set)

        # 3. Promote already-observed replay strata covering any uncovered active rows (zero extra episodes!).
        if rows:
            eligible_pids = [
                pid
                for pid in self.replay.policies_by_id
                if pid not in self.bank._ids and self.ledger.counts[pid] >= cfg.min_stratum_count
            ]
            for _ in range(len(rows) + 1):
                bank_data = self.replay.empirical_arrays(self.bank.policies, rows)
                covered = (
                    bank_data["information_lower"][:, bank_data["counts"] >= cfg.min_stratum_count].max(axis=1)
                    if np.any(bank_data["counts"] >= cfg.min_stratum_count)
                    else np.zeros(len(rows))
                )
                uncovered = np.flatnonzero(covered <= cfg.info_cover_threshold)
                if not len(uncovered) or not eligible_pids:
                    break
                cand_pols = [self.replay.policies_by_id[pid] for pid in eligible_pids]
                cand_data = self.replay.empirical_arrays(cand_pols, rows)
                gains = cand_data["information_lower"][uncovered, :].sum(axis=0)
                best_i = int(np.argmax(gains))
                if gains[best_i] <= cfg.info_cover_threshold:
                    break
                chosen_pol = cand_pols[best_i]
                if self.bank.add(chosen_pol):
                    self.replay_promotions += 1
                eligible_pids.pop(best_i)

            # Check if any active bank policy still needs its minimal stratum count.
            for p in self.bank.policies:
                if self.ledger.counts[p.identity] < cfg.min_stratum_count:
                    self.certified = False
                    self.prices.fill(0.0)
                    self.last_reason = "active_column_stratum_pending"
                    self.request_count(p.identity, cfg.min_stratum_count)
                    return

            # If rows are still uncovered, greedily admit the single maximally-covering candidate.
            bank_data = self.replay.empirical_arrays(self.bank.policies, rows)
            covered = bank_data["information_lower"].max(axis=1)
            uncovered = np.flatnonzero(covered <= cfg.info_cover_threshold)
            if len(uncovered):
                uncov_global = np.asarray([self.row_index[rows[u]] for u in uncovered], dtype=int)
                cov_scores = self.cand_row_kl[:, uncov_global].sum(axis=1)
                upper_ref = float(max(self.model.optimal_returns[list(refs)]))
                ref_rets = self.cand_returns[:, np.asarray(refs, dtype=int)].min(axis=1)
                scores = cov_scores - 0.01 * np.maximum(0.0, upper_ref - ref_rets)
                best_idx = int(np.argmax(scores))
                proposal = self.candidate_policies[best_idx]
                self.replay.register(proposal)
                if self.bank.add(proposal):
                    self.greedy_admissions += 1
                self.certified = False
                self.prices.fill(0.0)
                self.last_reason = "greedy_row_cover_pending"
                target = max(cfg.min_stratum_count, self.ledger.counts[proposal.identity] + cfg.min_stratum_count)
                self.request_count(proposal.identity, target)
                return

        # 4. Solve Graves--Lai replay master & dynamically re-rank buffer for instant column reuse.
        upper = float(max(self.model.optimal_returns[list(refs)]))
        design = solve_ranked_replay_master(self.replay, self.bank.policies, refs, upper, cfg.mass_bound)
        if design is None:
            self.certified = False
            self.prices.fill(0.0)
            self.last_reason = "ranked_master_infeasible"
            chosen = self.select_greedy_disagreement(refs) if len(refs) > 1 else self.anchor_policy
            self.request_count(chosen.identity, self.ledger.counts[chosen.identity] + cfg.min_stratum_count)
            return

        # Dynamic buffer reuse loop: promote top evidence-priced replay strata with negative reduced cost.
        for _ in range(cfg.max_replay_promotions_per_refresh):
            ranked = self.replay.rank_buffer(rows, design.beta, min_count=cfg.min_stratum_count)
            promoted = False
            for entry in ranked:
                if entry.policy_id in self.bank._ids:
                    continue
                rc = max(0.0, upper - entry.return_lcb) + design.mass_price - entry.evidence_value
                if rc < -cfg.column_tolerance:
                    if self.bank.add(self.replay.policies_by_id[entry.policy_id]):
                        self.replay_promotions += 1
                        promoted = True
                        break
            if not promoted:
                break
            re_solved = solve_ranked_replay_master(self.replay, self.bank.policies, refs, upper, cfg.mass_bound)
            if re_solved is None:
                break
            design = re_solved

        # Greedy counterfactual column check: admit at most 1 unexecuted high-leverage separator if beneficial.
        if rows:
            row_global = np.asarray([self.row_index[r] for r in rows], dtype=int)
            weights = np.asarray([design.beta[r] for r in rows], dtype=float)
            cand_gaps = np.maximum(0.0, upper - self.cand_returns[:, refs[0]])
            # Include finite-sample LCB slack matching min_stratum_count so already-tight designs don't churn.
            lcb_slack = cfg.return_lcb_scale * self.replay.return_width / math.sqrt(cfg.min_stratum_count)
            shrink = cfg.min_stratum_count / (cfg.min_stratum_count + 1.0)
            cand_ev = shrink * (self.cand_row_kl[:, row_global] @ weights)
            reduced = cand_gaps + lcb_slack + design.mass_price - cand_ev
            best_c = int(np.argmin(reduced))
            cand_pol = self.candidate_policies[best_c]
            design.column_residual = max(0.0, -float(reduced[best_c]))
            if reduced[best_c] < -cfg.column_tolerance and cand_pol.identity not in self.bank._ids:
                self.replay.register(cand_pol)
                if self.bank.add(cand_pol):
                    self.greedy_admissions += 1
                if self.ledger.counts[cand_pol.identity] < cfg.min_stratum_count:
                    self.certified = False
                    self.prices.fill(0.0)
                    self.last_reason = "greedy_separator_pending"
                    self.request_count(cand_pol.identity, cfg.min_stratum_count)
                    return
                design = solve_ranked_replay_master(self.replay, self.bank.policies, refs, upper, cfg.mass_bound) or design

        self.design = design
        self.prices = design.beta.copy()
        self.certified = True
        self.last_reason = "ranked_replay_design_active"
        self.ledger.add_design(design, evidence_scale(max(1, self.episodes)))

        # Warm-align actor from the #1 ranked replay policy on initial certification and once exploration decays.
        rho = self.effective_price_scale()
        if self.warm_stage == 0:
            self._warm_align_actor_from_buffer(rows)
            self.warm_stage = 1
        elif self.warm_stage == 1 and rho < 0.10:
            self._warm_align_actor_from_buffer(rows)
            self.warm_stage = 2

        ranked_now = self.replay.rank_buffer(rows, self.effective_prices(), min_count=cfg.min_stratum_count)
        self.last_ranked_buffer = [
            dict(
                policy_id=r.policy_id,
                count=r.count,
                return_mean=r.return_mean,
                evidence_value=r.evidence_value,
                priced_score=r.priced_score,
            )
            for r in ranked_now[:10]
        ]
        scores = self.replay.priced_scores(self.bank.policies, design.rows, self.effective_prices())
        self.last_replay_scores = scores.tolist()
        self.events.append(
            dict(
                episode=self.episodes,
                design=design.record(),
                effective_price_scale=rho,
                top_ranked_buffer=self.last_ranked_buffer[:5],
                bank_size=len(self.bank.policies),
                replay_promotions=self.replay_promotions,
                greedy_admissions=self.greedy_admissions,
                actor_updates=self.actor.updates,
            )
        )

    def _lookup_policy(self, pid: str) -> Policy:
        if pid in self.bank._ids:
            return self.bank.policies[self.bank._ids[pid]]
        return self.replay.policies_by_id[pid]

    def select(self, index: int):
        del index
        if (self.episodes and self.episodes % self.config.refresh_interval == 0) or self.replay.plausible != self.last_refs:
            self.refresh()
        pid, behavior = self.ledger.select()
        if pid is None and not self.certified:
            self.refresh()
            pid, behavior = self.ledger.select()
        if pid is not None:
            return self._lookup_policy(pid), behavior, self.effective_prices()
        if not self.certified:
            raise RuntimeError("Unfinished GreedyPricedCEAC step has no statistical request")
        # Trigger stage-2 warm reuse promptly as soon as effective price decays below 0.10.
        if self.warm_stage == 1 and self.effective_price_scale() < 0.10:
            self.refresh()
        policy = self.actor.snapshot()
        self.replay.register(policy, reusable=False)
        return policy, "actor", self.effective_prices()

    def observe(self, episode, policy: Policy, prices: np.ndarray):
        updated = self.actor.update(episode, policy, prices)
        if updated:
            self.actor_episodes_on_class += 1
        self.replay.append(episode)
        self.ledger.credit(episode)
        self.episodes += 1
        if self.episodes % self.config.snapshot_interval == 0 and self.design is not None:
            snap = self.actor.snapshot()
            self.replay.register(snap, reusable=True)
            if self.ledger.counts[snap.identity] >= self.config.min_stratum_count:
                data = self.replay.empirical_arrays([snap], self.design.rows)
                weights = np.asarray([self.design.beta[r] for r in self.design.rows], dtype=float)
                upper = float(max(self.model.optimal_returns[list(self.design.references)]))
                ev = float(weights @ data["information_lower"][:, 0]) if len(self.design.rows) else 0.0
                rc = max(0.0, upper - float(data["return_lower"][0])) + self.design.mass_price - ev
                if rc < -self.config.column_tolerance and self.bank.add(snap):
                    self.admissions += 1

    def diagnostics(self):
        return dict(
            algorithm=f"CEAC_GreedyPricedReplay_{self.config.optimizer}",
            heuristic_proxy=True,
            actor_updates=self.actor.updates,
            priced_actor_updates=self.actor.priced_updates,
            replay_episodes=len(self.replay.seen),
            policy_strata=len(self.replay.strata),
            bank_policies=len(self.bank.policies),
            replay_promotions=self.replay_promotions,
            greedy_admissions=self.greedy_admissions,
            refreshes=self.refreshes,
            certified=self.certified,
            plausible_count=len(self.replay.plausible),
            effective_price_scale=self.effective_price_scale(),
            reason=self.last_reason,
            top_ranked_buffer=self.last_ranked_buffer,
            priced_replay_scores=self.last_replay_scores,
            design=self.design.record() if self.design is not None else None,
        )
