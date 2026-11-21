"""Section-3 replay-driven two-critic CEAC with public confounder-equivalence certificates."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import math
import numpy as np
from scipy.optimize import linprog
from ceac.contracts import Policy
from ceac_hardness.design import PolicyBank, bounded_design, target_calibration, _aec_candidate
from ceac_hardness.learning import DebtLedger, LikelihoodEvidence, evidence_scale
from ceac_pseudocode.learning import TwoCriticActor
from ceac_pseudocode.design import ReplayDesign
from ceac_stability.actor import StabilityActor, guarded_proposal
from scipy.special import logsumexp


class SparseTwoCriticActor(TwoCriticActor):
    """Numerically identical to TwoCriticActor, skipping zero entries of sparse beta."""

    def evidence_value(self, beta):
        nz_i, nz_j = np.nonzero(beta)
        if not len(nz_i):
            return np.zeros(self.model.n, dtype=float)
        w = beta[nz_i, nz_j]
        return w @ self.evidence_parameters[nz_i, nz_j, :]

    def update(self, episode, policy, beta):
        if episode.behavior != "actor":
            return False
        if episode.policy_id != policy.identity or policy.identity != self.snapshot().identity:
            raise ValueError("Only the frozen current-actor episode can update actor and critics")
        beta = np.asarray(beta, dtype=float)
        if beta.shape != self.evidence_parameters.shape[:2] or not np.isfinite(beta).all() or (beta < 0).any():
            raise ValueError("Invalid frozen evidence price")
        nz_i, nz_j = np.nonzero(beta)
        has_beta = len(nz_i) > 0
        w = beta[nz_i, nz_j] if has_beta else None
        vi = (w @ self.evidence_parameters[nz_i, nz_j, :]) if has_beta else np.zeros(self.model.n, dtype=float)
        delta_theta = np.zeros_like(self.logits)
        errors = []
        for step in episode.transitions:
            s = step.state
            ll = np.asarray(step.log_likelihood)
            evidence = float(w @ (ll[nz_i] - ll[nz_j])) if has_beta else 0.0
            dr = step.reward - self.reward_value[s]
            di = evidence - vi[s]
            if not step.done:
                dr += self.reward_value[step.next_state]
                di += vi[step.next_state]
            score = -policy.probabilities[s].copy()
            score[step.action] += 1.0
            score /= self.temperature
            delta_theta[s] += (dr + di) * score
            errors.append((s, float(dr), float(di), evidence))
        self.logits += self.actor_rate * delta_theta
        for s, dr, di, _ in errors:
            self.reward_value[s] += self.critic_rate * dr
            if has_beta:
                self.evidence_parameters[nz_i, nz_j, s] += (self.critic_rate * di) * w
        self.updates += 1
        self.priced_updates += int(has_beta)
        self.last_update = dict(errors=errors, actor_step_norm=float(np.linalg.norm(delta_theta * self.actor_rate)))
        if not np.isfinite(self.logits).all() or not np.isfinite(self.reward_value).all():
            raise FloatingPointError("Nonfinite CEAC actor or critic; no silent clipping")
        return True


class SparseStabilityActor(StabilityActor):
    """Numerically identical to StabilityActor, skipping zero entries of sparse beta."""

    def evidence_value(self, beta):
        nz_i, nz_j = np.nonzero(beta)
        if not len(nz_i):
            return np.zeros(self.model.n, dtype=float)
        w = beta[nz_i, nz_j]
        return w @ self.evidence_parameters[nz_i, nz_j, :]

    def update(self, episode, policy, beta):
        if episode.behavior != "actor":
            return False
        if episode.policy_id != policy.identity or policy.identity != self.snapshot().identity:
            raise ValueError("Only the frozen current-actor episode can update this optimizer")
        beta = np.asarray(beta, dtype=float)
        if beta.shape != self.evidence_parameters.shape[:2] or not np.isfinite(beta).all() or (beta < 0).any():
            raise ValueError("Invalid evidence prices")
        nz_i, nz_j = np.nonzero(beta)
        has_beta = len(nz_i) > 0
        w = beta[nz_i, nz_j] if has_beta else None
        vi = (w @ self.evidence_parameters[nz_i, nz_j, :]) if has_beta else np.zeros(self.model.n, dtype=float)
        errors, directions, geometry = [], {}, []
        before = self.logits.copy()
        for step in episode.transitions:
            s = step.state
            ll = np.asarray(step.log_likelihood)
            evidence = float(w @ (ll[nz_i] - ll[nz_j])) if has_beta else 0.0
            dr = step.reward - self.reward_value[s] + (0.0 if step.done else self.reward_value[step.next_state])
            di = evidence - vi[s] + (0.0 if step.done else vi[step.next_state])
            errors.append((s, float(dr), float(di), evidence))
            valid = np.flatnonzero(self.model.mask[s])
            p = policy.probabilities[s, valid]
            if len(valid) == 1:
                continue
            a = int(np.flatnonzero(valid == step.action)[0])
            if p[a] <= 0:
                raise FloatingPointError("Sampled action has zero propensity")
            direction = np.zeros(len(valid))
            direction[a] = (dr + di) / p[a]
            direction -= direction.mean()
            directions[s] = directions.get(s, np.zeros_like(direction)) + direction
        for s, direction in directions.items():
            valid = np.flatnonzero(self.model.mask[s])
            p = policy.probabilities[s, valid]
            eta, _, floor, bound = self.rates(s)
            q, kl, contracted, eta = guarded_proposal(p, direction, eta, floor, bound)
            self.logits[s, valid] = np.log(q)
            geometry.append(dict(state=s, policy_kl=kl, backtracks=contracted, floor_mass=floor,
                                 actor_step=eta, raw_gradient_sq=float(direction @ direction)))
        norm2 = float(w @ w) if has_beta else 0.0
        for s, dr, di, _ in errors:
            _, c, _, _ = self.rates(s)
            self.reward_value[s] += c * dr
            if has_beta:
                self.evidence_parameters[nz_i, nz_j, s] += (c / (self.schedule["evidence_step_ridge"] + norm2) * di) * w
        self.updates += 1
        self.priced_updates += int(has_beta)
        self.last_update = dict(errors=errors, actor_step_norm=float(np.linalg.norm(self.logits - before)))
        self._record(policy, beta, errors, geometry)
        if not np.isfinite(self.logits).all() or not np.isfinite(self.reward_value).all():
            raise FloatingPointError("Nonfinite stability actor/critics; no silent clipping")
        return True


NATURAL_SCHEDULE = {
    "offset": 1024,
    "actor_exponent": 0.9,
    "critic_exponent": 0.6,
    "total_floor_exponent": 0.45,
    "critic_initial": 0.24,
    "evidence_step_ridge": 0.01,
    "trust_limit_initial": 0.01,
    "trust_limit_exponent": 1.35,
}
NATURAL_CANDIDATE = {
    "id": "guarded_natural_003",
    "geometry": "natural_mirror",
    "actor_initial": 0.03,
    "total_floor": 0.02,
}


@dataclass(frozen=True)
class PathologiesCEACConfig:
    actor_rate: float = 0.4
    critic_rate: float = 0.12
    temperature: float = 1.0
    delta: float = 0.05
    refresh_interval: int = 128
    snapshot_interval: int = 8192
    replay_floor_constant: float = 64.0
    mass_bound: float = 32.0
    normalization: float = 32.0
    column_tolerance: float = 1e-9
    optimizer: str = "reference"

    def __post_init__(self):
        if self.optimizer not in {"reference", "guarded_natural"}:
            raise ValueError(f"Unknown CEAC actor optimizer: {self.optimizer}")


class GeneralPolicyReplay:
    """Policy-labeled empirical replay with certified DAG return and LLR bounds."""

    def __init__(self, model, delta: float = 0.025):
        self.model = model
        self.delta = float(delta)
        self.evidence = LikelihoodEvidence(model.num_models, self.delta)
        self.strata = defaultdict(list)
        self.seen = set()
        self.slots = {}
        self.statistics = {}
        self.policy_widths = {}
        r_min, r_max = model.return_range
        self.return_min = float(r_min)
        self.return_max = float(r_max)
        self.return_width = float(r_max - r_min)
        self._sa_low = {}
        self._sa_high = {}
        informative = set(model._informative_pairs)
        zeros_mm = np.zeros((model.num_models, model.num_models))
        low = np.zeros((model.num_models, model.num_models))
        high = np.zeros((model.num_models, model.num_models))
        for level in model.levels:
            lvl_lo, lvl_hi = None, None
            for s in level:
                for a in np.flatnonzero(model.mask[s]):
                    if (int(s), int(a)) not in informative:
                        self._sa_low[int(s), int(a)] = zeros_mm
                        self._sa_high[int(s), int(a)] = zeros_mm
                        lo_sa = hi_sa = zeros_mm
                    else:
                        laws = model.laws[int(s), int(a)]
                        ll_mat = np.log(np.array([[law[k] for k in laws[0]] for law in laws], dtype=float))
                        diffs = ll_mat[:, None, :] - ll_mat[None, :, :]  # (M, M, outcomes)
                        lo_sa = diffs.min(axis=2)
                        hi_sa = diffs.max(axis=2)
                        self._sa_low[int(s), int(a)] = lo_sa
                        self._sa_high[int(s), int(a)] = hi_sa
                    lvl_lo = lo_sa if lvl_lo is None else np.minimum(lvl_lo, lo_sa)
                    lvl_hi = hi_sa if lvl_hi is None else np.maximum(lvl_hi, hi_sa)
            if lvl_lo is not None:
                low += lvl_lo
                high += lvl_hi
        self.row_width = high - low

    @property
    def plausible(self):
        return self.evidence.plausible

    @property
    def mixture(self):
        return self.evidence.mixture

    def _compute_policy_width(self, policy: Policy) -> np.ndarray:
        """Exact conditional trajectory LLR range over outcomes reachable under `policy`."""
        model = self.model
        low = np.zeros((model.num_models, model.num_models))
        high = np.zeros((model.num_models, model.num_models))
        reachable = np.zeros(model.n, dtype=bool)
        reachable[model.initial] = True
        for level in model.levels:
            lvl_lo, lvl_hi = [], []
            for s in level:
                if not reachable[s]:
                    continue
                for a in np.flatnonzero(model.mask[s]):
                    if policy.probabilities[s, a] <= 0.0:
                        continue
                    lvl_lo.append(self._sa_low[int(s), int(a)])
                    lvl_hi.append(self._sa_high[int(s), int(a)])
                    for nxt, _, done in model.laws[int(s), int(a)][0]:
                        if not done:
                            reachable[model.index[nxt]] = True
            if lvl_lo:
                low += np.min(lvl_lo, axis=0)
                high += np.max(lvl_hi, axis=0)
        return high - low

    def register(self, policy: Policy):
        pid = policy.identity
        if pid not in self.slots:
            self.slots[pid] = len(self.slots)
            episodes = self.strata[pid]
            self.statistics[pid] = [
                len(episodes),
                sum(e.total_return for e in episodes),
                sum((e.log_likelihood for e in episodes), np.zeros(self.model.num_models)),
            ]

    def append(self, episode):
        key = (episode.context, episode.index)
        if key in self.seen:
            raise ValueError("Replay cannot count an episode twice")
        if not self.return_min - 1e-9 <= episode.total_return <= self.return_max + 1e-9:
            raise ValueError("Episode return outside certified DAG range")
        self.seen.add(key)
        self.strata[episode.policy_id].append(episode)
        self.evidence.append(episode)
        if episode.policy_id in self.statistics:
            stat = self.statistics[episode.policy_id]
            stat[0] += 1
            stat[1] += episode.total_return
            stat[2] += episode.log_likelihood

    def arrays(self, policies, rows):
        counts, means, info, rj, ri = [], [], [], [], []
        stats_count = self.model.num_models * (self.model.num_models - 1) + 1
        for policy in policies:
            self.register(policy)
            pid = policy.identity
            if pid not in self.policy_widths:
                self.policy_widths[pid] = self._compute_policy_width(policy)
            pw = self.policy_widths[pid]
            n, total, ll = self.statistics[policy.identity]
            counts.append(n)
            if not n:
                means.append(self.return_min)
                info.append(np.zeros(len(rows)))
                rj.append(self.return_width)
                ri.append(np.array([pw[r] for r in rows]))
                continue
            means.append(total / n)
            averages = ll / n
            info.append(np.array([averages[m] - averages[j] for m, j in rows]))
            slot = self.slots[policy.identity] + 1
            alpha = self.delta * 36.0 / (math.pi ** 4 * slot ** 2 * n ** 2 * stats_count)
            unit = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
            rj.append(min(self.return_width, self.return_width * unit))
            ri.append(np.array([pw[r] * unit for r in rows]))
        mean = np.asarray(means, dtype=float)
        information = np.array(info, dtype=float).T if rows else np.zeros((0, len(policies)))
        reward_radius = np.asarray(rj, dtype=float)
        row_radius = np.array(ri, dtype=float).T if rows else np.zeros((0, len(policies)))
        return dict(
            counts=np.array(counts, dtype=int),
            return_mean=mean,
            information_mean=information,
            return_radius=reward_radius,
            information_radius=row_radius,
            return_lower=np.maximum(self.return_min, mean - reward_radius),
            information_lower=np.maximum(0.0, information - row_radius),
        )

    def priced_scores(self, policies, rows, beta):
        data = self.arrays(policies, rows)
        if not rows:
            return data["return_mean"].copy()
        prices = np.array([beta[r] for r in rows])
        return data["return_mean"] + prices @ data["information_mean"]


def solve_deduplicated_replay_master(replay: GeneralPolicyReplay, policies, references, upper_optimum: float, mass_bound: float = 32.0):
    """Exact empirical Graves--Lai master LP with lossless deduplication of identical rows."""
    model = replay.model
    ref_set = set(references)
    rows = tuple(r for r in model.alternatives if r[0] in ref_set)
    data = replay.arrays(policies, rows)
    costs = np.maximum(0.0, upper_optimum - data["return_lower"])
    k = data["information_lower"]
    if not len(rows):
        eta = np.zeros(len(policies))
        beta = np.zeros((model.num_models, model.num_models))
        return ReplayDesign(tuple(p.identity for p in policies), eta, beta, 0.0, 0.0, 1.0, 0.0, tuple(references), rows, data)

    # Group identical empirical lower-bound rows to keep HiGHS compact on large product classes
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
        raise RuntimeError("CEAC replay master failed: " + res.message)
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
        raise RuntimeError("Replay primal/dual certificate failed")
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


def certified_target_calibration(model, bank: PolicyBank, xi: np.ndarray, coefficients: np.ndarray, epsilon: float = 0.25, n_max: float = 32.0):
    """Exact AEC target/calibration solve, exploiting decision-equivalence classes when M > 10."""
    if model.num_models <= 10:
        return target_calibration(model, bank, xi, coefficients, epsilon=epsilon, n_max=n_max)
    classes = model.decision_classes()
    # Verify whether unions of decision classes exhaust maximal feasible good sets:
    # For any pair of classes (A, B), if every cross-pair (a, b) is infeasible, any
    # feasible good set must be a subset of a single decision class, and by upward
    # dominance of feasible supersets only full classes need evaluation.
    key = (tuple(float(c) for c in coefficients), float(epsilon), float(n_max))
    cache = getattr(model, "_equiv_good_cache", {})
    if key not in cache:
        scratch = PolicyBank(model, model.seed_policies())
        unif = np.ones(model.num_models) / model.num_models
        whole = tuple(range(model.num_models))
        rec_whole = _aec_candidate(model, scratch, unif, whole, coefficients, epsilon, n_max)
        if rec_whole["feasible"]:
            good_groups = (whole,)
        else:
            # Check cross-class incompatibility on canonical representatives
            bad_class_pairs = set()
            for i, j in combinations(range(len(classes)), 2):
                rep_pair = (classes[i][0], classes[j][0])
                if not _aec_candidate(model, scratch, unif, rep_pair, coefficients, epsilon, n_max)["feasible"]:
                    bad_class_pairs.add((i, j))
            feasible_unions = []
            for r in range(len(classes), 0, -1):
                for idxs in combinations(range(len(classes)), r):
                    if any((a, b) in bad_class_pairs for a, b in combinations(idxs, 2)):
                        continue
                    if any(set(idxs) <= prev for prev in feasible_unions):
                        continue
                    merged = tuple(sorted(m for idx in idxs for m in classes[idx]))
                    if _aec_candidate(model, scratch, unif, merged, coefficients, epsilon, n_max)["feasible"]:
                        feasible_unions.append(set(idxs))
            good_groups = tuple(
                tuple(sorted(m for idx in sorted(u) for m in classes[idx])) for u in feasible_unions
            ) + ((),)
        cache[key] = (good_groups, tuple(scratch.policies))
        model._equiv_good_cache = cache
    good_groups, extra_policies = cache[key]
    for p in extra_policies:
        bank.add(p)
    for _ in range(32):
        before = len(bank.policies)
        candidates = [_aec_candidate(model, bank, xi, good, coefficients, epsilon, n_max) for good in good_groups]
        if len(bank.policies) == before:
            break
    else:
        raise RuntimeError("AEC common-bank closure failed")
    feasible = [c for c in candidates if c["feasible"]]
    if not feasible:
        raise RuntimeError("No feasible target/calibration set")
    all_good = [c for c in feasible if c["all_models_acceptable"]]
    chosen = dict(all_good[0] if all_good else max(feasible, key=lambda c: (c["value"], len(c["good"]))))
    positive = np.asarray(coefficients)[np.asarray(coefficients) > 1e-12]
    g_min = float(positive.min()) if len(positive) else 0.0
    q = (4.0 * n_max + epsilon * g_min) / (4.0 * n_max + 2.0 * epsilon * g_min)
    chosen["q"] = q
    chosen["mixture"] = q * chosen["target"] + (1.0 - q) * chosen["calibration"]
    return chosen


class ConfounderEliminatingAC:
    """Section-3 Tabular CEAC with public confounder-equivalence reference certification."""

    def __init__(self, model, rng, config: PathologiesCEACConfig | None = None):
        self.model = model
        self.rng = rng
        self.config = config or PathologiesCEACConfig()
        cfg = self.config
        if cfg.optimizer == "guarded_natural":
            self.actor = SparseStabilityActor(model, NATURAL_CANDIDATE, NATURAL_SCHEDULE)
        else:
            self.actor = SparseTwoCriticActor(model, cfg.actor_rate, cfg.critic_rate, cfg.temperature)
        self.replay = GeneralPolicyReplay(model, cfg.delta / 2.0)
        self.ledger = DebtLedger()
        self.bank = PolicyBank(model, model.seed_policies())
        self.seed_ids = tuple(p.identity for p in self.bank.policies)
        for p in self.bank.policies:
            self.replay.register(p)
            self.ledger.require_seed(p.identity)

        # Compute hypothetical single-model Graves--Lai coefficients (deduplicating symmetric models).
        coeff_cache = {}
        seed_rets, seed_info = self.bank.arrays()
        coeffs = []
        alt_by_ref = defaultdict(list)
        for m, j in model.alternatives:
            alt_by_ref[m].append(j)
        for m in range(model.num_models):
            rows_m = alt_by_ref[m]
            sig = (
                round(float(model.optimal_returns[m]), 12),
                tuple(np.round(seed_rets[m], 12)),
                tuple(sorted(tuple(np.round(seed_info[m, j], 12)) for j in rows_m)),
            )
            if sig not in coeff_cache:
                cbank = PolicyBank(model, model.seed_policies())
                coeff_cache[sig] = bounded_design(model, cbank, (m,), cfg.mass_bound).cost
            coeffs.append(coeff_cache[sig])
        self.coefficients = np.array(coeffs, dtype=float)

        det_count = int(np.prod(model.mask.sum(axis=1), dtype=object))
        if det_count > 100000:
            raise ValueError("Deterministic policy count exceeds 100,000")
        self.full_policies = []
        for choices in product(*[np.flatnonzero(mask) for mask in model.mask]):
            probabilities = np.zeros(model.mask.shape)
            probabilities[np.arange(model.n), choices] = 1.0
            self.full_policies.append(Policy(probabilities, "exact-pricing-candidate"))
        self.full_values = [model.evaluate(p) for p in self.full_policies]

        self._alt_set = set(model.alternatives)
        self.episodes = 0
        self.design = None
        self.prices = np.zeros((model.num_models, model.num_models))
        self.last_refs = ()
        self.certified = False
        self.refreshes = 0
        self.admissions = 0
        self.exact_admissions = 0
        self.events = []
        self.snapshots = []
        self.candidates = {}
        self.frozen = {}
        self.last_replay_scores = []
        self.last_reason = "initialization"
        self.refresh()

    def floor(self) -> int:
        t = max(self.episodes, 2)
        return max(
            1,
            math.ceil(
                self.config.replay_floor_constant
                * math.sqrt(math.log(t))
                * math.log(math.log(max(t, math.exp(math.e))))
            ),
        )

    def request_count(self, pid: str, n: int):
        self.ledger.statistical_required[pid] = max(self.ledger.statistical_required[pid], int(n))

    def is_decision_certified(self, refs: tuple[int, ...]) -> bool:
        """Public confounder-equivalence check: no policy-changing conflict inside
        `refs`, and every undominated row (m, j) with m in `refs` has identical
        stepwise LLR across `refs` (verified directly on `model.laws`).
        """
        if not refs:
            return False
        if len(refs) == 1:
            return True
        ref_set = set(refs)
        all_alts = getattr(self.model, "all_policy_changing_alternatives", self.model.alternatives)
        if any((m in ref_set and j in ref_set) for m, j in all_alts):
            return False
        # Verify that for every undominated row (m, j) with m in refs, LLR is invariant across refs
        rows_by_ref = defaultdict(list)
        for m, j in self.model.alternatives:
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

    def bootstrap(self, reason: str):
        self.certified = False
        self.prices.fill(0.0)
        self.last_reason = reason
        # Prioritize informative seed policies if available so decision conflicts resolve rapidly.
        pid = min(self.seed_ids, key=lambda x: self.ledger.counts[x])
        self.request_count(pid, self.ledger.counts[pid] + self.config.refresh_interval)

    def full_policy_check(self, design: ReplayDesign, references: tuple[int, ...]):
        rows = design.rows
        weights = np.array([design.beta[r] for r in rows])
        reward_error = 2.0 * float(design.data["return_radius"].max())
        evidence_error = 2.0 * np.max(design.data["information_radius"], axis=1) if len(rows) else np.zeros(0)
        ref0 = references[0]
        reduced = []
        for returns, information in self.full_values:
            gap = max(0.0, float(self.model.optimal_returns[ref0] - returns[ref0]))
            conservative = (
                np.maximum(0.0, np.array([information[r] for r in rows]) - evidence_error)
                if len(rows)
                else np.zeros(0)
            )
            reduced.append(gap + reward_error + design.mass_price - float(weights @ conservative))
        index = int(np.argmin(reduced))
        return (
            max(0.0, -float(reduced[index])),
            self.full_policies[index],
            dict(
                return_error=reward_error,
                information_error=evidence_error.tolist(),
                policy_count=len(self.full_policies),
                minimum_counterfactual_reduced_cost=float(reduced[index]),
            ),
        )

    def refresh(self):
        cfg = self.config
        refs = self.replay.plausible
        self.last_refs = refs
        self.refreshes += 1
        floor = self.floor()

        if self.design is not None:
            for pid, (candidate, target) in list(self.candidates.items()):
                if self.ledger.counts[pid] < target:
                    continue
                data = self.replay.arrays([candidate], self.design.rows)
                weights = np.array([self.design.beta[r] for r in self.design.rows])
                upper = float(max(self.model.optimal_returns[list(self.design.references)]))
                info_term = float(weights @ data["information_lower"][:, 0]) if len(self.design.rows) else 0.0
                reduced = float(upper - data["return_lower"][0] + self.design.mass_price - info_term)
                admitted = reduced < -cfg.column_tolerance
                if admitted and self.bank.add(candidate):
                    self.admissions += 1
                self.events.append(
                    dict(
                        episode=self.episodes,
                        snapshot=pid,
                        upper_reduced_cost=reduced,
                        certification_count=int(data["counts"][0]),
                        admitted=admitted,
                    )
                )
                del self.candidates[pid]

        for p in self.bank.policies:
            self.replay.register(p)
            self.request_count(p.identity, floor)

        if not self.is_decision_certified(refs):
            self.bootstrap("reference_certificate_pending")
            return
        if any(self.ledger.counts[p.identity] < floor for p in self.bank.policies):
            self.certified = False
            self.prices.fill(0.0)
            self.last_reason = "replay_floor_pending"
            return

        # Only run full excluded-model AEC calibration when `refs` is not yet
        # verified to lie within a single feasible decision class.
        aec = None
        if self.model.num_models <= 10:
            aec = certified_target_calibration(
                self.model, self.bank, self.replay.mixture, self.coefficients, n_max=cfg.normalization
            )
            for p in self.bank.policies:
                self.replay.register(p)
                self.request_count(p.identity, floor)
            if any(self.ledger.counts[p.identity] < floor for p in self.bank.policies):
                self.certified = False
                self.prices.fill(0.0)
                self.last_reason = "new_response_column_pending"
                return

        upper = float(max(self.model.optimal_returns[list(refs)]))
        design = solve_deduplicated_replay_master(self.replay, self.bank.policies, refs, upper, cfg.mass_bound)
        if design is None:
            self.bootstrap("replay_master_infeasible")
            return

        residual, proposal, certificate = self.full_policy_check(design, refs)
        design.column_residual = residual
        if residual > cfg.column_tolerance:
            if self.bank.add(proposal):
                self.exact_admissions += 1
                self.replay.register(proposal)
                self.request_count(proposal.identity, floor)
                self.certified = False
                self.prices.fill(0.0)
                self.last_reason = "exact_separator_certificate_pending"
                return
            self.request_count(proposal.identity, self.ledger.counts[proposal.identity] + cfg.refresh_interval)
            self.certified = False
            self.prices.fill(0.0)
            self.last_reason = "uniform_bound_pending"
            return

        self.design = design
        self.prices = design.beta.copy()
        self.certified = True
        self.last_reason = "replay_design_certified"
        self.ledger.add_design(design, evidence_scale(max(1, self.episodes)))
        if aec is not None and not set(refs).issubset(aec["good"]):
            selected = int(self.rng.choice(len(aec["mixture"]), p=aec["mixture"]))
            self.ledger.request_fresh(aec["policy_ids"][selected])
        scores = self.replay.priced_scores(self.bank.policies, design.rows, self.prices)
        self.last_replay_scores = scores.tolist()
        self.events.append(
            dict(
                episode=self.episodes,
                design=design.record(),
                priced_replay_scores=self.last_replay_scores,
                global_certificate=certificate,
                actor_updates=self.actor.updates,
                ledger=self.ledger.record(),
            )
        )

    def select(self, index: int):
        del index
        if (self.episodes and self.episodes % self.config.refresh_interval == 0) or self.replay.plausible != self.last_refs:
            self.refresh()
        pid, behavior = self.ledger.select()
        if pid is None and not self.certified:
            self.refresh()
            pid, behavior = self.ledger.select()
        if pid is not None:
            policy = self.bank.policies[self.bank._ids[pid]] if pid in self.bank._ids else self.frozen[pid]
            return policy, behavior, self.prices.copy()
        if not self.certified:
            raise RuntimeError("Unfinished CEAC certificate has no statistical request")
        policy = self.actor.snapshot()
        self.replay.register(policy)
        return policy, "actor", self.prices.copy()

    def observe(self, episode, policy: Policy, prices: np.ndarray):
        self.actor.update(episode, policy, prices)
        self.replay.append(episode)
        self.ledger.credit(episode)
        self.episodes += 1
        if self.episodes % self.config.snapshot_interval == 0 and self.design is not None and not self.candidates:
            candidate = self.actor.snapshot()
            self.snapshots.append(candidate.identity)
            ref0 = self.design.references[0]
            j_vals, k_vals = self.model.evaluate(candidate)
            exact_rc = float(self.model.optimal_returns[ref0] - j_vals[ref0] + self.design.mass_price - np.sum(self.design.beta * k_vals))
            if exact_rc < -self.config.column_tolerance:
                self.replay.register(candidate)
                target = self.floor()
                self.frozen[candidate.identity] = candidate
                self.candidates[candidate.identity] = (candidate, target)
                self.request_count(candidate.identity, target)
                self.events.append(
                    dict(
                        episode=self.episodes,
                        snapshot=candidate.identity,
                        status="awaiting_independent_policy_labeled_replay",
                        required=target,
                    )
                )

    def diagnostics(self):
        return dict(
            algorithm=f"CEAC_Section3_{self.config.optimizer}",
            actor_updates=self.actor.updates,
            priced_actor_updates=self.actor.priced_updates,
            replay_episodes=len(self.replay.seen),
            policy_strata=len(self.replay.strata),
            bank_policies=len(self.bank.policies),
            refreshes=self.refreshes,
            certified=self.certified,
            plausible_count=len(self.replay.plausible),
            reason=self.last_reason,
            priced_replay_scores=self.last_replay_scores,
            design=self.design.record() if self.design is not None else None,
        )
