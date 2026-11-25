"""Composable `MetricObserver` persistent function classes for single-pass logging & Bellman auditing."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from ceac_bench.core.contracts import CriticSnapshot, HeldOutBatch, PolicyView, SelectionStep
from ceac_bench.core.primitives import FrozenFunctionClass, RNGStreamSet, seed_for
from ceac_bench.problems.interfaces import BoundProblem, FiniteDAGOracleEvaluator
from ceac_bench.runner.learners import Learner


CHECKPOINTS_32K: tuple[int, ...] = (16, 64, 256, 1024, 4096, 16384, 32768)


def evaluate_snapshot_bellman_errors(
    snapshot: CriticSnapshot,
    batch: HeldOutBatch,
    problem: BoundProblem,
    checkpoint: int,
) -> dict[str, float]:
    """Pure Bellman/TD evaluator operating solely on `CriticSnapshot` + `HeldOutBatch` (zero private learner access)."""
    pub = problem.public_view
    S = int(pub.num_states or 0)
    A = int(pub.num_actions or 0)
    ev_oracle = problem.evaluator
    assert isinstance(ev_oracle, FiniteDAGOracleEvaluator)
    truth = ev_oracle.truth_index

    c_on = batch.onpolicy_counts
    r_on = batch.onpolicy_reward_sum
    nx_on = batch.onpolicy_next_counts
    ll_on = batch.onpolicy_ll_sum

    c_pr = batch.probe_counts
    ll_pr = batch.probe_ll_sum

    c_mix = batch.mix_counts
    r_mix = batch.mix_reward_sum
    nx_mix = batch.mix_next_counts

    tot_on = max(1, int(c_on.sum()))
    tot_pr = max(1, int(c_pr.sum()))
    tot_mix = max(1, int(c_mix.sum()))
    s_counts_on = c_on.sum(axis=1)

    if snapshot.kind == "ceac_two_critic":
        assert snapshot.reward_v is not None and snapshot.evidence_v is not None
        vr = np.r_[snapshot.reward_v[:S], 0.0]
        vi = np.r_[snapshot.evidence_v[:S], 0.0]
        prices = snapshot.epistemic_state["prices"]
        nz_i, nz_j = np.nonzero(prices)
        has_b = len(nz_i) > 0
        bw = prices[nz_i, nz_j] if has_b else None

        rew_sq, ev_td_sq, ev_norm_sq = 0.0, 0.0, 0.0
        for s in np.flatnonzero(s_counts_on > 0):
            n_s = float(s_counts_on[s])
            mean_r = float(r_on[s].sum()) / n_s
            next_dist = nx_on[s].sum(axis=0) / n_s
            rew_sq += (n_s / tot_on) * (((mean_r + float(next_dist @ vr)) - vr[s]) ** 2)
            if has_b:
                ll_avg = ll_on[s].sum(axis=0) / n_s
                ev_r = float(bw @ (ll_avg[nz_i] - ll_avg[nz_j]))
            else:
                ev_r = 0.0
            td_err = (ev_r + float(next_dist @ vi)) - vi[s]
            ev_td_sq += (n_s / tot_on) * (td_err ** 2)
            ev_norm_sq += (n_s / tot_on) * (td_err ** 2 + vi[s] ** 2)
        return {
            "reward_bellman_error": float(math.sqrt(rew_sq)),
            "evidence_critic_error": float(math.sqrt(ev_td_sq)),
            "stage_epistemic_error": float(math.sqrt(ev_norm_sq)),
        }

    if snapshot.kind == "idac":
        assert snapshot.reward_v is not None and snapshot.evidence_v is not None
        vc = snapshot.reward_v
        vs = snapshot.evidence_v
        st = snapshot.epistemic_state
        w, hi, cf = st["width"], st["reward_upper"], st["cost_floor"]
        H, elapsed = st["horizon"], st["elapsed"]
        mu_c, mu_e = st["mu_cost"], st["mu_entropy"]
        rew_sq, ev_sq = 0.0, 0.0
        for s in np.flatnonzero(s_counts_on > 0):
            n_s = float(s_counts_on[s])
            mean_r = float(r_on[s].sum()) / n_s
            mean_c = (hi - mean_r) / w + cf
            nx_vec = nx_on[s].sum(axis=0) / n_s
            term_succ = pub.initial_state if (int(elapsed[s]) + 1 == H) else (S + int(elapsed[s]) + 1)
            v_next_c = float(nx_vec[:S] @ vc[:S] + nx_vec[S] * vc[term_succ])
            v_next_s = float(nx_vec[:S] @ vs[:S] + nx_vec[S] * vs[term_succ])
            td_c = w * (mean_c - mu_c + v_next_c - vc[s])
            sh_s = -math.log(max(n_s / tot_on, 1e-6))
            td_s = sh_s - mu_e + v_next_s - vs[s]
            rew_sq += (n_s / tot_on) * (td_c ** 2)
            ev_sq += (n_s / tot_on) * (td_s ** 2 + vs[s] ** 2)
        val_e = float(math.sqrt(ev_sq))
        return {"reward_bellman_error": float(math.sqrt(rew_sq)), "evidence_critic_error": val_e, "stage_epistemic_error": val_e}

    if snapshot.kind == "ucb_q":
        assert snapshot.reward_q is not None and snapshot.evidence_q is not None and pub.action_mask is not None
        q_rew = snapshot.reward_q
        bonus_all = snapshot.evidence_q
        H = snapshot.epistemic_state["horizon"]
        elapsed = snapshot.epistemic_state["elapsed"]
        v_rew_s = np.array([float(q_rew[s_i, pub.action_mask[s_i]].max()) for s_i in range(S)], dtype=float)
        rew_sq, ev_sq = 0.0, 0.0
        for s, a in zip(*np.nonzero(c_mix > 0)):
            cnt = float(c_mix[s, a])
            mean_r = float(r_mix[s, a]) / cnt
            nx_dist = nx_mix[s, a] / cnt
            t_s = int(elapsed[s])
            term_val = 0.0 if (t_s + 1 == H) else float(q_rew[S + t_s + 1, 0])
            cont = float(nx_dist[:S] @ v_rew_s + nx_dist[S] * term_val)
            bell_r = (mean_r + cont) - q_rew[s, a]
            rew_sq += (cnt / tot_mix) * (bell_r ** 2)
            ev_sq += (cnt / tot_mix) * (bonus_all[s, a] ** 2)
        val_e = float(math.sqrt(ev_sq))
        return {"reward_bellman_error": float(math.sqrt(rew_sq)), "evidence_critic_error": val_e, "stage_epistemic_error": val_e}

    if snapshot.kind == "pomd":
        assert snapshot.reward_q is not None and snapshot.evidence_q is not None and snapshot.reward_v is not None
        q_cost = snapshot.reward_q
        vals = snapshot.reward_v
        bonus = snapshot.evidence_q
        st = snapshot.epistemic_state
        w, hi, H, elapsed = st["width"], st["reward_upper"], st["horizon"], st["elapsed"]
        rew_sq, ev_sq = 0.0, 0.0
        for s, a in zip(*np.nonzero(c_mix > 0)):
            cnt = float(c_mix[s, a])
            mean_r = float(r_mix[s, a]) / cnt
            nx_dist = nx_mix[s, a] / cnt
            t_s = int(elapsed[s])
            cont_cost = float(nx_dist[:S] @ vals[:S] + (0.0 if (t_s + 1 == H) else nx_dist[S] * vals[S + t_s + 1]))
            mean_cost = (hi - mean_r) / w
            bell_r = w * ((mean_cost + cont_cost) - q_cost[s, a])
            rew_sq += (cnt / tot_mix) * (bell_r ** 2)
            ev_sq += (cnt / tot_mix) * (bonus[s, a] ** 2)
        val_e = float(math.sqrt(ev_sq))
        return {"reward_bellman_error": float(math.sqrt(rew_sq)), "evidence_critic_error": val_e, "stage_epistemic_error": val_e}

    if snapshot.kind == "bayesian_ids":
        assert snapshot.evidence_q is not None and pub.levels is not None
        model = pub.candidate_model
        st = snapshot.epistemic_state
        post = st["posterior"]
        eval_pol: PolicyView = st["eval_policy"]
        probs_on = eval_pol.probabilities
        assert probs_on is not None
        stage_r = np.einsum("m,ms,msa->sa", post, model.density, model.reward)
        stage_i = snapshot.evidence_q
        p_next = np.einsum("m,msan->san", post, ev_oracle.p_next_all)

        qr = np.zeros((S, A), dtype=float)
        vr = np.zeros(S + 1, dtype=float)
        for lvl in range(pub.horizon - 1, -1, -1):
            sts = list(pub.levels[lvl])
            cont = p_next[sts] @ vr[:S] if lvl < pub.horizon - 1 else np.zeros((len(sts), A))
            qr[sts] = stage_r[sts] + cont
            vr[sts] = np.sum(probs_on[sts] * qr[sts], axis=1)

        rew_sq, st_ep_sq, surprise_sq = 0.0, 0.0, 0.0
        for s, a in zip(*np.nonzero(c_mix > 0)):
            cnt = float(c_mix[s, a])
            mean_r = float(r_mix[s, a]) / cnt
            nx_dist = nx_mix[s, a] / cnt
            bell_r = (mean_r + float(nx_dist @ vr)) - qr[s, a]
            rew_sq += (cnt / tot_mix) * (bell_r ** 2)

        for s, a in zip(*np.nonzero(c_pr > 0)):
            cnt = float(c_pr[s, a])
            st_ep_sq += (cnt / tot_pr) * (stage_i[s, a] ** 2)
            ll_avg = ll_pr[s, a] / cnt
            surp = float(ll_avg[truth] - float(post @ ll_avg))
            surprise_sq += (cnt / tot_pr) * (surp ** 2)

        init_supp = st["initial_support"]
        supp_lw = st["log_weights"][init_supp]
        finite_lw = supp_lw[np.isfinite(supp_lw)]
        llr_div = float(np.std(finite_lw)) if len(finite_lw) > 1 else 0.0
        if not bool(init_supp[truth]):
            llr_div = float(checkpoint) * math.sqrt(max(surprise_sq, 1e-12))

        return {
            "reward_bellman_error": float(math.sqrt(rew_sq)),
            "evidence_critic_error": float(math.sqrt(llr_div ** 2 + st_ep_sq)),
            "stage_epistemic_error": float(math.sqrt(st_ep_sq + surprise_sq)),
        }

    return {"reward_bellman_error": 0.0, "evidence_critic_error": 0.0, "stage_epistemic_error": 0.0}


class _PerfSession:
    def __init__(self, window_size: int, log_every: int, include_first: bool, episodes: int) -> None:
        self.window: deque[float] = deque(maxlen=window_size)
        self.log_every = log_every
        self.include_first = include_first
        self.episodes = episodes
        self.cumulative = 0.0
        self.ep_series: list[float] = []
        self.gap_series: list[float] = []
        self.reg_series: list[float] = []

    def on_episode(self, *, episode_num: int, gap: float, **_kw: Any) -> None:
        self.cumulative += gap
        self.window.append(gap)
        if (self.include_first and episode_num == 1) or (episode_num % self.log_every == 0) or (episode_num == self.episodes):
            self.ep_series.append(float(episode_num))
            self.gap_series.append(round(float(np.mean(self.window)), 6))
            self.reg_series.append(round(float(self.cumulative), 4))

    def finalize(self) -> dict[str, Any]:
        return {
            "late_gap": float(np.mean(self.window)) if self.window else 0.0,
            "cumulative_regret": float(self.cumulative),
            "trace": {
                "episode": tuple(self.ep_series),
                "window_gap": tuple(self.gap_series),
                "cumulative_regret": tuple(self.reg_series),
            },
        }


@dataclass(frozen=True)
class PerformanceTraceObserver(FrozenFunctionClass):
    """Logs rolling policy gap and cumulative pseudo-regret trajectories."""

    window_size: int = 256
    log_every: int = 512
    include_episode_one: bool = True

    def start_session(self, problem: BoundProblem, learner: Learner, method_id: str, episodes: int) -> _PerfSession:
        return _PerfSession(self.window_size, self.log_every, self.include_episode_one, episodes)


class _EventSession:
    def __init__(self, event_names: tuple[str, ...], log_every: int, include_first: bool, episodes: int) -> None:
        self.counts = {name: 0 for name in event_names}
        self.log_every = log_every
        self.include_first = include_first
        self.episodes = episodes
        self.traces: dict[str, list[float]] = {f"event_rate_{name}": [] for name in event_names}

    def on_episode(self, *, episode_num: int, events: set[str], **_kw: Any) -> None:
        for ev in events:
            if ev in self.counts:
                self.counts[ev] += 1
        if (self.include_first and episode_num == 1) or (episode_num % self.log_every == 0) or (episode_num == self.episodes):
            for name, cnt in self.counts.items():
                self.traces[f"event_rate_{name}"].append(round(cnt / episode_num, 5))

    def finalize(self) -> dict[str, Any]:
        scalars = {f"event_{k}": int(v) for k, v in self.counts.items()}
        if self.episodes > 0:
            for k, v in self.counts.items():
                scalars[f"pct_{k}"] = round(100.0 * v / self.episodes, 4)
        return {
            "scalars": scalars,
            "trace": {k: tuple(v) for k, v in self.traces.items()},
        }


@dataclass(frozen=True)
class DomainEventObserver(FrozenFunctionClass):
    """Dynamically tracks domain-specific physical events declared in `DomainMetadata.event_names`."""

    log_every: int = 512
    include_episode_one: bool = True

    def start_session(self, problem: BoundProblem, learner: Learner, method_id: str, episodes: int) -> _EventSession:
        return _EventSession(problem.metadata.event_names, self.log_every, self.include_episode_one, episodes)


class _DiagnosticsSession:
    def __init__(self) -> None:
        self.actor_updates = 0
        self.priced_actor_updates = 0

    def on_episode(self, *, selection: SelectionStep, **_kw: Any) -> None:
        if selection.behavior == "actor":
            self.actor_updates += 1
            if selection.prices is not None and bool(np.any(selection.prices > 0)):
                self.priced_actor_updates += 1

    def finalize(self) -> dict[str, Any]:
        return {
            "scalars": {
                "actor_updates": int(self.actor_updates),
                "priced_actor_updates": int(self.priced_actor_updates),
            }
        }


@dataclass(frozen=True)
class LearnerDiagnosticsObserver(FrozenFunctionClass):
    """Records actor update counts, priced updates, and final learner diagnostics."""

    def start_session(self, problem: BoundProblem, learner: Learner, method_id: str, episodes: int) -> _DiagnosticsSession:
        return _DiagnosticsSession()


class _BellmanSession:
    def __init__(
        self,
        problem: BoundProblem,
        learner: Learner,
        method_id: str,
        checkpoints: set[int],
        n_onpolicy: int,
        n_probe: int,
        suite_seed_key: str,
    ) -> None:
        self.problem = problem
        self.learner = learner
        self.method_id = method_id
        self.checkpoints = checkpoints
        self.n_onpolicy = n_onpolicy
        self.n_probe = n_probe
        self.suite_seed_key = suite_seed_key
        self.ep_list: list[float] = []
        self.rew_list: list[float] = []
        self.ev_list: list[float] = []
        self.stg_list: list[float] = []

    def on_episode(self, *, episode_num: int, selection: SelectionStep, streams: RNGStreamSet, **_kw: Any) -> None:
        if self.method_id == "oracle" or episode_num not in self.checkpoints:
            return
        snap = self.learner.critic_snapshot(selection)
        eval_pol: PolicyView = snap.epistemic_state.get("eval_policy", selection.policy)
        cond = self.problem.condition
        if self.suite_seed_key:
            eval_rng = np.random.default_rng(
                seed_for("heldout_fast_eval", self.suite_seed_key, cond.variant_id, str(cond.dose), int(cond.seed), self.method_id, int(episode_num))
            )
        else:
            eval_rng = streams.make_checkpoint_eval_rng(episode_num)

        batch = self.problem.evaluator.sample_heldout_batch(eval_pol, self.n_onpolicy, self.n_probe, eval_rng)
        errs = evaluate_snapshot_bellman_errors(snap, batch, self.problem, episode_num)
        self.ep_list.append(float(episode_num))
        self.rew_list.append(errs["reward_bellman_error"])
        self.ev_list.append(errs["evidence_critic_error"])
        self.stg_list.append(errs["stage_epistemic_error"])

    def finalize(self) -> dict[str, Any]:
        if not self.ep_list:
            return {}
        return {
            "scalars": {
                "final_reward_bellman_error": self.rew_list[-1],
                "final_evidence_critic_error": self.ev_list[-1],
            },
            "critic_audit": {
                "episode": tuple(self.ep_list),
                "reward_bellman_error": tuple(self.rew_list),
                "evidence_critic_error": tuple(self.ev_list),
                "stage_epistemic_error": tuple(self.stg_list),
            },
        }


@dataclass(frozen=True)
class HeldOutBellmanObserver(FrozenFunctionClass):
    """Single-pass held-out Reward & Evidence Critic Bellman RMSE observer (zero training-RNG mutation)."""

    checkpoints: tuple[int, ...] = CHECKPOINTS_32K
    n_onpolicy: int = 512
    n_probe: int = 256
    compat_suite_key: str = ""

    def start_session(self, problem: BoundProblem, learner: Learner, method_id: str, episodes: int) -> _BellmanSession:
        cp_set = {int(c) for c in self.checkpoints if int(c) <= int(episodes)}
        return _BellmanSession(problem, learner, method_id, cp_set, self.n_onpolicy, self.n_probe, self.compat_suite_key)
