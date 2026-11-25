"""Fast vectorized multinomial Held-Out Reward & Evidence Critic Bellman evaluator and plotters.

Mathematically identical to rolling out N_onpolicy + N_probe independent episodes
in a separate `eval_env` under the realized true model `truth`: at each DAG level
and reachable state s, action counts and outcome counts follow exact multinomial
draws from `probs[s]` and `model.laws[s, a][truth]`, running ~1,000x faster than
Python step loops while maintaining zero mutation of any learner or training RNG.
"""
from __future__ import annotations
import gzip
import json
import math
from pathlib import Path
import numpy as np
from ceac_bench.reporting.theme import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    style_ceac_legend,
)
from ceac_pathologies.run import atomic_json, digest, seed_for


SUITE_COLORS = dict(METHOD_COLORS)
SUITE_LINESTYLES = dict(METHOD_LINESTYLES)
SUITE_MARKERS = dict(METHOD_MARKERS)
SUITE_LABELS = {
    **METHOD_LABELS,
    "ids_product": "IDS · model-MI, uniform prior",
    "ids_action_tree": "IDS · optimal-policy MI I(Π*; Z)",
    "ids_action_onpath": "IDS · on-path quotient MI I(C*; Z)",
}


def compile_outcome_tables(model, truth: int) -> dict:
    """Precompile true-model and per-model transition/reward/log-likelihood tables once per cell."""
    S, A, M = model.n, model.actions, model.num_models
    tables = {}
    p_next_true = np.zeros((S, A, S), dtype=float)
    p_next_all = np.zeros((M, S, A, S), dtype=float)
    for s in range(S):
        for a in np.flatnonzero(model.mask[s]):
            laws = model.laws[int(s), int(a)]
            keys = list(laws[0].keys())
            p_true = np.array([laws[truth][k] for k in keys], dtype=float)
            p_true /= p_true.sum()
            rews = np.array([float(k[1]) for k in keys], dtype=float)
            dests = np.array([S if k[2] else model.index[int(k[0])] for k in keys], dtype=int)
            probs_m = np.array([[laws[m][k] for m in range(M)] for k in keys], dtype=float)  # (K, M)
            ll_mat = np.log(probs_m)
            tables[int(s), int(a)] = (p_true, rews, dests, ll_mat)
            for k_idx, d in enumerate(dests):
                if d < S:
                    p_next_true[int(s), int(a), d] += p_true[k_idx]
                    p_next_all[:, int(s), int(a), d] += probs_m[k_idx]
    return dict(tables=tables, p_next_true=p_next_true, p_next_all=p_next_all)


def _multinomial_rollout_stats(model, compiled: dict, probs: np.ndarray, n_episodes: int, rng: np.random.Generator):
    """Exact multinomial rollout of `n_episodes` independent held-out episodes on the DAG."""
    S, A, M = model.n, model.actions, model.num_models
    tables = compiled["tables"]
    sa_counts = np.zeros((S, A), dtype=int)
    sa_rew_sum = np.zeros((S, A), dtype=float)
    sa_next_counts = np.zeros((S, A, S + 1), dtype=int)
    sa_ll_sum = np.zeros((S, A, M), dtype=float)

    s_entering = np.zeros(S + 1, dtype=int)
    s_entering[model.initial] = int(n_episodes)

    for level in model.levels:
        for s in level:
            s_i = int(s)
            cnt_s = int(s_entering[s_i])
            if cnt_s <= 0:
                continue
            p_a = np.clip(probs[s_i], 0.0, 1.0)
            p_a /= p_a.sum()
            a_draws = rng.multinomial(cnt_s, p_a)
            for a_i in np.flatnonzero(a_draws > 0):
                a_i = int(a_i)
                cnt_sa = int(a_draws[a_i])
                sa_counts[s_i, a_i] += cnt_sa
                p_true, rews, dests, ll_mat = tables[s_i, a_i]
                o_draws = rng.multinomial(cnt_sa, p_true)
                sa_rew_sum[s_i, a_i] += float(o_draws @ rews)
                sa_ll_sum[s_i, a_i] += o_draws @ ll_mat
                for o_cnt, d_idx in zip(o_draws, dests):
                    if o_cnt > 0:
                        sa_next_counts[s_i, a_i, d_idx] += int(o_cnt)
                        if d_idx < S:
                            s_entering[d_idx] += int(o_cnt)
    return sa_counts, sa_rew_sum, sa_next_counts, sa_ll_sum


def evaluate_heldout_critics_fast(
    suite_key: str,
    kind: str,
    method: str,
    learner,
    model,
    compiled: dict,
    truth: int,
    policy,
    prices: np.ndarray,
    checkpoint: int,
    seed: int,
    dose,
    n_onpolicy: int = 512,
    n_probe: int = 256,
    certified_vr: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute Held-Out Reward Bellman RMSE, Evidence Critic TD/Divergence RMSE, and Stage-Epistemic RMSE."""
    S, A = model.n, model.actions
    eval_rng = np.random.default_rng(
        seed_for("heldout_fast_eval", suite_key, kind, str(dose), int(seed), method, int(checkpoint))
    )

    # For Section-3 CEAC variants, evaluate V_R and V_I on the actor's own policy snapshot
    # (unless certified_vr is explicitly passed for v90 certified_deploy).
    if method in {"ceac", "ceac_natural", "ceac_ranked", "ceac_ranked_natural"} and certified_vr is None:
        probs_on = learner.actor.snapshot().probabilities
    else:
        probs_on = policy.probabilities
    probs_pr = model.mask / model.mask.sum(axis=1, keepdims=True)

    c_on, r_on, nx_on, ll_on = _multinomial_rollout_stats(model, compiled, probs_on, n_onpolicy, eval_rng)
    c_pr, r_pr, nx_pr, ll_pr = _multinomial_rollout_stats(model, compiled, probs_pr, n_probe, eval_rng)

    c_mix = c_on + c_pr
    r_mix = r_on + r_pr
    nx_mix = nx_on + nx_pr

    tot_on = max(1, int(c_on.sum()))
    tot_pr = max(1, int(c_pr.sum()))
    tot_mix = max(1, int(c_mix.sum()))

    s_counts_on = c_on.sum(axis=1)

    if method in {"ceac", "ceac_natural", "ceac_ranked", "ceac_ranked_natural"}:
        actor = learner.actor
        vr = np.r_[certified_vr[:S], 0.0] if certified_vr is not None else np.r_[actor.reward_value[:S], 0.0]
        if hasattr(actor, "evidence_value"):
            vi_raw = actor.evidence_value(prices)[:S]
        else:
            # Historical v90 ActorCritic (`row_values` of shape (M, M, S))
            vi_raw = np.einsum("ij,ijs->s", prices, actor.row_values[:, :, :S])
        vi = np.r_[vi_raw, 0.0]

        nz_i, nz_j = np.nonzero(prices)
        has_b = len(nz_i) > 0
        bw = prices[nz_i, nz_j] if has_b else None

        rew_sq = 0.0
        ev_td_sq = 0.0
        ev_norm_sq = 0.0
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

        return dict(
            reward_bellman_error=float(math.sqrt(rew_sq)),
            evidence_critic_error=float(math.sqrt(ev_td_sq)),
            stage_epistemic_error=float(math.sqrt(ev_norm_sq)),
        )

    if method == "idac":
        w = learner.spec.width
        hi = learner.spec.reward_upper
        cf = learner.config.cost_floor
        vc = learner.cost_value
        vs = learner.shadow_value
        H = learner.spec.horizon
        elapsed = learner.spec.elapsed
        rew_sq = 0.0
        ev_sq = 0.0
        for s in np.flatnonzero(s_counts_on > 0):
            n_s = float(s_counts_on[s])
            mean_r = float(r_on[s].sum()) / n_s
            mean_c = (hi - mean_r) / w + cf
            nx_vec = nx_on[s].sum(axis=0) / n_s
            term_succ = model.initial if (int(elapsed[s]) + 1 == H) else (S + int(elapsed[s]) + 1)
            v_next_c = float(nx_vec[:S] @ vc[:S] + nx_vec[S] * vc[term_succ])
            v_next_s = float(nx_vec[:S] @ vs[:S] + nx_vec[S] * vs[term_succ])
            td_c = w * (mean_c - learner.mu_cost + v_next_c - vc[s])
            sh_s = -math.log(max(n_s / tot_on, 1e-6))
            td_s = sh_s - learner.mu_entropy + v_next_s - vs[s]
            rew_sq += (n_s / tot_on) * (td_c ** 2)
            ev_sq += (n_s / tot_on) * (td_s ** 2 + vs[s] ** 2)
        val_e = float(math.sqrt(ev_sq))
        return dict(
            reward_bellman_error=float(math.sqrt(rew_sq)),
            evidence_critic_error=val_e,
            stage_epistemic_error=val_e,
        )

    if method == "ucb_q":
        w = learner.spec.width
        lo = learner.spec.reward_lower
        H = learner.spec.horizon
        elapsed = learner.spec.elapsed
        n_vis = np.maximum(1, learner.counts)
        bonus_all = w * learner.config.bonus_constant * np.sqrt((H ** 3) * learner.log_factor / n_vis)
        q_tot = lo + w * learner.q
        q_rew = q_tot - bonus_all
        v_rew_s = np.array([float(q_rew[s_i, model.mask[s_i]].max()) for s_i in range(S)], dtype=float)
        rew_sq = 0.0
        ev_sq = 0.0
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
        return dict(
            reward_bellman_error=float(math.sqrt(rew_sq)),
            evidence_critic_error=val_e,
            stage_epistemic_error=val_e,
        )

    if method == "pomd":
        w = learner.spec.width
        hi = learner.spec.reward_upper
        H = learner.spec.horizon
        elapsed = learner.spec.elapsed
        cb, tb = learner.bonuses()
        bonus = w * (cb[:S] + tb[:S])
        q_tot = hi - w * learner.q
        v_tot_s = np.sum(probs_on * q_tot[:S], axis=1)
        rew_sq = 0.0
        ev_sq = 0.0
        for s, a in zip(*np.nonzero(c_mix > 0)):
            cnt = float(c_mix[s, a])
            mean_r = float(r_mix[s, a]) / cnt
            nx_dist = nx_mix[s, a] / cnt
            t_s = int(elapsed[s])
            term_val = hi if (t_s + 1 == H) else float(q_tot[S + t_s + 1, 0])
            # In POMD cost space, terminal done at t_s + 1 == H has 0 remaining cost;
            # on reward scale q_tot[s,a] = hi - w * q_cost[s,a], continuation reward is v_tot_s - hi when non-terminal
            cont_cost = float(nx_dist[:S] @ learner.values[:S] + (0.0 if (t_s + 1 == H) else nx_dist[S] * learner.values[S + t_s + 1]))
            mean_cost = (hi - mean_r) / w
            bell_r = w * ((mean_cost + cont_cost) - learner.q[s, a])
            rew_sq += (cnt / tot_mix) * (bell_r ** 2)
            ev_sq += (cnt / tot_mix) * (bonus[s, a] ** 2)
        val_e = float(math.sqrt(ev_sq))
        return dict(
            reward_bellman_error=float(math.sqrt(rew_sq)),
            evidence_critic_error=val_e,
            stage_epistemic_error=val_e,
        )

    if method.startswith("ids"):
        post = learner.posterior
        stage_r = np.einsum("m,ms,msa->sa", post, model.density, model.reward)
        stage_i = (
            learner.action_information_stage(post)
            if hasattr(learner, "action_information_stage")
            else model.mutual_information_stage(post)
        )
        p_next = np.einsum("m,msan->san", post, compiled["p_next_all"])
        qr = np.zeros((S, A), dtype=float)
        vr = np.zeros(S + 1, dtype=float)
        for lvl in reversed(model.levels):
            sts = lvl
            if not len(sts):
                continue
            cont = p_next[sts] @ vr[:S]
            qr[sts] = stage_r[sts] + cont
            vr[sts] = np.sum(probs_on[sts] * qr[sts], axis=1)

        rew_sq = 0.0
        for s, a in zip(*np.nonzero(c_mix > 0)):
            cnt = float(c_mix[s, a])
            mean_r = float(r_mix[s, a]) / cnt
            nx_dist = nx_mix[s, a] / cnt
            bell_r = (mean_r + float(nx_dist @ vr)) - qr[s, a]
            rew_sq += (cnt / tot_mix) * (bell_r ** 2)

        st_ep_sq = 0.0
        surprise_sq = 0.0
        for s, a in zip(*np.nonzero(c_pr > 0)):
            cnt = float(c_pr[s, a])
            st_ep_sq += (cnt / tot_pr) * (stage_i[s, a] ** 2)
            ll_avg = ll_pr[s, a] / cnt
            surp = float(ll_avg[truth] - float(post @ ll_avg))
            surprise_sq += (cnt / tot_pr) * (surp ** 2)

        supp = getattr(learner, "initial_support", np.ones(model.num_models, dtype=bool))
        supp_lw = learner.log_weights[supp]
        finite_lw = supp_lw[np.isfinite(supp_lw)]
        llr_div = float(np.std(finite_lw)) if len(finite_lw) > 1 else 0.0
        if not bool(supp[truth]):
            llr_div = max(llr_div, float(checkpoint) * math.sqrt(max(surprise_sq, 1e-12)))

        return dict(
            reward_bellman_error=float(math.sqrt(rew_sq)),
            evidence_critic_error=float(math.sqrt(llr_div ** 2 + st_ep_sq)),
            stage_epistemic_error=float(math.sqrt(st_ep_sq + surprise_sq)),
        )

    raise ValueError(f"Unsupported method: {method}")


def save_suite_critic_audit(output_dir: Path | str, payload: dict, filename: str = "critic_bellman_audit.json.gz") -> str:
    """Write compressed critic audit bundle and return its SHA-256 digest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    with gzip.open(target, "wt") as f:
        json.dump(payload, f, allow_nan=False, separators=(",", ":"))
    return digest(target)


def load_suite_critic_audit(folder: Path | str, filename: str = "critic_bellman_audit.json.gz") -> dict:
    """Load and SHA-256-verify a portable held-out critic audit bundle."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = manifest["files"].get(filename)
    if not expected or digest(folder / filename) != expected:
        raise ValueError(f"Critic Bellman audit checksum mismatch: {folder / filename}")
    with gzip.open(folder / filename, "rt") as f:
        return json.load(f)


def _format_dose_tick(d) -> str:
    if isinstance(d, (int, np.integer)) or (isinstance(d, float) and d.is_integer() and d >= 2):
        return str(int(d))
    return f"{float(d):.2g}" if abs(float(d) - round(float(d), 2)) > 1e-9 else f"{float(d):.2f}".rstrip("0").rstrip(".")


def draw_suite_critic_traces(
    cells: list[dict],
    kinds: list[tuple[str, str]],
    target_dose,
    suptitle: str,
    method_order: list[str],
    ranked_ceac_present: bool = False,
    field: str | None = None,
):
    """Render clean 1x2 overlaid Held-Out Critic time traces (never >2 panels per figure)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    sub = [c for c in cells if math.isclose(float(c["dose"]), float(target_dose), abs_tol=1e-9)]
    methods = [m for m in method_order if any(c["method"] == m for c in sub)]

    field_meta = {
        "reward_bellman_error": ("Held-Out Reward Critic Bellman Error ↓", "Held-out Reward Bellman RMSE (log) ↓"),
        "evidence_critic_error": ("Held-Out Evidence Critic TD / Divergence ↓", "Held-out Evidence Critic RMSE (log) ↓"),
    }
    if len(kinds) == 1 and field is None:
        panel_specs = [
            (kinds[0][0], kinds[0][1], "reward_bellman_error", *field_meta["reward_bellman_error"]),
            (kinds[0][0], kinds[0][1], "evidence_critic_error", *field_meta["evidence_critic_error"]),
        ]
    else:
        chosen_field = field or "evidence_critic_error"
        sfx, ylab_default = field_meta[chosen_field]
        panel_specs = [
            (k_key, k_title, chosen_field, sfx, ylab_default)
            for k_key, k_title in kinds[:2]
        ]

    n_cols = len(panel_specs)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.8 * n_cols if n_cols > 1 else 8.8, 5.1), squeeze=False)

    for c_idx, (kind_key, kind_title, fld, panel_sfx, ylab) in enumerate(panel_specs):
        ax = axes[0, c_idx]
        xs_ref = None
        for m_i, m in enumerate(methods):
            grp = sorted([c for c in sub if c["kind"] == kind_key and c["method"] == m], key=lambda x: x["seed"])
            if not grp:
                continue
            xs = np.array([pt["episode"] for pt in grp[0]["checkpoints"]], dtype=float)
            xs_ref = xs
            vals = np.array([[max(1e-5, pt[fld]) for pt in g["checkpoints"]] for g in grp], dtype=float)
            mean = vals.mean(axis=0)
            sd = vals.std(axis=0, ddof=1) if len(grp) > 1 else np.zeros_like(mean)
            col = SUITE_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            ls = "--" if (ranked_ceac_present and m == "ceac") else SUITE_LINESTYLES.get(m, "-")
            lw = 3.25 if (is_ranked or (m == "ceac" and not ranked_ceac_present)) else (2.75 if is_ceac else 1.75)
            ms = 8.0 if is_ceac else 6.0
            zord = 11 if is_ranked else (9 if is_ceac else (5 if m.startswith("ids_action") else 4))
            step_mk = max(1, len(xs) // 7)
            mk_idx = list(range(m_i % step_mk, len(xs), step_mk)) or [0]
            ax.fill_between(xs, np.maximum(1e-5, mean - sd), mean + sd, color=col, alpha=0.13 if is_ceac else 0.08, linewidth=0)
            ax.plot(
                xs,
                mean,
                label=SUITE_LABELS[m],
                color=col,
                lw=lw,
                linestyle=ls,
                marker=SUITE_MARKERS[m],
                markersize=ms,
                markevery=mk_idx,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=zord,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        if xs_ref is not None:
            ax.set_xlim(xs_ref[0] * 0.86, xs_ref[-1] * 1.14)
            ticks = sorted({float(v) for v in xs_ref})
            if len(ticks) > 7:
                idxs = np.linspace(0, len(ticks) - 1, 6, dtype=int)
                ticks = [ticks[i] for i in idxs]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t)//1000}k" if t >= 1000 and int(t) % 1000 == 0 else str(int(t)) for t in ticks])
        ax.minorticks_off()
        ax.set_title(f"{kind_title} · {panel_sfx}", fontsize=10.5, pad=8)
        ax.set_xlabel("Training episodes (log scale)", fontsize=9.6, labelpad=4)
        if c_idx == 0 or len(kinds) == 1:
            ax.set_ylabel(ylab, fontsize=9.6)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    n_leg_rows = math.ceil(len(methods) / min(3, max(1, len(methods))))
    bot = 0.19 + 0.038 * max(0, n_leg_rows - 2)
    fig.suptitle(suptitle, fontsize=12.2, y=0.985)
    fig.tight_layout(rect=(0.0, bot, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, bot - 0.02),
        ncol=min(3, max(1, len(methods))),
        frameon=False,
        fontsize=8.9,
    )
    style_ceac_legend(leg)
    return fig


def draw_suite_critic_dose_scaling(
    cells: list[dict],
    kinds: list[tuple[str, str]],
    xlab: str,
    suptitle: str,
    method_order: list[str],
    ranked_ceac_present: bool = False,
):
    """Render clean 1x2 late-horizon Held-Out Critic Error scaling across pathology doses."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    doses = sorted({c["dose"] for c in cells})
    methods = [m for m in method_order if any(c["method"] == m for c in cells)]

    if len(kinds) == 1:
        panel_specs = [
            (kinds[0][0], kinds[0][1], "evidence_critic_error", "Final Evidence Critic TD / Divergence vs. Dose", "Final Evidence Critic RMSE (log) ↓"),
            (kinds[0][0], kinds[0][1], "stage_epistemic_error", "Final Stage-Epistemic / Bonus Error vs. Dose", "Bounded Stage-Critic RMSE (log) ↓"),
        ]
    else:
        panel_specs = [
            (k_key, k_title, "evidence_critic_error", "Final Held-Out Evidence Critic Error vs. Dose", "Final Evidence Critic RMSE (log) ↓")
            for k_key, k_title in kinds[:2]
        ]

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(13.4, 5.1), squeeze=False)
    for c_idx, (kind_key, kind_title, field, col_sfx, ylab) in enumerate(panel_specs):
        ax = axes[0, c_idx]
        for m in methods:
            xs, ys, sds = [], [], []
            for d in doses:
                grp = [c for c in cells if c["kind"] == kind_key and c["method"] == m and math.isclose(float(c["dose"]), float(d), abs_tol=1e-9)]
                if not grp:
                    continue
                v = np.array([max(1e-5, g["checkpoints"][-1][field]) for g in grp], dtype=float)
                xs.append(float(d))
                ys.append(float(v.mean()))
                sds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            if not xs:
                continue
            xs_a, ys_a, sds_a = np.array(xs), np.array(ys), np.array(sds)
            col = SUITE_COLORS[m]
            is_ceac = m.startswith("ceac")
            is_ranked = m.startswith("ceac_ranked")
            ls = "--" if (ranked_ceac_present and m == "ceac") else SUITE_LINESTYLES.get(m, "-")
            lw = 3.25 if (is_ranked or (m == "ceac" and not ranked_ceac_present)) else (2.75 if is_ceac else 1.75)
            ms = 8.0 if is_ceac else 6.0
            zord = 11 if is_ranked else (9 if is_ceac else 4)
            ax.fill_between(xs_a, np.maximum(1e-5, ys_a - sds_a), ys_a + sds_a, color=col, alpha=0.13 if is_ceac else 0.08, linewidth=0)
            ax.plot(
                xs_a,
                ys_a,
                label=SUITE_LABELS[m],
                color=col,
                lw=lw,
                linestyle=ls,
                marker=SUITE_MARKERS[m],
                markersize=ms,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=zord,
            )
        ax.set_yscale("log")
        ax.set_xticks([float(d) for d in doses])
        ax.set_xticklabels([_format_dose_tick(d) for d in doses])
        ax.minorticks_off()
        ax.set_title(f"{kind_title} · {col_sfx}", fontsize=10.4, pad=8)
        ax.set_xlabel(xlab, fontsize=9.5, labelpad=4)
        if c_idx == 0 or len(kinds) == 1:
            ax.set_ylabel(ylab, fontsize=9.5)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    n_leg_rows = math.ceil(len(methods) / min(3, max(1, len(methods))))
    bot = 0.19 + 0.038 * max(0, n_leg_rows - 2)
    fig.suptitle(suptitle, fontsize=12.2, y=0.985)
    fig.tight_layout(rect=(0.0, bot, 1.0, 0.925))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, bot - 0.02),
        ncol=min(3, max(1, len(methods))),
        frameon=False,
        fontsize=8.9,
    )
    style_ceac_legend(leg)
    return fig
