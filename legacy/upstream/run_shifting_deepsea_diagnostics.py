#!/usr/bin/env python3
"""Exact mechanism diagnostics for CEAC on a shifting Diagnostic DeepSea abstraction.

The script deliberately precedes neural actor--critic experiments.  It validates four
claims that can be tested without function-approximation confounds:
  1. the one-alternative Graves--Lai LP switches at the regret-per-nat crossover;
  2. decision-directed designs retain a constant coefficient as irrelevant gates grow;
  3. when the active deep gate shifts, CEAC buys the right policy-level evidence;
  4. Fisher information locally approximates exact Bernoulli KL.

The environment is an analytically transparent policy-level abstraction of Diagnostic
DeepSea.  A directed policy reaches gate h with probability p_forward**h.  A probe or
challenger action at that gate generates Bernoulli evidence against the associated
policy-changing alternative.  A stepwise-random explorer reaches and tests the same
gate with exponentially smaller probability.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

EPS = 1e-12


def bernoulli_kl(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)) with numerical clipping."""
    p = float(np.clip(p, EPS, 1.0 - EPS))
    q = float(np.clip(q, EPS, 1.0 - EPS))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def bernoulli_llr(y: int, p_true: float, p_alt: float) -> float:
    """log p_true(y) / p_alt(y)."""
    if y not in (0, 1):
        raise ValueError("Bernoulli outcome must be 0 or 1")
    if y == 1:
        return math.log(p_true / p_alt)
    return math.log((1.0 - p_true) / (1.0 - p_alt))


@dataclass(frozen=True)
class DiagnosticConfig:
    horizon: int = 12
    gates: Tuple[int, ...] = (2, 5, 9)
    p_forward: float = 0.95
    probe_p_true: float = 0.75
    probe_p_alt: float = 0.25
    challenge_p_true: float = 0.40
    challenge_p_alt: float = 0.90
    probe_cost: float = 0.030
    challenge_local_gap: float = 0.150
    tv_gap: float = 0.040
    random_wander_gap: float = 0.180
    evidence_target: float = 5.0
    block_length: int = 2000
    n_seeds: int = 250

    def reach(self, gate: int) -> float:
        if gate < 0 or gate >= self.horizon:
            raise ValueError(f"gate {gate} is outside horizon {self.horizon}")
        return self.p_forward**gate

    @property
    def probe_local_kl(self) -> float:
        return bernoulli_kl(self.probe_p_true, self.probe_p_alt)

    @property
    def challenge_local_kl(self) -> float:
        return bernoulli_kl(self.challenge_p_true, self.challenge_p_alt)


@dataclass(frozen=True)
class PolicyStat:
    name: str
    gate: int | None
    source: str
    regret: float
    expected_info: float
    event_probability: float
    p_true: float | None
    p_alt: float | None


def directed_policy_stats(cfg: DiagnosticConfig, gate: int) -> Dict[str, PolicyStat]:
    reach = cfg.reach(gate)
    return {
        "probe": PolicyStat(
            name=f"probe_{gate}",
            gate=gate,
            source="probe",
            regret=reach * cfg.probe_cost,
            expected_info=reach * cfg.probe_local_kl,
            event_probability=reach,
            p_true=cfg.probe_p_true,
            p_alt=cfg.probe_p_alt,
        ),
        "challenge": PolicyStat(
            name=f"challenge_{gate}",
            gate=gate,
            source="challenge",
            regret=reach * cfg.challenge_local_gap,
            expected_info=reach * cfg.challenge_local_kl,
            event_probability=reach,
            p_true=cfg.challenge_p_true,
            p_alt=cfg.challenge_p_alt,
        ),
    }


def solve_design(regrets: np.ndarray, info: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Solve min regrets^T x subject to info @ x >= 1, x >= 0."""
    regrets = np.asarray(regrets, dtype=float)
    info = np.asarray(info, dtype=float)
    if info.ndim == 1:
        info = info[None, :]
    if regrets.ndim != 1 or info.shape[1] != regrets.size:
        raise ValueError("incompatible design dimensions")
    result = linprog(
        c=regrets,
        A_ub=-info,
        b_ub=-np.ones(info.shape[0]),
        bounds=[(0.0, None)] * regrets.size,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"design LP failed: {result.message}")
    # HiGHS stores multipliers for -info x <= -1; negate to get evidence prices.
    dual = -np.asarray(result.ineqlin.marginals, dtype=float)
    return np.asarray(result.x), float(result.fun), dual


def run_crossover(cfg: DiagnosticConfig, results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    gate = cfg.gates[1]
    reach = cfg.reach(gate)
    challenge_regret = reach * cfg.challenge_local_gap
    challenge_info = reach * cfg.challenge_local_kl
    costs = np.geomspace(0.002, 0.35, 100)
    rows: List[Dict[str, float | str]] = []
    for cost in costs:
        probe_regret = reach * cost
        probe_info = reach * cfg.probe_local_kl
        x, objective, dual = solve_design(
            np.array([probe_regret, challenge_regret]),
            np.array([probe_info, challenge_info]),
        )
        chosen = "probe" if x[0] > x[1] + 1e-8 else "challenge" if x[1] > x[0] + 1e-8 else "tie"
        rows.append(
            {
                "gate": gate,
                "probe_cost": cost,
                "probe_regret_per_nat": probe_regret / probe_info,
                "challenge_regret_per_nat": challenge_regret / challenge_info,
                "x_probe": x[0],
                "x_challenge": x[1],
                "gl_coefficient": objective,
                "dual_price": dual[0],
                "chosen_source": chosen,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "crossover.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["probe_cost"], df["probe_regret_per_nat"], label="probe regret / nat")
    ax.axhline(df["challenge_regret_per_nat"].iloc[0], linestyle="--", label="challenger regret / nat")
    ax.set_xscale("log")
    ax.set_xlabel("probe cost")
    ax.set_ylabel("regret per expected nat")
    ax.set_title("Exact one-alternative design switches at the cost-per-nat crossover")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "deepsea_crossover.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "deepsea_crossover.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df


def run_irrelevant_gate_scaling(cfg: DiagnosticConfig, results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    active_gate = cfg.gates[0]
    active = directed_policy_stats(cfg, active_gate)["probe"]
    rows: List[Dict[str, float]] = []
    for n_gates in [1, 2, 4, 8, 16, 32, 64]:
        ceac_coeff = active.regret / active.expected_info
        # Uniformly sample one probe policy per gate. Only the active gate carries
        # evidence against the active alternative; all gates incur comparable cost.
        uniform_avg_regret = active.regret
        uniform_active_info = active.expected_info / n_gates
        uniform_coeff = uniform_avg_regret / uniform_active_info
        rows.append(
            {
                "candidate_gate_count": n_gates,
                "active_alternative_rank": 1,
                "ceac_coefficient": ceac_coeff,
                "uniform_policy_bank_coefficient": uniform_coeff,
                "inflation": uniform_coeff / ceac_coeff,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "irrelevant_gate_scaling.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["candidate_gate_count"], df["ceac_coefficient"], marker="o", label="active-confounder design")
    ax.plot(
        df["candidate_gate_count"],
        df["uniform_policy_bank_coefficient"],
        marker="s",
        label="uniform policy-bank coverage",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("candidate diagnostic gates")
    ax.set_ylabel("leading regret coefficient")
    ax.set_title("Irrelevant policy directions inflate coverage, not the active design")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "deepsea_effective_dimension.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "deepsea_effective_dimension.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df


def sample_policy_episode(rng: np.random.Generator, stat: PolicyStat, informative_gate: int) -> Tuple[float, float]:
    """Return (regret, realized LLR against the active alternative)."""
    if stat.source == "exploit":
        return stat.regret, 0.0
    if stat.source == "tv":
        return stat.regret, 0.0
    if stat.source == "wander":
        # The event probability already includes random prefix selection.
        reached = rng.random() < stat.event_probability
        if not reached or stat.gate != informative_gate:
            return stat.regret, 0.0
    else:
        reached = rng.random() < stat.event_probability
        if not reached or stat.gate != informative_gate:
            return stat.regret, 0.0
    assert stat.p_true is not None and stat.p_alt is not None
    y = int(rng.random() < stat.p_true)
    return stat.regret, bernoulli_llr(y, stat.p_true, stat.p_alt)


def policy_bank(cfg: DiagnosticConfig) -> Dict[str, PolicyStat]:
    bank: Dict[str, PolicyStat] = {
        "exploit": PolicyStat("exploit", None, "exploit", 0.0, 0.0, 1.0, None, None),
        "tv": PolicyStat("tv", None, "tv", cfg.tv_gap, 0.0, 1.0, None, None),
    }
    for gate in cfg.gates:
        stats = directed_policy_stats(cfg, gate)
        bank[stats["probe"].name] = stats["probe"]
        bank[stats["challenge"].name] = stats["challenge"]
        random_event_prob = (0.5 ** (gate + 1)) * cfg.reach(gate)
        bank[f"wander_{gate}"] = PolicyStat(
            name=f"wander_{gate}",
            gate=gate,
            source="wander",
            regret=cfg.random_wander_gap,
            expected_info=random_event_prob * cfg.challenge_local_kl,
            event_probability=random_event_prob,
            p_true=cfg.challenge_p_true,
            p_alt=cfg.challenge_p_alt,
        )
    return bank


def choose_policy(
    method: str,
    cfg: DiagnosticConfig,
    bank: Mapping[str, PolicyStat],
    active_gate: int,
    evidence: float,
    rng: np.random.Generator,
) -> str:
    if evidence >= cfg.evidence_target:
        return "exploit"
    active = directed_policy_stats(cfg, active_gate)
    if method == "CEAC":
        ratios = {source: stat.regret / stat.expected_info for source, stat in active.items()}
        return active[min(ratios, key=ratios.get)].name
    if method == "vanilla OED":
        return active[max(active, key=lambda source: active[source].expected_info)].name
    if method == "uniform policy bank":
        names = [f"probe_{gate}" for gate in cfg.gates] + [f"challenge_{gate}" for gate in cfg.gates]
        return str(rng.choice(names))
    if method == "occupancy-entropy proxy":
        # Deliberately a mechanism proxy, not a faithful implementation of OIR.
        if rng.random() < 0.70:
            return "tv"
        names = [f"probe_{gate}" for gate in cfg.gates]
        return str(rng.choice(names))
    if method == "stepwise random":
        return f"wander_{active_gate}"
    if method == "exploit only":
        return "exploit"
    raise KeyError(method)


def run_shifting(
    cfg: DiagnosticConfig, results_dir: Path, figures_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = [
        "CEAC",
        "vanilla OED",
        "uniform policy bank",
        "occupancy-entropy proxy",
        "stepwise random",
        "exploit only",
    ]
    gate_sequence = list(cfg.gates)
    bank = policy_bank(cfg)
    episode_rows: List[Dict[str, float | int | str]] = []
    summary_rows: List[Dict[str, float | int | str]] = []
    choice_rows: List[Dict[str, float | int | str]] = []

    for seed in range(cfg.n_seeds):
        root = np.random.SeedSequence(seed)
        method_sequences = root.spawn(len(methods))
        for method, method_seq in zip(methods, method_sequences):
            rng = np.random.default_rng(method_seq)
            cumulative_regret = 0.0
            global_episode = 0
            for block_index, active_gate in enumerate(gate_sequence):
                evidence = 0.0
                block_regret = 0.0
                completion_episode: int | None = None
                policy_counts: Dict[str, int] = {name: 0 for name in bank}
                for local_episode in range(1, cfg.block_length + 1):
                    global_episode += 1
                    choice = choose_policy(method, cfg, bank, active_gate, evidence, rng)
                    policy_counts[choice] += 1
                    regret, llr = sample_policy_episode(rng, bank[choice], active_gate)
                    evidence += llr
                    cumulative_regret += regret
                    block_regret += regret
                    if completion_episode is None and evidence >= cfg.evidence_target:
                        completion_episode = local_episode
                    if local_episode == 1 or local_episode % 20 == 0 or local_episode == cfg.block_length:
                        episode_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "block": block_index,
                                "active_gate": active_gate,
                                "local_episode": local_episode,
                                "global_episode": global_episode,
                                "evidence": evidence,
                                "evidence_margin": evidence - cfg.evidence_target,
                                "block_regret": block_regret,
                                "cumulative_regret": cumulative_regret,
                            }
                        )
                completed = completion_episode is not None
                summary_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "block": block_index,
                        "active_gate": active_gate,
                        "completed": int(completed),
                        "episodes_to_target": completion_episode if completed else cfg.block_length + 1,
                        "block_regret": block_regret,
                        "terminal_evidence": evidence,
                        "terminal_shortfall": max(cfg.evidence_target - evidence, 0.0),
                    }
                )
                for policy_name, count in policy_counts.items():
                    if count:
                        choice_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "block": block_index,
                                "active_gate": active_gate,
                                "policy": policy_name,
                                "count": count,
                            }
                        )

    episodes = pd.DataFrame(episode_rows)
    summaries = pd.DataFrame(summary_rows)
    choices = pd.DataFrame(choice_rows)
    episodes.to_csv(results_dir / "shifting_episode_trajectories.csv", index=False)
    summaries.to_csv(results_dir / "shifting_block_summary.csv", index=False)
    choices.to_csv(results_dir / "shifting_policy_counts.csv", index=False)

    # Regret-to-certification: include only methods that actually certify in the majority
    # of runs; plot the completion rate separately so zero-regret nonlearners are visible.
    agg = (
        summaries.groupby(["method", "block", "active_gate"], as_index=False)
        .agg(
            completion_rate=("completed", "mean"),
            median_episodes=("episodes_to_target", "median"),
            mean_regret=("block_regret", "mean"),
            se_regret=("block_regret", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
            mean_shortfall=("terminal_shortfall", "mean"),
        )
    )
    agg.to_csv(results_dir / "shifting_aggregate.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(gate_sequence), dtype=float)
    width = 0.12
    for j, method in enumerate(methods):
        subset = agg[agg["method"] == method].sort_values("block")
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, subset["completion_rate"], width=width, label=method)
    ax.set_xticks(x, [f"gate {gate}" for gate in gate_sequence])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("fraction reaching evidence target")
    ax.set_title("Shifting DeepSea: policy-level evidence completion by active gate")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "shifting_deepsea_completion.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "shifting_deepsea_completion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for method in methods:
        subset = agg[agg["method"] == method].sort_values("block")
        # The bar includes all exploration regret; completion rate must be read jointly.
        ax.plot(subset["active_gate"], subset["mean_regret"], marker="o", label=method)
    ax.set_xlabel("active diagnostic gate depth")
    ax.set_ylabel("mean exploration regret per block")
    ax.set_title("Regret is meaningful only together with the evidence certificate")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "shifting_deepsea_regret.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "shifting_deepsea_regret.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ceac_choices = choices[choices["method"] == "CEAC"].copy()
    ceac_heat = ceac_choices.groupby(["block", "policy"], as_index=False)["count"].mean()
    pivot = ceac_heat.pivot(index="policy", columns="block", values="count").fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"gate {gate_sequence[b]}" for b in pivot.columns])
    ax.set_title("CEAC redirects its diagnostic policy after each shift")
    ax.set_xlabel("active block")
    ax.set_ylabel("executed policy")
    fig.colorbar(image, ax=ax, label="mean episode count")
    fig.tight_layout()
    fig.savefig(figures_dir / "shifting_deepsea_policy_heatmap.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "shifting_deepsea_policy_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return episodes, summaries, agg


def logistic(theta: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(theta)))


def run_fisher_calibration(results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    p0 = 0.75
    theta0 = math.log(p0 / (1.0 - p0))
    deltas = np.concatenate([np.linspace(0.005, 0.15, 25), np.linspace(0.18, 2.0, 40)])
    fisher = p0 * (1.0 - p0)
    rows = []
    for delta in deltas:
        p1 = float(logistic(theta0 + delta))
        exact = bernoulli_kl(p0, p1)
        quad = 0.5 * fisher * delta**2
        rows.append(
            {
                "delta_natural_parameter": delta,
                "p_alt": p1,
                "exact_kl": exact,
                "fisher_quadratic": quad,
                "relative_error": abs(quad - exact) / max(exact, EPS),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "fisher_calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["delta_natural_parameter"], df["exact_kl"], label="exact Bernoulli KL")
    ax.plot(df["delta_natural_parameter"], df["fisher_quadratic"], linestyle="--", label="local Fisher quadratic")
    ax.set_xlabel("natural-parameter displacement")
    ax.set_ylabel("information")
    ax.set_title("Fisher evidence is accurate locally and must be trust-region controlled")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fisher_calibration.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fisher_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df


def write_report(
    cfg: DiagnosticConfig,
    crossover: pd.DataFrame,
    scaling: pd.DataFrame,
    shifting: pd.DataFrame,
    fisher: pd.DataFrame,
    output: Path,
) -> None:
    challenge_ratio = float(crossover["challenge_regret_per_nat"].iloc[0])
    probe_kl = cfg.probe_local_kl
    predicted_cost = challenge_ratio * probe_kl
    transition_rows = crossover[crossover["chosen_source"].ne(crossover["chosen_source"].shift())]
    empirical_transition = float(transition_rows["probe_cost"].iloc[-1]) if len(transition_rows) > 1 else float("nan")

    agg = shifting.copy()
    table = agg[["method", "active_gate", "completion_rate", "median_episodes", "mean_regret", "mean_shortfall"]]
    local = fisher[fisher["delta_natural_parameter"] <= 0.15]
    broad = fisher[fisher["delta_natural_parameter"] >= 1.0]

    text = f"""# Initial shifting Diagnostic DeepSea diagnostics

These experiments are exact mechanism checks, not a neural actor--critic benchmark.  The
policy-level abstraction preserves the two quantities used by the Graves--Lai design:
expected regret of a complete trajectory and trajectory KL against a policy-changing
alternative.  Directed policies must reach a gate before obtaining evidence; stepwise
random exploration has exponentially smaller reach-and-test probability.

## Configuration

```json
{json.dumps(asdict(cfg), indent=2)}
```

## 1. One-alternative crossover

The predicted switch occurs at probe cost approximately `{predicted_cost:.6f}`, where
probe and challenger have equal regret per expected nat.  The finite LP switched near
`{empirical_transition:.6f}` on the logarithmic sweep.  Away from the numerical tie, the
LP placed all mass on the lower-cost evidence source, exactly as the analytic
one-row Graves--Lai program predicts.

## 2. Irrelevant-gate scaling and effective problem dimension

With one active policy-changing alternative, the active information rank remains one.
The CEAC coefficient stayed at `{scaling['ceac_coefficient'].iloc[0]:.6f}` while the
uniform policy-bank proxy inflated by a factor of `{scaling['inflation'].iloc[-1]:.1f}`
when the candidate gate count increased to `{int(scaling['candidate_gate_count'].iloc[-1])}`.
This is a mechanism diagnostic for the proposed active-rank/effective-dimension claim;
it is not by itself a stochastic general-function-approximation regret theorem.

## 3. Shifting active gate

The table reports evidence completion, episodes to the target, exploration regret, and
remaining evidence shortfall.  Zero regret is not a successful outcome when the evidence
constraint remains violated; this is why `exploit only` is retained as a falsifying control.

{table.to_markdown(index=False, floatfmt='.4f')}

The occupancy-entropy row is intentionally a simple noisy-TV/coverage proxy, not a
faithful implementation of the OIR actor--critic.  It checks the mechanism that high
occupancy novelty can be orthogonal to the active likelihood-ratio constraint.

## 4. Fisher calibration

For natural-parameter displacements at most 0.15, the median relative error of the
quadratic Fisher approximation was `{local['relative_error'].median():.4f}` and the
maximum was `{local['relative_error'].max():.4f}`.  For displacements at least 1.0, the
median error rose to `{broad['relative_error'].median():.4f}`.  This supports using the
Fisher row only inside an explicit trust region and retaining exact-KL calibration as a
required diagnostic.

## Claim boundary

The diagnostics substantiate: (i) exact LP allocation and dual logic on a small model,
(ii) nuisance rejection and active-gate adaptation, and (iii) local Fisher calibration.
They do not establish the online likelihood-set theorem, actor-pricing correctness with
a neural function approximator, or replacement of global Eluder-type complexity in a
full stochastic regret bound.  Those remain theorem interfaces and future experiments.
"""
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    root = args.output_root.resolve()
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = DiagnosticConfig(n_seeds=args.seeds or DiagnosticConfig.n_seeds)
    crossover = run_crossover(cfg, results_dir, figures_dir)
    scaling = run_irrelevant_gate_scaling(cfg, results_dir, figures_dir)
    _, _, shifting = run_shifting(cfg, results_dir, figures_dir)
    fisher = run_fisher_calibration(results_dir, figures_dir)
    write_report(cfg, crossover, scaling, shifting, fisher, root / "EXPERIMENT_REPORT.md")
    (results_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(root / "EXPERIMENT_REPORT.md")


if __name__ == "__main__":
    main()
