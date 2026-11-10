"""Public, mode-independent benchmark specifications; no learner tuning."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    name: str
    nuisance_bins: int
    route_signal: float

    def overrides(self, kind):
        if self.name == "legacy":
            return {}
        if kind == "deepsea":
            return dict(terminal_reward=.5, challenge_scale=.5, probe_cost=.02,
                        challenge_prob_mode0=.5-self.route_signal,
                        challenge_prob_mode1=.5+self.route_signal,
                        tv_bins=self.nuisance_bins, distributed_nuisance=True)
        if kind == "frozenlake":
            # Outer route takes two extra steps. Offset its success mean by
            # .002 so its return gap is exactly +/- route_signal.
            return dict(safe_reward=.3, inner_reward=1., outer_reward=1.,
                        inner_success_normal=.5, inner_success_shifted=.5,
                        outer_success_normal=.502-self.route_signal,
                        outer_success_shifted=.502+self.route_signal,
                        deep_fee=.02, reed_bins=self.nuisance_bins)
        raise ValueError(kind)


BENCHMARKS = {
    "legacy": Benchmark("legacy", 0, 0.),
    "strong_signal_m4": Benchmark("strong_signal_m4", 4, .25),
    "weak_signal_m4": Benchmark("weak_signal_m4", 4, .05),
    "weak_signal_m16": Benchmark("weak_signal_m16", 16, .05),
    "weak_signal_m64": Benchmark("weak_signal_m64", 64, .05),
}
REFINED = tuple(name for name in BENCHMARKS if name != "legacy")


def specification(name, kind):
    try:
        benchmark = BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError("Unknown benchmark: " + str(name)) from exc
    return {"name": name, "environment": kind,
            "configuration_overrides": benchmark.overrides(kind),
            "shift_protocol": "paired_stationary_modes",
            "nuisance_mechanism": "mode_independent_observation_alphabet",
            "seed_bank_rule": "constant_size_across_nuisance_alphabets"}


def cell_folder(root, benchmark, kind, context, mode, method, seed):
    prefix = root if benchmark == "legacy" else root / benchmark
    return prefix / kind / f"context{context}_mode{mode}" / f"{method}_seed{seed}"


def study_cells(root, config):
    for benchmark in config.get("benchmarks", ["legacy"]):
        for kind in config["environments"]:
            for context in config["contexts"]:
                for mode in config["modes"]:
                    for seed in config["seeds"]:
                        for method in config["methods"]:
                            yield (benchmark, kind, context, mode, seed, method,
                                   cell_folder(root, benchmark, kind, context, mode, method, seed))
