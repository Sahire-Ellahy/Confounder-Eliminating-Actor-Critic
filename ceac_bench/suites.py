"""Top-level composition wiring Layer 1 ProblemBuilders, Layer 2 Runners, Layer 3 Observers, and Layer 4 Renderers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ceac_bench.core.contracts import ConditionKey
from ceac_bench.core.primitives import FrozenFunctionClass, ROOT
from ceac_bench.metrics import (
    CHECKPOINTS_32K,
    DomainEventObserver,
    HeldOutBellmanObserver,
    LearnerDiagnosticsObserver,
    PerformanceTraceObserver,
)
from ceac_bench.problems import (
    CanalizedGIProblemBuilder,
    EpistasisProblemBuilder,
    GridPathDepProblemBuilder,
    K562GIScreenProblemBuilder,
    LockProblemBuilder,
    ProblemBuilder,
    SignallingRewiringProblemBuilder,
)
from ceac_bench.reporting import (
    CriticAuditMarkdownTable,
    DoseTransitionMarkdownTable,
    PairedAdvantageMarkdownTable,
    SingleEnvCriticBellmanPlot,
    SingleEnvDoseScalingPlot,
    SingleEnvTimeTracePlot,
)
from ceac_bench.runner import (
    CalibrationGate,
    CellRunner,
    LearnerFactory,
    SpawnStudyExecutor,
    default_method_factories,
)
from ceac_bench.store import BundleStore, StudyBundle


@dataclass(frozen=True)
class SuiteSpec(FrozenFunctionClass):
    """Declarative end-to-end experimental suite definition."""

    suite_id: str
    title: str
    problem_builders: tuple[ProblemBuilder, ...]
    learner_factories: tuple[LearnerFactory, ...]
    doses: tuple[int | float, ...]
    tasks: tuple[int, ...] = (1,)
    seeds: tuple[int, ...] = (0, 1, 2)
    calibration_seeds: tuple[int, ...] = (1001, 1002)
    episodes: int = 32768
    runner: CellRunner = field(
        default_factory=lambda: CellRunner(
            observers=(
                PerformanceTraceObserver(window_size=256, log_every=512),
                DomainEventObserver(log_every=512),
                LearnerDiagnosticsObserver(),
                HeldOutBellmanObserver(checkpoints=CHECKPOINTS_32K, n_onpolicy=512, n_probe=256),
            ),
            compact_every=128,
        )
    )
    calibration_gate: CalibrationGate = field(default_factory=lambda: CalibrationGate(easy_gap_tolerance=0.05))

    def build_jobs(
        self,
        *,
        seeds: tuple[int, ...] | None = None,
        doses: tuple[int | float, ...] | None = None,
        tasks: tuple[int, ...] | None = None,
        episodes: int | None = None,
    ) -> list[tuple[ProblemBuilder, LearnerFactory, ConditionKey, int]]:
        use_seeds = seeds if seeds is not None else self.seeds
        use_doses = doses if doses is not None else self.doses
        use_tasks = tasks if tasks is not None else self.tasks
        use_eps = int(episodes if episodes is not None else self.episodes)
        jobs: list[tuple[ProblemBuilder, LearnerFactory, ConditionKey, int]] = []
        for pb in self.problem_builders:
            meta = pb.metadata
            for d in use_doses:
                for t in use_tasks:
                    for lf in self.learner_factories:
                        for s in use_seeds:
                            cond = ConditionKey(
                                suite_id=self.suite_id,
                                family_id=meta.family_id,
                                variant_id=meta.variant_id,
                                dose=d,
                                task=t,
                                seed=s,
                            )
                            jobs.append((pb, lf, cond, use_eps))
        return jobs

    def execute_and_save(
        self,
        out_dir: Path | str,
        *,
        cache_dir: Path | str | None = None,
        workers: int = 4,
        episodes: int | None = None,
        seeds: tuple[int, ...] | None = None,
        doses: tuple[int | float, ...] | None = None,
        render_figures: bool = True,
    ) -> StudyBundle:
        executor = SpawnStudyExecutor(workers=workers, maxtasksperchild=1)
        jobs = self.build_jobs(seeds=seeds, doses=doses, episodes=episodes)
        cells = executor.execute_jobs(
            self.runner,
            jobs,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            label=self.suite_id,
        )
        bundle = StudyBundle(
            suite_id=self.suite_id,
            families=tuple(pb.metadata for pb in self.problem_builders),
            cells=cells,
        )
        out = Path(out_dir)
        BundleStore.save_bundle(bundle, out, extra_metadata={"suite_spec": self.to_spec()})
        if render_figures:
            self.render_suite_artifacts(bundle, out)
        return BundleStore.load_bundle(out)

    def render_suite_artifacts(self, bundle: StudyBundle, out_dir: Path | str) -> dict[str, Path]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        max_dose_by_var = {v: bundle.doses(v)[-1] for v in bundle.variants() if bundle.doses(v)}

        md_parts = [f"# {self.title} (`{self.suite_id}`)\n"]
        for var in bundle.variants():
            if var not in max_dose_by_var:
                continue
            target_d = max_dose_by_var[var]
            plotters: list[tuple[str, Any]] = [
                (f"{var}_dose_scaling", SingleEnvDoseScalingPlot(variant_id=var)),
                (f"{var}_time_traces", SingleEnvTimeTracePlot(variant_id=var, dose=target_d)),
                (f"{var}_critic_bellman", SingleEnvCriticBellmanPlot(variant_id=var, dose=target_d)),
            ]
            for stem, plotter in plotters:
                fig = plotter(bundle)
                for ext in ("png", "pdf"):
                    p = out / f"{stem}.{ext}"
                    fig.savefig(p, dpi=180, bbox_inches="tight")
                    written[p.name] = p
                plt.close(fig)
                md_parts.append(f"![{stem}]({stem}.png)\n")

        md_parts.append("## Endpoint & Dose Transition Summary\n")
        md_parts.append(DoseTransitionMarkdownTable()(bundle))
        md_parts.append("\n## Held-Out Reward & Evidence Critic Bellman Audit\n")
        md_parts.append(CriticAuditMarkdownTable()(bundle))
        if any(m == "ceac_ranked" for m in bundle.methods()):
            md_parts.append("\n## Seed-Paired Regret Savings (`ceac_ranked`)\n")
            md_parts.append(PairedAdvantageMarkdownTable("ceac_ranked")(bundle))

        report_md = out / "report_v2.md"
        report_md.write_text("\n".join(md_parts))
        written[report_md.name] = report_md
        return written


SUITES: dict[str, SuiteSpec] = {
    "grid_pathdep": SuiteSpec(
        suite_id="grid_pathdep",
        title="Multi-Step Path-Dependent Undertow DeepSea & False Beacon Shore",
        problem_builders=(
            GridPathDepProblemBuilder("undertow_pathdep"),
            GridPathDepProblemBuilder("false_beacon_pathdep"),
        ),
        learner_factories=default_method_factories(pathdep=True, include_ranked=True),
        doses=(0, 1, 2, 3),
        episodes=32768,
        runner=CellRunner(
            observers=(
                PerformanceTraceObserver(window_size=256, log_every=512),
                DomainEventObserver(log_every=512),
                LearnerDiagnosticsObserver(),
                HeldOutBellmanObserver(checkpoints=CHECKPOINTS_32K, n_onpolicy=512, n_probe=256, compat_suite_key="grid_pathdep"),
            ),
            compact_every=128,
        ),
    ),
    "bio_pathdep": SuiteSpec(
        suite_id="bio_pathdep",
        title="Multi-Step Path-Dependent Signalling Rewiring & Canalized K562 Genetic Risk",
        problem_builders=(
            SignallingRewiringProblemBuilder("signalling_rewiring"),
            CanalizedGIProblemBuilder("canalized_gi"),
        ),
        learner_factories=default_method_factories(pathdep=True, include_ranked=True),
        doses=(0, 1, 2, 3),
        episodes=32768,
        runner=CellRunner(
            observers=(
                PerformanceTraceObserver(window_size=256, log_every=256),
                DomainEventObserver(log_every=256),
                LearnerDiagnosticsObserver(),
                HeldOutBellmanObserver(checkpoints=CHECKPOINTS_32K, n_onpolicy=512, n_probe=256, compat_suite_key="pathdep"),
            ),
            compact_every=128,
        ),
    ),
    "k562_gi_ranked": SuiteSpec(
        suite_id="k562_gi_ranked",
        title="Horlbeck 2018 K562 Pairwise Screen with Greedy Price-Ranked CEAC",
        problem_builders=(
            K562GIScreenProblemBuilder("gi_arrayed"),
            K562GIScreenProblemBuilder("gi_multiplexed"),
        ),
        learner_factories=default_method_factories(pathdep=False, include_ranked=True),
        doses=(0, 1, 2, 3),
        episodes=65536,
    ),
    "locks": SuiteSpec(
        suite_id="locks",
        title="Sequential & Bidirectional Combination Locks",
        problem_builders=(
            LockProblemBuilder("combination_lock"),
            LockProblemBuilder("bidirectional_lock"),
        ),
        learner_factories=default_method_factories(pathdep=False, include_ranked=False, include_action_ids=False),
        doses=(1, 2, 3, 4),
        episodes=65536,
    ),
    "epistasis": SuiteSpec(
        suite_id="epistasis",
        title="Epistatic Signalling Discovery under Off-Pathway Pleiotropy",
        problem_builders=(EpistasisProblemBuilder("epistatic_signalling"),),
        learner_factories=default_method_factories(pathdep=False, include_ranked=False),
        doses=(0, 1, 2, 3),
        episodes=32768,
    ),
}


def get_suite(suite_id: str) -> SuiteSpec:
    if suite_id not in SUITES:
        raise KeyError(f"Unknown suite {suite_id!r}; available: {sorted(SUITES)}")
    return SUITES[suite_id]


def verify_layer_import_boundaries(package_root: Path | None = None) -> list[str]:
    """AST import-direction audit enforcing strict unidirectional dependencies and zero matplotlib leaks."""
    import ast

    pkg = package_root or (ROOT / "ceac_bench")
    allowed_ceac_bench_deps: dict[str, set[str]] = {
        "core": set(),
        "problems": {"core"},
        "store": {"core"},
        "runner": {"core", "problems", "store"},
        "metrics": {"core", "problems", "runner"},
        "reporting": {"core", "store"},
    }
    forbidden_ui_pkgs = {"matplotlib", "IPython", "nbformat", "nbclient", "ipykernel"}
    violations: list[str] = []

    for sublayer, allowed in allowed_ceac_bench_deps.items():
        subdir = pkg / sublayer
        if not subdir.is_dir():
            continue
        for py_file in sorted(subdir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.level == 0:
                        imported = [node.module]
                for mod in imported:
                    top = mod.split(".")[0]
                    if sublayer != "reporting" and top in forbidden_ui_pkgs:
                        violations.append(f"{py_file.relative_to(pkg.parent)} imports UI library {mod!r} inside Layer `{sublayer}`")
                    if top == "ceac_bench":
                        parts = mod.split(".")
                        if len(parts) >= 2:
                            target_layer = parts[1]
                            if target_layer != sublayer and target_layer not in allowed:
                                violations.append(
                                    f"{py_file.relative_to(pkg.parent)} (`{sublayer}`) imports forbidden higher/peer layer `ceac_bench.{target_layer}`"
                                )
    return violations
