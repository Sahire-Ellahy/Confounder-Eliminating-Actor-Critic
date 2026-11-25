""" Comprehensive architectural, boundary, picklability, caching, and numerical-parity tests for `ceac_bench`."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
import numpy as np

from ceac_bench.cli import main as cli_main
from ceac_bench.core import ConditionKey, CriticSnapshot, DomainMetadata, FrozenFunctionClass, PersistentCallable, PolicyView, ROOT, SelectionStep
from ceac_bench.metrics import (
    CHECKPOINTS_32K,
    DomainEventObserver,
    HeldOutBellmanObserver,
    LearnerDiagnosticsObserver,
    PerformanceTraceObserver,
)
from ceac_bench.problems import (
    ALL_FAMILIES,
    BoundProblem,
    GridPathDepProblemBuilder,
    OracleEvaluator,
    PublicProblemView,
    SignallingRewiringProblemBuilder,
    get_problem_builder,
)
from ceac_bench.reporting import (
    CriticAuditMarkdownTable,
    DoseTransitionMarkdownTable,
    PairedAdvantageMarkdownTable,
    SingleEnvCriticBellmanPlot,
    SingleEnvDoseScalingPlot,
    SingleEnvTimeTracePlot,
    VisualTheme,
)
from ceac_bench.runner import (
    BayesianIDSFactory,
    CEACFactory,
    CalibrationGate,
    CellRunner,
    GreedyRankedCEACFactory,
    IDACFactory,
    OptimalActionIDSFactory,
    POMDFactory,
    SpawnStudyExecutor,
    UCBQFactory,
)
from ceac_bench.store import BundleStore, CellCacheStore, LegacyBundleAdapter, StudyBundle
from ceac_bench.suites import SUITES, verify_layer_import_boundaries


def _echo_callable_digest_in_worker(fn_obj: FrozenFunctionClass) -> tuple[str, str]:
    return fn_obj.type_id, fn_obj.digest()


class ArchitectureAndFactorizationTests(unittest.TestCase):
    def test_strict_four_layer_ast_import_boundaries(self) -> None:
        violations = verify_layer_import_boundaries()
        self.assertEqual(violations, [], f"Layer boundary violations detected: {violations}")

    def test_persistent_function_classes_picklable_across_spawn_and_content_addressed(self) -> None:
        callables: list[FrozenFunctionClass] = [
            GridPathDepProblemBuilder("undertow_pathdep"),
            SignallingRewiringProblemBuilder("signalling_rewiring"),
            IDACFactory(),
            UCBQFactory(),
            POMDFactory(),
            BayesianIDSFactory("ids_product", "reference"),
            OptimalActionIDSFactory("ids_action_onpath", "onpath_class"),
            CEACFactory("ceac", "reference"),
            GreedyRankedCEACFactory("ceac_ranked", "reference"),
            PerformanceTraceObserver(window_size=256, log_every=128),
            DomainEventObserver(log_every=128),
            HeldOutBellmanObserver(checkpoints=(16, 64, 256)),
            CalibrationGate(easy_gap_tolerance=0.05),
            VisualTheme(),
            SingleEnvDoseScalingPlot("undertow_pathdep"),
            SingleEnvTimeTracePlot("undertow_pathdep", dose=3),
            SingleEnvCriticBellmanPlot("undertow_pathdep", dose=3),
            DoseTransitionMarkdownTable(),
            PairedAdvantageMarkdownTable("ceac_ranked"),
            CriticAuditMarkdownTable(),
        ]
        for obj in callables:
            self.assertIsInstance(obj, PersistentCallable)

        # Mutating parameters must deterministically change digest()
        p1 = PerformanceTraceObserver(window_size=256, log_every=128)
        p2 = PerformanceTraceObserver(window_size=128, log_every=128)
        self.assertNotEqual(p1.digest(), p2.digest())

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=2, maxtasksperchild=4) as pool:
            worker_out = pool.map(_echo_callable_digest_in_worker, callables)
        for obj, (w_tid, w_dig) in zip(callables, worker_out):
            self.assertEqual(obj.type_id, w_tid)
            self.assertEqual(obj.digest(), w_dig)

    def test_public_problem_view_seals_ground_truth(self) -> None:
        for fam in ALL_FAMILIES:
            for builder in fam.builders:
                cond = ConditionKey(
                    suite_id="audit",
                    family_id=fam.family_id,
                    variant_id=builder.metadata.variant_id,
                    dose=builder.metadata.easy_dose,
                    task=1,
                    seed=0,
                )
                bound = builder(cond)
                pub_fields = set(vars(bound.public_view).keys())
                self.assertNotIn("truth", pub_fields)
                self.assertNotIn("truth_index", pub_fields)
                self.assertNotIn("realized_nuisance", pub_fields)
                self.assertGreater(bound.evaluator.optimal_return(), 0.45)

    def test_bit_exact_parity_with_pathdep_reference_runner_and_critic_audit(self) -> None:
        from ceac_pathdep.study import run_pathdep_cell

        builder = SignallingRewiringProblemBuilder("signalling_rewiring")
        runner = CellRunner(
            observers=(
                PerformanceTraceObserver(window_size=256, log_every=64),
                DomainEventObserver(log_every=64),
                LearnerDiagnosticsObserver(),
                HeldOutBellmanObserver(checkpoints=(16, 64, 256), n_onpolicy=512, n_probe=256, compat_suite_key="pathdep"),
            ),
            compact_every=128,
        )
        cond = ConditionKey("bio_pathdep", "signalling", "signalling_rewiring", dose=2, task=1, seed=0)

        with tempfile.TemporaryDirectory() as tmp:
            for factory in (
                GreedyRankedCEACFactory("ceac_ranked", "reference"),
                OptimalActionIDSFactory("ids_action_tree", "policy_tree", "reference"),
                IDACFactory(),
            ):
                rec = runner.run_cell(builder, factory, cond, episodes=256)
                ref = run_pathdep_cell("signalling_rewiring", factory.method_id, seed=0, task=1, dose=2, episodes=256, output=Path(tmp) / factory.method_id)
                self.assertAlmostEqual(rec.late_gap, ref["late_gap"], places=12)
                self.assertAlmostEqual(rec.cumulative_regret, ref["cumulative_regret"], places=10)
                self.assertEqual(len(rec.critic_audit["evidence_critic_error"]), 3)

    def test_every_suite_builder_and_learner_executes_cleanly(self) -> None:
        for sid, spec in sorted(SUITES.items()):
            jobs = spec.build_jobs(seeds=(0,), doses=(spec.doses[0],), tasks=(1,), episodes=6)
            self.assertGreater(len(jobs), 0)
            for pb, lf, cond, eps in jobs:
                rec = spec.runner.run_cell(pb, lf, cond, eps)
                self.assertEqual(rec.episodes, 6)
                self.assertTrue(np.isfinite(rec.cumulative_regret))

    def test_primitive_learners_receive_empty_likelihoods_at_runner_boundary(self) -> None:
        from dataclasses import dataclass

        observed_ll_lengths: list[int] = []

        class _SpyPrimitiveLearner:
            def __init__(self, inner):
                self._inner = inner

            def select(self, t: int) -> SelectionStep:
                return self._inner.select(t)

            def observe(self, episode, selection: SelectionStep):
                for tr in episode.transitions:
                    observed_ll_lengths.append(len(tr.log_likelihood))
                return self._inner.observe(episode, selection)

            def compact_memory(self, t: int) -> None:
                self._inner.compact_memory(t)

            def critic_snapshot(self, selection: SelectionStep) -> CriticSnapshot:
                return self._inner.critic_snapshot(selection)

            def diagnostics(self):
                return self._inner.diagnostics()

        @dataclass(frozen=True)
        class _SpyIDACFactory(IDACFactory):
            def __call__(self, view, episodes: int, rng):
                return _SpyPrimitiveLearner(super().__call__(view, episodes, rng))

        builder = GridPathDepProblemBuilder("undertow_pathdep")
        cond = ConditionKey("audit", "undertow", "undertow_pathdep", dose=1, task=1, seed=0)
        runner = CellRunner(observers=(PerformanceTraceObserver(window_size=8, log_every=4),))
        runner.run_cell(builder, _SpyIDACFactory(), cond, episodes=8)
        self.assertGreater(len(observed_ll_lengths), 0)
        self.assertEqual(set(observed_ll_lengths), {0})

    def test_continuous_vector_environment_and_callable_neural_policy_critic(self) -> None:
        from dataclasses import dataclass

        class _ContinuousBoxEnv:
            def __init__(self) -> None:
                self._rng = np.random.default_rng(0)
                self._step = 0

            def reset(self, seed: int | None = None):
                if seed is not None:
                    self._rng = np.random.default_rng(seed)
                self._step = 0
                return np.zeros(4, dtype=float), {}

            def step(self, action):
                self._step += 1
                act_arr = np.asarray(action, dtype=float)
                rew = float(0.95 - 0.25 * np.sum((act_arr - np.array([0.5, -0.5])) ** 2))
                done = self._step >= 3
                nxt = self._rng.normal(0.0, 0.1, size=4)
                return nxt, rew, done, False, {"event": "continuous_probe"}

        @dataclass(frozen=True)
        class _ContinuousOracle(OracleEvaluator):
            def optimal_return(self) -> float:
                return 3.0 * 0.95

            def expected_return(self, policy: PolicyView) -> float:
                assert policy.act_fn is not None
                act = np.asarray(policy.act_fn(np.zeros(4), np.random.default_rng(0)), dtype=float)
                return 3.0 * float(0.95 - 0.25 * np.sum((act - np.array([0.5, -0.5])) ** 2))

        @dataclass(frozen=True)
        class _ContinuousProblemBuilder(FrozenFunctionClass):
            @property
            def metadata(self) -> DomainMetadata:
                return DomainMetadata(
                    family_id="dexterous",
                    variant_id="toy_bayonet_box",
                    title="Continuous Contact-Gated Toy Box",
                    short_title="Toy Bayonet Box",
                    horizon=3,
                    dose_symbol="k",
                    dose_label="Clutter k",
                    event_names=("continuous_probe",),
                )

            def __call__(self, condition: ConditionKey) -> BoundProblem:
                meta = self.metadata
                pub = PublicProblemView(metadata=meta, horizon=3, obs_shape=(4,), continuous_action_dim=2)
                return BoundProblem(condition=condition, metadata=meta, env=_ContinuousBoxEnv(), public_view=pub, evaluator=_ContinuousOracle())

        class _NeuralMockLearner:
            def __init__(self) -> None:
                self.mu = np.zeros(2, dtype=float)

            def select(self, episode_idx: int) -> SelectionStep:
                target_mu = self.mu.copy()
                pol = PolicyView(
                    label=f"neural-{episode_idx}",
                    act_fn=lambda _obs, _rng: target_mu,
                    explicit_id=f"mu-{np.round(target_mu, 6).tolist()}",
                )
                return SelectionStep(policy=pol, behavior="actor")

            def observe(self, episode, selection: SelectionStep):
                self.mu = 0.7 * self.mu + 0.3 * np.array([0.5, -0.5])
                return {}

            def compact_memory(self, episode_idx: int) -> None:
                pass

            def critic_snapshot(self, selection: SelectionStep) -> CriticSnapshot:
                return CriticSnapshot(kind="neural_ceac", reward_fn=lambda s, a: 0.9, evidence_fn=lambda s, a: 0.0)

            def diagnostics(self):
                return {"bank_policies": 2}

        @dataclass(frozen=True)
        class _NeuralMockFactory(FrozenFunctionClass):
            method_id: str = "ceac_ranked"
            requires_calibration: bool = True
            uses_candidate_likelihoods: bool = False

            def __call__(self, view: PublicProblemView, episodes: int, rng) -> _NeuralMockLearner:
                return _NeuralMockLearner()

        runner = CellRunner(observers=(PerformanceTraceObserver(window_size=4, log_every=4), DomainEventObserver(log_every=4)))
        cond = ConditionKey("dex_suite", "dexterous", "toy_bayonet_box", dose=0, task=1, seed=0)
        rec = runner.run_cell(_ContinuousProblemBuilder(), _NeuralMockFactory(), cond, episodes=12)
        self.assertLess(rec.late_gap, 0.02)
        self.assertEqual(rec.scalars["event_continuous_probe"], 12)

    def test_cell_cache_and_bundle_store_roundtrip_and_tamper_rejection(self) -> None:
        builder = get_problem_builder("undertow_pathdep")
        runner = CellRunner(
            observers=(
                PerformanceTraceObserver(window_size=64, log_every=32),
                HeldOutBellmanObserver(checkpoints=(16, 64), n_onpolicy=64, n_probe=64),
            )
        )
        cond = ConditionKey("unit_suite", "undertow", "undertow_pathdep", dose=1, task=1, seed=0)
        factory = GreedyRankedCEACFactory("ceac_ranked", "reference")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = CellCacheStore(root / "cache")
            rec1 = runner.run_cell(builder, factory, cond, episodes=64, cache=cache)
            rec2 = runner.run_cell(builder, factory, cond, episodes=64, cache=cache)
            self.assertEqual(rec1, rec2)

            bundle = StudyBundle(suite_id="unit_suite", families=(builder.metadata,), cells=(rec1,))
            b_dir = root / "bundle"
            BundleStore.save_bundle(bundle, b_dir)
            loaded = BundleStore.load_bundle(b_dir)
            self.assertEqual(len(loaded.cells), 1)
            self.assertAlmostEqual(loaded.cells[0].cumulative_regret, rec1.cumulative_regret, places=9)

            # Tampering with bundle_v2.json.gz must fail closed
            (b_dir / BundleStore.BUNDLE_FILENAME).write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                BundleStore.load_bundle(b_dir)

    def test_legacy_bundle_adapters_and_layer4_renderers_on_verified_exports(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gp_bundle = LegacyBundleAdapter.load_grid_pathdep(ROOT / "notebooks/evidence/pathdep_undertow_falsebeacon")
        bp_bundle = LegacyBundleAdapter.load_bio_pathdep(ROOT / "notebooks/evidence/pathologies_pathdep")
        self.assertEqual(len(gp_bundle.cells), 288)
        self.assertEqual(len(bp_bundle.cells), 288)

        fig1 = SingleEnvDoseScalingPlot("undertow_pathdep")(gp_bundle)
        self.assertEqual(len(fig1.axes), 2)
        plt.close(fig1)

        fig2 = SingleEnvTimeTracePlot("signalling_rewiring", dose=3)(bp_bundle)
        self.assertEqual(len(fig2.axes), 2)
        plt.close(fig2)

        fig3 = SingleEnvCriticBellmanPlot("canalized_gi", dose=3)(bp_bundle)
        self.assertEqual(len(fig3.axes), 2)
        plt.close(fig3)

        md_tbl = DoseTransitionMarkdownTable()(gp_bundle)
        self.assertIn("CEAC · greedy disagreement + price-ranked replay", md_tbl)
        paired_tbl = PairedAdvantageMarkdownTable("ceac_ranked")(bp_bundle)
        self.assertIn("True", paired_tbl)

        self.assertEqual(cli_main(["check-boundaries"]), 0)
        self.assertEqual(cli_main(["list-suites"]), 0)


if __name__ == "__main__":
    unittest.main()
