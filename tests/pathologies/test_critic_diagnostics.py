"""Unit tests for Held-Out Reward & Evidence Critic TD/Bellman Diagnostics across suites."""
from __future__ import annotations
import unittest
import numpy as np

from ceac_critic_diagnostics.evaluator import compile_outcome_tables, evaluate_heldout_critics_fast
from ceac_critic_diagnostics.run_pathologies_audit import FastActionIDS, FastBayesianIDS
from ceac_hardness.ids import BayesianIDS
from ceac_pathologies.environments import EpistasisConfig, LockConfig
from ceac_pathologies.ids_action import ActionInformationIDS
from ceac_pathologies.model import EpistasisModel, LockModel
from ceac_pathologies.run import build_problem, construct_learner


class CriticDiagnosticsTests(unittest.TestCase):
    def test_fast_bayesian_ids_bit_exact_with_reference(self):
        cfg = EpistasisConfig(pleiotropy=2)
        model = EpistasisModel(cfg)
        prior = model.prior("product")
        rng_ref = np.random.default_rng(42)
        rng_fast = np.random.default_rng(42)
        ref = BayesianIDS(model, rng_ref, prior)
        fast = FastBayesianIDS(model, rng_fast, prior)

        # Test both high-uncertainty and near-converged posteriors
        for scale in (0.0, 2.5, 45.0):
            lw = np.full(model.num_models, -scale)
            lw[model.num_models - 1] = 0.0
            ref.log_weights[:] = lw
            fast.log_weights[:] = lw
            ref._cache = None
            fast._cache = None
            d_ref = ref.distribution()
            d_fast = fast.distribution()
            np.testing.assert_allclose(d_fast, d_ref, atol=1e-15, rtol=0.0)

    def test_fast_action_ids_bit_exact_with_reference(self):
        cfg = EpistasisConfig(pleiotropy=2)
        model = EpistasisModel(cfg)
        prior = model.prior("reference")
        ref = ActionInformationIDS(model, np.random.default_rng(7), prior)
        fast = FastActionIDS(model, np.random.default_rng(7), prior)

        for scale in (0.0, 3.0, 50.0):
            lw = np.full(model.num_models, -scale)
            lw[model.num_models - 1] = 0.0
            ref.log_weights[:] = lw
            fast.log_weights[:] = lw
            ref._cache = None
            fast._cache = None
            np.testing.assert_allclose(fast.distribution(), ref.distribution(), atol=1e-15, rtol=0.0)

    def test_evaluate_heldout_critics_fast_across_learners(self):
        _, model, _, truth, _ = build_problem("locks", dict(kind="bidirectional_lock", depth=2), task=1, seed=0)
        compiled = compile_outcome_tables(model, truth=truth)
        for method in ("idac", "ucb_q", "pomd", "ids_product", "ceac"):
            rng = np.random.default_rng(123)
            learner = construct_learner("locks", model, method, rng, episodes=64, truth=truth)
            pol, _, prices = learner.select(0)
            diag = evaluate_heldout_critics_fast(
                suite_key="unit_test",
                kind="bidirectional_lock",
                method=method,
                learner=learner,
                model=model,
                compiled=compiled,
                truth=truth,
                policy=pol,
                prices=prices,
                checkpoint=16,
                seed=0,
                dose=2,
                n_onpolicy=64,
                n_probe=64,
            )
            for key in ("reward_bellman_error", "evidence_critic_error", "stage_epistemic_error"):
                self.assertIn(key, diag)
                self.assertTrue(np.isfinite(diag[key]))
                self.assertGreaterEqual(diag[key], 0.0)


if __name__ == "__main__":
    unittest.main()
