"""Unit tests for path-dependent signalling rewiring and canalized K562 GI."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import numpy as np

from ceac_pathdep import (
    CanalizedGIModel,
    MultiStepActionIDS,
    PathDepGreedyCEAC,
    SignallingRewiringConfig,
    SignallingRewiringModel,
    build_canalized_gi_config,
    relabel_canalized_gi,
    relabel_signalling,
)
from ceac_pathdep.study import run_pathdep_cell


class PathDepBenchmarkTests(unittest.TestCase):
    def test_signalling_rewiring_invariants_and_ids_separation(self):
        m0 = SignallingRewiringModel(relabel_signalling(SignallingRewiringConfig(complexity=0), 0))
        m2 = SignallingRewiringModel(relabel_signalling(SignallingRewiringConfig(complexity=2), 0))
        self.assertAlmostEqual(m0.optimal_returns[m0.true_index(0)], 0.50, places=9)
        self.assertAlmostEqual(m0.optimal_returns[m0.true_index(1)], 0.92, places=9)
        self.assertAlmostEqual(m2.optimal_returns[m2.true_index(1)], 0.92, places=9)
        # Policy-tree classes blow up by 2^k while on-path classes remain (1 + R + k)
        self.assertEqual(len({p.identity for p in m2.optimal_policies}), m2.num_models)
        self.assertEqual(len(m2.decision_classes()), 1 + m2.cfg.total_pathways)

        rng = np.random.default_rng(7)
        tree_ids = MultiStepActionIDS(m2, rng, m2.prior("reference"), target="policy_tree")
        onpath_ids = MultiStepActionIDS(m2, rng, m2.prior("reference"), target="onpath_class")
        d_tree = tree_ids.distribution()
        d_onpath = onpath_ids.distribution()
        self.assertAlmostEqual(float(d_tree.sum()), 1.0, places=9)
        self.assertAlmostEqual(float(d_onpath.sum()), 1.0, places=9)

        ceac_r = PathDepGreedyCEAC(m2, rng)
        self.assertEqual(len(ceac_r.bank.policies), 1)
        scores = ceac_r.disagreement_scores(tuple(range(m2.num_models)))
        # Root actions for modifier assays (3+k .. 3+2k-1) must score strictly 0 causal disagreement
        k = m2.cfg.complexity
        for idx, pol in enumerate(ceac_r.candidate_policies):
            a0 = int(np.argmax(pol.probabilities[0]))
            if 3 + k <= a0 < 3 + 2 * k:
                self.assertEqual(float(scores[idx]), 0.0)

    def test_canalized_gi_k562_panel_and_smoke_cell(self):
        cfg2 = relabel_canalized_gi(build_canalized_gi_config(2), 1)
        self.assertEqual(cfg2.hub_gene, "PPP2R1A")
        self.assertIn("FDPS", cfg2.modifier_genes[: cfg2.primary_count])
        model = CanalizedGIModel(cfg2)
        self.assertEqual(len(model.decision_classes()), 1 + cfg2.total_modifiers)
        self.assertGreater(model.optimal_returns[model.true_index(1)], 0.95)

        with tempfile.TemporaryDirectory() as tmp:
            res = run_pathdep_cell("canalized_gi", "ceac_ranked", seed=0, task=1, dose=1, episodes=256, output=Path(tmp) / "cell")
            self.assertLess(res["late_gap"], 0.10)
            self.assertLessEqual(res["learner_diagnostics"]["bank_policies"], 6)
            for prim in ("idac", "ucb_q", "pomd"):
                r_p = run_pathdep_cell("signalling_rewiring", prim, seed=0, task=1, dose=1, episodes=32, output=Path(tmp) / f"p_{prim}")
                self.assertEqual(r_p["settings"]["episodes"], 32)

    def test_heldout_critic_bellman_audit_parity(self):
        from ceac_pathdep.critic_audit import run_critic_audit_cell

        c_rank = run_critic_audit_cell(("signalling_rewiring", "ceac_ranked", 0, 2, 256))
        c_ids = run_critic_audit_cell(("signalling_rewiring", "ids_action_tree", 0, 2, 256))
        self.assertGreaterEqual(len(c_rank["checkpoints"]), 4)
        self.assertLess(c_rank["checkpoints"][-1]["evidence_critic_error"], 0.15)
        self.assertGreater(c_ids["checkpoints"][-1]["evidence_critic_error"], c_rank["checkpoints"][-1]["evidence_critic_error"])


if __name__ == "__main__":
    unittest.main()
