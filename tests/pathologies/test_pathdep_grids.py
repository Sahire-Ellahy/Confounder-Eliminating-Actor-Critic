"""Unit tests for Path-Dependent Undertow DeepSea & False Beacon Shore."""
from __future__ import annotations
import unittest
import numpy as np

from ceac_pathdep.learners import MultiStepActionIDS, PathDepGreedyCEAC
from ceac_pathdep_grids import (
    PathDepFalseBeaconConfig,
    PathDepFalseBeaconModel,
    PathDepUndertowConfig,
    PathDepUndertowModel,
    relabel_grid_pathdep,
)
from ceac_pathdep_grids.study import run_grid_pathdep_cell


class GridPathDepTests(unittest.TestCase):
    def test_undertow_and_false_beacon_pathdep_models_and_separation(self):
        for cfg_cls, model_cls, opt_r in [
            (PathDepUndertowConfig, PathDepUndertowModel, 0.93),
            (PathDepFalseBeaconConfig, PathDepFalseBeaconModel, 0.95),
        ]:
            m0 = model_cls(relabel_grid_pathdep(cfg_cls(complexity=0), 0))
            m2 = model_cls(relabel_grid_pathdep(cfg_cls(complexity=2), 0))
            self.assertAlmostEqual(m0.optimal_returns[m0.true_index(0)], 0.50, places=9)
            self.assertAlmostEqual(m2.optimal_returns[m2.true_index(1)], opt_r, places=9)
            self.assertEqual(len({p.identity for p in m2.optimal_policies}), m2.num_models)

            onpath = MultiStepActionIDS(m2, np.random.default_rng(3), m2.prior("reference"), target="onpath_class")
            dist = onpath.distribution()
            # Verify pure zero-mixing onto echo plumes / false-beacon glints (root actions 3..3+n_echo-1)
            chosen_idx = int(np.argmax(dist))
            self.assertAlmostEqual(float(dist[chosen_idx]), 1.0, places=6)
            a0 = int(np.argmax(onpath.bank.policies[chosen_idx].probabilities[0]))
            self.assertTrue(3 <= a0 < 3 + m2.cfg.echo_count)

            ceac_r = PathDepGreedyCEAC(m2, np.random.default_rng(3))
            scores = ceac_r.disagreement_scores(tuple(range(m2.num_models)))
            n_echo = m2.cfg.echo_count
            k = m2.cfg.complexity
            for idx, pol in enumerate(ceac_r.candidate_policies):
                ra = int(np.argmax(pol.probabilities[0]))
                if 3 + n_echo <= ra < 3 + n_echo + k:
                    self.assertEqual(float(scores[idx]), 0.0)

    def test_unified_cell_runner_returns_trace_and_critic_checkpoints(self):
        res = run_grid_pathdep_cell(("undertow_pathdep", "ceac_ranked", 0, 1, 1, 512))
        self.assertLess(res["late_gap"], 0.05)
        self.assertGreaterEqual(len(res["checkpoints"]), 3)
        self.assertLess(res["checkpoints"][-1]["evidence_critic_error"], 0.15)


if __name__ == "__main__":
    unittest.main()
