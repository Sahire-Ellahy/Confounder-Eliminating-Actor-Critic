"""Real-data GI screen benchmark and optimal-action IDS regression tests."""
from __future__ import annotations
import math
import unittest
from dataclasses import asdict
import numpy as np
from ceac_gi.data import load_k562_map, select_panel
from ceac_gi.environment import GIScreenModel, build_config, config_from_dict, public_blocks
from ceac_pathologies.agent import ConfounderEliminatingAC
from ceac_pathologies.environments import EpistasisConfig
from ceac_pathologies.ids_action import ActionInformationIDS
from ceac_pathologies.model import EpistasisModel
from ceac_hardness.ids import BayesianIDS


class GIScreenTests(unittest.TestCase):

    def test_map_is_verified_and_panel_has_one_rare_pair(self):
        names, matrix = load_k562_map()
        self.assertEqual(matrix.shape, (448, 448))
        panel = select_panel(10)
        values = np.array(panel["values"])
        self.assertEqual(int((values <= -5.0).sum()), 1)
        self.assertTrue((np.delete(values, panel["sl_pair_index"]) > -2.5).all())
        self.assertEqual(panel["genes"][:2], ("FDPS", "PPP2R1A"))

    def test_true_hypothesis_is_exactly_the_published_panel(self):
        for kind in ("gi_arrayed", "gi_multiplexed"):
            cfg = build_config(kind, 1)
            model = GIScreenModel(cfg)
            truth = model._models[model.true_index(1, 0)]
            np.testing.assert_allclose(truth.gi, cfg.values)
            null = model._models[model.true_index(0, 0)]
            self.assertEqual(null.gi[cfg.sl_pair_index], cfg.neutral_value)
            self.assertGreater(model.optimal_returns[model.true_index(1)], 0.95)
            self.assertAlmostEqual(model.optimal_returns[model.true_index(0)], 0.5)

    def test_profiling_singles_carry_zero_kl_on_undominated_rows(self):
        cfg = build_config("gi_arrayed", 2)
        model = GIScreenModel(cfg)
        for gene in cfg.nuisance_genes:
            kl = model.stage_kl(0, 1 + gene)
            self.assertGreater(kl.max(), 0.0)
            for m, j in model.alternatives:
                self.assertEqual(float(kl[m, j]), 0.0)
        ceac = ConfounderEliminatingAC(model, np.random.default_rng(0))
        slice_ = tuple(model.hypotheses.index((cfg.sl_pair_index, e)) for e in range(cfg.passenger_states))
        self.assertTrue(ceac.is_decision_certified(slice_))

    def test_blocks_identify_pairs_and_configs_roundtrip(self):
        blocks = public_blocks(10)
        pairs = [(a, b) for a in range(10) for b in range(a + 1, 10)]
        sig = {tuple(int(a in blk and b in blk) for blk in blocks) for a, b in pairs}
        self.assertEqual(len(sig), 45)
        cfg = build_config("gi_multiplexed", 1)
        import json
        self.assertEqual(config_from_dict(json.loads(json.dumps(asdict(cfg)))), cfg)

    def test_action_ids_ignores_profiling_where_model_mi_ids_does_not(self):
        cfg = build_config("gi_arrayed", 2)
        model = GIScreenModel(cfg)
        g = len(cfg.genes)

        def profiling_mass(learner):
            dist = learner.distribution()
            return sum(w * pol.probabilities[0, [1 + x for x in cfg.nuisance_genes]].sum()
                       for w, pol in zip(dist, learner.bank.policies) if w > 0)
        model_mi = BayesianIDS(model, np.random.default_rng(0), model.prior("empirical"))
        action_mi = ActionInformationIDS(model, np.random.default_rng(0), model.prior("reference"))
        self.assertGreater(profiling_mass(model_mi), 0.99)
        self.assertLess(profiling_mass(action_mi), 1e-9)

    def test_relabelling_preserves_the_landscape_and_moves_the_sl_index(self):
        from ceac_gi.environment import relabel
        from ceac_gi.data import load_k562_map
        names, matrix = load_k562_map()
        base = build_config("gi_arrayed", 2)
        indices = set()
        for seed in (0, 1, 2, 1001, 1002):
            cfg = relabel(base, seed)
            self.assertEqual(sorted(cfg.genes), sorted(base.genes))
            self.assertEqual(sorted(cfg.values), sorted(base.values))
            a, b = cfg.pairs[cfg.sl_pair_index]
            self.assertEqual({cfg.genes[a], cfg.genes[b]}, {"FDPS", "PPP2R1A"})
            self.assertAlmostEqual(cfg.values[cfg.sl_pair_index], base.strong_value)
            self.assertNotIn(a, cfg.nuisance_genes)
            indices.add(cfg.sl_pair_index)
        self.assertGreater(len(indices), 1)

    def test_action_ids_matches_model_ids_when_optima_are_distinct(self):
        model = EpistasisModel(EpistasisConfig(subtypes=2, pleiotropy=0))
        posterior = np.array([0.3, 0.7])
        a = ActionInformationIDS(model, np.random.default_rng(0), posterior)
        np.testing.assert_allclose(a.action_information_stage(posterior), model.mutual_information_stage(posterior), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
