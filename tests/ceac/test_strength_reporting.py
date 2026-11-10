"""Matched overlays cannot silently pool doses, sources, budgets or seeds."""
from copy import deepcopy
import unittest
import numpy as np
from ceac.benchmarks import STRENGTH_PROFILES, specification
from ceac.pathology_report import strength_groups, strength_statistics, draw_method_overlay, draw_strength_learning, draw_strength_diagnostics


class StrengthReportingTests(unittest.TestCase):
    def runs(self):
        runs=[]
        for b,rho in STRENGTH_PROFILES.items():
            for method in ("idac","ucb_q","pomd","ids","ids_zero","ceac","oracle"):
                for seed in (0,1):
                    regret=rho+seed
                    metrics=[dict(episode=t, expected_return=.5-regret/4, optimal_return=.5,
                        cumulative_pseudo_regret=regret*(t+1)/4, terminal_success=bool(t%2),
                        cumulative_nuisance_visits=t+1, cumulative_diversions=t,
                        cumulative_lost_steps=t, cumulative_expected_model_information=0,
                        diagnostics={"all_optimistic_q_zero":method=="pomd"}) for t in range(4)]
                    config=dict(benchmark=specification(b,"deepsea"),environment="deepsea",method=method,
                                context=0,seed=seed,true_mode_evaluator_only=1,episodes=4)
                    result=dict(benchmark=b,environment="deepsea",context=0,true_mode=1,
                                method=method,seed=seed,episodes=4,pseudo_regret=regret)
                    runs.append(dict(manifest={"configuration":config,"sources":{"runtime":"fixed"}},result=result,metrics=metrics))
        return runs

    def test_paired_increase_and_failure_flags_are_derived_from_data(self):
        rows=strength_statistics(self.runs())
        high=[r for r in rows if r["rho"]==.75]
        np.testing.assert_allclose([r["added_regret"] for r in high],.7)
        for row in rows:
            self.assertEqual("POMD_BONUS_SATURATION" in row["diagnostic_flags"],row["method"]=="pomd")
            self.assertEqual(row["success"],.5)

    def test_missing_seed_or_dose_and_changed_budget_or_source_are_rejected(self):
        runs=self.runs()
        variants=[runs[:-1],runs+[runs[0]], [r for r in runs if not (r["result"]["method"]=="idac" and r["result"]["benchmark"]=="pathology_rho075")]]
        for field,value in (("episodes",5),("method","different")):
            changed=deepcopy(runs);changed[-1]["manifest"]["configuration"][field]=value;variants.append(changed)
        changed=deepcopy(runs);changed[-1]["manifest"]["sources"]={"runtime":"different"};variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError):strength_groups(variant)

    def test_added_regret_subtracts_paired_traces_before_seed_variation(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig=draw_strength_learning(strength_groups(self.runs()),"deepsea",plt=plt,incremental=True)
        for axis in fig.axes:
            if not axis.axison:
                continue
            # Four strength curves, then a zero reference line.
            np.testing.assert_allclose(axis.lines[0].get_ydata(),0)
            np.testing.assert_allclose(axis.lines[3].get_ydata(),.7*np.arange(1,5)/4)
        plt.close(fig)

    def test_standard_overlay_contains_all_seven_methods_and_both_priors(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig=draw_method_overlay(strength_groups(self.runs()),"deepsea",plt=plt)
        self.assertEqual(len(fig.axes),4)
        for axis in fig.axes:
            self.assertEqual(len(axis.lines),7)
            self.assertEqual(axis.get_yscale(),"symlog")
            self.assertTrue(any("IDAC" in l.get_label() for l in axis.lines))
            self.assertIn("Bayesian IDS: full support", [l.get_label() for l in axis.lines])
            self.assertIn("Bayesian IDS: misspecified prior", [l.get_label() for l in axis.lines])
        plt.close(fig)

    def test_strength_layout_does_not_truncate_seven_methods_to_six_panels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ceac.visualization import LABELS
        expected = {LABELS[m] for m in ("ucb_q", "pomd", "idac", "ids", "ids_zero", "ceac", "oracle")}
        for incremental in (False, True):
            fig = draw_strength_learning(strength_groups(self.runs()), "deepsea", plt=plt,
                                         incremental=incremental)
            axes = [axis for axis in fig.axes if axis.axison]
            self.assertEqual(len(axes), 7)
            self.assertEqual({axis.get_title().replace("\n", " ") for axis in axes}, expected)
            self.assertTrue(all(len(axis.lines) == (5 if incremental else 4) for axis in axes))
            plt.close(fig)

    def test_strength_diagnostics_keep_both_bayesian_treatments(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ceac.visualization import LABELS
        fig = draw_strength_diagnostics(strength_statistics(self.runs()), "deepsea", plt=plt)
        for axis in fig.axes:
            # Diagnostics use a shared figure legend, with seven actual curves per axis.
            self.assertEqual(len(axis.lines), 7)
        labels = {text.get_text() for text in fig.legends[0].get_texts()}
        self.assertIn(LABELS["ids"], labels)
        self.assertIn(LABELS["ids_zero"], labels)
        plt.close(fig)


if __name__=="__main__":unittest.main()
