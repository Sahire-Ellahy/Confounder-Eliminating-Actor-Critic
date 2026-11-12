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


    def comparison_rows(self):
        rows=strength_statistics(self.runs())
        for row in rows:
            seed, rho = row["seed"], row["rho"]
            ceac=100+50*seed+2*rho
            row["regret"]=ceac
            if row["method"]=="ucb_q":
                row["regret"]+=(seed+1)*(2+10*(rho-.05))
            elif row["method"]=="ids":
                row["regret"]-=seed+1
        return rows

    def test_ceac_advantage_is_paired_before_variation_and_preserves_losses(self):
        from ceac.pathology_report import ceac_advantage_statistics, summarize_ceac_advantages
        rows=self.comparison_rows()
        paired=ceac_advantage_statistics(rows)
        self.assertEqual(paired,ceac_advantage_statistics(list(reversed(rows))))
        summary=summarize_ceac_advantages(paired)
        high=next(r for r in summary if r["method"]=="ucb_q" and r["rho"]==.75)
        self.assertAlmostEqual(high["regret_saved"]["mean"],13.5)
        self.assertAlmostEqual(high["regret_saved"]["sample_sd"],np.std([9.,18.],ddof=1))
        self.assertAlmostEqual(high["extra_regret_saved"]["mean"],10.5)
        self.assertAlmostEqual(high["extra_regret_saved"]["sample_sd"],np.std([7.,14.],ddof=1))
        self.assertEqual(high["regret_saved"]["ceac_better"],2)
        ids=next(r for r in summary if r["method"]=="ids" and r["rho"]==.75)
        self.assertEqual(ids["regret_saved"]["mean"],-1.5)
        self.assertEqual(ids["regret_saved"]["ceac_worse"],2)
        oracle=next(r for r in summary if r["method"]=="oracle")
        self.assertEqual(oracle["regret_saved"]["tied"],2)
        low=[r["extra_regret_saved"] for r in paired if r["rho"]==.05]
        np.testing.assert_array_equal(low,0)

    def test_ceac_pairing_rejects_incomplete_or_incompatible_endpoints(self):
        from ceac.pathology_report import ceac_advantage_statistics
        rows=self.comparison_rows()
        variants=[rows[:-1],rows+[rows[0]],
                  [r for r in rows if r["method"]!="ceac"],
                  [r for r in rows if not (r["method"]=="ids" and r["rho"]==.75)]]
        for field,value in (("episodes",5),("regret",float("nan"))):
            changed=deepcopy(rows);changed[-1][field]=value;variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError):ceac_advantage_statistics(variant)

    def test_single_seed_has_no_spread_and_text_keeps_losses_and_ties(self):
        from ceac.pathology_report import (ceac_advantage_statistics, summarize_ceac_advantages,
                                           ceac_comparison_section)
        rows=[r for r in self.comparison_rows() if r["seed"]==0]
        summaries=summarize_ceac_advantages(ceac_advantage_statistics(rows))
        self.assertTrue(all(r["regret_saved"]["sample_sd"] is None for r in summaries))
        text="\n".join(ceac_comparison_section(rows))
        self.assertIn("-1.00 ± n/a",text)
        self.assertIn("**+9.00 ± n/a**",text)
        self.assertIn("0 / 1 / 0",text)
        self.assertIn("not beacon breadth",text)
        self.assertIn("Full-support IDS has lower",text)

    def test_response_and_advantage_show_all_controls_and_signed_effects(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ceac.pathology_report import draw_pathology_response, draw_ceac_advantage, METHODS
        from ceac.visualization import LABELS
        rows=self.comparison_rows()
        # A method can improve as dose increases; its negative delta must survive.
        for row in rows:
            if row["method"]=="idac":row["regret"]-=10*row["rho"]
        fig=draw_pathology_response(rows,plt=plt)
        self.assertEqual(len(fig.axes),3)
        for axis in fig.axes:
            self.assertTrue({LABELS[m] for m in METHODS} <= {line.get_label() for line in axis.lines})
        idac=next(line for line in fig.axes[1].lines if line.get_label()==LABELS["idac"])
        self.assertLess(idac.get_ydata()[-1],0)
        self.assertLess(fig.axes[1].get_ylim()[0],0)
        plt.close(fig)
        fig=draw_ceac_advantage(rows,plt=plt)
        self.assertEqual(len(fig.axes[0].get_yticklabels()),6)
        self.assertTrue((fig.axes[0].images[0].get_array()<0).any())
        self.assertTrue(any(t.get_text().startswith("-1.50") for t in fig.axes[0].texts))
        plt.close(fig)

    def test_coincident_trace_markers_survive_all_method_lines_without_value_offsets(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection
        from ceac.visualization import plot_seed_trace, pad_plot_axis
        from ceac.pathology_report import METHODS
        fig,axis=plt.subplots()
        # Identical curves are a realistic exact-tie case, including a zero start.
        values=np.tile(np.arange(256,dtype=float),(3,1))
        for method in METHODS:plot_seed_trace(axis,values,method,"sd")
        markers=[c for c in axis.collections if isinstance(c,PathCollection)]
        self.assertEqual(len(markers),len(METHODS))
        sampled=[]
        for line,points in zip(axis.lines,markers):
            np.testing.assert_array_equal(line.get_ydata(),values[0])
            np.testing.assert_array_equal(line.get_xdata(),np.arange(1,257))
            xy=np.asarray(points.get_offsets())
            np.testing.assert_array_equal(xy[:,1],values[0,xy[:,0].astype(int)-1])
            self.assertGreater(points.get_zorder(),max(l.get_zorder() for l in axis.lines))
            sampled.append(set(xy[:,0]))
        self.assertTrue(all(not a.intersection(b) for i,a in enumerate(sampled) for b in sampled[i+1:]))
        axis.set_yscale("symlog",linthresh=.01)
        pad_plot_axis(axis)
        low,high=axis.get_ylim()
        self.assertLess(low,0)
        self.assertGreater(high,255)
        # Display-space padding must not introduce a large negative blank region.
        self.assertGreater(low,-.01)
        plt.close(fig)

    def test_shared_axis_padding_keeps_negative_bands_and_zero(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ceac.visualization import pad_plot_axis, plot_seed_trace
        fig,axes=plt.subplots(1,2,sharey=True)
        plot_seed_trace(axes[0],np.zeros((2,32)),"ceac","sd")
        plot_seed_trace(axes[1],np.array([[-30.]*32,[10.]*32]),"ids","sd")
        axes[0].set_yscale("symlog",linthresh=.01)
        pad_plot_axis(axes[0])
        self.assertEqual(axes[0].get_ylim(),axes[1].get_ylim())
        low,high=axes[0].get_ylim()
        self.assertLess(low,-10-np.std([-30,10],ddof=1))
        self.assertGreater(high,-10+np.std([-30,10],ddof=1))
        plt.close(fig)
        fig=draw_strength_learning(strength_groups(self.runs()),"deepsea",plt=plt)
        active=[axis for axis in fig.axes if axis.axison]
        self.assertTrue(all(axis.get_yscale()=="symlog" for axis in active))
        self.assertEqual(len({axis.get_ylim() for axis in active}),1)
        plt.close(fig)

    def test_endpoint_rows_preserve_exact_ties_and_paired_negative_changes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ceac.pathology_report import draw_endpoint_comparison, METHODS
        from ceac.visualization import LABELS
        rows=strength_statistics(self.runs())
        # All methods tie at strongest dose, but improve from the low dose.
        for row in rows:row["regret"]=0. if row["rho"]==.75 else 1.+row["seed"]
        fig=draw_endpoint_comparison(rows,"deepsea",plt=plt)
        for col,axis in enumerate(fig.axes):
            self.assertEqual(len(axis.containers),len(METHODS))
            expected_mean=(0.,-1.5,.3125)[col]
            for i,container in enumerate(axis.containers):
                point=container.lines[0]
                self.assertEqual(container.get_label(),LABELS[METHODS[i]])
                np.testing.assert_allclose(np.asarray(point.get_xdata(),dtype=float),[expected_mean])
                np.testing.assert_array_equal(point.get_ydata(),[i])
            self.assertEqual(len(axis.texts),len(METHODS))
        self.assertLess(fig.axes[1].get_xlim()[0],-1.5-np.std([-1,-2],ddof=1))
        self.assertLess(fig.axes[0].get_xlim()[0],0)
        self.assertGreater(fig.axes[0].get_xlim()[1],0)
        plt.close(fig)
        with self.assertRaisesRegex(ValueError,"not recorded"):
            draw_endpoint_comparison(rows,"deepsea",plt=plt,rho=.9)


if __name__=="__main__":unittest.main()
