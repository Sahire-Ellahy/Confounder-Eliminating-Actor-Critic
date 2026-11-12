"""Breadth changes physical opportunities, never useful task laws or hidden access."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
from ceac.breadth import make_model,make_environment,specification,audit_breadth_model,audit_breadth_family
from ceac.model import FiniteModel
from ceac.contracts import Policy
from ceac.environments import frozenlake as shore
from ceac.environments.beacon_breadth import BeaconBreadthConfig,FIELD_ORDER,GATEWAY,ENTRY,MOVES
from ceac.run import collect_episode
from ceac.breadth_report import breadth_statistics


class BreadthEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={b:make_model("frozenlake",benchmark=f"beacon_breadth_b{b:02d}") for b in (0,2,8)}

    def test_zero_breadth_is_exact_original_shoreline(self):
        original=FiniteModel("frozenlake",benchmark="weak_signal_m4")
        zero=self.models[0]
        for field in ("state_ids","mask","reward","kl","edge","destination","prob","optimal_returns"):
            np.testing.assert_array_equal(getattr(zero,field),getattr(original,field))
        self.assertEqual(zero.laws,original.laws)

    def test_field_has_only_one_old_new_connection_and_is_nested(self):
        for b,model in self.models.items():
            env=model._models[0]
            self.assertEqual(len(env.field_cells),b)
            self.assertFalse(env.field_cells & shore.OPEN_CELLS)
            cross=set()
            for row,col in env.open_cells:
                state=shore.DecodedState(0,row,col,-1,-1,-1,0)
                for action in np.flatnonzero(env.valid_action_mask(env.encode(state))):
                    dr,dc=MOVES[int(action)];other=(row+dr,col+dc)
                    self.assertIn(other,env.open_cells)
                    if ((row,col) in env.field_cells)!=(other in env.field_cells):
                        cross.add(tuple(sorted(((row,col),other))))
            self.assertEqual(cross,{tuple(sorted((GATEWAY,ENTRY)))} if b else set())
        self.assertTrue(set(self.models[0].state_ids)<set(self.models[2].state_ids)<set(self.models[8].state_ids))

    def test_preflight_preserves_task_and_oracle_and_rejects_confounds(self):
        rows=[audit_breadth_model(m) for m in self.models.values()]
        checked=audit_breadth_family(rows)[0]
        self.assertEqual(checked["breadths"],[0,2,8])
        self.assertFalse(checked["fixed_domain"])
        self.assertTrue(checked["fixed_optimum_and_oracle_cost"])
        for field,value in (("beacon_cells",-1),("beacon_cells",1.5),("nuisance_probability",.5)):
            with self.assertRaises(ValueError):replace(BeaconBreadthConfig(),**{field:value})
        changed=deepcopy(rows);changed[-1]["oracle_design_costs"][0]+=.01
        with self.assertRaises(AssertionError):audit_breadth_family(changed)
        changed=deepcopy(rows);changed[-1]["breadth"]["field_transition_max_task_kl"]=.1
        with self.assertRaises(ValueError):audit_breadth_family(changed)

    def test_actual_field_steps_and_unique_locations_are_charged_and_likelihood_valid(self):
        model=self.models[8]
        actions=[shore.UP,shore.RIGHT,shore.RIGHT,shore.UP,shore.LEFT,shore.LEFT,
                 shore.RIGHT,shore.RIGHT,shore.DOWN,shore.DOWN]+[shore.RIGHT]*6
        p=np.zeros(model.mask.shape)
        for s,obs in enumerate(model.state_ids):
            state=model._models[0].decode(int(obs));action=actions[state.elapsed]
            if not model.mask[s,action]:action=int(np.flatnonzero(model.mask[s])[0])
            p[s,action]=1.
        policy=Policy(p,"field-logging-check")
        for mode in (0,1):
            env=make_environment("frozenlake",(mode,mode,mode),benchmark=model.benchmark)
            episode,info=collect_episode(env,model,policy,0,"check",np.random.default_rng(2),3)
            self.assertEqual(len(episode.transitions),16)
            self.assertEqual(info["events"]["beacon_steps"],5)
            self.assertEqual(len(info["beacon_locations"]),3)
            self.assertTrue(info["terminal_success"])
            self.assertAlmostEqual(episode.log_likelihood[0],episode.log_likelihood[1])
        np.testing.assert_allclose(model.evaluate(policy)[1],0,atol=1e-12)

    def test_sampler_and_enumerator_agree_on_all_seed_routes(self):
        for b in (0,8):
            model=self.models[b]
            for mode in (0,1):
                env=make_environment("frozenlake",(mode,mode,mode),benchmark=model.benchmark)
                for policy in model.seed_policies():
                    episode,info=collect_episode(env,model,policy,0,"check",np.random.default_rng(3),4)
                    self.assertTrue(np.isfinite(episode.log_likelihood).all())
                    self.assertEqual(info["events"]["beacon_steps"],0)
                    self.assertEqual(info["beacon_locations"],set())


class BreadthReportingTests(unittest.TestCase):
    def runs(self):
        runs=[]
        for b in (0,8):
            for method in ("ceac","oracle","ids","ids_zero","idac","ucb_q","pomd"):
                for seed in (0,1):
                    name=f"beacon_breadth_b{b:02d}";ceac=10+seed+.1*b
                    regret=ceac if method=="ceac" else ceac-1 if method=="ids" else 20+10*seed+b*(seed+1)
                    cfg=dict(benchmark=specification(name,"frozenlake"),environment="frozenlake",method=method,
                             context=0,seed=seed,true_mode_evaluator_only=1,episodes=4)
                    result=dict(benchmark=name,environment="frozenlake",context=0,true_mode=1,method=method,
                                seed=seed,episodes=4,pseudo_regret=regret)
                    metrics=[dict(episode=t,cumulative_pseudo_regret=regret*(t+1)/4,optimal_return=.539,
                        expected_return=.3,cumulative_beacon_steps=t if b else 0,unique_beacon_cells=2 if b else 0,
                        cumulative_expected_model_information=.2*t,diagnostics={}) for t in range(4)]
                    runs.append(dict(manifest={"configuration":cfg,"sources":{"fixed":"hash"}},result=result,metrics=metrics))
        return runs

    def test_paired_effects_keep_ceac_losses_and_baseline_subtraction(self):
        rows=breadth_statistics(self.runs())
        strong=[r for r in rows if r["method"]=="ucb_q" and r["breadth"]==8]
        np.testing.assert_allclose([r["added_regret"] for r in strong],[8,16])
        np.testing.assert_allclose([r["additional_ceac_advantage"] for r in strong],[7.2,15.2])
        ids=[r for r in rows if r["method"]=="ids"]
        np.testing.assert_allclose([r["regret_saved_by_ceac"] for r in ids],-1)

    def test_incomplete_or_non_breadth_changes_fail(self):
        runs=self.runs()
        variants=[runs[:-1],runs+[runs[0]],[r for r in runs if r["result"]["benchmark"]!="beacon_breadth_b00"]]
        for field,value in (("episodes",5),("seed",99)):
            changed=deepcopy(runs);changed[-1]["manifest"]["configuration"][field]=value;variants.append(changed)
        changed=deepcopy(runs);changed[-1]["manifest"]["sources"]={"different":"source"};variants.append(changed)
        changed=deepcopy(runs);changed[0]["metrics"][-1]["cumulative_beacon_steps"]=1;variants.append(changed)
        for variant in variants:
            with self.assertRaises(ValueError):breadth_statistics(variant)

    def test_all_seven_methods_are_visible_and_smoke_is_labeled(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ceac.breadth_report import draw_breadth_response,breadth_report_section
        from ceac.pathology_report import METHODS
        from ceac.visualization import LABELS
        rows=breadth_statistics(self.runs());fig=draw_breadth_response(rows,plt=plt)
        for axis in fig.axes:
            self.assertTrue({LABELS[m] for m in METHODS}<={line.get_label() for line in axis.lines})
        plt.close(fig)
        text="\n".join(breadth_report_section(rows))
        self.assertIn("Short integration run",text)
        self.assertIn("-1.000",text)
        self.assertIn("reachable state/action domain grows",text)


if __name__=="__main__":unittest.main()
