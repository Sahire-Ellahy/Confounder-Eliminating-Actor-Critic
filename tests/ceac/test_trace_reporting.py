"""Scientific aggregation and saved-history reconstruction, without training."""
from copy import deepcopy
from pathlib import Path
import csv
import gzip
import json
import tempfile
import unittest
import numpy as np
from ceac.report import reconstruct_evidence_trace
from ceac.visualization import (seed_summary, trace_values, export_trace_statistics,
                                nuisance_sensitivity_rows, summarize_nuisance)
from ceac.benchmarks import specification


class TraceStatisticsTests(unittest.TestCase):
    def test_sample_sd_uses_seed_replication(self):
        stats=seed_summary([[1,4],[2,8],[3,12]])
        self.assertEqual(stats["n"],3)
        np.testing.assert_allclose(stats["mean"],[2,8])
        np.testing.assert_allclose(stats["sample_sd"],[1,4])
        np.testing.assert_allclose(stats["lower"],[1,4])
        np.testing.assert_allclose(stats["upper"],[3,12])

    def test_seed_range_is_not_sd(self):
        values=[[0],[0],[9]]
        stats=seed_summary(values,"seed_range")
        np.testing.assert_allclose(stats["lower"],[0])
        np.testing.assert_allclose(stats["upper"],[9])
        self.assertAlmostEqual(stats["sample_sd"][0],np.sqrt(27))
        self.assertLess(seed_summary(values)["lower"][0],0.)

    def test_single_seed_has_no_estimated_variability(self):
        for band in ("sd","seed_range"):
            stats=seed_summary([[2,5]],band)
            self.assertEqual(stats["n"],1)
            for field in ("sample_sd","lower","upper"):
                self.assertIsNone(stats[field])

    def test_rejects_invalid_statistics(self):
        for values in ([],[1,2],[[]],[[1,np.nan]],[[np.inf]]):
            with self.assertRaises(ValueError):seed_summary(values)
        with self.assertRaises(ValueError):seed_summary([[1]],"ci")

    def runs(self):
        return [{"result":{"benchmark":"test","environment":"deepsea","context":0,
                            "true_mode":0,"method":"ceac","seed":seed},
                 "metrics":[{"episode":i,"expected_return":v,"diagnostics":{}}
                            for i,v in enumerate(values)]}
                for seed,values in enumerate(([0,0,6],[3,6,9]))]

    def test_causal_smoothing_preserves_seed_count(self):
        values=trace_values(self.runs(),"expected_return")
        np.testing.assert_allclose(values,[[0,0,2],[3,4.5,6]])
        stats=seed_summary(values)
        self.assertEqual(stats["n"],2)
        np.testing.assert_allclose(stats["mean"],[1.5,2.25,4])
        with self.assertRaises(ValueError):
            trace_values(self.runs()+self.runs(),"expected_return")
        runs=self.runs();runs[0]["metrics"].pop()
        with self.assertRaises(ValueError):trace_values(runs,"expected_return")

    def test_export_keeps_descriptive_statistics_and_smoothing(self):
        with tempfile.TemporaryDirectory() as temp:
            path=export_trace_statistics(temp,self.runs())
            with gzip.open(path,"rt") as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[-1]["independent_unit"],"training_seed")
            self.assertEqual(rows[-1]["causal_smoothing_width"],"16")
            self.assertEqual(rows[-1]["n_seeds"],"2")
            self.assertEqual(float(rows[-1]["mean"]),4.)
            path=export_trace_statistics(temp,self.runs()[:1])
            with gzip.open(path,"rt") as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]["sample_sd"],"")
            self.assertEqual(rows[0]["lower"],"")


class HistoryReconstructionTests(unittest.TestCase):
    def fixture(self,folder):
        rows=[];episodes=[];scores=np.zeros(2);actors=deployments=0;counts={}
        for i,(policy,behavior,ll) in enumerate((
                ("a","actor",[-1.,-2.]),
                ("b","certified_deploy",[-.7,-.2]),
                ("a","actor",[-.1,-4.1]))):
            scores+=ll
            probability=np.exp(scores-scores.max());probability/=probability.sum()
            counts[policy]=counts.get(policy,0)+1
            actors+=int(behavior=="actor");deployments+=int(behavior=="certified_deploy")
            rows.append({"episode":i,"policy_id":policy,"behavior":behavior,
                         "diagnostics":{"model_mixture":probability.tolist(),
                            "plausible":np.flatnonzero(scores.max()-scores<=np.log(20)).tolist(),
                            "actor_updates":actors,"deployment_episodes":deployments,
                            # Policy b has no reported quota; it must still enter evidence.
                            "ledger":{"a":{"count":counts["a"],"statistical_assigned":counts["a"]}}}})
            episodes.append({"index":i,"context":0,"policy_id":policy,"behavior":behavior,
                             "transitions":[{"log_likelihood":(np.asarray(ll)/2).tolist()}]*2})
        (folder/"trajectories.jsonl").write_text("\n".join(json.dumps(e) for e in episodes)+"\n")
        return {"result":{"environment":"deepsea","context":0,"true_mode":0,"method":"ceac",
                          "seed":0,"diagnostics":{"actor_admissions":0}},
                "manifest":{"configuration":{"ceac":{"confidence_delta":.05}}},"metrics":rows}

    def test_all_history_contributes_once_with_correct_policy_credits(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);run=self.fixture(folder)
            audit=reconstruct_evidence_trace(run,folder)
            self.assertEqual(audit["episodes"],3)
            self.assertEqual(audit["saved_policy_strata"],2)
            self.assertEqual(audit["actor_updates"],2)
            self.assertEqual(audit["certified_deployment_episodes"],1)
            np.testing.assert_allclose([r["cumulative_observed_llr"] for r in run["metrics"]],[1,.5,4.5])
            self.assertLess(audit["maximum_probability_residual"],1e-14)

    def test_inconsistent_mixture_credit_or_behavior_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);original=self.fixture(folder)
            for mutation in (
                lambda r:r["diagnostics"].update(model_mixture=[.5,.5]),
                lambda r:r["diagnostics"].update(model_mixture=[float("nan"),.5]),
                lambda r:r["diagnostics"]["ledger"]["a"].update(count=2),
                lambda r:r["diagnostics"].update(actor_updates=100),
                lambda r:r.update(behavior="design")):
                run=deepcopy(original);mutation(run["metrics"][0])
                with self.assertRaises(ValueError):reconstruct_evidence_trace(run,folder)

    def test_missing_trajectory_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);run=self.fixture(folder)
            file=folder/"trajectories.jsonl"
            file.write_text("\n".join(file.read_text().splitlines()[:-1])+"\n")
            with self.assertRaises(ValueError):reconstruct_evidence_trace(run,folder)


class NuisanceSensitivityTests(unittest.TestCase):
    def runs(self):
        runs=[]
        for size,values in ((4,{0:1.,1:10.}),(64,{1:14.,0:3.})):
            for seed,regret in values.items():
                benchmark=f"weak_signal_m{size}"
                runs.append({"result":{"benchmark":benchmark,"environment":"deepsea","context":0,
                              "true_mode":0,"method":"ceac","seed":seed,"episodes":512,
                              "pseudo_regret":regret,"event_counts":{"nuisance":5}},
                             "manifest":{"configuration":{"benchmark":specification(benchmark,"deepsea"),
                              "environment":"deepsea","context":0,"true_mode_evaluator_only":0,
                              "method":"ceac","seed":seed,"episodes":512,"ceac":{"mass_bound":128}}}})
        return runs

    def test_subtracts_matched_seeds_before_aggregation(self):
        rows=nuisance_sensitivity_rows(self.runs())
        deltas={r["seed"]:r["added_regret"] for r in rows if r["nuisance_bins"]==64}
        self.assertEqual(deltas,{0:2.,1:4.})
        summary=next(r for r in summarize_nuisance(rows)
                     if r["nuisance_bins"]==64 and r["metric"]=="added_regret")
        self.assertEqual(summary["mean"],3.)
        self.assertAlmostEqual(summary["sample_sd"],np.sqrt(2))
        self.assertEqual(summary["n_seeds"],2)

    def test_unmatched_or_duplicate_seed_fails(self):
        runs=self.runs()
        with self.assertRaises(ValueError):nuisance_sensitivity_rows(runs[:-1])
        with self.assertRaises(ValueError):nuisance_sensitivity_rows(runs+[runs[0]])

    def test_a_second_experimental_change_fails(self):
        for mutate in (
            lambda c:c.update(episodes=1024),
            lambda c:c["ceac"].update(mass_bound=20),
            lambda c:c["benchmark"]["configuration_overrides"].update(challenge_prob_mode0=.25),
            lambda c:c["benchmark"]["configuration_overrides"].update(tv_bins=128)):
            runs=self.runs();mutate(runs[-1]["manifest"]["configuration"])
            with self.assertRaises(ValueError):nuisance_sensitivity_rows(runs)

    def test_one_dose_does_not_imply_a_sensitivity_curve(self):
        self.assertEqual(nuisance_sensitivity_rows(self.runs()[:2]),[])


if __name__=="__main__":unittest.main()
