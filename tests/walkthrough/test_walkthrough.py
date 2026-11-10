"""The walkthrough preserves kernel geometry and immutable cached evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from ceac_walkthrough.evidence import verified_bundle
from ceac_walkthrough.visuals import rollout_route
from ceac.environments import frozenlake

ROOT = Path(__file__).resolve().parents[2]


class WalkthroughTests(unittest.TestCase):
    def test_undertow_spends_depth_without_progress(self):
        direct = rollout_route("deepsea", "exploit")
        noisy = rollout_route("deepsea", "sonar-sweep")
        probe = rollout_route("deepsea", "probe")
        self.assertTrue(direct["terminal_success"])
        self.assertFalse(noisy["terminal_success"])
        np.testing.assert_array_equal(direct["cells"][-1], [8, 8])
        np.testing.assert_array_equal(noisy["cells"][-1], [0, 8])
        np.testing.assert_array_equal(probe["cells"][-1], [8, 8])
        self.assertEqual(probe["events"].count("probe"), 1)
        self.assertEqual(len(probe["events"]), len(direct["events"])+1)
        self.assertEqual(noisy["events"].count("tv"), 8)

    def test_shore_paths_follow_real_corridors_and_diagnostic_sites(self):
        for name, endpoint in (("inner-bridge", frozenlake.INNER_DOCK),
                               ("outer-causeway", frozenlake.OUTER_DOCK),
                               ("deep-survey", frozenlake.SAFE_DOCK),
                               ("reed-scan", frozenlake.SAFE_DOCK)):
            route = rollout_route("frozenlake", name)
            self.assertTrue(all((row, col) in frozenlake.OPEN_CELLS for col, row in route["cells"]))
            np.testing.assert_array_equal(route["cells"][-1], endpoint[::-1])
        self.assertIn("deep", rollout_route("frozenlake", "deep-survey")["events"])
        self.assertIn("reeds", rollout_route("frozenlake", "reed-scan")["events"])

    def test_published_export_has_unique_complete_seed_curves(self):
        bundle, manifest = verified_bundle(ROOT / "notebooks")
        self.assertEqual(len(bundle["endpoints"]), 336)
        self.assertEqual(len(bundle["curves"]), 336)
        self.assertEqual(set(manifest["curves_benchmarks"]),
                         {"pathology_rho005", "pathology_rho025", "pathology_rho050", "pathology_rho075"})
        self.assertEqual(sum(s["run_count"] for s in manifest["source_studies"]), 336)
        self.assertEqual(manifest["excluded_runs"], 0)
        self.assertEqual(len(bundle["strength_checks"]), 2)
        seen = set()
        for run in bundle["curves"]:
            r = run["result"]
            key = tuple(r[k] for k in ("benchmark", "environment", "context", "true_mode", "method", "seed"))
            self.assertNotIn(key, seen); seen.add(key)
            self.assertEqual([m["episode"] for m in run["metrics"]], list(range(2048)))
        self.assertEqual({r["result"]["method"] for r in bundle["curves"]},
                         {"ceac", "oracle", "ids", "idac", "ucb_q", "pomd", "ids_zero"})

    def test_corrupted_export_is_rejected_before_display(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / "notebooks/evidence", root / "evidence")
            (root / "assets").mkdir()
            manifest = json.loads((root / "evidence/manifest.json").read_text())
            for name in manifest["artifacts"]:
                if name.startswith("assets/"):
                    shutil.copyfile(ROOT / "notebooks" / name, root / name)
            bundle = root / "evidence/curves.json.gz"
            bundle.write_bytes(bundle.read_bytes()+b"corrupted")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verified_bundle(root)

    def test_manifest_cannot_escape_notebook_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "evidence").mkdir()
            manifest = {"schema": 1, "artifacts": {"../outside": "unused"}}
            (root / "evidence/manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leaves notebook"):
                verified_bundle(root)


if __name__ == "__main__":
    unittest.main()
