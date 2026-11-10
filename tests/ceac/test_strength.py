"""Scientific controls for physical strength, distinct from symbol diversity."""
from dataclasses import replace
import unittest
import numpy as np
from ceac.benchmarks import STRENGTH_PROFILES
from ceac.model import FiniteModel, environment, OutcomeRNG
from ceac.benchmark_audit import audit_model, audit_strength_family
from ceac.environments import deepsea, frozenlake
from ceac.run import collect_episode


class StrengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {(k,b): FiniteModel(k,benchmark=b) for k in ("deepsea","frozenlake")
                      for b in ("pathology_rho005","pathology_rho075")}

    def test_only_physical_strength_changes(self):
        for kind in ("deepsea","frozenlake"):
            models = [self.models[kind,b] for b in ("pathology_rho005","pathology_rho075")]
            rows = [audit_model(m) for m in models]
            checked = audit_strength_family(rows)
            self.assertTrue(checked[0]["fixed_domain"])
            np.testing.assert_array_equal(models[0].mask,models[1].mask)
            wrong = [dict(r) for r in rows]
            wrong[1]["state_domain_sha256"] = "different"
            with self.assertRaisesRegex(ValueError,"domain"):
                audit_strength_family(wrong)

    def test_deepsea_drift_law_and_closed_form_noise_route(self):
        for b in ("pathology_rho005","pathology_rho075"):
            m = self.models["deepsea",b];rho=STRENGTH_PROFILES[b]
            policy = next(p for p in m.seed_policies() if p.label=="sonar-sweep")
            j,k = m.evaluate(policy)
            np.testing.assert_allclose(j,.5*(1-rho)**8-8*.005,atol=1e-12)
            np.testing.assert_array_equal(k,0.)
            law=m.laws[m.initial,deepsea.TV][0]
            lost=sum(p for (obs,r,d),p in law.items() if m._models[0].decode(obs).column==0)
            self.assertAlmostEqual(lost,rho)

    def test_beacon_dwell_is_one_charged_step_then_route_resumes(self):
        env=environment("frozenlake",benchmark="pathology_rho075")
        env.reset(seed=3);env.row,env.col=(3,3);env.elapsed=5
        env.np_random=OutcomeRNG(env.config.reed_bins)
        obs,r,term,trunc,info=env.step(frozenlake.RIGHT)
        self.assertTrue(info["nuisance_activated"])
        self.assertEqual((env.row,env.col),frozenlake.REEDS)
        np.testing.assert_array_equal(env.valid_action_mask(obs),[0,0,1,0])
        elapsed=env.elapsed
        obs,r,term,trunc,info=env.step(frozenlake.RIGHT)
        self.assertEqual(env.elapsed,elapsed+1)
        self.assertEqual((env.row,env.col),frozenlake.REEDS)
        self.assertTrue(info["nuisance_delay_step"])
        self.assertEqual(r,-env.config.step_cost)
        # Reed-scan's intended next leg is DOWN, despite the extra elapsed step.
        self.assertEqual(frozenlake.policy_action("reed-scan",env.decode(obs),env.config),frozenlake.DOWN)
        env.step(frozenlake.DOWN)
        self.assertEqual((env.row,env.col),(4,4))

    def test_simulator_samples_are_in_exact_laws(self):
        for (kind,b),m in self.models.items():
            for mode in (0,1):
                env=environment(kind,(mode,mode,mode),benchmark=b)
                p=next(p for p in m.seed_policies() if p.label==("sonar-sweep" if kind=="deepsea" else "reed-scan"))
                for seed in range(12):
                    episode,info=collect_episode(env,m,p,seed,"check",np.random.default_rng(seed),seed)
                    self.assertTrue(np.isfinite(episode.log_likelihood).all())
                    self.assertEqual(len(episode.transitions),m._models[0].decode(info["terminal_observation"]).elapsed)
                    self.assertGreaterEqual(info["events"]["diversions"],info["events"]["lost_steps"])


if __name__=="__main__":unittest.main()
