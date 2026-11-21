"""Unit tests for greedy maximally-disagreeing admission and price-ranked replay CEAC."""
from __future__ import annotations
from collections import deque
import unittest
import numpy as np
from ceac.contracts import Episode, Transition
from ceac_gi.environment import GIScreenEnv, GIScreenModel, build_config, relabel
from ceac_ranked import GreedyPricedCEAC, RankedCEACConfig


def _simulate_ranked(cfg, task: int, episodes: int, seed: int = 1001, optimizer: str = "reference"):
    cfg = relabel(cfg, seed)
    model = GIScreenModel(cfg)
    truth = model.true_index(task, passenger=0)
    env = GIScreenEnv(cfg, pair=model.hypotheses[truth][0], passenger=0)
    env.reset(seed=seed + 100)
    learner = GreedyPricedCEAC(model, np.random.default_rng(seed), RankedCEACConfig(optimizer=optimizer))
    act_rng = np.random.default_rng(seed + 200)
    regret = 0.0
    window = deque(maxlen=256)
    priced = 0
    for t in range(episodes):
        policy, behavior, prices = learner.select(t)
        expected = float(model.returns(policy)[truth])
        gap = max(0.0, float(model.optimal_returns[truth] - expected))
        obs, _ = env.reset()
        trans = []
        for _ in range(model.horizon):
            s = model.index[obs]
            a = int(act_rng.choice(model.actions, p=policy.probabilities[s]))
            nxt, rew, done, _, _ = env.step(a)
            ll = model.log_likelihood(s, a, nxt, rew, done)
            trans.append(Transition(s, a, rew, -1 if done else model.index[nxt], done, ll))
            obs = nxt
            if done:
                break
        ep = Episode(t, 0, policy.identity, behavior, tuple(trans))
        learner.observe(ep, policy, prices)
        regret += gap
        window.append(gap)
        if behavior == "actor" and np.any(prices > 0):
            priced += 1
    return learner, float(np.mean(window)), regret, priced


class RankedCEACTests(unittest.TestCase):

    def test_avoids_pre_registering_combinatorial_seed_bank(self):
        cfg = build_config("gi_arrayed", 1)
        model = GIScreenModel(cfg)
        learner = GreedyPricedCEAC(model, np.random.default_rng(0))
        self.assertEqual(len(model.seed_policies()), 46)
        self.assertEqual(len(learner.bank.policies), 1)
        self.assertEqual(learner.bank.policies[0].identity, model.seed_policies()[0].identity)

    def test_disagreement_scores_zero_on_confounders_and_favor_multiplexed_blocks(self):
        for kind in ("gi_arrayed", "gi_multiplexed"):
            cfg = build_config(kind, 2)
            model = GIScreenModel(cfg)
            learner = GreedyPricedCEAC(model, np.random.default_rng(0))
            scores = learner.disagreement_scores(learner.replay.plausible)
            g = len(cfg.genes)
            for idx, pol in enumerate(learner.candidate_policies):
                root_action = int(np.argmax(pol.probabilities[0]))
                if root_action <= g:
                    self.assertEqual(float(scores[idx]), 0.0)
            if kind == "gi_multiplexed":
                chosen = learner.select_greedy_disagreement(learner.replay.plausible)
                chosen_action = int(np.argmax(chosen.probabilities[0]))
                self.assertGreater(chosen_action, g + len(cfg.values))

    def test_dynamic_price_ranking_reorders_replay_without_new_episodes(self):
        cfg = build_config("gi_arrayed", 0)
        model = GIScreenModel(cfg)
        learner = GreedyPricedCEAC(model, np.random.default_rng(0))
        env = GIScreenEnv(cfg, pair=-1, passenger=0)
        env.reset(seed=7)
        g = len(cfg.genes)
        # Collect 3 episodes on pair 0 (action g+1) and 3 episodes on pair 1 (action g+2)
        p1 = next(p for p in learner.candidate_policies if int(np.argmax(p.probabilities[0])) == g + 1)
        p2 = next(p for p in learner.candidate_policies if int(np.argmax(p.probabilities[0])) == g + 2)
        ep_idx = 0
        for pol in (p1, p2):
            learner.replay.register(pol)
            for _ in range(3):
                obs, _ = env.reset()
                trans = []
                for _ in range(model.horizon):
                    s = model.index[obs]
                    a = int(np.argmax(pol.probabilities[s]))
                    nxt, rew, done, _, _ = env.step(a)
                    ll = model.log_likelihood(s, a, nxt, rew, done)
                    trans.append(Transition(s, a, rew, -1 if done else model.index[nxt], done, ll))
                    obs = nxt
                    if done:
                        break
                learner.replay.append(Episode(ep_idx, 0, pol.identity, "statistical", tuple(trans)))
                ep_idx += 1

        rows = ((0, 1), (0, 2))
        beta_a = np.zeros((model.num_models, model.num_models))
        beta_a[0, 1] = 1.0
        ranked_a = learner.replay.rank_buffer(rows, beta_a, min_count=2)
        self.assertEqual(ranked_a[0].policy_id, p1.identity)

        beta_b = np.zeros((model.num_models, model.num_models))
        beta_b[0, 2] = 1.0
        seen_before = len(learner.replay.seen)
        ranked_b = learner.replay.rank_buffer(rows, beta_b, min_count=2)
        self.assertEqual(len(learner.replay.seen), seen_before)
        self.assertEqual(ranked_b[0].policy_id, p2.identity)

    def test_solves_both_tasks_and_cuts_arrayed_regret(self):
        for kind in ("gi_arrayed", "gi_multiplexed"):
            cfg = build_config(kind, 1)
            for task in (0, 1):
                for opt in ("reference", "guarded_natural"):
                    learner, late_gap, regret, priced = _simulate_ranked(cfg, task, episodes=4096, seed=1001, optimizer=opt)
                    self.assertLessEqual(late_gap, 0.025, msg=f"{kind} task={task} opt={opt} late_gap={late_gap}")
                    self.assertGreater(learner.actor.updates, 0)
                    self.assertGreater(priced, 0)
                    if kind == "gi_arrayed" and task == 1:
                        self.assertLess(len(learner.bank.policies), 10)
                        self.assertLess(regret, 500.0)


if __name__ == "__main__":
    unittest.main()
