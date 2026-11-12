"""Protect the mechanism equations, saved-update audit and within-method plots."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from ceac_walkthrough.mechanisms import occupancy_ratio,population_nuisance,decompose_update,reconstruct_cell,load_mechanisms
from ceac_walkthrough.failure_report import seed_curve,draw_failure_fans
from ceac_hardness.environment import JunctionConfig
from ceac_hardness.model import JunctionModel
from ceac_pseudocode.learning import TwoCriticActor
from ceac.contracts import Episode,Transition
from ceac_hardness.run import digest


class MechanismTests(unittest.TestCase):
    def test_population_ratio_matches_normalized_two_step_occupancy(self):
        for b in (4,16,64):
            for p in (.1,.5,.9):
                occupancy=np.array([.5,.5*(1-p)]+[.5*p/b]*b)
                entropy=-np.sum(occupancy*np.log(occupancy))
                average_cost=1.01-.5*.9*(1-p)
                self.assertAlmostEqual(occupancy_ratio(p,b),average_cost/(entropy+.1))
        self.assertAlmostEqual(occupancy_ratio(0,0),.56/(np.log(2)+.1))
        self.assertEqual(population_nuisance(0),0)
        self.assertTrue(population_nuisance(4)<population_nuisance(16)<population_nuisance(64))
        with self.assertRaises(ValueError):occupancy_ratio(.1,0)

    def test_frozen_errors_decompose_actual_update_and_zero_evidence(self):
        model=JunctionModel(JunctionConfig(burden=4))
        for price in (0,.2):
            actor=TwoCriticActor(model);p=actor.snapshot();beta=np.array([[0.,0.],[price,0.]])
            episode=Episode(0,0,p.identity,'actor',(
                Transition(0,1,1.,1,False,tuple(np.log([.1,.9]))),
                Transition(1,0,0.,-1,True,(0.,0.))))
            before=actor.logits.copy();actor.update(episode,p,beta)
            reward,evidence=decompose_update(actor,episode,p)
            np.testing.assert_allclose(reward+evidence,actor.logits-before,atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(evidence)==0,price==0)
            if price:self.assertGreater(evidence[0,1]-evidence[0,3],0)

    def fixture(self,root):
        from dataclasses import asdict
        folder=root/'cell';folder.mkdir()
        config=JunctionConfig(burden=0);model=JunctionModel(config);actor=TwoCriticActor(model)
        episodes=[];designs=[]
        for index in range(6):
            beta=np.array([[0.,0.],[.2 if index<4 else .3,0.]])
            if index in (2,4):
                designs.append(dict(episode=index,design=dict(beta=beta.tolist(),rows=[[1,0]],
                    references=[1],policy_ids=['fixed'],arrays={'return_mean':[.4],'information_mean':[[1.]]}),
                    priced_replay_scores=[.4+beta[1,0]]))
            p=actor.snapshot();behavior='statistical' if index<2 else 'actor'
            e=Episode(index,0,p.identity,behavior,(
                Transition(0,1,1.,1,False,tuple(np.log([.1,.9]))),Transition(1,0,0.,-1,True,(0.,0.))))
            episodes.append(dict(episode=index,policy_id=p.identity,behavior=behavior,
                transitions=[asdict(t) for t in e.transitions]))
            actor.update(e,p,beta)
        with gzip.open(folder/'episodes.jsonl.gz','wt') as handle:
            for e in episodes:handle.write(json.dumps(e)+'\n')
        (folder/'design_history.json').write_text(json.dumps(designs))
        settings=dict(environment=asdict(config),ceac_config={'actor_rate':.4,'critic_rate':.12,'temperature':1.},
            task_evaluator_only=1,seed=0,episodes=6,log_every=1)
        (folder/'summary.json').write_text(json.dumps(dict(identity='test',settings=settings,actor_updates=4,priced_actor_updates=4)))
        (folder/'manifest.json').write_text(json.dumps({'sources':{}}))
        (folder/'status.json').write_text(json.dumps(dict(status='complete',artifacts={p.name:digest(p) for p in folder.iterdir()})))
        return folder,actor

    def test_reconstruction_freezes_prices_and_skips_diagnostic_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder,actor=self.fixture(root)
            with patch('ceac_walkthrough.mechanisms.ROOT',root):result=reconstruct_cell(folder)
            self.assertEqual(result['checked_actor_policy_ids'],4)
            self.assertEqual(result['checked_replay_scores'],2)
            self.assertEqual(result['first_actor_episode'],2)
            self.assertEqual(result['trace'][-1]['behaviors'],{'statistical':2,'actor':4})
            self.assertAlmostEqual(result['trace'][-1]['logit_contrast']['risky_vs_safe'],actor.logits[0,1]-actor.logits[0,0])
            # Even a coherently rehashed changed policy is rejected by reconstruction.
            with gzip.open(folder/'episodes.jsonl.gz','rt') as f:lines=[json.loads(line) for line in f]
            lines[3]['policy_id']='wrong-policy'
            with gzip.open(folder/'episodes.jsonl.gz','wt') as f:
                for line in lines:f.write(json.dumps(line)+'\n')
            state=json.loads((folder/'status.json').read_text());state['artifacts']['episodes.jsonl.gz']=digest(folder/'episodes.jsonl.gz')
            (folder/'status.json').write_text(json.dumps(state))
            with patch('ceac_walkthrough.mechanisms.ROOT',root),self.assertRaisesRegex(ValueError,'diverged'):
                reconstruct_cell(folder)

    def test_raw_and_portable_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder,_=self.fixture(root)
            (folder/'summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'input mismatch'):reconstruct_cell(folder)
            (root/'updates.json.gz').write_bytes(b'changed')
            (root/'manifest.json').write_text(json.dumps({'files':{'updates.json.gz':'wrong'}}))
            with self.assertRaisesRegex(ValueError,'evidence mismatch'):load_mechanisms(root)
            (root/'manifest.json').write_text(json.dumps({'files':{'../escape':'wrong'}}))
            with self.assertRaisesRegex(ValueError,'leaves export'):load_mechanisms(root)

    def test_fans_have_one_method_and_all_doses_on_a_shared_time_scale(self):
        import matplotlib.pyplot as plt
        runs=[]
        for kind in ('undertow_junction','false_beacon_junction'):
            for method in ('idac','ucb_q','pomd','ceac'):
                for dose in (0,4,16,64):
                    for seed in (0,1,2):
                        runs.append(dict(kind=kind,task=1,method=method,dose=dose,seed=seed,
                            trace=[dict(interactions=t,cumulative_regret=(dose*t+seed)*(3 if method=='pomd' else 1)) for t in (2,100,200)],
                            summary={'cumulative_regret':dose*200+seed}))
        bundle={'runs':runs,'registration':{'protocol':'burden'}}
        fig=draw_failure_fans(bundle)
        self.assertEqual(len(fig.axes),8)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines),4);self.assertEqual(len(ax.collections),4)
            for line in ax.lines:np.testing.assert_array_equal(line.get_xdata(),[.002,.1,.2])
            self.assertEqual([line.get_label() for line in ax.lines],['B = 0','B = 4','B = 16','B = 64'])
            self.assertEqual(ax.get_ylim(),fig.axes[0].get_ylim())
            for collection in ax.collections:
                self.assertLessEqual(max(path.vertices[:,1].max() for path in collection.get_paths()),ax.get_ylim()[1])
        plt.close(fig)

    def test_seed_is_replication_unit_and_misaligned_time_fails(self):
        cells=[dict(trace=[dict(interactions=2,value=v),dict(interactions=4,value=v)]) for v in (1,2,3)]
        x,y,sd=seed_curve(cells,lambda t:t['value'])
        np.testing.assert_array_equal(y,[2,2]);np.testing.assert_array_equal(sd,[1,1])
        cells[1]['trace'][0]['interactions']=3
        with self.assertRaisesRegex(ValueError,'Unequal'):seed_curve(cells,lambda t:t['value'])

if __name__=='__main__':unittest.main()
