"""Reporting tests protect the unit of replication and intervention grid."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from ceac_hardness.report import endpoints,draw_learning,draw_response,load_export

class ReportingTests(unittest.TestCase):
    def bundle(self):
        runs=[]
        for method in ('idac','ucb_q','pomd','ids_product','ceac','oracle'):
            for dose in (0,4,16,64):
                for seed in (0,1,2):
                    gap=.1*seed+(.2 if dose else 0)
                    summary=dict(late_gap=gap,cumulative_regret=10*gap,events={'nuisance':dose,'probe':1},
                        settings={'episodes':100},actor_updates=0,priced_actor_updates=0,
                        task1_posterior=1.,prior_contains_true_joint_model=True)
                    trace=[dict(interactions=x,window_gap=gap,cumulative_regret=gap*x/2) for x in (2,100,200)]
                    runs.append(dict(protocol='burden',kind='undertow_junction',task=1,method=method,dose=dose,seed=seed,summary=summary,trace=trace))
        return {'runs':runs,'registration':{'protocol':'burden'}}

    def test_error_bars_use_seed_sd_and_keep_all_algorithm_panels(self):
        import matplotlib.pyplot as plt
        bundle=self.bundle();rows=endpoints(bundle)
        assert len(rows)==24
        for row in rows:self.assertAlmostEqual(row['late_gap_sd'],.1)
        fig=draw_learning(bundle,'undertow_junction')
        self.assertEqual(len(fig.axes),6)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines),4);self.assertEqual(len(ax.collections),4)
            self.assertEqual(len(ax.texts),1)
        plt.close(fig)

    def test_main_response_uses_pathology_on_x_with_all_six_methods(self):
        import matplotlib.pyplot as plt
        fig=draw_response(self.bundle())
        self.assertEqual(len(fig.axes[0].lines),6)
        for line in fig.axes[0].lines:np.testing.assert_array_equal(line.get_xdata(),[0,4,16,64])
        self.assertTrue(any('CEAC' in line.get_label() for line in fig.axes[0].lines))
        plt.close(fig)

    def test_unequal_time_grids_fail_instead_of_interpolating(self):
        bundle=self.bundle();bundle['runs'][1]['trace'][-1]['interactions']=201
        with self.assertRaisesRegex(ValueError,'Unequal trace grids'):draw_learning(bundle,'undertow_junction')
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_export_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'manifest.json').write_text(json.dumps({'files':{'../secret':'bad'}}))
            with self.assertRaisesRegex(ValueError,'leaves export'):load_export(root)

if __name__=='__main__':unittest.main()
