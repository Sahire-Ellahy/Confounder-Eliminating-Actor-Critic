import unittest
import numpy as np
from ceac_stability.report import paired_rows, family_summary


def example():
    rows=[]
    for seed,deltas in [(1,[1.,9.]),(2,[3.,15.])]:
        for burden,delta in zip([0,4],deltas):
            for method,value in [('original_reference',20.),('guarded_natural_003',20.+delta)]:
                rows.append(dict(protocol='burden',kind='undertow_junction',burden=burden,dose=0.,
                    task=0,seed=seed,candidate=method,cumulative_regret=value,late_gap=value/100,
                    final_actor_augmented_gap=value/1000,final_actor_constrained_augmented_gap=value/2000))
    return dict(runs=rows)


class ReportingTests(unittest.TestCase):
    def test_pair_seed_then_average_doses_before_dispersion(self):
        summary=family_summary(example())[0]
        # Seed averages are 5 and 9: mean 7, sample SD sqrt(8).
        self.assertEqual(summary['seeds'],2)
        self.assertAlmostEqual(summary['difference_cumulative_regret'],7.)
        self.assertAlmostEqual(summary['sd_difference_cumulative_regret'],np.sqrt(8))
        self.assertEqual(summary['reference_cumulative_regret'],20.)
        self.assertEqual(summary['variant_cumulative_regret'],27.)

    def test_missing_and_duplicate_conditions_fail_closed(self):
        bundle=example();bundle['runs'].pop()
        with self.assertRaises(ValueError):paired_rows(bundle)
        bundle=example();bundle['runs'].append(bundle['runs'][-1])
        with self.assertRaises(ValueError):paired_rows(bundle)
        bundle=example();bundle['runs'].append(bundle['runs'][0])
        with self.assertRaises(ValueError):paired_rows(bundle)

    def test_reference_only_primary_has_no_fabricated_comparison(self):
        bundle=example();bundle['runs']=[r for r in bundle['runs'] if r['candidate']=='original_reference']
        self.assertEqual(paired_rows(bundle),[])
        self.assertEqual(family_summary(bundle),[])


if __name__=='__main__':unittest.main()
