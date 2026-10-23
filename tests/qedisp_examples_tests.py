"""Input contracts for external examples; no QEDispInv checkout is required."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_qedisp_examples import model_table, data_table, check_queries
from benchmark_crust_depth import variants
from benchmark_crust_reference import reference_config, depth_rms
from benchmark_crust_regularization import sweep_configs, summarize_case, compare_weights, fd_passed, metric_range


class ExternalContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'input.txt'

    def test_model_units_order_and_duplicate_display_index(self):
        self.path.write_text('7 0 1.8 .4 .8\n7 .01 2.1 .2 .5\n')
        model = model_table(self.path)
        self.assertEqual(model, dict(depth=[0., .01], vs=[.4, .2], vp=[.8, .5], rho=[1.8, 2.1]))
        # Low velocity layers are allowed; crossing interfaces are not.
        self.path.write_text('0 0 1.8 .4 .8\n1 0 2.1 .2 .5\n')
        with self.assertRaises(ValueError):
            model_table(self.path)

    def test_errors_are_explicit_and_input_order_is_preserved(self):
        original = np.array([[12., .3, 2], [6., .4, 0], [12., .3, 2]])
        np.savetxt(self.path, original)
        for sigma in [None, 0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                data_table(self.path, sigma)
        converted = data_table(self.path, .005)
        np.testing.assert_array_equal(converted[:, :3], original)
        np.testing.assert_array_equal(converted[:, 3], [.005]*3)
        converted[:, 3] = [.001, .002, .003]
        np.savetxt(self.path, converted)
        np.testing.assert_array_equal(data_table(self.path, .1), converted)

    def test_fractional_mode_rejected(self):
        self.path.write_text('12 .3 1.5\n')
        with self.assertRaises(ValueError):
            data_table(self.path, .005)

    def test_no_dropped_or_reordered_forward_rows(self):
        rows = [['0', '6', '1'], ['1', '12', '0']]
        check_queries(rows, [6., 12.], [1, 0])
        for bad in [rows[:1], rows[::-1], [['0', '6', '0'], rows[1]]]:
            with self.assertRaises(ValueError):
                check_queries(bad, [6., 12.], [1, 0])

    def test_depth_controls_preserve_common_inputs(self):
        doc = dict(model=dict(depth=[0., 100., 200.], vs=[3., 4., 4.], vp=[6., 7., 7.], rho=[2., 3., 3.]),
            inversion=dict(lower_vs=[2., 3., 3.], upper_vs=[4., 5., 5.],
                           starts_vs=[[2.8, 3.8, 4.2], [3.1, 4.1, 3.9]], max_iterations=200))
        controls = variants(doc, 100.)
        self.assertEqual(controls['A_free']['inversion']['starts_vs'], doc['inversion']['starts_vs'])
        self.assertEqual(controls['B_fixed_halfspace']['inversion']['starts_vs'], [[2.8, 3.8, 4.], [3.1, 4.1, 4.]])
        self.assertEqual(controls['C_merged_tail']['inversion']['starts_vs'], [[2.8, 4.], [3.1, 4.]])
        self.assertEqual(controls['C_merged_tail']['model']['depth'], [0., 100.])
        self.assertEqual(doc['inversion']['starts_vs'][0], [2.8, 3.8, 4.2])
        for c in controls.values():
            self.assertEqual(c['inversion']['max_iterations'], 200)
        doc['model']['vs'][-1] = 4.01
        with self.assertRaises(ValueError):
            variants(doc, 100.)

    def test_reference_case_free_halfspace_and_depth_based_starts(self):
        model=dict(depth=[0.,1.,25.,200.],vs=[3.,3.1,4.,4.5],vp=[5.5,5.6,7.,8.],rho=[2.5,2.6,3.,3.2])
        config,labels=reference_config(model,.5,6.,1.2,.2,'reference_damping',100.,200)
        self.assertEqual(config['model'],model)
        self.assertEqual(config['inversion']['fixed_vs_layers'],[])
        self.assertEqual(config['inversion']['starts_vs'][0],model['vs'])
        self.assertEqual(labels,['reference','uniform_minus','uniform_plus','depth_sine','depth_cosine'])
        np.testing.assert_allclose(np.array(config['inversion']['starts_vs'][1])-model['vs'],-.2,atol=1e-15)
        # Quarter-period in physical depth, regardless of how many layers precede it.
        self.assertAlmostEqual(config['inversion']['starts_vs'][3][2],4.2)
        self.assertEqual(config['inversion']['starts_vs'][2][-1],4.7)
        self.assertNotIn('lower_vs',config['inversion'])
        self.assertNotIn('upper_vs',config['inversion'])
        with self.assertRaises(ValueError):
            reference_config(model,.5,6.,.2,.2,'reference_damping',100.,200)

    def test_structure_comparison_weights_depth_not_layer_count(self):
        # Integral over [0,4]: 1 km * 1^2 + 3 km * 2^2; halfspace excluded.
        self.assertAlmostEqual(depth_rms([1.,2.,20.],[0.,1.,4.],0.,4.),np.sqrt(13/4))
        self.assertAlmostEqual(depth_rms([1.,2.,20.],[0.,1.,4.],.5,2.),np.sqrt(4.5/1.5))
        self.assertAlmostEqual(depth_rms([1.,2.,20.],[0.,1.,4.],4.,5.),20.)

    def test_weight_scan_changes_only_lambda_including_zero(self):
        model=dict(depth=[0.,1.,4.],vs=[3.,3.5,4.],vp=[5.5,6.,7.],rho=[2.5,2.6,3.])
        original=copy.deepcopy(model)
        configs,labels=sweep_configs(model,[0.,10.,30.,100.,300.])
        self.assertEqual(model,original)
        for config,weight in zip(configs,[0.,10.,30.,100.,300.]):
            self.assertEqual(config['inversion']['regularization'],dict(type='reference_damping',**{'lambda':weight}))
            expected=copy.deepcopy(configs[0]);expected['inversion']['regularization']['lambda']=weight
            self.assertEqual(config,expected)
            self.assertEqual(len(config['inversion']['starts_vs']),len(labels))
        # Ensure this is also accepted by the actual per-weight runner's builder.
        zero,_=reference_config(model,.5,6.,1.2,.2,'reference_damping',0.,200)
        self.assertEqual(zero,configs[0])
        configs[0]['model']['vs'][0]=99
        self.assertEqual(configs[1]['model']['vs'][0],3.)
        for bad in [[],[0,10],[100,100],[-1,100],[float('nan'),100],[100,float('inf')]]:
            with self.subTest(weights=bad),self.assertRaises(ValueError):sweep_configs(model,bad)

    def test_scan_preserves_failed_runs_and_checks_both_fd_steps(self):
        model=dict(depth=[0.,1.,4.],vs=[3.,3.5,4.],vp=[5.5,6.,7.],rho=[2.5,2.6,3.])
        configs,_=sweep_configs(model,[0.,100.])
        audit=dict(invalid_rows=[],unavailable_gradients=[],fd_passed_at_1e_6=True,
                   steps=[dict(h=h,unresolved_fd_entries=0,max_scaled_error=.1) for h in [1e-5,1e-6]])
        records=[dict(run=i,converged=i!=1,audit=copy.deepcopy(audit)) for i in range(5)]
        records[2]['audit']['steps'][0]['max_scaled_error']=2.  # Fine step alone cannot certify.
        records[3]['audit']['steps'][0]['unresolved_fd_entries']=1
        records[4]['audit']['steps']=[]
        profiles={i:np.array(model['vs'])+np.array([.1,.2,.3]) for i in range(5)}
        group=summarize_case(configs[0],records,profiles)
        self.assertEqual(len(group['runs']),5)
        self.assertEqual(group['converged_count'],4)
        self.assertEqual(group['fd_passed_count'],2)
        self.assertEqual(group['validated_count'],1)
        self.assertIsNone(group['within_weight_max_layer_spread_km_s'])
        self.assertFalse(group['runs'][1]['validated'])
        self.assertEqual(group['runs'][2]['audit'],records[2]['audit'])
        r=group['runs'][0]
        self.assertAlmostEqual(r['reference_depth_rms'][0]['rms_km_s'],np.sqrt((.1**2+3*.2**2)/4))
        self.assertAlmostEqual(r['halfspace_reference_change_km_s'],.3)
        for broken in [{},dict(audit,invalid_rows=[0]),dict(audit,unavailable_gradients=[1])]:
            self.assertFalse(fd_passed(broken))
        with self.assertRaises(ValueError):summarize_case(configs[0],records[:-1],profiles)
        with self.assertRaises(ValueError):summarize_case(configs[0],records[:-1]+[records[0]],profiles)
        reference_profiles={i:np.array(model['vs']) for i in range(5)}
        baseline=summarize_case(configs[1],records,reference_profiles)
        comparison=compare_weights([group,baseline],model)
        self.assertEqual(len(comparison[0]['pairs']),1)
        self.assertAlmostEqual(comparison[0]['pairs'][0]['max_layer_difference_km_s'],.3)
        for r in group['runs']:r['validated']=False
        comparison=compare_weights([group,baseline],model)
        self.assertFalse(comparison[0]['available'])
        self.assertEqual(comparison[0]['pairs'],[])
        for values in [[.01,None],[.01,float('nan')],[.01,float('inf')]]:
            self.assertIn('未解析',metric_range(values))


if __name__ == '__main__':
    unittest.main()
