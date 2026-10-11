# SPDX-License-Identifier: GPL-3.0-or-later
"""Counterexamples for the precision evaluator, not camera quality assertions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('precision_budget_tool',
    Path(__file__).resolve().parents[1]/'tools/validate_sensor_precision.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class PrecisionBudgetTests(unittest.TestCase):
    def test_local_error_cannot_disappear_in_zero_mean(self):
        reference=np.ones((512,512,3),np.float32)
        actual=reference.copy();actual[40:48,20:28,0]=2;actual[40:48,20:28,1]=0
        result=tool.precision_error(reference,actual)
        self.assertEqual(result['bias'],0)
        self.assertEqual(result['maximum_absolute_error'],1)
        self.assertGreater(result['rms'],0)

    def test_half_bound_includes_negative_and_subnormal_values(self):
        x=np.r_[np.linspace(-2,8,65536),np.linspace(-2.**-14,2.**-14,4096)]
        result=tool.half_rounding_audit(x)
        self.assertTrue(result['within_rounding_bound'])
        self.assertEqual(result['out_of_finite_half_range'],0)

    def test_renderer_change_is_not_licensed_as_half_rounding(self):
        reference=np.array([0.,.00001,-.2,.5,2.],np.float32)
        actual=reference.astype(np.float16);actual[3]=.49
        result=tool.half_rounding_audit(reference,actual)
        self.assertFalse(result['within_rounding_bound'])
        self.assertEqual(result['bound_violations'],1)

    def test_overflow_is_separate_from_rounding(self):
        result=tool.half_rounding_audit(np.array([-70000.,0.,70000.]))
        self.assertEqual(result['checked_values'],1)
        self.assertEqual(result['out_of_finite_half_range'],2)
        self.assertTrue(result['within_rounding_bound'])

    def test_nonfinite_pairs_are_reported(self):
        result=tool.precision_error(np.array([1.,2.,3.]),np.array([1.,np.nan,np.inf]))
        self.assertEqual(result['nonfinite_pairs'],2)
        self.assertEqual(result['finite_pairs'],1)

    def test_same_formula_isolates_clipping_from_storage(self):
        result=tool.production_boundary_experiments(32)
        self.assertTrue(result['spatial_black']['same_formula_float64_to_uint16']['budget_passed'])
        self.assertGreater(result['spatial_black']['lower_clipped_fraction'],0)
        self.assertGreater(result['spatial_black']['clipping_policy_error_separate']['maximum_absolute_error'],.5)
        self.assertTrue(result['late_warp']['same_kernel_single_storage']['budget_passed'])

    def test_raw_output_conflict_precedes_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.dng';source.write_bytes(b'private original')
            from dngscan.protected_output import SourceOutputConflictError
            with patch.object(tool.sys,'argv',['validate_sensor_precision.py','--output',str(source),'--apple-raw',str(source)]), \
                 patch.object(tool,'black_experiments') as run:
                with self.assertRaises(SourceOutputConflictError):tool.main()
                run.assert_not_called()
            self.assertEqual(source.read_bytes(),b'private original')

    def test_quarter_midpoint_negative_and_directed_rounding_are_distinct(self):
        unit = 2.**-10
        reference = np.array([1.+unit*f for f in (.25,.5,.75,1.5)] +
                             [-1.-unit*f for f in (.25,.5,.75,1.5)],np.float32)
        # The midpoint with an odd lower mantissa rounds up to its even neighbour.
        expected = {'nearest_even': [1.,1.,1.+unit,1.+2*unit,-1.,-1.,-1.-unit,-1.-2*unit],
                    'toward_zero': [1.,1.,1.,1.+unit,-1.,-1.,-1.,-1.-unit],
                    'toward_negative': [1.,1.,1.,1.+unit,-1.-unit,-1.-unit,-1.-unit,-1.-2*unit],
                    'toward_positive': [1.+unit,1.+unit,1.+unit,1.+2*unit,-1.,-1.,-1.,-1.-unit]}
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                actual=np.array(values,np.float16)
                result=tool.half_conversion_diagnostics(reference,actual,chunk_values=3)
                group=result['groups']['finite_in_half_range']
                self.assertEqual(group['exact_bit_match_counts'][mode],8)
                self.assertEqual(group['unmatched_any_rounding_mode'],0)
                self.assertEqual(group['ulp_error_over_one'],0)
        actual=np.array(expected['toward_zero'],np.float16)
        # Recognizing RTZ must not turn the old nearest-even budget green.
        self.assertFalse(tool.half_rounding_audit(reference,actual)['within_rounding_bound'])
        self.assertEqual(tool.half_rounding_audit(reference,actual)['assumed_rounding_mode'],'nearest-even')
        result=tool.half_conversion_diagnostics(reference,actual)
        self.assertEqual(result['groups']['positive_reference']['positive_error_count'],0)
        self.assertEqual(result['groups']['negative_reference']['negative_error_count'],0)

    def test_subnormal_rounding_and_extra_flushing_are_different(self):
        unit=2.**-24
        positive=np.array([.25,.5,.75,1.,1.75,2.],np.float32)*unit
        half_positive=np.array([0.,0.,0.,unit,unit,2*unit],np.float16)
        reference=np.r_[positive,-positive]
        actual=np.r_[half_positive,-half_positive]
        result=tool.half_conversion_diagnostics(reference,actual)
        group=result['groups']['subnormal_reference']
        self.assertEqual(group['exact_bit_match_counts']['toward_zero'],12)
        self.assertEqual(group['actual_positive_zero'],3)
        self.assertEqual(group['actual_negative_zero'],3)
        self.assertEqual(group['actual_subnormal_values'],6)
        result=tool.half_conversion_diagnostics(np.array([2*unit,-2*unit],np.float32),np.array([0.,-0.],np.float16))
        group=result['groups']['subnormal_reference']
        self.assertEqual(group['unmatched_any_rounding_mode'],2)
        self.assertEqual(group['ulp_error_over_one'],2)
        self.assertEqual(group['maximum_absolute_ulp_error'],2.)

    def test_large_change_and_nonfinite_output_do_not_pass_as_rounding(self):
        reference=np.array([1.+2.**-12,2.,70000.,np.inf,np.nan],np.float32)
        actual=np.array([1.+4*2.**-10,np.nan,np.inf,np.inf,np.nan],np.float16)
        result=tool.half_conversion_diagnostics(reference,actual)
        group=result['groups']['finite_in_half_range']
        self.assertEqual(group['unmatched_any_rounding_mode'],2)
        self.assertEqual(group['ulp_error_over_one'],1)
        self.assertEqual(group['nonfinite_actual'],1)
        self.assertEqual(result['special']['finite_reference_outside_half_range'],1)
        self.assertEqual(result['special']['reference_nan'],1)
        self.assertEqual(result['special']['reference_infinite'],1)

    def test_signed_zero_and_chunking_preserve_exact_classification(self):
        reference=np.array([[0.,-0.,1.000732421875],[-1.000732421875,2.**-25,-2.**-25]],np.float32).T
        actual=reference.astype(np.float16)
        self.assertEqual(tool.half_conversion_diagnostics(reference,actual,chunk_values=1),
                         tool.half_conversion_diagnostics(reference,actual,chunk_values=64))
        actual[0,0]=np.float16(-0.)
        result=tool.half_conversion_diagnostics(reference,actual)
        self.assertEqual(result['groups']['zero_reference']['unmatched_any_rounding_mode'],1)

    def test_coreimage_entry_retains_rgba_and_separate_nearest_budget(self):
        from dngscan import coreimage_decode as ci
        image=SimpleNamespace(extent=lambda:SimpleNamespace(origin=SimpleNamespace(x=3.,y=5.),
                                                            size=SimpleNamespace(width=8.,height=8.)))
        floating=np.full((8,8,4),1.+.75*2.**-10,np.float32);floating[...,3]=.75
        half=np.ones((8,8,4),np.float16);half[...,3]=.75
        formats=[]
        class Context:
            def render_toBitmap_rowBytes_bounds_format_colorSpace_(self, source, data, stride, bounds, fmt, color):
                self_outer.assertIs(source,image)
                self_outer.assertEqual(bounds,(3.,5.,8,8))
                formats.append(fmt)
                data[:]=(floating if fmt=='RGBAf' else half).tobytes()
        self_outer=self
        quartz=SimpleNamespace(kCGColorSpaceExtendedLinearITUR_2020='linear2020',
            CGColorSpaceCreateWithName=lambda name:name, CGRectMake=lambda *values:values,
            kCIFormatRGBAf='RGBAf',kCIFormatRGBAh='RGBAh')
        config={'actual_decoder_token':'9','scale_factor':1.,'controls':{}}
        with patch.object(ci,'_recipe_source_identity',return_value={'status':'hashed','sha256':'synthetic'}), \
             patch.object(ci,'available',return_value=True),patch.object(ci,'supported_versions',return_value=('9',)), \
             patch.object(ci,'_open_filter',return_value=SimpleNamespace(outputImage=lambda:image)), \
             patch.object(ci,'configure_linear_filter',return_value=config), \
             patch.object(ci,'_scene_input_qualification',return_value={'status':'qualified'}), \
             patch.object(ci,'_require_quartz',return_value=quartz),patch.object(ci,'_render_context',return_value=Context()), \
             patch.object(ci,'decoder_runtime_id',return_value='synthetic-runtime'), \
             patch.object(ci,'decoder_hardware_id',return_value={'status':'synthetic'}), \
             patch.object(ci,'_assert_recipe_source_current') as source_guard:
            result=tool.coreimage_precision_experiment(Path('synthetic.dng'),'9',full=True)
        self.assertEqual(formats,['RGBAf','RGBAh','RGBAf'])
        self.assertFalse(result['bitmap_rounding']['within_rounding_bound'])
        self.assertEqual(result['bitmap_conversion_modes']['groups']['finite_in_half_range']['exact_bit_match_counts']['toward_zero'],192)
        rgba=result['rgba_readback']
        self.assertEqual(rgba['shape'],[8,8,4])
        self.assertTrue(rgba['float_repeat_bitwise_equal'])
        self.assertEqual(rgba['alpha']['float_minmax'],[.75,.75])
        self.assertEqual(rgba['alpha']['float_not_one_count'],64)
        self.assertTrue(all(row['gain_rounding']['within_rounding_bound'] and
                            row['resize_storage_rounding']['within_rounding_bound'] for row in result['later_half_boundaries']))
        source_guard.assert_called_once()
