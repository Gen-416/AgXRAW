# SPDX-License-Identifier: GPL-3.0-or-later
"""Counterexamples for the precision evaluator, not camera quality assertions."""
import importlib.util
from pathlib import Path
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
