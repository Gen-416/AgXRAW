# SPDX-License-Identifier: GPL-3.0-or-later
"""External review batch 25 (2026-09-02 handoff, R-P2-6): the native HDR
kernel's output stage carried float32 matrices while its header claimed to
match NumPy's operation order (NumPy accumulates the two rec2020->output
stages in float64 and materializes float32 per stage — the contract the SDR
output plan has carried since ABI v8). ABI v10 gives NativeHdrPlan the same
exact float64 two-stage matrices.

Measured on test_hdr_native's ten 60k-pixel sweeps: max |native - numpy|
8.46e-5 -> 8.27e-5, p99 5.2e-6 unchanged, bit-identical share 21.7% -> 21.9%.
The stage was one of several float32 residual sources, not the dominant one.
Math review 2026-09-03 (ABI v11) then converted the remaining float64-matrix
stages (inset/outset, punch/Oklab) in both kernels: HDR 2.36e-5 max / 2.0e-6
p99 / 81% bit-identical, SDR 47% -> 92% bit-identical; gates tightened
accordingly (see test_hdr_native / test_fast_backend).
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HdrOutputStageIsFloat64(unittest.TestCase):
    def test_abi_v10_on_both_sides(self) -> None:
        from dngscan.fast_plan import NATIVE_ABI_VERSION

        self.assertGreaterEqual(NATIVE_ABI_VERSION, 10)
        header = (ROOT / "cpp" / "include" / "dngscan_fast" / "agx_core.h").read_text(encoding="utf-8")
        self.assertRegex(header, r"NATIVE_ABI_VERSION = 1[01];")

    def test_kernel_reads_exact_float64_stages(self) -> None:
        hdr_h = (ROOT / "cpp" / "include" / "dngscan_fast" / "hdr_core.h").read_text(encoding="utf-8")
        self.assertIn("double rec2020_to_xyz[9];", hdr_h)
        self.assertIn("double xyz_to_output[9];", hdr_h)
        self.assertNotIn("matching the\n  // NumPy", hdr_h)
        kernel = (ROOT / "cpp" / "src" / "hdr_core.cpp").read_text(encoding="utf-8")
        self.assertIn("mat3_exact_f64(plan.rec2020_to_xyz, mapped)", kernel)
        self.assertIn("mat3_exact_f64(plan.xyz_to_output, xyz)", kernel)
        bindings = (ROOT / "cpp" / "src" / "bindings.cpp").read_text(encoding="utf-8")
        self.assertIn('copy_matrix9_f64(obj.attr("rec2020_to_xyz"), plan.rec2020_to_xyz)', bindings)
        self.assertIn('copy_matrix9_f64(obj.attr("xyz_to_output"), plan.xyz_to_output)', bindings)
        # one shared definition of the exact stage, used by both kernels
        shared = (ROOT / "cpp" / "include" / "dngscan_fast" / "exact_matrix.h").read_text(encoding="utf-8")
        self.assertIn("inline Rgb mat3_exact_f64(const double matrix[9], const Rgb& value)", shared)
        output = (ROOT / "cpp" / "src" / "output_core.cpp").read_text(encoding="utf-8")
        self.assertNotIn("inline Rgb mat3_exact_f64", output)
        self.assertIn('#include "dngscan_fast/exact_matrix.h"', output)

    def test_built_extension_matches_when_present(self) -> None:
        from dngscan import _fast as fast_backend

        if not fast_backend.available():
            self.skipTest("native extension not built")
        from dngscan import _dngscan_fast as ext

        self.assertGreaterEqual(int(ext.native_abi_version()), 10)


if __name__ == "__main__":
    unittest.main()
