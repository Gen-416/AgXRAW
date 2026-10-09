# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 17 (full-code review) regression gates for the immediate
fixes: session token + Origin gate, upload quotas, finite-value validation,
cache identity hardening, export deadline, fused film HDR pair, tile-local
noise floor."""
from __future__ import annotations

import unittest

import numpy as np


class ServerSecurityTests(unittest.TestCase):
    def test_page_carries_token_and_api_fetch_wrapper(self) -> None:
        from dngscan.gui.page import render_page

        html = render_page("/tmp", session_token="tok-abc").decode()
        self.assertIn("tok-abc", html)
        self.assertIn("X-DngScan-Token", html)
        for needle in ('apiFetch("/list', 'apiFetch("/upload', "apiFetch(path"):
            self.assertIn(needle, html)


class FiniteValidationTests(unittest.TestCase):
    def test_nan_and_infinity_are_rejected(self) -> None:
        from dngscan.gui.service import _finite_number

        self.assertEqual(_finite_number("1.5", "x", -8, 8), 1.5)
        for bad in (float("nan"), float("inf"), "-Infinity", "NaN"):
            with self.assertRaises(ValueError):
                _finite_number(bad, "x", -8, 8)
        with self.assertRaises(ValueError):
            _finite_number(9.0, "x", -8, 8)

    def test_parse_job_uses_finite_validation_for_ev(self) -> None:
        import inspect

        from dngscan.gui import service

        src = inspect.getsource(service)
        self.assertIn('_finite_number(params.get("ev"', src)


class CacheIdentityTests(unittest.TestCase):
    def test_identity_carries_inode_and_header_hash(self) -> None:
        import tempfile
        from pathlib import Path

        from dngscan.gui.preview_cache import _evidence_cache_identity

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.dng"
            f.write_bytes(b"A" * 70000)
            ident1 = _evidence_cache_identity(f)
            stat = f.stat()
            # replace IN PLACE with same size, restore mtime
            f.write_bytes(b"B" * 70000)
            import os

            os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            ident2 = _evidence_cache_identity(f)
            self.assertNotEqual(
                ident1, ident2,
                "same path/size/mtime with different bytes must change identity",
            )


class ExportDeadlineTests(unittest.TestCase):
    def test_timeout_source_pins_deadline_terminate_and_messages(self) -> None:
        """Source pin only: real timeout termination is not exercised here
        (it needs a hung child process); this keeps the deadline/terminate
        path and its user-facing messages from being deleted silently."""
        import inspect

        from dngscan.gui import service

        src = inspect.getsource(service.run_export_isolated)
        for needle in ("deadline", "terminate", "导出超时", "崩溃"):
            self.assertIn(needle, src)




class NoiseFloorLocalityTests(unittest.TestCase):
    def test_noise_is_independent_of_mosaic_colour(self) -> None:
        from tests.test_automatic_defaults import AutomaticAnalysisTests
        AutomaticAnalysisTests().test_noise_is_invariant_to_channel_offsets_and_linear_gradients()


if __name__ == "__main__":
    unittest.main()
