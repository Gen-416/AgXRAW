# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


class DebugTracebackTest(unittest.TestCase):
    def test_cli_debug_prints_traceback(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["DNGSCAN_DEBUG"] = "1"
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.run(
            [sys.executable, "-m", "dngscan", "/no/such/file.dng"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Traceback", proc.stderr)
        self.assertIn("error:", proc.stderr)

    def test_cli_without_debug_hides_traceback(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("DNGSCAN_DEBUG", None)
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.run(
            [sys.executable, "-m", "dngscan", "/no/such/file.dng"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("error:", proc.stderr)

    def test_maybe_print_exc_respects_flag(self) -> None:
        from dngscan.debug_util import maybe_print_exc
        import io

        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"DNGSCAN_DEBUG": "1"}):
            try:
                raise ValueError("boom")
            except ValueError:
                maybe_print_exc(file=buf)
        self.assertIn("ValueError", buf.getvalue())
        buf2 = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True):
            try:
                raise ValueError("boom")
            except ValueError:
                maybe_print_exc(file=buf2)
        self.assertEqual(buf2.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
