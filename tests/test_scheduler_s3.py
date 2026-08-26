# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduler plan S3 acceptance gates: one CPU budget owner, rented down
the stack instead of each layer claiming the machine.

The headline gate is measured, not asserted from source: a render in a
CHILD process is sampled by the parent (a same-process sampler is starved
by the GIL and silently reports a false low peak), and the peak OS thread
count must stay inside the machine budget plus a small constant.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BudgetArithmeticTests(unittest.TestCase):
    def test_split_follows_who_does_the_work(self) -> None:
        from dngscan.cpu_budget import TOTAL, split_for

        native_outer, native_inner = split_for(True)
        numpy_outer, numpy_inner = split_for(False)
        # native-core renders: few outer workers, wide native budget
        self.assertLessEqual(native_outer, 3)
        self.assertGreaterEqual(native_inner, 2)
        # NumPy-core renders: the outer pipeline is what scales
        self.assertGreaterEqual(numpy_outer, min(4, max(2, TOTAL - 2)))
        # both stay inside the machine plus a small constant
        for outer, inner in ((native_outer, native_inner), (numpy_outer, numpy_inner)):
            self.assertLessEqual(outer * inner, TOTAL + max(2, TOTAL // 2))

    def test_inner_budget_is_thread_local_and_restored(self) -> None:
        from dngscan.cpu_budget import TOTAL, current_inner, inner

        self.assertEqual(current_inner(), TOTAL)
        seen: list[int] = []
        with inner(2):
            self.assertEqual(current_inner(), 2)
            t = threading.Thread(target=lambda: seen.append(current_inner()))
            t.start(); t.join()
        self.assertEqual(current_inner(), TOTAL)
        self.assertEqual(seen, [TOTAL], "a sibling thread keeps the machine")

    def test_native_claims_nest_tightest_wins(self) -> None:
        from unittest import mock

        from dngscan import cpu_budget

        published: list[int] = []
        with mock.patch(
            "dngscan._fast.set_thread_budget", published.append
        ):
            with cpu_budget.native_budget(4):
                with cpu_budget.native_budget(2):
                    pass
        # 4 claimed, 2 tightens, releasing 2 restores 4, releasing 4 -> 0
        self.assertEqual(published, [4, 2, 4, 0])


class OperatorBypassTests(unittest.TestCase):
    def test_operators_go_serial_when_their_share_is_one(self) -> None:
        import inspect

        from dngscan import gated_drt, scene_transform

        self.assertIn("current_inner()", inspect.getsource(gated_drt))
        self.assertIn("current_inner()", inspect.getsource(scene_transform))

    def test_gated_core_bypasses_its_pool_under_a_share_of_one(self) -> None:
        from unittest import mock

        from dngscan import gated_drt

        submits: list[int] = []
        real_submit = gated_drt._GATED_POOL.submit

        def spy(*a, **k):
            submits.append(1)
            return real_submit(*a, **k)

        import numpy as np

        from tests.golden_support import build_daylight_wide_dr
        from dngscan.tone import build_render_plan

        scene = build_daylight_wide_dr()
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb", tone_core="gated"
        )
        rgb = np.random.default_rng(0).uniform(
            0.0, 1.0, (100_000, 3)
        ).astype(np.float32)
        from dngscan.cpu_budget import inner

        with mock.patch.object(gated_drt._GATED_POOL, "submit", spy):
            with inner(1):
                gated_drt.apply_gated_core(rgb, plan.tone, plan.color, None, None)
        self.assertEqual(
            submits, [],
            "a share of 1 must run the serial oracle, not stack a pool on "
            "top of the outer render pool",
        )


def _thread_count(pid: int) -> int:
    lib = ctypes.CDLL(
        ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
    )

    class _TI(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_uint64 * 6),
            ("b", ctypes.c_int32 * 9),
            ("pti_threadnum", ctypes.c_int32),
            ("c", ctypes.c_int32 * 2),
        ]

    ti = _TI()
    if lib.proc_pidinfo(pid, 4, 0, ctypes.byref(ti), ctypes.sizeof(ti)) <= 0:
        return 0
    return int(ti.pti_threadnum)


_CHILD = r'''
import dataclasses, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from tests.golden_support import build_daylight_wide_dr
import dngscan.render as R
from dngscan.tone import build_render_plan
scene = build_daylight_wide_dr()
b = scene.bundle
big = np.repeat(np.repeat(b.scene_rec2020_render, 24, axis=0), 24, axis=1)
b = dataclasses.replace(b, scene_rec2020_render=np.ascontiguousarray(big))
plan = build_render_plan(b, scene.analysis, "agx", "srgb")
R.render_output_u8(b, scene.analysis, "srgb", plan)
'''


@unittest.skipUnless(sys.platform == "darwin", "libproc thread sampling is macOS")
class MeasuredThreadBoundTests(unittest.TestCase):
    def test_render_thread_peak_stays_inside_the_budget(self) -> None:
        from dngscan.cpu_budget import TOTAL

        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT)),
        )
        peak = 0
        while proc.poll() is None:
            peak = max(peak, _thread_count(proc.pid))
            time.sleep(0.001)
        _, err = proc.communicate()
        self.assertEqual(proc.returncode, 0, err[-2000:])
        self.assertGreater(peak, 0, "sampler never saw the child")
        # Machine budget plus a constant for the interpreter's own threads.
        # Measured before S3 on a 10-core host: 49 threads for one render,
        # 88 for two concurrent ones.
        self.assertLessEqual(
            peak, TOTAL + 8,
            f"peak {peak} threads exceeds the CPU budget ({TOTAL}) + 8 — "
            "the outer pool, the operator pools and the native kernels are "
            "stacking their own widths again",
        )


if __name__ == "__main__":
    unittest.main()
