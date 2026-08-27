# SPDX-License-Identifier: GPL-3.0-or-later
"""Byte-pinned golden renders for whole-image regression."""

from __future__ import annotations

import os
import unittest

import numpy as np

from dngscan.render import render_output_u8
from tests.golden_support import (
    GoldenCase,
    all_scenes,
    iter_cases,
    encoded_lab_stats,
    render_plan_for_case,
)


def _stats_match(
    expected: dict[str, dict[str, float]],
    actual: dict[str, dict[str, float]],
    *,
    atol: float = 1e-3,
) -> bool:
    for roi, exp_stats in expected.items():
        act_stats = actual.get(roi, {})
        for key, exp_val in exp_stats.items():
            act_val = float(act_stats.get(key, float("nan")))
            # inverted comparison: a missing key yields NaN, and NaN must
            # FAIL the match, not slip through a False `>` comparison
            if not abs(act_val - float(exp_val)) <= atol:
                return False
    return True


def _format_stats_delta(
    expected: dict[str, dict[str, float]],
    actual: dict[str, dict[str, float]],
) -> str:
    lines = ["Perceptual delta (Tier 2):"]
    for roi, exp_stats in expected.items():
        act_stats = actual.get(roi, {})
        parts = []
        for key, exp_val in exp_stats.items():
            act_val = float(act_stats.get(key, float("nan")))
            parts.append(f"{key} {exp_val:.4f}->{act_val:.4f}")
        lines.append(f"  {roi}: " + ", ".join(parts))
    return "\n".join(lines)


class GoldenRenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fast_env = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "0"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fast_env is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = cls._fast_env

    def test_golden_fixtures_present(self) -> None:
        missing = [case.fixture_name for case in iter_cases() if not case.fixture_path.is_file()]
        if missing:
            self.fail(
                "Missing golden fixtures. Run: .venv/bin/python tools/regen_golden.py\n"
                + "\n".join(f"  - {name}" for name in missing[:8])
                + (f"\n  ... and {len(missing) - 8} more" if len(missing) > 8 else "")
            )

    def test_golden_byte_parity(self) -> None:
        import json

        scenes = all_scenes()
        for case in iter_cases():
            with self.subTest(case=case.fixture_name):
                # allow_pickle stays False: fixtures come in via PRs, and a pickled
                # object array would execute on whoever runs the suite.
                fixture = np.load(case.fixture_path, allow_pickle=False)
                expected = fixture["u8"]
                scene = scenes[case.scene_id]
                plan = render_plan_for_case(scene, case)
                actual = render_output_u8(
                    scene.bundle,
                    scene.analysis,
                    "srgb",
                    plan,
                    tone_core=case.tone_core,
                    agx_primaries=case.agx_primaries if case.tone_core == "agx" else "base",
                ).reshape(expected.shape)
                if np.array_equal(actual, expected):
                    continue
                exp_stats = json.loads(str(fixture["stats"])) if "stats" in fixture else {}
                act_stats = {name: encoded_lab_stats(actual, mask) for name, mask in scene.rois.items()}
                diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                max_delta = int(diff.max())
                # Cross-platform libm can move a handful of pixels by one dither LSB while
                # Oklab scene stats stay unchanged; reject only real regressions. The
                # escape is bounded in pixel count too, so a broad 1-LSB drift (which a
                # per-ROI mean could dilute) still fails.
                differing_frac = float(np.count_nonzero(diff)) / float(diff.size)
                if max_delta <= 1 and differing_frac <= 0.01 and _stats_match(exp_stats, act_stats):
                    continue
                msg = _format_stats_delta(exp_stats, act_stats)
                msg = (
                    f"{case.fixture_name}: max byte delta {max_delta}, "
                    f"{int(np.count_nonzero(diff))} px differ\n{msg}"
                )
                self.fail(msg)


if __name__ == "__main__":
    unittest.main()
