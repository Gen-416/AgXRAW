# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-0 SDR freeze gate for the ACES 2 HDR dual-rendition migration.

Any HDR work that changes these P3 SDR bytes, or the existing tests/golden tree hash,
must fail here. Do not regenerate fixtures to silence a later phase.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

from dngscan.tone import compute_exposure_gain, exposure_mode_for_tone_core
from tests.golden_support import all_scenes

ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "tests" / "sdr_freeze"
MANIFEST_PATH = FREEZE_DIR / "MANIFEST.json"

def _load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise unittest.SkipTest(
            f"missing {MANIFEST_PATH}; run: .venv/bin/python tools/regen_sdr_freeze.py"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class SdrFreezeGateTest(unittest.TestCase):
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

    def test_manifest_pins_golden_tree(self) -> None:
        from tools.regen_sdr_freeze import golden_tree_digest

        manifest = _load_manifest()
        actual = golden_tree_digest()
        self.assertEqual(
            actual,
            manifest["golden_tree_sha256"],
            "tests/golden tree hash drifted. HDR phases must not change SDR golden "
            "fixtures; restore them or open an explicit Phase-0 review before regenerating.",
        )
        self.assertEqual(
            int(manifest["golden_npz_count"]),
            len(list((ROOT / "tests" / "golden").glob("*.npz"))),
        )

    def test_p3_sdr_freeze_byte_parity(self) -> None:
        from tools.regen_sdr_freeze import FreezeCase, render_freeze_case

        manifest = _load_manifest()
        scenes = all_scenes()
        for entry in manifest["freeze_cases"]:
            case = FreezeCase(**entry)
            with self.subTest(case=case.stem):
                if case.scene_id not in scenes:
                    self.fail(f"freeze scene missing from golden_support: {case.scene_id}")
                if not case.path.is_file():
                    self.fail(
                        f"missing freeze fixture {case.path.name}; "
                        "run tools/regen_sdr_freeze.py"
                    )
                expected = np.load(case.path, allow_pickle=False)
                linear, u8 = render_freeze_case(case)
                exp_u8 = np.asarray(expected["u8"])
                exp_linear = np.asarray(expected["linear"], dtype=np.float32)
                if not np.array_equal(u8, exp_u8):
                    diff = np.abs(u8.astype(np.int16) - exp_u8.astype(np.int16))
                    changed = int(np.count_nonzero(diff))
                    # The freeze is unconditional: the fixtures were authored on
                    # macOS and macOS is the only supported platform (a retired
                    # ubuntu CI leg once needed a one-LSB libm exemption here).
                    self.fail(
                        f"{case.stem}: P3 SDR u8 drifted "
                        f"(max_delta={int(diff.max())}, "
                        f"changed={changed}/{diff.size})"
                    )
                # float16 storage; allow a few ULPs of half round-trip.
                if not np.allclose(
                    linear, exp_linear, rtol=0.0, atol=float(np.finfo(np.float16).eps) * 4
                ):
                    delta = float(np.max(np.abs(linear - exp_linear)))
                    self.fail(f"{case.stem}: P3 SDR linear drifted (max_abs={delta:.6g})")

    def test_user_ev_plus_one_is_not_identity(self) -> None:
        """Sanity: EV+1 freeze cases must differ from EV0 (exposure contract alive)."""
        from tools.regen_sdr_freeze import FreezeCase

        manifest = _load_manifest()
        by_key: dict[tuple[str, float], Path] = {}
        for entry in manifest["freeze_cases"]:
            case = FreezeCase(**entry)
            by_key[(case.scene_id, case.ev)] = case.path
        for scene_id in {entry["scene_id"] for entry in manifest["freeze_cases"]}:
            with self.subTest(scene_id=scene_id):
                zero = np.load(by_key[(scene_id, 0.0)], allow_pickle=False)["u8"]
                plus = np.load(by_key[(scene_id, 1.0)], allow_pickle=False)["u8"]
                self.assertFalse(np.array_equal(zero, plus))

    def test_exposure_gain_contract_for_freeze_evs(self) -> None:
        g0 = compute_exposure_gain(exposure_mode_for_tone_core("agx"), 0.0)
        g1 = compute_exposure_gain(exposure_mode_for_tone_core("agx"), 1.0)
        self.assertAlmostEqual(g1 / g0, 2.0, places=12)

    def test_coreimage_boundary_documented_in_freeze_matrix(self) -> None:
        """RAW9 stays a separate pipeline; the freeze matrix is LibRaw-only by design.

        RAW9's own regression coverage lives in tests/test_coreimage_decode.py
        (production-parity alignment against the LibRaw reference) rather than
        in this byte-freeze matrix, whose fixtures must stay decoder-portable.
        """
        from dngscan import coreimage_decode

        self.assertIsInstance(coreimage_decode.available(), bool)
        manifest = _load_manifest()
        decoders = {entry["decoder"] for entry in manifest["freeze_cases"]}
        self.assertEqual(decoders, {"libraw"})


if __name__ == "__main__":
    unittest.main()
