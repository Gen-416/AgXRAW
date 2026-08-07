# SPDX-License-Identifier: GPL-3.0-or-later
"""Film v2 P0 freeze gate (FILM_PRINT_RENDERING_PLAN §12 P0).

The current none/observe/full rendering surface is pinned byte-exactly across
the six-way decomposition ladder. v2 work must not move these bytes until the
declared migration point (plan §7.2: the old single-LUT backend keeps this
freeze; v2 validates against the direct-chain oracle instead). Regenerating
the manifest to silence a diff requires an explicit review gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "film_freeze" / "MANIFEST.json"


def _load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise unittest.SkipTest(
            f"missing {MANIFEST_PATH}; run: tools/regen_film_freeze.py"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class FilmFreezeGateTest(unittest.TestCase):
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

    def test_manifest_covers_the_declared_ladder(self) -> None:
        from tools.regen_film_freeze import BASELINE, SCHEMA, STOCKS, freeze_configs

        manifest = _load_manifest()
        self.assertEqual(int(manifest["schema"]), SCHEMA)
        cases = manifest["cases"]
        scenes = {c["scene"] for c in cases}
        for scene in scenes:
            rows = [c for c in cases if c["scene"] == scene]
            self.assertEqual(
                {(c["stock"], c["config"]) for c in rows},
                {("none", "agx_baseline")}
                | {
                    (stock, config)
                    for stock in STOCKS
                    for config in freeze_configs(stock)
                },
                f"{scene}: ladder incomplete",
            )
        # The declared baseline params are part of the freeze contract.
        for c in cases:
            if c["config"] == "agx_baseline":
                self.assertEqual(c["params"], BASELINE)

    def test_every_case_is_byte_identical(self) -> None:
        from tests.golden_support import all_scenes
        from tools.regen_film_freeze import render_case

        manifest = _load_manifest()
        scenes = all_scenes()
        for c in manifest["cases"]:
            with self.subTest(scene=c["scene"], stock=c["stock"], config=c["config"]):
                scene = scenes.get(c["scene"])
                if scene is None:
                    self.fail(f"freeze scene missing from golden_support: {c['scene']}")
                u8 = np.ascontiguousarray(render_case(scene, c["params"]))
                self.assertEqual(list(u8.shape), c["shape"])
                digest = hashlib.sha256(u8.tobytes()).hexdigest()
                self.assertEqual(
                    digest,
                    c["sha256"],
                    f"{c['scene']}/{c['stock']}/{c['config']}: film surface "
                    "drifted — v2 changes must go through the declared "
                    "migration gate, not silently move the v1 freeze",
                )


if __name__ == "__main__":
    unittest.main()
