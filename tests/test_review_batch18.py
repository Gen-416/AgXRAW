# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 18 regression gates.

1. [P0] The GUI export filename call must not carry the seed (it crashed
   every export with TypeError).
2. The export child gets its seed RESOLVED BY THE PARENT — a spawned
   process has a fresh PREVIEW_STORE and would otherwise mint a new one.
3. Conservative scatter with a source taken from the FULL-RESOLUTION
   pre-bloom print: a sparse highlight sheds energy AND its neighbourhood
   gains it (the decimated-proxy source gave up energy to nobody).
5. inner budget 1 silences every nested pool, not just the outer one.
6. The GUI export releases its analysis buffers when no dashboard follows.
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np



class ExportFilenameTests(unittest.TestCase):

    def test_suffix_signature_rejects_the_seed(self) -> None:
        from dngscan.gui.service import export_suffix_parts

        with self.assertRaises(TypeError):
            export_suffix_parts("clip", "srgb", "sdr", film_optics_seed=7)






class NestedBudgetTests(unittest.TestCase):
    def test_share_of_one_silences_every_pool(self) -> None:
        from unittest import mock

        from dngscan import agx, gated_drt, look
        from dngscan.cpu_budget import TOTAL, inner
        from dngscan.render import apply_tone_core
        from dngscan.tone import build_render_plan
        from tests.golden_support import build_daylight_wide_dr

        scene = build_daylight_wide_dr()
        submits: list[str] = []

        def spy(pool, name):
            real = pool.submit
            return lambda *a, **k: (submits.append(name), real(*a, **k))[1]

        rgb = np.random.default_rng(0).uniform(
            0.01, 0.9, (200_000, 3)
        ).astype(np.float32)
        plan = build_render_plan(scene.bundle, scene.analysis, "agx", "srgb")
        gated = build_render_plan(
            scene.bundle, scene.analysis, "agx", "srgb", tone_core="gated"
        )
        lab = rgb.astype(np.float32)
        look_name = next(iter(look.LOOK_FIELDS))

        def exercise() -> None:
            apply_tone_core(rgb, plan.tone, plan.color, None, None)
            apply_tone_core(rgb, gated.tone, gated.color, None, None)
            look.apply_look_oklab(lab[:, 0], lab[:, 1], lab[:, 2], look_name, 1.0)

        with mock.patch.object(
            agx._FORMATION_POOL, "submit", spy(agx._FORMATION_POOL, "formation")
        ), mock.patch.object(
            look._LOOK_POOL, "submit", spy(look._LOOK_POOL, "look")
        ), mock.patch.object(
            gated_drt._GATED_POOL, "submit", spy(gated_drt._GATED_POOL, "gated")
        ):
            with inner(1):
                exercise()
            starved = list(submits)
            submits.clear()
            with inner(max(TOTAL, 2)):
                exercise()
            full = list(submits)
        self.assertEqual(
            starved, [],
            "a share of 1 must silence the formation, look and gated pools — "
            "the review measured 1 and 4 stray submits",
        )
        self.assertGreater(len(full), 0, "the pools must still work when budgeted")


class GuiStagedReleaseTests(unittest.TestCase):
    def test_gui_export_releases_before_encoding(self) -> None:
        from dngscan.gui import service

        src = inspect.getsource(service.run_export)
        release = src.find("release_analysis_buffers")
        encode = src.find("export_result = dg.export_jpeg")
        self.assertGreater(release, 0, "the GUI export must stage the release")
        self.assertLess(
            release, encode,
            "release must precede the JPEG/HDR encode (the dashboard holds "
            "its own export slot earlier, which is not the anchor here)",
        )
        # batch 19: the dashboard now runs BEFORE the export, so the
        # release is unconditional — a png=1 export used to encode with
        # xyz_render / y / ev_img still resident.
        dashboard = src.find("plot_dashboard")
        self.assertGreater(dashboard, 0)
        self.assertLess(
            dashboard, release,
            "the dashboard (the last consumer) must run before the release",
        )


if __name__ == "__main__":
    unittest.main()
