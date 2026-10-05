# SPDX-License-Identifier: GPL-3.0-or-later
"""Export preview encoding uses the final display pixels exactly once."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from dngscan.gui import service
from dngscan.models import AutoEvResult
from tests.test_preview_cache import _analysis, _bundle


class ExportPreviewReuseTests(unittest.TestCase):
    def exercise(self, *, decoded_available, auto_ev, dashboard):
        rgb = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        auto = AutoEvResult(.25, .25, 0., False, 1., 0.)
        lines = ["Auto EV +0.25"]
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.dng"
            source.write_bytes(b"fixture")
            encoded_path = []

            def export(**kwargs):
                encoded_path.append(kwargs["out_path"])
                Image.fromarray(rgb).save(kwargs["out_path"], format="JPEG", quality=100, subsampling=0)
                return True, rgb if decoded_available else None

            def plot(bundle, analysis, y, ev, path, **kwargs):
                path.write_bytes(b"dashboard")

            with ExitStack() as stack:
                stack.enter_context(patch.object(service.dg, "require_dependencies"))
                stack.enter_context(patch.object(service.PREVIEW_STORE, "peek", return_value=None))
                stack.enter_context(patch.object(service, "_load_export_scene", return_value=_bundle()))
                stack.enter_context(patch.object(service, "_cached_full_analysis", return_value=None))
                stack.enter_context(patch.object(service.dg, "analyze", return_value=(_analysis(), None, None)))
                stack.enter_context(patch.object(service.dg, "build_render_plan", return_value=object()))
                stack.enter_context(patch.object(service.dg, "compute_auto_ev", return_value=auto))
                stack.enter_context(patch.object(service.dg, "auto_ev_overlay_lines", return_value=lines))
                stack.enter_context(patch.object(service.dg, "plot_dashboard", side_effect=plot))
                stack.enter_context(patch.object(service.dg, "export_jpeg", side_effect=export))
                metrics = stack.enter_context(patch.object(service, "output_luminance_metrics_u8", return_value={"marker": 1}))
                stack.enter_context(patch.object(service, "output_luminance_metrics", return_value={"marker": 2}))
                stack.enter_context(patch.object(service, "estimate_ev_headroom", return_value={}))
                encode = stack.enter_context(patch.object(service, "preview_b64_from_u8", wraps=service.preview_b64_from_u8))
                from_file = stack.enter_context(patch.object(service, "make_preview_b64", wraps=service.make_preview_b64))
                annotate = stack.enter_context(patch.object(service, "annotate_preview_rgb_u8", wraps=service.annotate_preview_rgb_u8))
                open_image = stack.enter_context(patch.object(Image, "open", wraps=Image.open))
                result = service.run_export({"input": str(source), "outdir": td,
                                             "evAuto": auto_ev, "png": dashboard})
                self.assertTrue(result["ok"])
                self.assertEqual(len(result["saved"]), 2 if dashboard else 1)
                self.assertEqual(annotate.call_count, int(auto_ev and dashboard))
                if decoded_available or (auto_ev and dashboard):
                    encode.assert_called_once()
                    from_file.assert_not_called()
                else:
                    encode.assert_not_called()
                    from_file.assert_called_once()
                self.assertEqual(open_image.call_count, 0 if decoded_available else 1)
                if decoded_available:
                    self.assertIs(metrics.call_args.args[0], rgb)
                    self.assertEqual(result["metrics"]["marker"], 1)
                else:
                    self.assertEqual(result["metrics"]["marker"], 2)

            # Compare actual final JPEG bytes, independent of call-count checks.
            if decoded_available:
                expected_rgb = rgb
            else:
                with Image.open(encoded_path[0]) as image:
                    expected_rgb = np.asarray(image.convert("RGB"))
            if auto_ev and dashboard:
                expected_rgb = service.annotate_preview_rgb_u8(expected_rgb, lines)
            expected = service.preview_b64_from_u8(expected_rgb,
                icc_profile=service.dg.output_icc_profile_bytes("srgb"), width=service.PROXY_LONG_EDGE)
            self.assertEqual(result["preview"], expected)

    def test_decoded_final_frame_is_encoded_once_with_optional_annotation(self):
        for auto_ev, dashboard in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(auto_ev=auto_ev, dashboard=dashboard):
                self.exercise(decoded_available=True, auto_ev=auto_ev, dashboard=dashboard)

    def test_fallback_file_is_read_once_and_preserves_final_preview(self):
        for auto_ev, dashboard in ((False, False), (True, True)):
            with self.subTest(auto_ev=auto_ev, dashboard=dashboard):
                self.exercise(decoded_available=False, auto_ev=auto_ev, dashboard=dashboard)


if __name__ == "__main__":
    unittest.main()
