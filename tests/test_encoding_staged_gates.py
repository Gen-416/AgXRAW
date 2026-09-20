# SPDX-License-Identifier: GPL-3.0-or-later
"""Staged HEIF gates must preserve selection and report only measured results."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dngscan.auto_encode import (
    EncodingStageRejected,
    select_heif_encoding,
)


def _sdr_metrics():
    return {
        "base_mean_code_error": .2,
        "coding_luma_rmse": .5,
        "coding_chroma_rmse": .5,
        "coding_local_luma_p99": .5,
    }


def _complete_metrics(q, chroma, auxiliary):
    return {
        **_sdr_metrics(),
        "delivery_quality": q,
        "delivery_chroma_requested": chroma,
        "gainmap_encoding_quality": auxiliary,
        "block_p95_luma_error": .02,
        "highlight_max_luma_error": .05,
        "chroma_error": .03,
    }


class StagedEncodingGateTests(unittest.TestCase):
    def test_default_callback_keeps_four_argument_protocol(self):
        calls = []

        def encode(q, chroma, auxiliary, path):
            calls.append((q, chroma, auxiliary))
            path.write_bytes(b"x" * q)
            return _complete_metrics(q, chroma, auxiliary)

        with tempfile.TemporaryDirectory() as td:
            result = select_heif_encoding(Path(td) / "out.heic", encode)
        self.assertEqual(calls[0], (95, "444", None))
        self.assertEqual(result["delivery_quality"], 70)

    def test_coding_failure_skips_hdr_and_records_available_metrics_and_bytes(self):
        calls, hdr_calls, selected = [], [], []

        def encode(q, chroma, auxiliary, path, *, sdr_precheck):
            key = (q, chroma, auxiliary)
            calls.append(key)
            path.write_bytes(b"x" * (1000 if q == 95 else 900))
            metrics = _sdr_metrics()
            if sdr_precheck is not None:
                metrics["coding_chroma_rmse"] = 10.
                sdr_precheck(metrics, encoded_bytes=path.stat().st_size)
            hdr_calls.append(key)
            return _complete_metrics(q, chroma, auxiliary)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.heic"
            result = select_heif_encoding(
                out, encode, gainmap=True, staged_sdr=True,
                on_selected=lambda info: selected.append(info["delivery_quality"]),
            )
            self.assertEqual(list(Path(td).iterdir()), [out])
            self.assertEqual(out.stat().st_size, 1000)
        self.assertEqual(hdr_calls, [(95, "444", 100)])
        self.assertEqual(selected, [95])
        self.assertEqual(len(result["auto_attempts"]), len(calls))
        for attempt in result["auto_attempts"][1:]:
            self.assertFalse(attempt["accepted"])
            self.assertEqual(attempt["rejected_at"], "sdr_additional")
            self.assertIn("bytes", attempt)
            self.assertEqual(attempt["metrics"]["coding_chroma_rmse"], 10.)
            self.assertEqual(attempt["metrics"]["base_mean_code_error"], .2)
            self.assertNotIn("chroma_error", attempt["metrics"])
            self.assertNotIn("block_p95_luma_error", attempt["metrics"])
            self.assertNotIn("highlight_max_luma_error", attempt["metrics"])

    def test_missing_or_nonfinite_candidate_metrics_reject_before_hdr(self):
        missing = object()
        for key in ("coding_luma_rmse", "coding_chroma_rmse", "coding_local_luma_p99"):
            for value in (missing, float("nan"), float("inf"), -float("inf"), None):
                with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as td:
                    hdr_calls = []

                    def encode(q, chroma, auxiliary, path, *, sdr_precheck):
                        path.write_bytes(b"x" * q)
                        metrics = _sdr_metrics()
                        if sdr_precheck is not None:
                            if value is missing:
                                del metrics[key]
                            else:
                                metrics[key] = value
                            # No byte count is required by the callback protocol.
                            sdr_precheck(metrics)
                        hdr_calls.append(q)
                        return _complete_metrics(q, chroma, auxiliary)

                    result = select_heif_encoding(Path(td) / "out.heic", encode, staged_sdr=True)
                    self.assertEqual(hdr_calls, [95])
                    for attempt in result["auto_attempts"][1:]:
                        self.assertFalse(attempt["accepted"])
                        self.assertNotIn("bytes", attempt)
                        self.assertEqual(attempt["rejected_at"], "sdr_additional")
                        if value is missing:
                            self.assertNotIn(key, attempt["metrics"])

    def test_reference_is_fully_verified_before_invalid_coding_metrics_abort_search(self):
        for value in (None, float("nan"), float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as td:
                out = Path(td) / "out.heic"
                out.write_bytes(b"previous")
                hdr_calls, selected = [], []

                def encode(q, chroma, auxiliary, path, *, sdr_precheck):
                    self.assertIsNone(sdr_precheck)
                    hdr_calls.append((q, chroma, auxiliary))
                    path.write_bytes(b"candidate")
                    metrics = _complete_metrics(q, chroma, auxiliary)
                    if value is None:
                        del metrics["coding_luma_rmse"]
                    else:
                        metrics["coding_luma_rmse"] = value
                    return metrics

                with self.assertRaisesRegex(RuntimeError, "参考"):
                    select_heif_encoding(
                        out, encode, gainmap=True, staged_sdr=True,
                        on_selected=lambda info: selected.append(info),
                    )
                self.assertEqual(hdr_calls, [(95, "444", 100)])
                self.assertEqual(selected, [])
                self.assertEqual(out.read_bytes(), b"previous")
                self.assertEqual(list(Path(td).iterdir()), [out])

    def test_reference_hdr_failure_never_promotes_lower_quality(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.heic"
            out.write_bytes(b"previous")
            calls = []

            def encode(q, chroma, auxiliary, path, *, sdr_precheck):
                calls.append((q, chroma, auxiliary))
                self.assertIsNone(sdr_precheck)
                path.write_bytes(b"invalid")
                raise RuntimeError("HDR reconstruction failed")

            with self.assertRaisesRegex(RuntimeError, "参考.*HDR reconstruction failed"):
                select_heif_encoding(out, encode, gainmap=True, staged_sdr=True)
            self.assertEqual(calls, [(95, "444", 100)])
            self.assertEqual(out.read_bytes(), b"previous")

    def test_shared_sdr_floor_accepts_boundary_and_rejects_next_float(self):
        # The one-code RMS floor must stay identical in the staged and full gates.
        from dngscan.auto_encode import additional_error_acceptable

        for value, expected in ((1., True), (float(np.nextafter(1., np.inf)), False)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as td:
                hdr_calls = []

                def encode(q, chroma, auxiliary, path, *, sdr_precheck):
                    metrics = _complete_metrics(q, chroma, auxiliary)
                    path.write_bytes(b"x" * q)
                    if sdr_precheck is not None:
                        metrics["coding_luma_rmse"] = value
                        self.assertEqual(
                            additional_error_acceptable(metrics, _complete_metrics(95, "444", None)),
                            expected,
                        )
                        sdr_precheck(metrics)
                    hdr_calls.append(q)
                    return metrics

                result = select_heif_encoding(Path(td) / "out.heic", encode, staged_sdr=True)
                self.assertEqual(len(hdr_calls) > 1, expected)
                self.assertEqual(result["delivery_quality"], 70 if expected else 95)

    def test_sdr_precheck_success_does_not_bypass_final_hdr_budget(self):
        hdr_calls = []

        def encode(q, chroma, auxiliary, path, *, sdr_precheck):
            metrics = _complete_metrics(q, chroma, auxiliary)
            path.write_bytes(b"x" * (1000 if sdr_precheck is None else 100))
            if sdr_precheck is not None:
                sdr_precheck(_sdr_metrics())
                metrics["highlight_max_luma_error"] = .5
            hdr_calls.append((q, chroma, auxiliary))
            return metrics

        with tempfile.TemporaryDirectory() as td:
            result = select_heif_encoding(
                Path(td) / "out.heic", encode, gainmap=True, staged_sdr=True,
            )
        self.assertEqual(len(hdr_calls), len(result["auto_attempts"]))
        self.assertEqual(result["delivery_quality"], 95)
        self.assertEqual(result["gainmap_encoding_quality"], 100)
        self.assertTrue(all(not a["accepted"] for a in result["auto_attempts"][1:]))

    def test_staging_preserves_candidate_order_size_ties_and_winner_notifications(self):
        results = []
        for staged in (False, True):
            with tempfile.TemporaryDirectory() as td:
                calls, selected = [], []

                def encode(q, chroma, auxiliary, path, **kwargs):
                    key = (q, chroma, auxiliary)
                    calls.append(key)
                    if not staged:
                        self.assertEqual(kwargs, {})
                    size = {95: 1000, 92: 900, 90: 900, 85: 1000, 80: 1100, 70: 950}[q]
                    path.write_bytes(bytes([q]) * size)
                    metrics = _complete_metrics(q, chroma, auxiliary)
                    callback = kwargs.get("sdr_precheck")
                    if callback is not None:
                        callback(_sdr_metrics(), encoded_bytes=size)
                    return metrics

                result = select_heif_encoding(
                    Path(td) / "out.heic", encode, staged_sdr=staged,
                    on_selected=lambda info: selected.append(
                        (info["delivery_quality"], info["delivery_chroma_requested"])),
                )
                self.assertEqual(selected, [(95, "444"), (92, "444")])
                self.assertEqual(result["delivery_quality"], 92)
                self.assertEqual(result["file_size_bytes"], 900)
                self.assertTrue(result["auto_attempts"][2]["accepted"])  # Equal winner size.
                self.assertFalse(result["auto_attempts"][3]["accepted"])  # Equal reference size.
                self.assertTrue(result["auto_attempts"][5]["accepted"])  # Larger than winner.
                results.append((calls, result["auto_attempts"], result["auto_saved_pct"]))
        self.assertEqual(results[0], results[1])

    def test_completed_candidate_cannot_drop_prechecked_coding_metrics(self):
        def encode(q, chroma, auxiliary, path, *, sdr_precheck):
            path.write_bytes(b"x" * q)
            metrics = _complete_metrics(q, chroma, auxiliary)
            if sdr_precheck is not None:
                sdr_precheck(_sdr_metrics())
                del metrics["coding_local_luma_p99"]
            return metrics

        with tempfile.TemporaryDirectory() as td:
            result = select_heif_encoding(Path(td) / "out.heic", encode, staged_sdr=True)
        self.assertEqual(result["delivery_quality"], 95)
        self.assertTrue(all(a["rejected_at"] == "sdr_additional"
                            for a in result["auto_attempts"][1:]))

    def test_auxiliary_improvements_notify_winner_before_next_candidate(self):
        selected = []

        def encode(q, chroma, auxiliary, path, *, sdr_precheck):
            if auxiliary == 90:
                self.assertEqual(selected[-1], (70, "444", 95))
            if auxiliary in (85, 80):
                self.assertEqual(selected[-1], (70, "444", 90))
            size = q * 10 + (1000 if auxiliary == 100 else 700 if auxiliary == 95 else 600)
            path.write_bytes(b"x" * size)
            metrics = _complete_metrics(q, chroma, auxiliary)
            if sdr_precheck is not None:
                sdr_precheck(_sdr_metrics())
            return metrics

        with tempfile.TemporaryDirectory() as td:
            result = select_heif_encoding(
                Path(td) / "out.heic", encode, gainmap=True, staged_sdr=True,
                on_selected=lambda info: selected.append(
                    (info["delivery_quality"], info["delivery_chroma_requested"],
                     info["gainmap_encoding_quality"])),
            )
        self.assertEqual(selected[-2:], [(70, "444", 95), (70, "444", 90)])
        self.assertEqual(result["gainmap_encoding_quality"], 90)

    def test_absolute_sdr_stage_failure_retains_only_completed_measurements(self):
        def encode(q, chroma, auxiliary, path, *, sdr_precheck):
            path.write_bytes(b"x" * q)
            if sdr_precheck is not None:
                raise EncodingStageRejected(
                    "base readback failed", metrics={"base_mean_code_error": 10.},
                    rejected_at="sdr_base", file_size_bytes=q,
                )
            return _complete_metrics(q, chroma, auxiliary)

        with tempfile.TemporaryDirectory() as td:
            result = select_heif_encoding(Path(td) / "out.heic", encode, staged_sdr=True)
        for attempt in result["auto_attempts"][1:]:
            self.assertEqual(attempt["rejected_at"], "sdr_base")
            self.assertEqual(attempt["metrics"], {"base_mean_code_error": 10.})
            self.assertEqual(attempt["bytes"], attempt["quality"])

    def test_winner_callback_error_is_not_swallowed_as_candidate_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.heic"
            out.write_bytes(b"previous")
            calls = []

            def encode(q, chroma, auxiliary, path):
                calls.append(q)
                path.write_bytes(b"candidate")
                return _complete_metrics(q, chroma, auxiliary)

            def on_selected(info):
                raise RuntimeError("private donor retention failed")

            with self.assertRaisesRegex(RuntimeError, "private donor retention failed"):
                select_heif_encoding(out, encode, on_selected=on_selected)
            self.assertEqual(calls, [95])
            self.assertEqual(out.read_bytes(), b"previous")
            self.assertEqual(list(Path(td).iterdir()), [out])


if __name__ == "__main__":
    unittest.main()
