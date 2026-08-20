# SPDX-License-Identifier: GPL-3.0-or-later
"""Review R6 gates.

1. film_interimage=custom must actually RENDER (the runtime whitelist had
   not learned the latitude dial).
2. DNG stage-1 linearization tags are detected and degrade evidence claims.
3. DNG calibration composes AnalogBalance x CameraCalibration x ColorMatrix
   with the signature rule, and the 1.6 third illuminant interpolates.
4. Non-2x2 CFA guidance bins by CFA period, so a 2x2 block without a red
   sensel can no longer read as full headroom.
5. (comment-only fix, no gate needed: hdr_agx_math subdivision story.)
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _plan(**kw) -> SimpleNamespace:
    base = dict(
        curve_preset="portra400", film_mode="full", film_crossover="datasheet",
        film_exposure_ev=0.0, film_print_timing="fixed", film_print_medium="",
        film_print_exposure_ev=0.0, color_head_y=0.0, color_head_m=0.0,
        film_development="measured_default", film_dev_contrast=0.0,
        film_dev_fog=0.0, film_dev_density=0.0, film_compression=0.0,
        film_compression_knee=2.0, film_highlight_density=0.0,
        film_grain=0.0, film_halation=0.0, film_bloom=0.0,
        film_optics_seed=0, film_media_scatter="off",
        film_interimage="declared", film_interimage_beta=0.62,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class CustomInterimageRendersTests(unittest.TestCase):
    def test_custom_beta_actually_renders(self) -> None:
        from dngscan.film_develop import apply_film_core

        rng = np.random.default_rng(0)
        rgb = (rng.random((4000, 3)).astype(np.float32) * 0.8 + 0.02)
        declared = apply_film_core(rgb, _plan())
        same = apply_film_core(
            rgb, _plan(film_interimage="custom", film_interimage_beta=0.62)
        )
        hot = apply_film_core(
            rgb, _plan(film_interimage="custom", film_interimage_beta=1.4)
        )
        self.assertTrue(np.array_equal(declared, same),
                        "custom at the declared beta must render identically")
        self.assertGreater(float(np.abs(hot - declared).max()), 1e-3,
                           "a different beta must change the render")
        with self.assertRaises(ValueError):  # custom without a beta
            apply_film_core(
                rgb, _plan(film_interimage="custom", film_interimage_beta=None)
            )


# ---------------------------------------------------------------------------
# Minimal DNG-ish TIFF writer for parser gates (IFD0 only).
# ---------------------------------------------------------------------------

_SRATIONAL = 10
_RATIONAL = 5
_SHORT = 3
_ASCII = 2


def _entry(tag: int, typ: int, values) -> tuple[bytes, bytes]:
    """(12-byte IFD entry with placeholder offset, payload bytes)."""
    if typ in (_RATIONAL, _SRATIONAL):
        sub = "l" if typ == _SRATIONAL else "L"
        payload = b"".join(
            struct.pack("<" + sub * 2, int(round(v * 10000)), 10000) for v in values
        )
        count = len(values)
    elif typ == _SHORT:
        payload = struct.pack("<" + "H" * len(values), *values)
        count = len(values)
    elif typ == _ASCII:
        payload = values.encode("ascii") + b"\x00"
        count = len(payload)
    else:
        raise AssertionError(typ)
    head = struct.pack("<HHL", tag, typ, count)
    return head, payload


def _write_tiff(path: Path, entries: list[tuple[int, int, object]]) -> None:
    parts = [_entry(tag, typ, vals) for tag, typ, vals in entries]
    n = len(parts)
    ifd_off = 8
    data_off = ifd_off + 2 + n * 12 + 4
    body = bytearray()
    out_entries = []
    for head, payload in parts:
        if len(payload) <= 4:
            out_entries.append(head + payload.ljust(4, b"\x00"))
        else:
            out_entries.append(head + struct.pack("<L", data_off + len(body)))
            body += payload
    blob = bytearray()
    blob += b"II" + struct.pack("<H", 42) + struct.pack("<L", ifd_off)
    blob += struct.pack("<H", n)
    for e in sorted(out_entries, key=lambda e: struct.unpack("<H", e[:2])[0]):
        blob += e
    blob += struct.pack("<L", 0)
    blob += body
    path.write_bytes(bytes(blob))


_IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
_CM = [0.9, 0.1, 0.0, 0.05, 0.8, 0.05, 0.0, 0.1, 0.7]


class DngCalibrationCompositionTests(unittest.TestCase):
    def _cal(self, entries):
        from dngscan.metadata import read_dng_color_calibration

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.dng"
            _write_tiff(p, entries)
            return read_dng_color_calibration(p)

    def test_analog_balance_scales_rows(self) -> None:
        cal = self._cal([
            (50721, _SRATIONAL, _CM),          # ColorMatrix1
            (50778, _SHORT, [21]),             # CalibrationIlluminant1 = D65
            (50727, _RATIONAL, [2.0, 1.0, 1.0]),  # AnalogBalance
        ])
        self.assertIsNotNone(cal)
        np.testing.assert_allclose(
            np.asarray(cal.matrix1)[0], 2.0 * np.asarray(_CM[:3]), atol=1e-3
        )
        np.testing.assert_allclose(
            np.asarray(cal.matrix1)[1], np.asarray(_CM[3:6]), atol=1e-3
        )

    def test_camera_calibration_respects_signatures(self) -> None:
        cc = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
        base = [
            (50721, _SRATIONAL, _CM),
            (50778, _SHORT, [21]),
            (50723, _SRATIONAL, cc),           # CameraCalibration1
        ]
        matched = self._cal(base)  # both signatures absent -> match
        np.testing.assert_allclose(
            np.asarray(matched.matrix1), 0.5 * np.asarray(_CM).reshape(3, 3),
            atol=1e-3,
        )
        mismatched = self._cal(base + [
            (50931, _ASCII, "com.body.cal"),   # CameraCalibrationSignature
            (50932, _ASCII, "com.profile"),    # ProfileCalibrationSignature
        ])
        np.testing.assert_allclose(
            np.asarray(mismatched.matrix1), np.asarray(_CM).reshape(3, 3),
            atol=1e-3,
        )

    def test_third_illuminant_parses_and_interpolates(self) -> None:
        from dngscan.wb import interpolated_color_matrix

        m_a = [1.0] * 9
        m_d65 = [2.0] * 9
        m_d75 = [4.0] * 9
        cal = self._cal([
            (50721, _SRATIONAL, m_a), (50778, _SHORT, [17]),    # A (2856K)
            (50722, _SRATIONAL, m_d65), (50779, _SHORT, [21]),  # D65
            (52531, _SRATIONAL, m_d75), (52529, _SHORT, [22]),  # D75
        ])
        self.assertIsNotNone(cal.matrix3)
        self.assertAlmostEqual(float(cal.cct3), 7504.0)
        # Between D65 and D75 the answer must come from THAT bracket.
        mid = interpolated_color_matrix(cal, 7000.0)
        self.assertTrue(2.0 < float(mid[0][0]) < 4.0)
        # Outside the span: clamped to the nearest calibration.
        np.testing.assert_allclose(
            interpolated_color_matrix(cal, 12000.0), np.full((3, 3), 4.0),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            interpolated_color_matrix(cal, 2000.0), np.full((3, 3), 1.0),
            atol=1e-6,
        )


class Stage1DetectionTests(unittest.TestCase):
    def _flags(self, entries):
        from dngscan.metadata import read_dng_stage1_flags

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.dng"
            _write_tiff(p, entries)
            return read_dng_stage1_flags(p)

    def test_stage1_tags_detected_and_default_limit_ignored(self) -> None:
        base = [(50721, _SRATIONAL, _CM), (50778, _SHORT, [21])]
        self.assertEqual(self._flags(base), ())
        got = self._flags(base + [
            (50712, _SHORT, [0, 1, 2, 3]),       # LinearizationTable
            (50715, _SRATIONAL, [0.5, -0.5]),    # BlackLevelDeltaH
            (50734, _RATIONAL, [1.0]),           # default limit: ignored
        ])
        self.assertEqual(got, ("BlackLevelDeltaH", "LinearizationTable"))
        got2 = self._flags(base + [(50734, _RATIONAL, [0.8])])
        self.assertEqual(got2, ("LinearResponseLimit",))


class XTransGuidanceBinTests(unittest.TestCase):
    def test_missing_channel_block_no_longer_reads_full_headroom(self) -> None:
        """R6 item 4 repro: an X-Trans 2x2 block without a red sensel was
        filled with neutral 1.0 and binned as FULL red headroom. Period
        binning must surface the true per-period minimum instead."""
        from dngscan.guidance import _raw_headroom_rgb

        xtrans = np.array([
            [1, 1, 0, 1, 1, 2],
            [1, 1, 2, 1, 1, 0],
            [2, 0, 1, 0, 2, 1],
            [1, 1, 2, 1, 1, 0],
            [1, 1, 0, 1, 1, 2],
            [0, 2, 1, 2, 0, 1],
        ], dtype=np.uint8)
        h = w = 12
        colors = np.tile(xtrans, (2, 2))
        raw = np.full((h, w), 100.0, dtype=np.float32)
        # Drive every RED sensel near clip: red headroom is genuinely ~0.
        raw[colors == 0] = 1000.0
        bundle = SimpleNamespace(
            raw_image=raw,
            raw_colors=colors,
            raw_pattern=xtrans.tolist(),
            color_desc="RGBG",
            black_levels=[0.0, 0.0, 0.0, 0.0],
            white_level=1023,
            camera_white_levels=[1023.0] * 4,
            orientation_flip=0,
        )
        out = _raw_headroom_rgb(bundle, (2, 2))
        red = np.asarray(out)[..., 0]
        self.assertLess(
            float(red.max()), 0.05,
            "every output cell must report the near-clipped red it contains "
            "(the old 2x2 bin read 1.0 where the block had no red sensel)",
        )


if __name__ == "__main__":
    unittest.main()
