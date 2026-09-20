"""Exact bounded intermediates in the non-film render/delivery boundary."""
import gc
import dataclasses
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dngscan import hdr_agx, heif_encoder, retreat


class RenderMaskChunksTests(unittest.TestCase):
    def test_same_shape_storage_is_borrowed_and_each_slice_matches_full_oracle(self):
        for dtype in (np.float16, np.float32):
            masks = np.random.default_rng(12).uniform(-1, 2, (13, 11, 3)).astype(dtype)
            masks[0, 0] = (np.nan, np.inf, -np.inf)
            before = masks.tobytes()
            bundle = SimpleNamespace(clip_masks=masks)
            chunks = retreat.clip_masks_for_render(bundle, (13, 11))
            self.assertTrue(np.shares_memory(chunks._flat, masks))
            expected = retreat.resize_clip_masks(masks, (13, 11)).reshape(-1, 3)
            for start in range(0, 143, 17):
                self.assertEqual(chunks[start:start + 17].tobytes(),
                                 expected[start:start + 17].tobytes())
            self.assertFalse(hasattr(bundle, "_clip_masks_resized"))
            self.assertEqual(masks.tobytes(), before)

    def test_crop_resize_and_strided_mask_keep_public_resampler(self):
        masks = np.random.default_rng(13).random((8, 10, 3)).astype(np.float16)
        for view, shape, crop in ((masks, (8, 10), (1., 1., 7., 9.)),
                                   (masks, (5, 7), None),
                                   (masks[:, ::-1], (8, 10), None)):
            bundle = SimpleNamespace(clip_masks=view, scene_geometry_crop=crop,
                                     evidence_shape=(8, 10))
            with mock.patch.object(retreat, "clip_masks_for_shape", wraps=retreat.clip_masks_for_shape) as old:
                got = retreat.clip_masks_for_render(bundle, shape)
            old.assert_called_once()
            self.assertEqual(got.tobytes(), retreat.clip_masks_for_shape(bundle, shape).reshape(-1, 3).tobytes())


class HdrPackingTests(unittest.TestCase):
    def test_production_packed_pair_matches_f32_pair_with_partial_quantize_group(self):
        from tests.golden_support import build_staggered_clip
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.tone import build_render_plan

        scene = build_staggered_clip()
        bundle = dataclasses.replace(scene.bundle,
            scene_rec2020_render=scene.bundle.scene_rec2020_render[:7, :19], clip_masks=None)
        plan = build_render_plan(bundle, scene.analysis, "agx", "p3")
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        with mock.patch.multiple(hdr_agx, STREAM_RENDER_CHUNK=20,
                                  STREAM_QUANTIZE_CHUNK=40, STREAM_THREAD_MIN_PIXELS=80):
            base, linear = hdr_agx.render_ultrahdr_agx_pair(bundle, scene.analysis, plan, hdr_plan)
            packed_base, packed, usage = hdr_agx.render_ultrahdr_agx_pair_packed(
                bundle, scene.analysis, plan, hdr_plan)
        self.assertEqual(packed_base.tobytes(), base.tobytes())
        self.assertEqual(packed.tobytes(), hdr_agx.to_gainmap_alternate(linear, hdr_plan.tone.peak_linear).tobytes())
        self.assertEqual(struct.pack("d", usage), struct.pack("d", hdr_agx.achieved_headroom(linear)))

    def test_native_quantize_slices_preserve_full_group_noise_and_partial_tail(self):
        from tests.golden_support import build_staggered_clip
        from dngscan import _fast, render
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.tone import build_render_plan
        import os

        if _fast._load_extension() is None:
            self.skipTest("native extension required")
        scene = build_staggered_clip()
        bundle = dataclasses.replace(scene.bundle,
            scene_rec2020_render=scene.bundle.scene_rec2020_render[:7, :19], clip_masks=None)
        plan = build_render_plan(bundle, scene.analysis, "agx", "p3")
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        for mode in ('1', '0'):
            outputs = []
            for chunk in (20, 40):
                with mock.patch.dict(os.environ, DNGSCAN_FAST=mode), \
                     mock.patch.multiple(render, STREAM_RENDER_CHUNK=chunk,
                         STREAM_QUANTIZE_CHUNK=40, STREAM_THREAD_MIN_PIXELS=80), \
                     mock.patch.multiple(hdr_agx, STREAM_RENDER_CHUNK=chunk,
                         STREAM_QUANTIZE_CHUNK=40, STREAM_THREAD_MIN_PIXELS=80):
                    outputs.append((render.render_output_u8(bundle, scene.analysis, "p3", plan),
                        *hdr_agx.render_ultrahdr_agx_pair(bundle, scene.analysis, plan, hdr_plan)))
            for actual, expected in zip(*outputs):
                self.assertEqual(actual.tobytes(), expected.tobytes(), mode)

    def test_pair_finalizer_failure_preserves_old_sanitization_noise_and_strict_policy(self):
        from tests.golden_support import build_staggered_clip
        from dngscan import _fast, render
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.tone import build_render_plan
        import os

        if _fast._load_extension() is None:
            self.skipTest("native extension required")
        scene = build_staggered_clip()
        bundle = dataclasses.replace(scene.bundle,
            scene_rec2020_render=scene.bundle.scene_rec2020_render[:7, :19], clip_masks=None)
        plan = build_render_plan(bundle, scene.analysis, "agx", "p3")
        hdr_plan = compile_hdr_agx_plan(plan, analysis=scene.analysis)
        edges = np.array([[np.nan, 0., 1.], [np.inf, np.inf, np.inf],
            [0., -np.inf, 1.], [np.finfo(np.float32).max, 0., 0.],
            [1.2, -.4, .2], [.3, .7, 1.7], [-0., 0., .18]], np.float32)
        attempts = []

        def fail_finalizer(mapped, noise_a, noise_b, _plan):
            attempts.append((mapped.copy(), noise_a.copy(), noise_b.copy()))
            raise RuntimeError("injected SDR finalizer failure")

        with mock.patch.multiple(hdr_agx, STREAM_RENDER_CHUNK=20,
                STREAM_QUANTIZE_CHUNK=40, STREAM_THREAD_MIN_PIXELS=80), \
             mock.patch.object(hdr_agx, "apply_tone_core",
                side_effect=lambda rgb, *_: np.resize(edges, rgb.shape)), \
             mock.patch.object(hdr_agx, "_form_hdr_chunk",
                side_effect=lambda rgb, *_args, **_kwargs: np.zeros_like(rgb)), \
             mock.patch.object(_fast, "finalize_rec2020_u8_f32", side_effect=fail_finalizer), \
             np.errstate(all='ignore'):
            with mock.patch.dict(os.environ, DNGSCAN_FAST='auto'):
                actual, _ = hdr_agx.render_ultrahdr_agx_pair(bundle, scene.analysis, plan, hdr_plan)
            self.assertGreater(len(attempts), 1)
            mapped = np.concatenate([attempt[0] for attempt in attempts])
            self.assertEqual(len(mapped), 133)
            # Execute the old explicit conversion/sanitization and grouped RNG
            # graph, independently of the new slice finalizer's exception path.
            expected, noise_as, noise_bs = [], [], []
            rng = np.random.default_rng(0)
            for start in range(0, len(mapped), 40):
                part = mapped[start:start + 40]
                output = hdr_agx.rec2020_to_output(part, 'p3')
                output = np.nan_to_num(output, nan=0., posinf=1e6, neginf=-1e6).astype(np.float32)
                output = render._apply_output_color_ops(output, 'p3', 'none', 1., plan.color)
                output = hdr_agx.fit_to_output_gamut(output, 'p3', alpha=plan.color.gamut_fit_alpha)
                na, nb = hdr_agx.generate_dither_noise(rng, part.shape)
                noise_as.append(na)
                noise_bs.append(nb)
                expected.append(hdr_agx.dither_quantize_u8_with_noise(
                    hdr_agx.encode_display_linear(output, 'p3'), na, nb))
            self.assertEqual(actual.reshape(-1, 3).tobytes(), np.concatenate(expected).tobytes())
            self.assertEqual(np.concatenate([attempt[1] for attempt in attempts]).tobytes(),
                             np.concatenate(noise_as).tobytes())
            self.assertEqual(np.concatenate([attempt[2] for attempt in attempts]).tobytes(),
                             np.concatenate(noise_bs).tobytes())
            with mock.patch.dict(os.environ, DNGSCAN_FAST='1'):
                with self.assertRaisesRegex(_fast.NativeKernelError, 'injected SDR finalizer failure'):
                    hdr_agx.render_ultrahdr_agx_pair(bundle, scene.analysis, plan, hdr_plan)

    def test_streamed_tail_matches_numpy_with_boundaries_and_exceptional_values(self):
        original = np.random.default_rng(18).uniform(-1, 9, (211, 3)).astype(np.float32)
        for value in (0., np.nan, np.inf, -np.inf):
            data = original.copy()
            data[-1] = value
            for chunk in (1, 17, 64, 211):
                tail = hdr_agx._HeadroomTail(len(data))
                for start in range(0, len(data), chunk):
                    tail.add(data[start:start + chunk])
                with np.errstate(invalid="ignore"):
                    expected = hdr_agx.achieved_headroom(data.reshape(1, -1, 3))
                    self.assertEqual(struct.pack("d", tail.headroom()), struct.pack("d", expected))

    def test_half_packing_matches_full_expression_including_payloads_and_strides(self):
        rng = np.random.default_rng(16)
        for dtype in (np.float16, np.float32):
            data = rng.uniform(-.1, 12, (131, 9, 3)).astype(dtype)
            data[0, 0] = (np.nan, np.inf, -np.inf)
            data[1, 0] = (-0., np.nextafter(np.float32(1), np.float32(2)), 65504)
            for source in (data, data[::-1, ::-1], data.transpose(1, 0, 2)):
                for peak in (0.5, 1., 7.3):
                    before = source.tobytes()
                    clipped = np.clip(np.asarray(source, np.float32), 0., float(peak))
                    expected = np.empty(source.shape[:2] + (4,), np.float16)
                    expected[..., :3] = clipped.astype(np.float16, copy=False)
                    expected[..., 3] = np.float16(1)
                    self.assertEqual(hdr_agx.to_gainmap_alternate(source, peak).tobytes(), expected.tobytes())
                    self.assertEqual(source.tobytes(), before)

    def test_upper_headroom_is_exact_with_partial_bands_and_rank_interpolation(self):
        data = np.random.default_rng(17).uniform(0., 32., (513, 512, 4)).astype(np.float32)
        data[..., 3] = np.nan  # Alpha cannot affect the RGB usage statistic.
        for source in (data, data[::-1, ::-1], np.full_like(data, 3.1)):
            for q in (99., 99.5, 99.99, 100.):
                top = float(np.percentile(np.max(source[..., :3], axis=-1), q))
                expected = float(np.log2(top)) if top > 1. else 0.
                self.assertEqual(struct.pack("d", hdr_agx.achieved_headroom(source, q)),
                                 struct.pack("d", expected))
        for value in (np.nan, np.inf, -np.inf):
            data[-1, -1, :3] = value
            with np.errstate(invalid="ignore"):
                top = float(np.percentile(np.max(data[..., :3], axis=-1), 99.99))
                expected = float(np.log2(top)) if top > 1. else 0.
                self.assertEqual(hdr_agx.achieved_headroom(data), expected)


class HeifPlaneQuantizationTests(unittest.TestCase):
    def test_all_u8_codes_and_strides_match_old_float32_quantization(self):
        source = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (3, 256, 3))
        for data in (source, source[::-1, ::-1], source.transpose(1, 0, 2)):
            for bits in (8, 10):
                old = data.astype(np.float32)
                old /= 255.0
                expected = np.rint(np.clip(old, 0, 1) * ((1 << bits) - 1)).astype(
                    np.uint8 if bits == 8 else '<u2')
                got = heif_encoder._quantized_band(data, bits)
                self.assertEqual(got.tobytes(), expected.tobytes())
                self.assertTrue(got.flags.c_contiguous)
        lut = heif_encoder._u8_to_u10_lut()
        with self.assertRaises(ValueError):
            lut.flags.writeable = True

    def test_float_and_other_integer_input_keep_original_math(self):
        for dtype in (np.float16, np.float32, np.float64, np.int32):
            source = np.array([[[0, .5, 1], [-1, 2, 1]]], dtype=dtype)
            for bits in (8, 10):
                expected = np.rint(np.clip(source.astype(np.float32), 0, 1) * ((1 << bits) - 1)).astype(
                    np.uint8 if bits == 8 else '<u2')
                self.assertEqual(heif_encoder._quantized_band(source, bits).tobytes(), expected.tobytes())


class HdrReadbackOwnerTests(unittest.TestCase):
    def test_readonly_view_retains_private_bitmap_without_copy_or_cross_call_alias(self):
        from dngscan import gainmap
        from pathlib import Path

        expected = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        extent = SimpleNamespace(size=SimpleNamespace(width=3, height=2))
        image = mock.Mock()
        image.imageByApplyingGainMap_.return_value = image
        image.extent.return_value = extent
        context = mock.Mock()
        owners = []

        def render(_image, buf, *_args):
            buf[:] = expected.tobytes()
            owners.append(buf)

        context.render_toBitmap_rowBytes_bounds_format_colorSpace_.side_effect = render
        quartz = SimpleNamespace(
            CGColorSpaceCreateWithName=lambda _: object(), kCGColorSpaceExtendedLinearDisplayP3=1,
            kCIImageAuxiliaryHDRGainMap=2, kCIContextCacheIntermediates=3, kCIFormatRGBAh=4,
            CIImage=SimpleNamespace(imageWithContentsOfURL_=lambda _: image,
                                    imageWithContentsOfURL_options_=lambda *_: image),
            CIContext=SimpleNamespace(contextWithOptions_=lambda _: context),
        )
        foundation = SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda x: x))
        with mock.patch.dict(sys.modules, Quartz=quartz, Foundation=foundation), \
             mock.patch.object(gainmap, "_nsnumber_bool", side_effect=bool):
            first = gainmap._read_expanded_hdr_rgba_half(Path("one"))
            second = gainmap._read_expanded_hdr_rgba_half(Path("two"))
        self.assertTrue(np.shares_memory(first, np.frombuffer(owners[0], np.float16)))
        self.assertFalse(np.shares_memory(first, second))
        with self.assertRaises(ValueError):
            first.flags.writeable = True
        owners.clear()
        del context, image
        gc.collect()
        self.assertEqual(first.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
