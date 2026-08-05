#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the fixed-resolution interactive preview path on one RAW file."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import resource
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dngscan.gui.preview_scheduler import PREVIEW_COORDINATOR
from dngscan.gui import preview_cache as preview_cache_module
from dngscan.gui import service as service_module
from dngscan.gui.constants import REALTIME_PREVIEW_LONG_EDGE


prepare_preview = service_module.prepare_preview
run_preview = service_module.run_preview

SCENARIOS: dict[str, dict[str, object]] = {
    "baseline": {},
    "wb-5500k": {"wb": "5500k"},
    "look": {"grade": "look:optic_warm_cyan", "gradeStrength": 1.0},
    "scene-transform": {
        "sceneTransform": "portra400_d55",
        "sceneTransformStrength": 1.0,
    },
    "film-portra400": {
        "wb": "5500k",
        "sceneTransform": "portra400_d55",
        "sceneTransformStrength": 1.3,
        "filmCurve": "portra400",
        "agxPrimaries": "base",
    },
    "raw-gated": {"toneCore": "gated"},
}


def _elapsed_ms(callable_) -> tuple[object, float]:
    started = time.perf_counter()
    result = callable_()
    return result, (time.perf_counter() - started) * 1000.0


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _dimensions(data_url: str) -> list[int]:
    encoded = data_url.split(",", 1)[-1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        return [image.width, image.height]


def _median_elapsed_ms(callable_, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        samples.append((time.perf_counter() - started) * 1000.0)
    return _percentile(samples, 0.50)


def _profile_native_output_substages(
    common: dict[str, object], generation: int
) -> dict[str, float]:
    """Isolate work inside the fused output kernel on one representative frame.

    These numbers deliberately materialize intermediate output/fitted buffers, so they
    explain cost distribution but must not be added to the fused production wall time.
    """
    from dngscan import _fast as fast_module
    from dngscan import render as render_module

    captured: list[tuple[object, object, object, object]] = []
    original_rec2020 = fast_module.finalize_rec2020_u8_noise_f32
    original_output = fast_module.finalize_output_u8_noise_f32

    def capture(kind: str, original):
        def wrapped(rgb, noise, plan):
            captured.append((kind, rgb, noise, plan))
            return original(rgb, noise, plan)

        return wrapped

    params = {**common, "generation": generation, "ev": 0.371987}
    with patch.object(
        fast_module,
        "finalize_rec2020_u8_noise_f32",
        capture("rec2020", original_rec2020),
    ), patch.object(
        fast_module,
        "finalize_output_u8_noise_f32",
        capture("output", original_output),
    ):
        result = run_preview(params)
    if not result.get("ok") or result.get("superseded") or not captured:
        return {}

    output_gamut = str(common.get("gamut", "srgb"))
    prepared: list[tuple[str, object, object, object, object, object]] = []
    for kind, rgb, noise, plan in captured:
        output_rgb = (
            render_module.rec2020_to_output(rgb, output_gamut)
            if kind == "rec2020"
            else rgb
        )
        fitted = fast_module.fit_output_gamut_f32(output_rgb, plan)
        prepared.append((kind, rgb, noise, plan, output_rgb, fitted))

    def fused() -> None:
        for kind, rgb, noise, plan, _, _ in prepared:
            function = original_rec2020 if kind == "rec2020" else original_output
            function(rgb, noise, plan)

    def gamut_transfer_quantize() -> None:
        for _, _, noise, plan, output_rgb, _ in prepared:
            original_output(output_rgb, noise, plan)

    def gamut_only() -> None:
        for _, _, _, plan, output_rgb, _ in prepared:
            fast_module.fit_output_gamut_f32(output_rgb, plan)

    def transfer_quantize() -> None:
        for _, _, noise, plan, _, fitted in prepared:
            original_output(fitted, noise, plan)

    fused_ms = _median_elapsed_ms(fused)
    gamut_transfer_ms = _median_elapsed_ms(gamut_transfer_quantize)
    gamut_ms = _median_elapsed_ms(gamut_only)
    transfer_ms = _median_elapsed_ms(transfer_quantize)
    return {
        "captured_chunks": len(prepared),
        "fused_total_ms": round(fused_ms, 3),
        "rec2020_matrix_estimate_ms": round(max(0.0, fused_ms - gamut_transfer_ms), 3),
        "gamut_fit_only_ms": round(gamut_ms, 3),
        "transfer_dither_quantize_ms": round(transfer_ms, 3),
        "materialized_gamut_plus_tail_ms": round(gamut_transfer_ms, 3),
    }


def _profile_native_agx_substages(
    common: dict[str, object], generation: int
) -> dict[str, float]:
    """Estimate AgX base/hue/punch shares using identical native inputs."""
    from dngscan import _fast as fast_module

    captured: list[tuple[object, object]] = []
    original = fast_module.apply_agx_core_f32

    def capture(rgb, plan):
        captured.append((rgb, plan))
        return original(rgb, plan)

    params = {**common, "generation": generation, "ev": 0.418321}
    with patch.object(fast_module, "apply_agx_core_f32", capture):
        result = run_preview(params)
    if not result.get("ok") or result.get("superseded") or not captured:
        return {}

    no_punch = [
        (rgb, SimpleNamespace(**{**vars(plan), "punch_strength": 0.0}))
        for rgb, plan in captured
    ]
    base = [
        (
            rgb,
            SimpleNamespace(
                **{
                    **vars(plan),
                    "punch_strength": 0.0,
                    "hue_restore": 0.0,
                }
            ),
        )
        for rgb, plan in captured
    ]

    def execute(items) -> None:
        for rgb, plan in items:
            original(rgb, plan)

    full_ms = _median_elapsed_ms(lambda: execute(captured))
    no_punch_ms = _median_elapsed_ms(lambda: execute(no_punch))
    base_ms = _median_elapsed_ms(lambda: execute(base))
    return {
        "captured_chunks": len(captured),
        "full_agx_ms": round(full_ms, 3),
        "base_compress_matrix_curve_ms": round(base_ms, 3),
        "hue_restore_estimate_ms": round(max(0.0, no_punch_ms - base_ms), 3),
        "oklab_punch_estimate_ms": round(max(0.0, full_ms - no_punch_ms), 3),
    }


class _StageRecorder:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.frame_samples: dict[str, list[float]] = defaultdict(list)
        self._frame_totals: dict[str, float] | None = None
        self._lock = threading.Lock()

    def clear(self) -> None:
        self.samples.clear()
        self.frame_samples.clear()
        self._frame_totals = None

    def begin_frame(self) -> None:
        with self._lock:
            if self._frame_totals is not None:
                raise RuntimeError("stage frame already active")
            self._frame_totals = defaultdict(float)

    def end_frame(self) -> None:
        with self._lock:
            if self._frame_totals is None:
                raise RuntimeError("no active stage frame")
            for stage, elapsed_ms in self._frame_totals.items():
                self.frame_samples[stage].append(elapsed_ms)
            self._frame_totals = None

    def wrapper(self, stage: str, callable_):
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return callable_(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self.samples[stage].append(elapsed_ms)
                    if self._frame_totals is not None:
                        self._frame_totals[stage] += elapsed_ms

        return measured

    def report(self) -> dict[str, dict[str, float]]:
        return {
            stage: {
                "calls": len(samples),
                "total_ms": round(sum(samples), 3),
                "p50_ms": round(_percentile(samples, 0.50), 3),
                "p95_ms": round(_percentile(samples, 0.95), 3),
            }
            for stage, samples in sorted(self.samples.items())
            if samples
        }

    def frame_report(self) -> dict[str, dict[str, float]]:
        return {
            stage: {
                "p50_ms": round(_percentile(samples, 0.50), 3),
                "p95_ms": round(_percentile(samples, 0.95), 3),
            }
            for stage, samples in sorted(self.frame_samples.items())
            if samples
        }


@contextmanager
def _record_stages():
    """Instrument preview stages without adding timers to the production path."""
    from dngscan import render as render_module
    from dngscan import _fast as fast_module

    recorder = _StageRecorder()
    targets = (
        (service_module.PREVIEW_STORE, "get", "session_lookup"),
        (service_module.dg, "with_intent_exposure", "bundle_exposure_copy"),
        (service_module, "_preview_pixel_key", "pixel_key"),
        (service_module, "_preview_frame_key", "frame_key"),
        (preview_cache_module.PreviewEntry, "get_frame", "frame_cache_lookup"),
        (preview_cache_module.PreviewEntry, "get_pixels", "pixel_cache_lookup"),
        (preview_cache_module.PreviewEntry, "put_pixels", "pixel_cache_store"),
        (preview_cache_module.PreviewEntry, "put_frame", "frame_cache_store"),
        (
            preview_cache_module.PreviewEntry,
            "get_or_build_dither_noise",
            "dither_cache_lookup",
        ),
        (service_module, "_cached_render_plan", "plan_lookup"),
        (service_module.dg, "render_output_u8", "pixel_pipeline_total"),
        (render_module, "plan_with_look_overrides", "look_plan_override"),
        (fast_module, "compile_output_plan", "native_output_plan"),
        (fast_module, "compile_agx_plan", "native_agx_plan"),
        (fast_module, "apply_agx_core_f32", "native_agx_kernel"),
        (
            render_module.retreat_engine,
            "clip_masks_for_shape",
            "clip_mask_lookup",
        ),
        (
            render_module.scene_transform_engine,
            "wb_adaptation_ratios",
            "wb_adaptation",
        ),
        (render_module, "scene_intent_rec2020", "scene_intent"),
        (
            render_module.scene_transform_engine,
            "apply_scene_transform_rec2020",
            "scene_transform",
        ),
        (render_module.retreat_engine, "apply_clip_retreat_rec2020", "clip_retreat"),
        (render_module, "apply_tone_core", "tone_core"),
        (render_module, "generate_dither_noise", "dither_noise_generation"),
        (render_module, "rec2020_to_output", "output_matrix"),
        (render_module, "finalize_output_linear", "gamut_finalize"),
        (render_module, "fit_to_output_gamut", "gamut_fit"),
        (render_module, "encode_display_linear", "transfer_encode"),
        (
            render_module,
            "dither_quantize_u8_with_noise",
            "dither_quantize",
        ),
        (
            fast_module,
            "finalize_rec2020_u8_f32",
            "native_output_finalize",
        ),
        (
            fast_module,
            "finalize_output_u8_f32",
            "native_output_finalize",
        ),
        (
            fast_module,
            "finalize_rec2020_u8_noise_f32",
            "native_output_finalize_cached_noise",
        ),
        (
            fast_module,
            "finalize_output_u8_noise_f32",
            "native_output_finalize_cached_noise",
        ),
        (service_module, "preview_metrics_from_u8", "preview_metrics"),
        (service_module.dg, "output_icc_profile_bytes", "icc_profile"),
        (service_module, "preview_b64_from_u8", "jpeg_base64"),
    )
    with ExitStack() as stack:
        for owner, name, stage in targets:
            original = getattr(owner, name)
            stack.enter_context(patch.object(owner, name, recorder.wrapper(stage, original)))
        yield recorder


@contextmanager
def _record_prepare_stages():
    """Instrument cold/disk/memory prepare work and first plan compilation."""
    from dngscan import analysis as analysis_module
    from dngscan import evidence as evidence_module
    from dngscan import raw_io as raw_io_module
    from dngscan import tone as tone_module

    recorder = _StageRecorder()
    targets = (
        (preview_cache_module, "_evidence_cache_identity", "evidence_identity"),
        (preview_cache_module, "_read_disk_entry", "disk_proxy_read"),
        (preview_cache_module, "_write_disk_entry", "disk_proxy_write"),
        (preview_cache_module, "build_proxy_entry", "proxy_build_total"),
        (preview_cache_module, "downsample_mean", "proxy_scene_lanczos"),
        (preview_cache_module, "resize_clip_masks", "proxy_mask_resize"),
        (service_module.dg, "load_raw", "raw_decode"),
        (service_module.dg, "analyze", "raw_analysis"),
        (raw_io_module, "acquire_raw_evidence", "raw_evidence_acquire"),
        (raw_io_module, "solve_wb_for_mode", "raw_wb_solve"),
        (
            raw_io_module,
            "render_to_scene_rec2020",
            "raw_scene_demosaic_postprocess",
        ),
        (raw_io_module, "scene_rec2020_to_xyz_render", "raw_xyz_render"),
        (raw_io_module, "build_clip_masks", "raw_clip_mask_build"),
        (evidence_module.rawpy, "imread", "libraw_open"),
        (analysis_module, "detect_ceilings", "analysis_detect_ceilings"),
        (
            raw_io_module,
            "refresh_clip_masks_from_fullwell",
            "analysis_refresh_clip_masks",
        ),
        (
            analysis_module,
            "compute_clip_pct_by_thresholds",
            "analysis_clip_percent",
        ),
        (analysis_module, "compute_cell_metrics", "analysis_cfa_cell_metrics"),
        (
            analysis_module,
            "luminance_from_xyz_render",
            "analysis_luminance",
        ),
        (analysis_module, "compute_ev_metrics", "analysis_ev_metrics"),
        (
            analysis_module,
            "estimate_raw_noise_floor",
            "analysis_noise_floor",
        ),
        (analysis_module, "compute_snr_curves", "analysis_snr_curves"),
        (analysis_module, "compute_gamut_metrics", "analysis_gamut_metrics"),
        (analysis_module, "raw_health_metrics", "analysis_raw_health"),
        (service_module.dg, "build_render_plan", "render_plan_build"),
        (tone_module, "scene_tone_metrics", "plan_scene_metrics"),
        (
            tone_module,
            "tone_plan_sample_scene_rec2020",
            "plan_scene_sample",
        ),
        (
            tone_module,
            "build_tone_compression_plan",
            "tone_plan_build",
        ),
        (service_module.dg, "apply_render_adjustments", "render_adjustments"),
        (service_module, "_cached_render_plan", "prepare_plan_total"),
        (service_module, "detected_scene_params", "detected_scene_params"),
    )
    with ExitStack() as stack:
        for owner, name, stage in targets:
            original = getattr(owner, name)
            stack.enter_context(patch.object(owner, name, recorder.wrapper(stage, original)))
        yield recorder


@contextmanager
def _output_backend(mode: str):
    """Select only the finalizer implementation while leaving tone native policy intact."""
    if mode == "numpy":
        from dngscan import _fast as fast_module

        with patch.object(fast_module, "supports_output_finalizer", return_value=False):
            yield
        return
    yield


@contextmanager
def _preview_geometry(long_edge: int | None, isolated_cache: bool = False):
    """Use an isolated cache when profiling a non-product candidate geometry."""
    if not isolated_cache and (long_edge is None or long_edge == REALTIME_PREVIEW_LONG_EDGE):
        yield REALTIME_PREVIEW_LONG_EDGE
        return
    if long_edge is None:
        long_edge = REALTIME_PREVIEW_LONG_EDGE
    if long_edge < 320:
        raise ValueError("--long-edge must be at least 320")
    old_edge = preview_cache_module.PROXY_LONG_EDGE
    old_cache_dir = os.environ.get("DNGSCAN_PREVIEW_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="dngscan-preview-profile-") as cache_dir:
        try:
            preview_cache_module.PROXY_LONG_EDGE = int(long_edge)
            os.environ["DNGSCAN_PREVIEW_CACHE_DIR"] = cache_dir
            preview_cache_module.PREVIEW_STORE.clear_memory()
            yield int(long_edge)
        finally:
            preview_cache_module.PREVIEW_STORE.clear_memory()
            preview_cache_module.PROXY_LONG_EDGE = old_edge
            if old_cache_dir is None:
                os.environ.pop("DNGSCAN_PREVIEW_CACHE_DIR", None)
            else:
                os.environ["DNGSCAN_PREVIEW_CACHE_DIR"] = old_cache_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--long-edge",
        type=int,
        help="profile an isolated candidate size without changing the product constant",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="baseline",
        help="representative preview mode to measure",
    )
    parser.add_argument(
        "--output-backend",
        choices=("auto", "numpy", "native"),
        default="auto",
        help="isolate the finalizer backend without disabling the native tone core",
    )
    parser.add_argument(
        "--isolated-cache",
        action="store_true",
        help="use an empty temporary proxy cache to measure full RAW cold prepare",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    input_path = args.input.expanduser().resolve()
    if args.output_backend == "native":
        os.environ["DNGSCAN_FAST"] = "1"
    with _preview_geometry(args.long_edge, args.isolated_cache) as long_edge, _output_backend(
        args.output_backend
    ):
        session = f"benchmark:{uuid.uuid4()}"
        common = {
            "input": str(input_path),
            "previewSession": session,
            "includeMetrics": False,
            **SCENARIOS[args.scenario],
        }
        try:
            prepare_usage_before = resource.getrusage(resource.RUSAGE_SELF)
            with _record_prepare_stages() as prepare_stages:
                prepared, prepare_ms = _elapsed_ms(lambda: prepare_preview(common))
            prepare_usage_after = resource.getrusage(resource.RUSAGE_SELF)
            if not prepared.get("ok"):
                raise RuntimeError(prepared)

            with _record_stages() as stages:
                first, first_frame_ms = _elapsed_ms(
                    lambda: run_preview({**common, "generation": 1, "ev": 0.0})
                )
                if not first.get("ok") or first.get("superseded"):
                    raise RuntimeError(first)
                metrics_variant, metrics_variant_ms = _elapsed_ms(
                    lambda: run_preview(
                        {
                            **common,
                            "generation": 2,
                            "ev": 0.0,
                            "includeMetrics": True,
                        }
                    )
                )
                if not metrics_variant.get("ok") or metrics_variant.get("superseded"):
                    raise RuntimeError(metrics_variant)
                stages.clear()

                continuous_ms: list[float] = []
                last_params: dict[str, object] | None = None
                usage_before = resource.getrusage(resource.RUSAGE_SELF)
                for index in range(args.iterations):
                    step = (index // 2 + 1) * 0.05
                    ev = -step if index % 2 == 0 else step
                    last_params = {
                        **common,
                        "generation": index + 3,
                        "ev": ev,
                    }
                    stages.begin_frame()
                    try:
                        result, elapsed_ms = _elapsed_ms(
                            lambda p=last_params: run_preview(p)
                        )
                    finally:
                        stages.end_frame()
                    if not result.get("ok") or result.get("superseded"):
                        raise RuntimeError(result)
                    continuous_ms.append(elapsed_ms)

                usage_after = resource.getrusage(resource.RUSAGE_SELF)
                stage_report = stages.report()
                stage_frame_report = stages.frame_report()

            assert last_params is not None
            repeated, cache_hit_ms = _elapsed_ms(
                lambda: run_preview(
                    {**last_params, "generation": args.iterations + 3}
                )
            )
            native_output_substages = (
                _profile_native_output_substages(
                    common, args.iterations + 10_000
                )
                if args.output_backend != "numpy"
                else {}
            )
            native_agx_substages = (
                _profile_native_agx_substages(
                    common, args.iterations + 20_000
                )
                if args.output_backend != "numpy"
                else {}
            )
            report = {
                "input": input_path.name,
                "scenario": args.scenario,
                "output_backend": args.output_backend,
                "long_edge": long_edge,
                "dimensions": _dimensions(first["preview"]),
                "prepare_ms": round(prepare_ms, 2),
                "prepare_stage_ms": prepare_stages.report(),
                "prepare_resource": {
                    "cpu_ms": round(
                        1000.0
                        * (
                            prepare_usage_after.ru_utime
                            + prepare_usage_after.ru_stime
                            - prepare_usage_before.ru_utime
                            - prepare_usage_before.ru_stime
                        ),
                        3,
                    ),
                    "cpu_to_wall_ratio": round(
                        1000.0
                        * (
                            prepare_usage_after.ru_utime
                            + prepare_usage_after.ru_stime
                            - prepare_usage_before.ru_utime
                            - prepare_usage_before.ru_stime
                        )
                        / prepare_ms,
                        3,
                    ),
                    "filesystem_input_blocks": int(
                        prepare_usage_after.ru_inblock - prepare_usage_before.ru_inblock
                    ),
                    "filesystem_output_blocks": int(
                        prepare_usage_after.ru_oublock - prepare_usage_before.ru_oublock
                    ),
                    "major_page_faults": int(
                        prepare_usage_after.ru_majflt - prepare_usage_before.ru_majflt
                    ),
                    "minor_page_faults": int(
                        prepare_usage_after.ru_minflt - prepare_usage_before.ru_minflt
                    ),
                },
                "first_frame_ms": round(first_frame_ms, 2),
                "metrics_variant_pixel_cache_hit": bool(
                    metrics_variant.get("pixel_cache_hit")
                ),
                "metrics_variant_ms": round(metrics_variant_ms, 2),
                "continuous_iterations": len(continuous_ms),
                "continuous_p50_ms": round(_percentile(continuous_ms, 0.50), 2),
                "continuous_p95_ms": round(_percentile(continuous_ms, 0.95), 2),
                "continuous_p99_ms": round(_percentile(continuous_ms, 0.99), 2),
                "continuous_max_ms": round(max(continuous_ms), 2),
                "continuous_stage_call_ms": stage_report,
                "continuous_stage_frame_ms": stage_frame_report,
                "continuous_resource": {
                    "cpu_ms_per_frame": round(
                        1000.0
                        * (
                            usage_after.ru_utime
                            + usage_after.ru_stime
                            - usage_before.ru_utime
                            - usage_before.ru_stime
                        )
                        / len(continuous_ms),
                        3,
                    ),
                    "cpu_to_wall_ratio": round(
                        1000.0
                        * (
                            usage_after.ru_utime
                            + usage_after.ru_stime
                            - usage_before.ru_utime
                            - usage_before.ru_stime
                        )
                        / sum(continuous_ms),
                        3,
                    ),
                    "filesystem_input_blocks": int(
                        usage_after.ru_inblock - usage_before.ru_inblock
                    ),
                    "filesystem_output_blocks": int(
                        usage_after.ru_oublock - usage_before.ru_oublock
                    ),
                    "major_page_faults": int(
                        usage_after.ru_majflt - usage_before.ru_majflt
                    ),
                    "minor_page_faults": int(
                        usage_after.ru_minflt - usage_before.ru_minflt
                    ),
                },
                "native_output_isolated_ms": native_output_substages,
                "native_agx_isolated_ms": native_agx_substages,
                "revisit_cache_hit": bool(repeated.get("cache_hit")),
                "revisit_ms": round(cache_hit_ms, 2),
            }
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        finally:
            PREVIEW_COORDINATOR.clear()


if __name__ == "__main__":
    raise SystemExit(main())
