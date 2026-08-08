# SPDX-License-Identifier: GPL-3.0-or-later
"""Small persistent cache for proxy preview sessions."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable

import dngscan as dg
from dngscan.guidance import raw_color_permission, raw_guidance_for_shape
from dngscan.models import Analysis, RawBundle, RawGuidanceMaps
from dngscan.retreat import resize_clip_masks

from .constants import PROXY_LONG_EDGE


# v11: RAW guidance gained the compiled permission map. Older entries can still
# recompute it, but forcing one rebuild keeps the preview hot path and its exactness
# contract independent of which cache version happened to be on disk.
PREVIEW_CACHE_VERSION = 12
PROXY_RESAMPLER = "lanczos"
MAX_DISK_CACHE_FILES = 24
MAX_DISK_CACHE_BYTES = 768 * 1024 * 1024
MAX_MEMORY_PROXY_ITEMS = 2
MAX_PLAN_CACHE_ITEMS = 32
MAX_PIXEL_CACHE_ITEMS = 2
MAX_FRAME_CACHE_ITEMS = 24
MAX_BALANCE_CACHE_ITEMS = 8


@dataclass
class PreviewEntry:
    """Everything needed to render a proxy, without retaining the RAW mosaic."""

    bundle: RawBundle
    analysis: Analysis
    # P3 seed lifecycle (review batch 15): one grain realization per loaded
    # RAW, minted here and reused by every preview, probe and export this
    # entry serves. A re-loaded entry (cache eviction) mints a new one —
    # "fully random per photo" — while an explicit seed always wins.
    realization_id: int = field(
        default_factory=lambda: __import__("secrets").randbits(32) | 1,
        init=False, repr=False,
    )
    _plan_cache: OrderedDict[Hashable, Any] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _frame_cache: OrderedDict[Hashable, dict[str, Any]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _pixel_cache: OrderedDict[Hashable, Any] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _balance_cache: OrderedDict[str, "PreviewEntry"] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _dither_noise: tuple[Any, Any] | None = field(default=None, init=False, repr=False)
    _runtime_cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def get_or_build_plan(self, key: Hashable, builder: Callable[[], Any]) -> Any:
        """Return an immutable base plan, compiling it at most once per key."""
        with self._runtime_cache_lock:
            cached = self._plan_cache.get(key)
            if cached is not None:
                self._plan_cache.move_to_end(key)
                return cached
            plan = builder()
            self._plan_cache[key] = plan
            while len(self._plan_cache) > MAX_PLAN_CACHE_ITEMS:
                self._plan_cache.popitem(last=False)
            return plan

    def get_or_build_balance(
        self,
        wb: str,
        builder: Callable[[], "PreviewEntry"],
    ) -> "PreviewEntry":
        """Build each user WB once from the immutable proxy DecodeContext."""
        if wb == "camera":
            return self
        with self._runtime_cache_lock:
            cached = self._balance_cache.get(wb)
            if cached is not None:
                self._balance_cache.move_to_end(wb)
                return cached
            balanced = builder()
            self._balance_cache[wb] = balanced
            self._balance_cache.move_to_end(wb)
            while len(self._balance_cache) > MAX_BALANCE_CACHE_ITEMS:
                self._balance_cache.popitem(last=False)
            return balanced

    def get_frame(self, key: Hashable) -> dict[str, Any] | None:
        """Return a shallow payload copy so request metadata can be added safely."""
        with self._runtime_cache_lock:
            payload = self._frame_cache.get(key)
            if payload is None:
                return None
            self._frame_cache.move_to_end(key)
            result = dict(payload)
            if isinstance(payload.get("metrics"), dict):
                result["metrics"] = dict(payload["metrics"])
            return result

    def get_pixels(self, key: Hashable) -> Any | None:
        """Return immutable rendered pixels shared by representation variants."""
        with self._runtime_cache_lock:
            pixels = self._pixel_cache.get(key)
            if pixels is not None:
                self._pixel_cache.move_to_end(key)
            return pixels

    def put_pixels(self, key: Hashable, pixels: Any) -> Any:
        """Keep only the newest full preview frames; each is several MiB."""
        np = dg.np
        stored = np.array(pixels, dtype=np.uint8, copy=True, order="C")
        stored.setflags(write=False)
        with self._runtime_cache_lock:
            self._pixel_cache[key] = stored
            self._pixel_cache.move_to_end(key)
            while len(self._pixel_cache) > MAX_PIXEL_CACHE_ITEMS:
                self._pixel_cache.popitem(last=False)
        return stored

    def put_frame(self, key: Hashable, payload: dict[str, Any]) -> None:
        """Keep a bounded exact-frame LRU; predicted combinations never explode."""
        stored = dict(payload)
        if isinstance(payload.get("metrics"), dict):
            stored["metrics"] = dict(payload["metrics"])
        with self._runtime_cache_lock:
            self._frame_cache[key] = stored
            self._frame_cache.move_to_end(key)
            while len(self._frame_cache) > MAX_FRAME_CACHE_ITEMS:
                self._frame_cache.popitem(last=False)

    def get_or_build_dither_noise(self) -> tuple[Any, Any]:
        """Reuse both fixed seed-0 TPDF planes without changing operation order."""
        with self._runtime_cache_lock:
            if self._dither_noise is None:
                self._dither_noise = dg.deterministic_dither_planes(
                    self.bundle.scene_rec2020_render.shape[:2] + (3,)
                )
            return self._dither_noise


INT_KEY_ANALYSIS_FIELDS = {
    "labels",
    "ceilings",
    "ceil_spike_counts",
    "ceil_near_counts",
    "ceil_spike_ok",
    "saturation_levels",
    "channel_fullwell",
    "channel_thresholds",
    "clip_pct",
    "cell_k_of_clipped_pct",
    "cell_k_of_all_pct",
    "snr1_dr",
    "snr1_stop",
}


def proxy_target_size(width: int, height: int, max_long_edge: int) -> tuple[int, int]:
    """Fit inside one long-edge bound while preserving the decoded source ratio."""
    if width < 1 or height < 1:
        raise ValueError("preview source dimensions must be positive")
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return width, height
    scale = float(max_long_edge) / float(long_edge)
    return max(1, round(width * scale)), max(1, round(height * scale))


def downsample_mean(image: object, max_long_edge: int = PROXY_LONG_EDGE) -> object:
    """High-quality fixed-geometry proxy in scene-linear code values."""
    np = dg.np
    if np is None:
        return image
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    target = proxy_target_size(w, h, max_long_edge)
    if target == (w, h):
        return arr
    from PIL import Image

    channels = []
    for idx in range(arr.shape[2]):
        plane = Image.fromarray(arr[:, :, idx].astype(np.float32, copy=False), mode="F")
        channels.append(
            np.asarray(plane.resize(target, Image.Resampling.LANCZOS), dtype=np.float32)
        )
    return np.stack(channels, axis=2)


def _cache_dir() -> Path:
    override = os.environ.get("DNGSCAN_PREVIEW_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "posix" and (Path.home() / "Library" / "Caches").is_dir():
        return Path.home() / "Library" / "Caches" / "dngscan" / "preview-v9"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dngscan" / "preview-v9"


def _evidence_cache_identity(path: Path) -> tuple[str, int, int, int, str, str]:
    """Decoder-independent identity for the Evidence input and provider.

    Beyond path/mtime/size, the identity carries the inode and a hash of the
    file's first 64 KiB (review batch 17): sync tools and batch converters
    can replace a RAW in place while preserving size and timestamps, and the
    header hash catches a different capture wearing the same clothes at the
    cost of one small read."""
    from dngscan.evidence import libraw_runtime_id

    stat = path.stat()
    with path.open("rb") as fh:
        head = fh.read(65536)
    return (
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        int(getattr(stat, "st_ino", 0)),
        hashlib.sha256(head).hexdigest()[:16],
        str(libraw_runtime_id() or "libraw-unknown"),
    )


def _scene_decoder_runtime_id(decoder: str) -> str:
    """The actual system decoder behind the requested one, or "" when LibRaw owns it.

    Apple revises the Core Image RAW models by OS build, so cached Core Image pixels
    are only valid for the build that decoded them; the LibRaw runtime already rides
    the evidence identity.
    """
    if str(decoder) != "coreimage":
        return ""
    from dngscan.coreimage_decode import decoder_runtime_id

    return str(decoder_runtime_id())


def _cache_identity(
    path: Path,
    highlight: str,
    wb: str,
    decoder: str = "libraw",
    coreimage_version: str = "auto",
    demosaic: str = "auto",
) -> tuple[tuple, str]:
    evidence_key = _evidence_cache_identity(path)
    key = (
        *evidence_key,
        highlight,
        str(decoder),
        str(coreimage_version),
        str(demosaic),
        _scene_decoder_runtime_id(decoder),
    )
    encoded = "\0".join(
        (
            str(PREVIEW_CACHE_VERSION),
            str(PROXY_LONG_EDGE),
            PROXY_RESAMPLER,
            *(str(value) for value in key),
        )
    ).encode("utf-8")
    return key, hashlib.sha256(encoded).hexdigest()


def _analysis_to_json(analysis: Analysis) -> dict[str, Any]:
    return asdict(analysis)


def _analysis_from_json(data: dict[str, Any]) -> Analysis:
    restored = dict(data)
    for field in INT_KEY_ANALYSIS_FIELDS:
        values = restored.get(field)
        if isinstance(values, dict):
            restored[field] = {int(key): value for key, value in values.items()}
    return Analysis(**restored)


def _bundle_metadata(bundle: RawBundle) -> dict[str, Any]:
    return {
        "render_scale": float(bundle.render_scale),
        "scene_scale": float(bundle.scene_scale),
        "white_level": int(bundle.white_level),
        "black_levels": [float(value) for value in bundle.black_levels],
        "camera_wb": [float(value) for value in bundle.camera_wb],
        "color_desc": str(bundle.color_desc),
        "raw_pattern": bundle.raw_pattern,
        "camera_white_levels": [float(value) for value in bundle.camera_white_levels],
        "scene_highlight_mode": str(bundle.scene_highlight_mode),
        "orientation_flip": int(bundle.orientation_flip),
        "wb_mode": str(bundle.wb_mode),
        "applied_wb": (
            [float(value) for value in bundle.applied_wb]
            if bundle.applied_wb is not None
            else None
        ),
        "decode_wb": (
            [float(value) for value in bundle.decode_wb]
            if bundle.decode_wb is not None
            else None
        ),
        "wb_xyz_to_cam": (
            dg.np.asarray(bundle.wb_xyz_to_cam, dtype=dg.np.float64).tolist()
            if bundle.wb_xyz_to_cam is not None
            else None
        ),
        "wb_color_matrix": (
            dg.np.asarray(bundle.wb_color_matrix, dtype=dg.np.float64).tolist()
            if getattr(bundle, "wb_color_matrix", None) is not None
            else None
        ),
        "wb_degradation": bundle.wb_degradation,
        "daylight_wb": (
            [float(value) for value in bundle.daylight_wb]
            if bundle.daylight_wb is not None
            else None
        ),
        "shot_make": bundle.shot_make,
        "shot_model": bundle.shot_model,
        "shot_iso": bundle.shot_iso,
        "baseline_exposure": bundle.baseline_exposure,
        "baseline_exposure_baked_in": bool(bundle.baseline_exposure_baked_in),
        "evidence_provider": str(
            getattr(bundle, "evidence_provider", "libraw") or "libraw"
        ),
        "evidence_provider_version": getattr(
            bundle, "evidence_provider_version", None
        ),
        "scene_decoder": str(getattr(bundle, "scene_decoder", "libraw") or "libraw"),
        "scene_decoder_version": getattr(bundle, "scene_decoder_version", None),
        "scene_decoder_runtime": getattr(bundle, "scene_decoder_runtime", None),
        "scene_scale_mode": getattr(bundle, "scene_scale_mode", None),
        "scene_align_factor": float(getattr(bundle, "scene_align_factor", 1.0)),
        "scene_align_error": getattr(bundle, "scene_align_error", None),
        "scene_opcode_names": list(getattr(bundle, "scene_opcode_names", ()) or ()),
        "evidence_shape": (
            [int(v) for v in bundle.evidence_shape]
            if getattr(bundle, "evidence_shape", None) is not None
            else None
        ),
        "scene_geometry_crop": (
            [float(v) for v in bundle.scene_geometry_crop]
            if getattr(bundle, "scene_geometry_crop", None) is not None
            else None
        ),
        "scene_geometry_corr": getattr(bundle, "scene_geometry_corr", None),
    }


def _bundle_from_cache(
    path: Path,
    metadata: dict[str, Any],
    scene: Any,
    masks: Any | None,
    guidance: RawGuidanceMaps | None,
) -> RawBundle:
    np = dg.np
    evidence_shape = metadata.get("evidence_shape")
    crop = metadata.get("scene_geometry_crop")
    return RawBundle(
        path=path,
        raw_image=None,
        raw_colors=None,
        xyz_render=None,
        render_scale=float(metadata["render_scale"]),
        scene_rec2020_render=scene,
        scene_scale=float(metadata["scene_scale"]),
        white_level=int(metadata["white_level"]),
        black_levels=[float(value) for value in metadata["black_levels"]],
        camera_wb=[float(value) for value in metadata["camera_wb"]],
        color_desc=str(metadata["color_desc"]),
        raw_pattern=metadata["raw_pattern"],
        camera_white_levels=[float(value) for value in metadata["camera_white_levels"]],
        scene_highlight_mode=str(metadata["scene_highlight_mode"]),
        orientation_flip=int(metadata["orientation_flip"]),
        wb_mode=str(metadata["wb_mode"]),
        applied_wb=metadata.get("applied_wb"),
        decode_wb=metadata.get("decode_wb"),
        wb_xyz_to_cam=(
            np.asarray(metadata["wb_xyz_to_cam"], dtype=np.float64)
            if metadata.get("wb_xyz_to_cam") is not None
            else None
        ),
        wb_color_matrix=(
            np.asarray(metadata["wb_color_matrix"], dtype=np.float64)
            if metadata.get("wb_color_matrix") is not None
            else None
        ),
        wb_degradation=metadata.get("wb_degradation"),
        daylight_wb=metadata["daylight_wb"],
        shot_make=metadata["shot_make"],
        shot_model=metadata["shot_model"],
        shot_iso=metadata["shot_iso"],
        baseline_exposure=metadata.get("baseline_exposure"),
        baseline_exposure_baked_in=bool(
            metadata.get("baseline_exposure_baked_in", False)
        ),
        evidence_provider=str(
            metadata.get("evidence_provider", "libraw") or "libraw"
        ),
        evidence_provider_version=metadata.get("evidence_provider_version"),
        clip_masks=masks,
        raw_guidance=guidance,
        _raw_guidance_has_sensor_snr=(
            guidance is not None and guidance.snr_confidence is not None
        ),
        scene_decoder=str(metadata.get("scene_decoder", "libraw") or "libraw"),
        scene_decoder_version=metadata.get("scene_decoder_version"),
        scene_decoder_runtime=metadata.get("scene_decoder_runtime"),
        scene_scale_mode=metadata.get("scene_scale_mode"),
        scene_align_factor=float(metadata.get("scene_align_factor", 1.0)),
        scene_align_error=metadata.get("scene_align_error"),
        scene_opcode_names=tuple(metadata.get("scene_opcode_names", ()) or ()),
        evidence_shape=(
            (int(evidence_shape[0]), int(evidence_shape[1]))
            if evidence_shape is not None
            else None
        ),
        scene_geometry_crop=(
            (float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3]))
            if crop is not None
            else None
        ),
        scene_geometry_corr=(
            float(metadata["scene_geometry_corr"])
            if metadata.get("scene_geometry_corr") is not None
            else None
        ),
    )


def _copy_guidance(maps: RawGuidanceMaps | None) -> RawGuidanceMaps | None:
    if maps is None:
        return None
    np = dg.np
    headroom = np.asarray(maps.headroom).copy()
    clip_class = np.asarray(maps.clip_class).copy()
    permission = (
        np.asarray(maps.raw_permission).copy()
        if maps.raw_permission is not None
        else raw_color_permission(
            headroom_rgb=headroom.reshape(-1, 3),
            clip_class=clip_class.reshape(-1),
        ).reshape(headroom.shape[:2])
    )
    return RawGuidanceMaps(
        headroom=headroom,
        clip_class=clip_class,
        snr_confidence=(
            np.asarray(maps.snr_confidence).copy()
            if maps.snr_confidence is not None
            else None
        ),
        raw_permission=permission,
    )


def build_proxy_entry(
    source: RawBundle,
    analysis: Analysis,
    include_guidance: bool = False,
) -> PreviewEntry:
    """Discard full RAW state after reducing the scene and evidence to proxy geometry."""
    np = dg.np
    proxy_scene = downsample_mean(source.scene_rec2020_render, PROXY_LONG_EDGE)
    proxy_shape = proxy_scene.shape[:2]
    # Proxy masks are already in scene space after resize_clip_masks with the bundle crop.
    proxy_masks = resize_clip_masks(
        source.clip_masks,
        proxy_shape,
        crop=getattr(source, "scene_geometry_crop", None),
    )
    if proxy_masks is not None:
        proxy_masks = proxy_masks.astype(np.float16, copy=False)
    proxy_guidance = None
    if include_guidance:
        proxy_guidance = _copy_guidance(raw_guidance_for_shape(source, proxy_shape, analysis))
    meta = _bundle_metadata(source)
    # After proxying, masks live at proxy geometry; clear the evidence crop so later
    # resizes treat them as already scene-aligned.
    meta["evidence_shape"] = [int(proxy_shape[0]), int(proxy_shape[1])]
    meta["scene_geometry_crop"] = None
    bundle = _bundle_from_cache(
        source.path,
        meta,
        proxy_scene,
        proxy_masks,
        proxy_guidance,
    )
    return PreviewEntry(bundle=bundle, analysis=analysis)


def _read_disk_entry(
    cache_path: Path,
    source_path: Path,
    require_guidance: bool,
    expected_runtime: str | None = None,
) -> PreviewEntry | None:
    np = dg.np
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            if int(metadata.get("version", -1)) != PREVIEW_CACHE_VERSION:
                return None
            if require_guidance and not bool(metadata.get("has_guidance", False)):
                return None
            if expected_runtime:
                stored = str(metadata.get("bundle", {}).get("scene_decoder_runtime") or "")
                if stored != expected_runtime:
                    return None
            scene = np.asarray(payload["scene"]).copy()
            masks = np.asarray(payload["masks"]).copy() if bool(metadata.get("has_masks", False)) else None
            guidance = None
            if bool(metadata.get("has_guidance", False)):
                snr = (
                    np.asarray(payload["guidance_snr"]).copy()
                    if bool(metadata.get("guidance_has_snr", False))
                    else None
                )
                guidance = RawGuidanceMaps(
                    headroom=np.asarray(payload["guidance_headroom"]).copy(),
                    clip_class=np.asarray(payload["guidance_clip_class"]).copy(),
                    snr_confidence=snr,
                    raw_permission=np.asarray(
                        payload["guidance_raw_permission"]
                    ).copy(),
                )
            bundle = _bundle_from_cache(source_path, metadata["bundle"], scene, masks, guidance)
            return PreviewEntry(bundle=bundle, analysis=_analysis_from_json(metadata["analysis"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _trim_disk_cache(directory: Path) -> None:
    try:
        files = sorted(directory.glob("*.npz"), key=lambda item: item.stat().st_mtime)
    except OSError:
        return
    total = sum(item.stat().st_size for item in files)
    while files and (len(files) > MAX_DISK_CACHE_FILES or total > MAX_DISK_CACHE_BYTES):
        oldest = files.pop(0)
        try:
            size = oldest.stat().st_size
            oldest.unlink()
            total -= size
        except OSError:
            continue


def _write_disk_entry(cache_path: Path, entry: PreviewEntry) -> None:
    np = dg.np
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = entry.bundle
        maps = bundle.raw_guidance
        metadata = {
            "version": PREVIEW_CACHE_VERSION,
            "bundle": _bundle_metadata(bundle),
            "analysis": _analysis_to_json(entry.analysis),
            "has_masks": bundle.clip_masks is not None,
            "has_guidance": maps is not None,
            "guidance_has_snr": maps is not None and maps.snr_confidence is not None,
        }
        fd, temp_name = tempfile.mkstemp(prefix=".preview-", suffix=".npz", dir=cache_path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                values: dict[str, Any] = {
                    "scene": np.asarray(bundle.scene_rec2020_render),
                    "masks": (
                        np.asarray(bundle.clip_masks)
                        if bundle.clip_masks is not None
                        else np.empty((0, 0, 0), dtype=np.float16)
                    ),
                    "metadata": np.asarray(json.dumps(metadata, allow_nan=True)),
                }
                if maps is not None:
                    values["guidance_headroom"] = np.asarray(maps.headroom)
                    values["guidance_clip_class"] = np.asarray(maps.clip_class)
                    values["guidance_raw_permission"] = (
                        np.asarray(maps.raw_permission)
                        if maps.raw_permission is not None
                        else raw_color_permission(
                            headroom_rgb=np.asarray(maps.headroom).reshape(-1, 3),
                            clip_class=np.asarray(maps.clip_class).reshape(-1),
                        ).reshape(np.asarray(maps.headroom).shape[:2])
                    )
                    if maps.snr_confidence is not None:
                        values["guidance_snr"] = np.asarray(maps.snr_confidence)
                np.savez(handle, **values)
            os.replace(temp_name, cache_path)
        finally:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        _trim_disk_cache(cache_path.parent)
    except OSError:
        return


class PreviewCache:
    """A small in-memory proxy LRU plus a bounded, validated on-disk cache."""

    # How many cold builds (decode + full analysis + proxy) may run at the
    # same time. Each holds roughly one full decode's working set, so the
    # quota is the memory contract for concurrent cold starts (scheduler
    # plan S1); further keys queue on the semaphore, and REQUESTS FOR THE
    # SAME KEY never build twice — they wait on that key's in-flight event.
    MAX_CONCURRENT_BUILDS = 2

    def __init__(self) -> None:
        self.entries: OrderedDict[
            tuple, PreviewEntry
        ] = OrderedDict()
        self.lock = threading.Lock()
        self._inflight: dict[tuple, threading.Event] = {}
        self._build_slots = threading.BoundedSemaphore(self.MAX_CONCURRENT_BUILDS)

    def clear_memory(self) -> None:
        with self.lock:
            self.entries.clear()

    def peek(
        self,
        path: Path,
        highlight: str,
        wb: str,
        require_guidance: bool = False,
        decoder: str = "libraw",
        coreimage_version: str = "auto",
        demosaic: str = "auto",
    ) -> PreviewEntry | None:
        """The loaded entry for this identity WITHOUT building one — the
        export path uses it to reuse the preview's grain realization (batch
        15 seed lifecycle); a cold export mints its own instead."""
        if decoder == "coreimage":
            highlight = "reconstruct"
            demosaic = "auto"
        key, _ = _cache_identity(
            path, highlight, wb, decoder, coreimage_version, demosaic
        )
        with self.lock:
            return self.entries.get(key)

    def get(
        self,
        path: Path,
        highlight: str,
        wb: str,
        require_guidance: bool = False,
        decoder: str = "libraw",
        coreimage_version: str = "auto",
        demosaic: str = "auto",
    ) -> PreviewEntry:
        if decoder == "coreimage":
            highlight = "reconstruct"
            demosaic = "auto"
        key, digest = _cache_identity(
            path, highlight, wb, decoder, coreimage_version, demosaic
        )
        cached: PreviewEntry | None = None
        with self.lock:
            cached = self.entries.get(key)
            if cached is not None and (not require_guidance or cached.bundle.raw_guidance is not None):
                self.entries.move_to_end(key)
            else:
                cached = None
        if cached is not None:
            return cached.get_or_build_balance(
                wb,
                lambda: self._build_balance(cached, wb),
            )

        # Single-flight per key (scheduler plan S1): the FIRST requester of
        # a key becomes its builder; concurrent requesters of the SAME key
        # wait on the key's event and reuse the result; DIFFERENT keys build
        # in parallel up to MAX_CONCURRENT_BUILDS (the old global build_lock
        # made every cold start queue behind every other file's cold start).
        while True:
            with self.lock:
                cached = self.entries.get(key)
                if cached is not None and (
                    not require_guidance or cached.bundle.raw_guidance is not None
                ):
                    self.entries.move_to_end(key)
                    winner = cached
                    break
                waiter = self._inflight.get(key)
                if waiter is None:
                    waiter = threading.Event()
                    self._inflight[key] = waiter
                    builder = True
                else:
                    builder = False
            if not builder:
                waiter.wait()
                err = getattr(waiter, "error", None)
                if err is not None:
                    # the error rides THIS flight's event object, so a later
                    # flight on the same key can never serve a stale failure
                    raise err
                continue  # re-check the memory entry the builder installed
            try:
                with self._build_slots:
                    cache_path = _cache_dir() / f"{digest}.npz"
                    built = _read_disk_entry(
                        cache_path,
                        path,
                        require_guidance,
                        expected_runtime=_scene_decoder_runtime_id(decoder) or None,
                    )
                    if built is None:
                        # The cold entry is always the one fixed as-shot
                        # DecodeContext. WB no longer participates in the
                        # disk/memory identity or decoder call.
                        source = dg.load_raw(
                            path,
                            highlight,
                            scene_half_size=False,
                            demosaic=demosaic,
                            wb_mode="camera",
                            decoder=decoder,
                            coreimage_version=coreimage_version,
                        )
                        analysis, _, _ = dg.analyze(source, 4, diagnostics=False)
                        built = build_proxy_entry(source, analysis, require_guidance)
                        _write_disk_entry(cache_path, built)
                with self.lock:
                    self.entries[key] = built
                    self.entries.move_to_end(key)
                    while len(self.entries) > MAX_MEMORY_PROXY_ITEMS:
                        self.entries.popitem(last=False)
                winner = built
                break
            except BaseException as exc:
                waiter.error = exc
                raise
            finally:
                with self.lock:
                    self._inflight.pop(key, None)
                waiter.set()
        return winner.get_or_build_balance(
            wb,
            lambda: self._build_balance(winner, wb),
        )

    @staticmethod
    def _build_balance(base: PreviewEntry, wb: str) -> PreviewEntry:
        bundle = dg.rebalance_raw_bundle(base.bundle, wb)
        if bundle.wb_mode == "camera":
            # The requested balance degraded to camera AsShot (missing multipliers or
            # calibration).  The scene pixels are exactly the base proxy's, so the
            # persisted camera Analysis is already the truth for them — and the proxy
            # DecodeContext deliberately carries no xyz_render, so a scene-only
            # reanalysis is both impossible and unnecessary here.  Keep the degraded
            # bundle (it carries the wb_degradation note the UI must surface) and
            # reuse the base analysis instead of recomputing it.
            child = PreviewEntry(bundle=bundle, analysis=base.analysis)
            # inherit the grain realization (review batch 16): the preview
            # renders through this balanced entry while export peeks the
            # BASE entry — a freshly minted id here silently changed the
            # exported grain under any non-AsShot white balance
            child.realization_id = base.realization_id
            return child
        analysis = dg.reanalyze_balanced_scene(base.analysis, bundle)
        child = PreviewEntry(bundle=bundle, analysis=analysis)
        child.realization_id = base.realization_id
        return child


PREVIEW_STORE = PreviewCache()
