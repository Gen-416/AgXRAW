# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoder-independent RAW sensor evidence acquisition."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._deps import np, rawpy
from .models import RawEvidence


class EvidenceAcquisitionError(RuntimeError):
    """LibRaw could not provide the evidence contract for a RAW file."""

    def __init__(self, message: str, *, unsupported_format: bool = False) -> None:
        super().__init__(message)
        self.unsupported_format = bool(unsupported_format)


def libraw_runtime_id() -> str | None:
    """Return the actual rawpy/LibRaw runtime used for evidence."""
    if rawpy is None:
        return None
    rawpy_version = str(getattr(rawpy, "__version__", "unknown"))
    version = getattr(rawpy, "libraw_version", None)
    if version is None:
        return f"rawpy {rawpy_version}/LibRaw unknown"
    try:
        libraw_version = ".".join(str(int(part)) for part in version)
    except (TypeError, ValueError):
        libraw_version = str(version)
    return f"rawpy {rawpy_version}/LibRaw {libraw_version}"


def _decode_color_desc(desc: Any) -> str:
    if isinstance(desc, bytes):
        text = desc.decode("ascii", errors="replace")
    else:
        text = str(desc)
    return text.replace("\x00", "").strip()


def acquire_raw_evidence(path: Path) -> RawEvidence:
    """Acquire the one evidence source shared by every scene decoder.

    Deliberately no scene-decoder argument: selecting Apple RAW, LibRaw, or a future
    renderer cannot affect evidence acquisition or its provenance.

    R6 item 2 — the evidence MODEL is "per-channel black + linear DN", which
    is an assumption, not a verified fact: a legal DNG may carry
    LinearizationTable, BlackLevelDeltaH/V or a non-default
    LinearResponseLimit, none of which this layer applies. load_raw detects
    those tags (metadata.read_dng_stage1_flags) and stamps
    bundle.evidence_stage1_note so every precision claim downstream (noise
    floor, clip statistics, reliable tail, RAW gating) degrades honestly in
    the report instead of silently proceeding. Applying the corrections (or
    sourcing stage-1-corrected buffers) is the recorded follow-up.
    """
    path = Path(path)
    try:
        with rawpy.imread(str(path)) as raw:
            raw_image = np.asarray(raw.raw_image_visible).copy()
            raw_colors = np.asarray(raw.raw_colors_visible).copy()
            if raw_image.size == 0 or raw_colors.size == 0:
                raise RuntimeError("decoded RAW has no visible sensor pixels")
            if raw_image.shape != raw_colors.shape:
                raise RuntimeError(
                    "raw_image_visible and raw_colors_visible shapes differ"
                )

            white_level = getattr(raw, "white_level", None)
            if white_level is None:
                white_level = int(np.max(raw_image))

            raw_pattern_attr = getattr(raw, "raw_pattern", None)
            raw_pattern = (
                []
                if raw_pattern_attr is None
                else np.asarray(raw_pattern_attr).astype(int).tolist()
            )
            daylight_attr = getattr(raw, "daylight_whitebalance", None)
            black_attr = getattr(raw, "black_level_per_channel", None)
            wb_attr = getattr(raw, "camera_whitebalance", None)
            white_pc_attr = getattr(raw, "camera_white_level_per_channel", None)
            orientation_flip = int(
                getattr(getattr(raw, "sizes", object), "flip", 0) or 0
            )
            matrix = getattr(raw, "rgb_xyz_matrix", None)
            xyz_to_cam = None if matrix is None else np.asarray(matrix).copy()
            # rgb_cam: the camera -> linear-sRGB matrix LibRaw actually decodes
            # through.  Some DNGs (Sigma fp) leave rgb_xyz_matrix all-zero while
            # this matrix is valid, so both are evidence.
            decode_matrix = getattr(raw, "color_matrix", None)
            color_matrix = (
                None if decode_matrix is None else np.asarray(decode_matrix).copy()
            )

            return RawEvidence(
                path=path,
                raw_image=raw_image,
                raw_colors=raw_colors,
                white_level=int(white_level),
                black_levels=(
                    [float(value) for value in black_attr]
                    if black_attr is not None
                    else []
                ),
                camera_wb=(
                    [float(value) for value in wb_attr]
                    if wb_attr is not None
                    else []
                ),
                daylight_wb=(
                    [float(value) for value in daylight_attr]
                    if daylight_attr is not None
                    else None
                ),
                color_desc=_decode_color_desc(getattr(raw, "color_desc", "")),
                raw_pattern=raw_pattern,
                camera_white_levels=(
                    [float(value) for value in white_pc_attr]
                    if white_pc_attr is not None
                    else []
                ),
                orientation_flip=orientation_flip,
                xyz_to_cam=xyz_to_cam,
                provider_version=libraw_runtime_id(),
                color_matrix=color_matrix,
            )
    except FileNotFoundError:
        raise
    except rawpy.LibRawFileUnsupportedError as exc:
        raise EvidenceAcquisitionError(
            str(exc), unsupported_format=True
        ) from exc
    except EvidenceAcquisitionError:
        raise
    except Exception as exc:
        raise EvidenceAcquisitionError(str(exc)) from exc
