# SPDX-License-Identifier: GPL-3.0-or-later
"""Select high-quality delivery using codec-specific candidates and readback.

These bounded error budgets are engineering policy, not a claim of perceptual
equivalence. Formation is fixed. Every candidate uses the actual output codec;
HDR candidates additionally pass the existing SDR/HDR round-trip gates.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

from ._deps import np


def coding_metrics(decoded, intended):
    """Full-resolution luma/chroma error with bounded row-band temporaries."""
    if (decoded.shape != intended.shape or decoded.dtype != np.uint8
            or intended.dtype != np.uint8 or decoded.ndim != 3
            or decoded.shape[2] != 3 or not decoded.size):
        raise ValueError("编码回读尺寸或像素格式不符")
    n = decoded.shape[0] * decoded.shape[1]
    luma_sq = chroma_sq = 0.0
    blocks = []
    for y in range(0, decoded.shape[0], 128):
        diff = decoded[y:y+128].astype(np.float32) - intended[y:y+128]
        # Code-domain JPEG luma; this measures additional coding error, not
        # scene radiometry or an HDR display luminance.
        dy = diff[..., 0]*.299 + diff[..., 1]*.587 + diff[..., 2]*.114
        luma_sq += float(np.sum(dy*dy, dtype=np.float64))
        for c in (0, 2):
            dc = diff[..., c] - dy
            chroma_sq += float(np.sum(dc*dc, dtype=np.float64))
        # Non-cancelling local error: ringing of opposite signs cannot hide.
        hh, ww = dy.shape[0]//8*8, dy.shape[1]//8*8
        if hh and ww:
            blocks.append(np.abs(dy[:hh, :ww]).reshape(hh//8, 8, ww//8, 8).mean(axis=(1, 3)).ravel())
            if hh < dy.shape[0]:
                blocks.append(np.abs(dy[hh:, :ww]).reshape(dy.shape[0]-hh, ww//8, 8).mean(axis=(0, 2)))
            if ww < dy.shape[1]:
                blocks.append(np.abs(dy[:hh, ww:]).reshape(hh//8, 8, dy.shape[1]-ww).mean(axis=(1, 2)))
            if hh < dy.shape[0] and ww < dy.shape[1]:
                blocks.append(np.asarray([np.abs(dy[hh:, ww:]).mean()]))
        else:
            blocks.append(np.asarray([np.mean(np.abs(dy))]))
    return {
        "coding_luma_rmse": math.sqrt(luma_sq/max(n, 1)),
        "coding_chroma_rmse": math.sqrt(chroma_sq/max(2*n, 1)),
        "coding_local_luma_p99": float(np.percentile(np.concatenate(blocks), 99)),
    }


def additional_error_acceptable(candidate, reference):
    """Modest additional error only; never loosen the delivery's absolute gates."""
    # One 8-bit code of luma RMS and 1.5 codes of local MAE are the default
    # engineering budget. Chroma stays close to the q99 reference; increasing
    # JPEG quality cannot restore detail discarded by chroma subsampling.
    # HDR local MAE has a 0.04 reference-white floor: a 0.004 rise from
    # 0.020 to 0.024 should not veto an otherwise faithful q98 delivery.
    # Absolute HDR gates, worst-highlight and chroma budgets still apply.
    for key, ratio, offset, floor in (
        ("coding_luma_rmse", 1.12, .05, 1.0),
        ("coding_chroma_rmse", 1.10, .10, 0.0),
        ("coding_local_luma_p99", 1.12, .10, 1.5),
        ("block_p95_luma_error", 1.10, .002, .04),
        ("highlight_max_luma_error", 1.00, .02, 0.0),
        ("chroma_error", 1.05, .002, 0.0),
    ):
        if key not in reference:
            continue
        value = float(candidate.get(key, float("inf")))
        if not math.isfinite(value) or value > max(floor, float(reference[key])*ratio + offset):
            return False
    return True


def select_encoding(out_path: Path, encode, *, encode_420=None, qualities=(99,98,97,96,95)):
    """encode(quality, private_path) returns verified metrics for one candidate.

    The first verified quality in the supplied descending list is the reference
    (JPEG defaults to q99–q95; HEIF supplies its own range). Smaller candidates
    must save at least 5% and satisfy the measured error budget. SDR may try
    4:2:0 against its 4:2:2 reference, at the selected quality. HDR keeps its
    selected reference sampling. No automatic q100 escalation, resizing or
    re-render; the destination is untouched until a verified result exists.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".agxraw-encode-", dir=out_path.parent) as td:
        reference = selected = info = None
        reference_bytes = 0
        attempts = []
        for quality in qualities:
            path = Path(td) / (f"q{quality}" + out_path.suffix)
            try:
                candidate = encode(quality, path)
                size = path.stat().st_size
                if reference is None:
                    reference, reference_bytes = candidate, size
                    selected, info = path, candidate
                    accepted = True
                else:
                    accepted = size <= reference_bytes*.95 and additional_error_acceptable(candidate, reference)
                attempts.append({"quality": quality, "bytes": size, "accepted": accepted,
                                 "chroma": candidate.get("delivery_chroma_requested"),
                                 "metrics": {k: v for k, v in candidate.items()
                                             if k.startswith("coding_")}})
                if accepted and size < selected.stat().st_size:
                    selected, info = path, candidate
            except (RuntimeError, ValueError) as exc:
                # A failed candidate cannot destroy the existing destination.
                attempts.append({"quality": quality, "accepted": False, "reason": str(exc)})
        if selected is None:
            raise RuntimeError("所有编码候选均未通过回读：" + attempts[-1]["reason"])
        if encode_420 is not None:
            path = Path(td) / ("alternate420" + out_path.suffix)
            quality = int(info["delivery_quality"])
            try:
                candidate = encode_420(quality, path)
                size = path.stat().st_size
                accepted = size <= selected.stat().st_size*.95 and additional_error_acceptable(candidate, reference)
                attempts.append({"quality": quality, "chroma": "420", "bytes": size,
                                 "accepted": accepted,
                                 "metrics": {k: v for k, v in candidate.items() if k.startswith("coding_")}})
                if accepted:
                    selected, info = path, candidate
            except (RuntimeError, ValueError) as exc:
                attempts.append({"quality": quality, "chroma": "420", "accepted": False, "reason": str(exc)})
        size = selected.stat().st_size
        os.replace(selected, out_path)
        return {
            **info, "delivery_profile": "auto", "file_size_bytes": size,
            "output_path": str(out_path), "auto_reference_quality": reference["delivery_quality"],
            "auto_reference_chroma": reference.get("delivery_chroma_requested"),
            "auto_reference_bytes": reference_bytes,
            "auto_saved_pct": 100.0*(1-size/max(reference_bytes, 1)),
            "auto_attempts": attempts,
        }
