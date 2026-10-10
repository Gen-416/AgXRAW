#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-plan three-way RAW comparison. Ordinary photos are not calibration.

Private PNG sheets show 100%, 200% nearest-neighbour and sharing-size SDR.
JSON records actual application/skips, SDR/HDR changes and source identities.
Single-image statistics are descriptive, never estimates of temporal noise or
proof of texture fidelity. No personal pixel map is applied without a qualified
signed decoder chain; missing calibration is recorded rather than fabricated.
"""
from __future__ import annotations
import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import rawpy
from PIL import Image, ImageDraw
from dngscan import _fast
from dngscan.analysis import analyze
from dngscan.auto_ev import compute_auto_ev
from dngscan.calibration import calibration_diagnostics, calibration_fingerprint
from dngscan.capture_identity import public_capture
from dngscan.calibration_registry import library_status
from dngscan.chroma_policy import resolve_chroma_policy
from dngscan.hdr_agx import render_ultrahdr_agx_pair
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.models import HdrDisplayTarget
from dngscan.raw_io import load_raw, release_analysis_buffers
from dngscan.scene_scale import with_intent_exposure
from dngscan.tone import build_render_plan
from validate_native_raw import digest, difference, finite_summary, region_bounds, implementation_identity


def preview_image(array):
    # Display sheet only: output/container precision is tested elsewhere.
    return Image.fromarray(np.rint(np.clip(array, 0, 1)*255).astype(np.uint8))


def run_file(path, *, half, pictures, scratch):
    bundle = load_raw(path, "clip", scene_half_size=half)
    analysis, _, _ = analyze(bundle, 4, diagnostics=True, gamut_names=("P3",))
    auto = compute_auto_ev(bundle, analysis)
    bundle = with_intent_exposure(bundle, user_ev=auto.ev)
    row = {"file": path.name, "sha256": digest(path), "half_size": half,
        "camera": {"make": bundle.shot_make, "model": bundle.shot_model, "iso": bundle.shot_iso},
        "scene_decoder": bundle.scene_decoder, "capture_readout": public_capture(bundle.capture_readout),
        "noise_decode": bundle.noise_decode, "noise_model": dataclasses.asdict(analysis.noise_model),
        "calibrations": calibration_diagnostics(bundle.shot_make, bundle.shot_model,
            bundle.shot_shutter, bundle.shot_iso, bundle.capture_readout, analysis=analysis),
        "fixed_parameters": {"ev": auto.ev, "wb": "camera", "gamut": "p3", "tone": "agx/base",
            "highlight": "clip", "scene_transform": "none", "look": "none", "sharpening": False},
        "personal_system_correction": {"status": "not-applied",
            "reason": "no-qualified-body-dark-flat-captures; production-signed-decode-not-qualified"},
        "groups": {}, "regions_yxxy": region_bounds(bundle.scene_rec2020_render)}
    bundle = release_analysis_buffers(bundle)
    names = (("new-corrections-off", "off"), ("system-only", "off"),
             ("system-and-conservative-chroma", "conservative"))
    sheet = Image.new("RGB", (1280, 3*540), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, (name, mode) in enumerate(names):
        policy = resolve_chroma_policy(mode)
        plan = build_render_plan(bundle, analysis, "agx", "p3", chroma_nr=policy.amount)
        hdr_plan = compile_hdr_agx_plan(plan, HdrDisplayTarget(peak_nits=800),
            analysis=analysis, scene_decoder=bundle.scene_decoder)
        base, hdr = render_ultrahdr_agx_pair(bundle, analysis, plan, hdr_plan, "p3", sdr_float=True)
        group = {"policy": policy.to_dict(), "chroma_status": bundle.chroma_nr_status,
            "chroma_reason": bundle.chroma_nr_reason, "sdr": finite_summary(base),
            "hdr": finite_summary(hdr), "black_ev": plan.tone.black_ev, "white_ev": plan.tone.white_ev}
        group["calibrations"] = calibration_diagnostics(
            bundle.shot_make, bundle.shot_model, bundle.shot_shutter,
            bundle.shot_iso, bundle.capture_readout, analysis=analysis,
            chroma_nr={"status": bundle.chroma_nr_status, "reason": bundle.chroma_nr_reason})
        for stage, array in (("sdr", base), ("hdr", hdr)):
            reference_path = scratch / (stage+".npy")
            if index == 0: np.save(reference_path, array)
            else:
                reference = np.load(reference_path, mmap_mode="r")
                group[stage+"_difference_from_off"] = difference(array, reference)
                group[stage+"_regions"] = {key: difference(array[y0:y1, x0:x1], reference[y0:y1, x0:x1])
                    for key, (y0,x0,y1,x1) in row["regions_yxxy"].items()}
                del reference
        offset = index*540
        draw.text((8, offset+5), name+" | "+bundle.chroma_nr_status+" | "+str(bundle.chroma_nr_reason or ""), fill="white")
        draw.text((8, offset+25), "100% (decoded pixels) / 200% nearest neighbour / sharing size", fill="white")
        image = preview_image(base)
        thumb = image.copy(); thumb.thumbnail((640, 480), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (640, offset+50))
        for number, key in enumerate(("centre", "dark", "edge", "highlight")):
            y0,x0,y1,x1 = row["regions_yxxy"][key]
            crop = image.crop((x0,y0,x1,y1))
            col, top = number%2*300, offset+55+number//2*230
            draw.text((col+8, top), key, fill="white")
            sheet.paste(crop, (col+8, top+20))
            sheet.paste(crop.resize((crop.width*2, crop.height*2), Image.Resampling.NEAREST), (col+100, top+20))
        row["groups"][name] = group
        del base, hdr, array, image, thumb, plan, hdr_plan
        gc.collect()
    pictures.mkdir(parents=True, exist_ok=True)
    image_path = pictures/(path.stem+"-three-way.png")
    sheet.save(image_path)
    row["private_comparison_sheet"] = str(image_path.resolve())
    row["interpretation"] = "System-only equals off when no qualified personal correction exists. No real-camera calibration or noise/texture benefit is inferred. Half-size sheets use decoded, not native sensor pixels."
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pictures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "agxraw-calibrated-three-way-1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_changes": True, "implementation": implementation_identity(),
        "calibration_fingerprint": calibration_fingerprint(),
        "public_library": library_status(), "runtime": {"python": platform.python_version(), "numpy": np.__version__,
            "rawpy": rawpy.__version__, "libraw": list(rawpy.libraw_version), "native": _fast.available(),
            "libraw_commit": "e419de08001de28ae6988ecb22df47e52b9c5eaa"},
        "scope": "fixed-plan real RAW SDR/HDR comparison; no actual independent dark/flat or low-noise truth",
        "missing_measurements": ["body-specific dark/flat/PTC and independent heldout", "fine chroma decoder covariance", "fp SSF/lens spectral measurements"],
        "samples": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agxraw-three-way-") as folder:
        for path in args.files:
            print("Comparing "+path.name, flush=True)
            try: row = run_file(path, half=not args.full, pictures=args.pictures, scratch=Path(folder))
            except Exception as error: row = {"file": path.name, "error": str(error)}
            report["samples"].append(row)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return int(any("error" in row for row in report["samples"]))


if __name__ == "__main__":
    raise SystemExit(main())
