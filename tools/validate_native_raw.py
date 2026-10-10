#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Private RAW acceptance manifest; no photos or full-resolution buffers persist.

Runs production decode/analysis/AgX SDR/HDR with NumPy and strict native kernels.
Their comparison establishes implementation parity, not independent decoder
accuracy or recovery of samples discarded by the camera's RAW compression.
An independent reference can be added to the manifest only after its geometry,
linear units, black/white convention and decoder version have been established.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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

from dngscan import _fast
from dngscan.analysis import analyze
from dngscan.auto_ev import compute_auto_ev
from dngscan.hdr_agx import render_ultrahdr_agx_pair
from dngscan.hdr_agx_plan import compile_hdr_agx_plan, compile_tail_snr_gate
from dngscan.models import HdrDisplayTarget
from dngscan.raw_io import load_raw, release_analysis_buffers
from dngscan.scene_scale import with_intent_exposure
from dngscan.tone import build_render_plan


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def finite_summary(array):
    flat = np.asarray(array).reshape(-1, array.shape[-1])
    low, high, finite = np.full(flat.shape[1], np.inf), np.full(flat.shape[1], -np.inf), 0
    for start in range(0, len(flat), 250000):
        part = flat[start:start+250000]
        finite += int(np.isfinite(part).sum())
        low, high = np.minimum(low, np.min(part, axis=0)), np.maximum(high, np.max(part, axis=0))
    stride = max(1, len(flat)//100000)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "finite_fraction": finite/array.size, "min_rgb": low.tolist(), "max_rgb": high.tolist(),
            "sample_quantiles_rgb": np.quantile(flat[::stride], [.001,.01,.5,.99,.999], axis=0).tolist()}


def region_bounds(array):
    """Deterministic support regions; extreme blocks are descriptive, not truth."""
    h, w = array.shape[:2]
    size = min(64, h, w)
    regions = {"centre": (h//2-size//2, w//2-size//2),
               "top_left": (0,0), "top_right": (0,w-size),
               "bottom_left": (h-size,0), "bottom_right": (h-size,w-size)}
    thumb = array[::max(1,h//128), ::max(1,w//128)].astype(np.float64)
    y = thumb @ np.array([.2627,.6780,.0593])
    for name, index in (("dark", np.argmin(y)), ("highlight", np.argmax(y)),
                        ("edge", np.argmax(np.hypot(*np.gradient(y))))):
        ty, tx = np.unravel_index(index, y.shape)
        regions[name] = (min(max(0,int(ty*h/y.shape[0])-size//2),h-size),
                         min(max(0,int(tx*w/y.shape[1])-size//2),w-size))
    return {name: [row,col,row+size,col+size] for name,(row,col) in regions.items()}


def difference(actual, reference):
    if actual.shape != reference.shape:
        return {"shape_match": False}
    channels = actual.shape[-1]
    a, b = actual.reshape(-1,channels), reference.reshape(-1,channels)
    square, signed, maximum = np.zeros(channels), np.zeros(channels), np.zeros(channels)
    unequal = 0
    for start in range(0,len(a),250000):
        delta = a[start:start+250000].astype(np.float64)-b[start:start+250000]
        square += np.square(delta).sum(axis=0)
        signed += delta.sum(axis=0)
        maximum = np.maximum(maximum, np.abs(delta).max(axis=0))
        unequal += int(np.count_nonzero(delta))
    return {"shape_match": True, "max_abs_rgb": maximum.tolist(),
            "rmse_rgb": np.sqrt(square/len(a)).tolist(), "mean_signed_rgb": (signed/len(a)).tolist(),
            "unequal_fraction": unequal/actual.size}


def phase_summary(bundle):
    result = {}
    raw = bundle.raw_image
    if raw.ndim != 2:
        return {"status": "not-a-CFA-mosaic"}
    for cid in np.unique(bundle.raw_colors):
        cid = int(cid)
        values = raw[bundle.raw_colors == cid]
        occupied = np.unique(values)
        white = bundle.camera_white_levels[cid] if cid < len(bundle.camera_white_levels) else bundle.white_level
        result[str(cid)] = {"colour": bundle.color_desc[cid], "count": int(values.size),
            "min": int(values.min()), "max": int(values.max()),
            "quantiles": np.quantile(values,[0,.001,.01,.5,.99,.999,1]).tolist(),
            "black_dn": bundle.black_levels[cid], "linear_validity_white_dn": white,
            "coding_endpoint_fraction": float(np.mean(values >= bundle.white_level)),
            "linear_ceiling_fraction": float(np.mean(values >= white)),
            "occupied_codes": len(occupied),
            "occupied_step_quantiles": np.quantile(np.diff(occupied),[0,.5,.95,1]).tolist() if len(occupied)>1 else []}
    return result


def run_file(path, half, scratch):
    row = {"file": path.name, "sha256": digest(path), "size_bytes": path.stat().st_size,
           "half_size": half, "independent_decoder_reference": {"status": "not-available"},
           "parameters": {"decoder": "libraw", "highlight": "clip", "demosaic": "auto",
                          "wb": "camera", "gamut": "p3", "tone": "agx/base", "target_peak_nits": 800,
                          "chroma_nr": 0, "look": "none", "scene_transform": "none"}, "runs": {}}
    for name, fast in (("numpy","0"), ("native","1")):
        started = time.monotonic()
        os.environ["DNGSCAN_FAST"] = fast
        if fast == "1" and not _fast.available():
            raise RuntimeError("strict native extension unavailable")
        bundle = load_raw(path, "clip", scene_half_size=half)
        analysis, _, _ = analyze(bundle, 4, diagnostics=True, gamut_names=("P3",))
        auto = compute_auto_ev(bundle, analysis)
        bundle = with_intent_exposure(bundle, user_ev=auto.ev)
        plan = build_render_plan(bundle, analysis, "agx", "p3")
        hdr_plan = compile_hdr_agx_plan(plan, HdrDisplayTarget(peak_nits=800),
                                        analysis=analysis, scene_decoder=bundle.scene_decoder)
        run = {"decode_supported": True, "render_supported": False,
               "noise_model_usable": analysis.noise_model.status == "valid",
               "measurement_qualified": (analysis.noise_model.status == "valid"
                                          and analysis.noise_model.source != "DNG NoiseProfile"),
               "capture_readout": bundle.capture_readout, "raw_shape": list(bundle.raw_image.shape),
               "coding_white_levels": bundle.coding_white_levels, "coding_black_levels": bundle.coding_black_levels,
               "linear_validity_white_levels": bundle.camera_white_levels,
               "scene_scale": bundle.scene_scale, "scene_decoder": bundle.scene_decoder,
               "actual_demosaic": bundle.noise_decode.get("demosaic_algorithm"), "noise_status": analysis.noise_model.status,
               "noise_source": analysis.noise_model.source, "noise_reason": analysis.noise_model.reason,
               "noise_hdr_factor": compile_tail_snr_gate(analysis), "noise_decode": bundle.noise_decode,
               "auto_ev": auto.ev, "hdr_peak_linear": hdr_plan.tone.peak_linear,
               "scene": finite_summary(bundle.scene_rec2020_render)}
        if name == "numpy":
            row["camera"] = {"make": bundle.shot_make, "model": bundle.shot_model, "iso": bundle.shot_iso,
                             "shutter": bundle.shot_shutter, "firmware": None}
            row["raw_phases"] = phase_summary(bundle)
            row["regions_yxxy"] = region_bounds(bundle.scene_rec2020_render)
            row["scene_cfa_pattern"] = np.asarray(bundle.raw_pattern).tolist()
        regions = row["regions_yxxy"]
        run["scene_regions"] = {key: finite_summary(bundle.scene_rec2020_render[y0:y1,x0:x1])
                                 for key,(y0,x0,y1,x1) in regions.items()}
        for stage, array in (("scene",bundle.scene_rec2020_render),):
            reference_file = scratch / (stage+".npy")
            if name == "numpy":
                np.save(reference_file,array)
            else:
                reference = np.load(reference_file,mmap_mode="r")
                run[stage+"_numpy_difference"] = difference(array,reference)
                del reference
        bundle = release_analysis_buffers(bundle)
        base, hdr = render_ultrahdr_agx_pair(bundle,analysis,plan,hdr_plan,"p3",sdr_float=True)
        for stage, array in (("sdr",base),("hdr",hdr)):
            run[stage] = finite_summary(array)
            reference_file = scratch / (stage+".npy")
            if name == "numpy":
                np.save(reference_file,array)
            else:
                reference = np.load(reference_file,mmap_mode="r")
                run[stage+"_numpy_difference"] = difference(array,reference)
                run[stage+"_numpy_regions"] = {key:difference(array[y0:y1,x0:x1],reference[y0:y1,x0:x1])
                    for key,(y0,x0,y1,x1) in regions.items()}
                del reference
        run["render_supported"] = True
        run["elapsed_seconds"] = time.monotonic()-started
        row["runs"][name] = run
        del base,hdr,array,bundle,analysis,plan,hdr_plan
        gc.collect()
    row["parity_checks"] = parity_checks(row)
    return row


def parity_checks(row):
    """Implementation budgets, not visual-quality or decoder-truth limits."""
    checks = {}
    native = row["runs"]["native"]
    for name in ("numpy","native"):
        checks[name+"_finite"] = all(row["runs"][name][stage]["finite_fraction"] == 1.
                                      for stage in ("scene","sdr","hdr"))
    for stage in ("scene","sdr","hdr"):
        metric = native[stage+"_numpy_difference"]
        divisor = native["scene_scale"] if stage == "scene" else 1.
        checks[stage+"_normalized_max_abs_le_1e-4"] = (metric["shape_match"]
            and max(metric["max_abs_rgb"])/divisor <= 1e-4)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files",type=Path,nargs="+")
    parser.add_argument("--sizes",choices=("full","half"),nargs="+",default=["half"])
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    report = {"schema":"agxraw-native-raw-acceptance-1", "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "working_changes":True, "runtime":{"python":platform.python_version(),"platform":platform.platform(),
                "numpy":np.__version__,"rawpy":rawpy.__version__,"libraw_version":list(rawpy.libraw_version),
                "libraw_commit":"e419de08001de28ae6988ecb22df47e52b9c5eaa"},
              "comparison_scope":"NumPy/native pipeline parity; not independent RAW decoder accuracy or compression fidelity",
              "qualification_contract":"measurement_qualified means matched external model; noise_model_usable also includes file-declared DNG variance",
              "missing_coverage":["Sony other codecs/APS-C/shutter/ISO modes", "Sony controlled dark/flat/near-saturation calibration pairs",
                                  "Nikon native NEF, HE and HE*", "independent manufacturer/Adobe reference in equivalent units"], "samples":[]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agxraw-native-acceptance-") as directory:
        for path in args.files:
            for size in args.sizes:
                print(f"Validating {path.name} ({size})",flush=True)
                try:
                    row = run_file(path,size=="half",Path(directory))
                except Exception as exc:
                    row = {"file":path.name,"half_size":size=="half","error":str(exc)}
                report["samples"].append(row)
                args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
                print(f"{'FAILED' if 'error' in row else 'Completed'} {path.name} ({size})",flush=True)
    return 1 if any("error" in row or not all(row.get("parity_checks",{}).values())
                    for row in report["samples"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
