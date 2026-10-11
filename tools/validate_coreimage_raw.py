#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Private Apple RAW version/output matrix; no image buffers persist.

Fixed RAW9/RAW8, LibRaw and auto are separate cases. A fallback cannot pass a
fixed-version case. Native interactive previews and full-decode resized
previews are different paths; decoder-native and relative aligned scale are
also separate. Ordinary photos establish execution, not noise calibration,
compression fidelity or independent decoder accuracy.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import rawpy

from dngscan import coreimage_decode
from dngscan.analysis import analyze
from dngscan.capture_identity import public_capture
from dngscan.auto_ev import compute_auto_ev
from dngscan.decoder_residual_research import CHROMA_BASIS, LUMA
from dngscan.calibration import calibration_diagnostics, calibration_fingerprint
from dngscan.calibration_registry import library_status
from dngscan.hdr_agx import render_ultrahdr_agx_pair
from dngscan.hdr_agx_plan import compile_hdr_agx_plan, compile_tail_snr_gate
from dngscan.models import HdrDisplayTarget
from dngscan.raw_io import load_raw, release_analysis_buffers
from dngscan.scene_scale import with_intent_exposure
from dngscan.scene_transform import decoder_anchor_fingerprint
from dngscan.tone import build_render_plan
from dngscan.protected_output import ProtectedSources
from dngscan.delivery_transaction import DeliveryTransaction
from validate_native_raw import digest, finite_summary, implementation_identity, region_bounds

MATRIX = ("raw9", "raw8", "libraw", "auto")
SIZES = ("full", "native-preview", "resized-full")


def _implementation_digest():
    value = {"processing": implementation_identity(), "validation_tool_sha256": digest(Path(__file__).resolve()),
             "decoder_anchor_asset_sha256": decoder_anchor_fingerprint()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _window_provenance(bundle):
    """Keep decoder-window applicability separate from sensor profile status."""
    return {"scene_window_calibration": copy.deepcopy(getattr(bundle, "scene_window_calibration", {}) or {}),
            "decoder_anchor_asset_sha256": decoder_anchor_fingerprint()}


def build_cases(matrix, sizes, scales, preview_max_edge=1280):
    """Stable case IDs; LibRaw has no Apple scale-policy axis."""
    cases = []
    for decoder in dict.fromkeys(matrix):
        if decoder not in MATRIX: raise ValueError("unknown decoder matrix member")
        for size in dict.fromkeys(sizes):
            if size not in SIZES: raise ValueError("unknown decode-size path")
            for scale in (["libraw"] if decoder == "libraw" else dict.fromkeys(scales)):
                if scale not in ("native", "aligned", "libraw"): raise ValueError("unknown scale policy")
                cases.append({"id": f"{decoder}/{size}/{scale}", "decoder": decoder,
                    "size": size, "scale": scale,
                    "resized_full_max_edge": preview_max_edge if size == "resized-full" else None,
                    "native_preview_target_long_edge": (coreimage_decode.COREIMAGE_PREVIEW_LONG_EDGE
                        if size == "native-preview" and decoder != "libraw" else None)})
    if not cases: raise ValueError("empty matrix")
    return cases


def resize_full(array, max_edge):
    """Deterministic area-average RGB resampling after a complete decode.

    Integer blocks are chosen by ceil(max_dimension/max_edge); incomplete
    edge blocks keep their actual sample count. This deliberately does not
    imitate Apple's native preview or promise its exact output dimensions.
    """
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[-1] != 3 or max_edge < 8:
        raise ValueError("expected HxWx3 and preview edge >=8")
    factor = max(1, math.ceil(max(image.shape[:2])/max_edge))
    if factor == 1: return image.copy(), {"algorithm": "area-block-mean-v1", "integer_factor": 1}
    height, width = image.shape[:2]
    y, x = np.arange(0, height, factor), np.arange(0, width, factor)
    # Block sums use float64 to keep their order/precision explicit and do
    # not clip signed or above-one linear samples.
    result = np.add.reduceat(np.add.reduceat(image.astype(np.float64), y, axis=0), x, axis=1)
    counts = np.minimum(factor, height-y)[:, None]*np.minimum(factor, width-x)[None, :]
    result /= counts[..., None]
    return result.astype(np.float32), {"algorithm": "area-block-mean-v1", "integer_factor": factor,
        "source_shape": list(image.shape), "output_shape": list(result.shape), "clipping": False}


def output_summary(array):
    image = np.asarray(array)
    finite = np.isfinite(image)
    # Preserve the original helper's labels. Counts supplement its summary;
    # they do not rename a decoder output as raw/sensor truth.
    if finite.all(): result = finite_summary(image)
    else:
        result = {"shape": list(image.shape), "dtype": str(image.dtype),
                  "finite_fraction": float(finite.mean()), "min_rgb": None,
                  "max_rgb": None, "sample_quantiles_rgb": None}
    result.update(negative_fraction=float(np.count_nonzero((image < 0)&finite)/image.size),
        above_one_fraction=float(np.count_nonzero((image > 1)&finite)/image.size),
        zero_fraction=float(np.count_nonzero((image == 0)&finite)/image.size))
    return result


def version_outcome(case, *, actual_decoder, actual_token, fallback, recipe):
    """Separate unavailable execution from an exact fixed-version success."""
    if case["decoder"] == "libraw":
        return (("passed", "actual-libraw-result") if actual_decoder == "libraw"
                else ("failed", "libraw-case-used-another-decoder"))
    if case["decoder"] == "auto" and actual_decoder == "libraw":
        return "passed", "actual-auto-libraw-fallback-not-apple-version-pass"
    if not isinstance(recipe, dict) or not recipe.get("configuration_id"):
        return "failed", "decoder-recipe-missing"
    if actual_token is None or recipe.get("actual_decoder_token") != actual_token:
        return "failed", "actual-decoder-token-unverified"
    if (recipe.get("linear_access") or {}).get("status") != "parameter-state-verified":
        return "failed", "linear-parameter-state-unverified"
    requested = {"raw9": "9", "raw8": "8"}.get(case["decoder"])
    if requested is not None:
        if actual_decoder != "coreimage" or str(actual_token).split(".")[0] != requested or fallback:
            return "failed", "fixed-version-did-not-run-without-fallback"
    return "passed", "actual-fixed-version" if requested else "actual-auto-result" if case["decoder"] == "auto" else "actual-libraw-result"


def scene_structure_summary(array, max_edge=1280):
    """Single-image spatial variation, explicitly not temporal noise evidence."""
    image, resampling = resize_full(np.asarray(array), max_edge)
    if not np.isfinite(image).all():
        return {"status": "unavailable-nonfinite", "noise_estimate": False}
    height, width = image.shape[:2]
    flat = image.reshape(-1, 3).astype(np.float64)
    luminance = flat @ LUMA
    chroma = flat @ CHROMA_BASIS
    centered = chroma-chroma.mean(0)
    covariance = centered.T @ centered/len(flat)
    field = centered.reshape(height,width,2)
    fft = np.fft.fft2(field,axes=(0,1))
    power = np.real(fft[..., :, None]*fft[..., None, :].conj())/(height*width)**2
    radial = np.hypot(np.fft.fftfreq(height)[:,None],np.fft.fftfreq(width)[None,:])
    edges=(0.,1/128.,1/64.,1/32.,1/16.,1/8.,math.inf)
    bands=[]
    for low,high in zip(edges[:-1],edges[1:]):
        matrix=power[(radial>=low)&(radial<high)].sum(0)
        bands.append({"minimum_cycles_per_diagnostic_pixel":low,
            "maximum_cycles_per_diagnostic_pixel":high if math.isfinite(high) else None,
            "zero_y_structure_covariance_power_2x2":matrix.tolist(),
            "zero_y_structure_energy":float(np.trace(matrix))})
    return {"status":"descriptive-scene-plus-noise", "noise_estimate":False,
        "input_shape":list(array.shape),"diagnostic_shape":list(image.shape),"resampling":resampling,
        "domain":"normalized-scene-linear-rec2020", "frequency_unit":"cycles/diagnostic-output-pixel",
        "Y_mean":float(luminance.mean()),"Y_median":float(np.median(luminance)),
        "Y_spatial_variance":float(luminance.var()),
        "Y_quantiles":np.quantile(luminance,[.001,.01,.1,.5,.9,.99,.999]).tolist(),
        "zero_y_mean_2":chroma.mean(0).tolist(),"zero_y_spatial_covariance_2x2":covariance.tolist(),
        "zero_y_basis_rgb_columns":CHROMA_BASIS.tolist(),"bands":bands,
        "interpretation":"contains scene structure, noise, decoder processing and geometry; not noise covariance or texture-loss evidence"}


def distribution_change(reference, actual):
    """Compare distributions without assuming spatial registration or truth."""
    a,b=reference["scene_structure"],actual["scene_structure"]
    ay,by=a["Y_median"],b["Y_median"]
    return {"reference_case":reference["case"]["id"],"actual_case":actual["case"]["id"],
        "reference_diagnostic_shape":a["diagnostic_shape"],"actual_diagnostic_shape":b["diagnostic_shape"],
        "Y_mean_difference":b["Y_mean"]-a["Y_mean"],
        "Y_median_EV_difference":math.log2(by/ay) if ay>0 and by>0 else None,
        "Y_quantile_differences":(np.array(b["Y_quantiles"])-a["Y_quantiles"]).tolist(),
        "zero_y_mean_difference_2":(np.array(b["zero_y_mean_2"])-a["zero_y_mean_2"]).tolist(),
        "zero_y_spatial_covariance_difference_2x2":(np.array(b["zero_y_spatial_covariance_2x2"])-a["zero_y_spatial_covariance_2x2"]).tolist(),
        "zero_y_band_structure_energy_differences":[y["zero_y_structure_energy"]-x["zero_y_structure_energy"]
                                                    for x,y in zip(a["bands"],b["bands"])],
        "pixel_comparison_qualified":False,
        "interpretation":"unregistered distribution/structure comparison; geometry or scene content changes can dominate; no noise/texture damage conclusion"}


def case_comparisons(samples):
    available={}
    for row in samples:
        if row.get("status")=="passed" and row.get("scene_structure",{}).get("status")=="descriptive-scene-plus-noise":
            available[(row["sha256"],row["case"]["decoder"],row["case"]["size"],row["case"]["scale"])]=row
    result=[]
    for (source,decoder,size,scale),row in available.items():
        partners=[]
        if size=="native-preview": partners.append(("native-preview-vs-full-downsample",(source,decoder,"resized-full",scale)))
        if scale=="native": partners.append(("native-vs-aligned-scale",(source,decoder,size,"aligned")))
        if decoder=="raw9": partners.append(("raw9-vs-raw8",(source,"raw8",size,scale)))
        for purpose,key in partners:
            if key in available:
                result.append({"purpose":purpose,"file":row["file"],"sha256":source,
                               **distribution_change(row,available[key])})
    return result


def _rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak*1024)


def _preflight(path, case):
    if case["decoder"] == "libraw": return {"status": "attempt", "versions_offered": []}
    if sys.platform != "darwin" or not coreimage_decode.available():
        return {"status": "unavailable", "reason": "coreimage-runtime-unavailable-on-this-host",
                "qualification": "host-unavailable-not-file-unsupported", "versions_offered": []}
    try: offered = coreimage_decode.supported_versions(path)
    except Exception as exc:
        return {"status": "unavailable", "reason": "coreimage-file-probe-failed",
                "error": f"{type(exc).__name__}: {exc}",
                "qualification": "probe-failure-not-a-file-support-conclusion", "versions_offered": []}
    requested = {"raw9": "9", "raw8": "8"}.get(case["decoder"])
    if requested and not any(str(token).split(".")[0] == requested for token in offered):
        return {"status": "unavailable", "reason": "fixed-version-not-offered-for-file",
                "qualification": "file-advertised-version-support-only", "versions_offered": list(offered)}
    if not coreimage_decode.runtime_available(interactive=case["size"] == "native-preview"):
        return {"status": "unavailable", "reason": "coreimage-render-context-unavailable",
                "qualification": "execution-context-unavailable-not-file-unsupported", "versions_offered": list(offered)}
    return {"status": "attempt", "versions_offered": list(offered)}


def run_case(path, case):
    started = time.monotonic()
    row = {"case": case, "file": path.name, "sha256": digest(path),
        "working_implementation_sha256": _implementation_digest(),
        "decoder_anchor_asset_sha256": decoder_anchor_fingerprint(),
        "calibration_fingerprint": calibration_fingerprint(),
        "size_bytes": path.stat().st_size, "status": "not-run",
        "parameters": {"highlight": "clip", "wb": "camera", "demosaic": "auto",
            "gamut": "p3", "tone": "agx/base", "target_peak_nits": 800,
            "look": "none", "scene_transform": "none", "chroma_nr": 0},
        "personal_pixel_correction": {"status": "not-applied", "reason": "no-production-hook-or-qualified-independent-body-captures"},
        "post_decoder_noise_calibration": {"status": "not-measured", "reason": "ordinary-RAW-is-not-independent-dark-flat-repeat-data"}}
    preflight = _preflight(path, case)
    row["preflight"] = preflight
    if preflight["status"] == "unavailable":
        row.update(status="unavailable", reason=preflight["reason"])
        row.update(elapsed_seconds=time.monotonic()-started, peak_process_rss_bytes=_rss_bytes())
        return row
    decoder = "libraw" if case["decoder"] == "libraw" else "coreimage"
    version = {"raw9": "9", "raw8": "8"}.get(case["decoder"], "auto")
    decode_started = time.monotonic()
    bundle = load_raw(path, "clip", scene_half_size=case["size"] == "native-preview",
        decoder=decoder, coreimage_version=version,
        coreimage_scale="unity" if case["scale"] in ("native", "libraw") else "aligned")
    row["decode_seconds"] = time.monotonic()-decode_started
    recipe = getattr(bundle, "scene_decoder_recipe", None)
    row.update(_window_provenance(bundle))
    actual_token = (recipe or {}).get("actual_decoder_token")
    status, reason = version_outcome(case, actual_decoder=bundle.scene_decoder,
        actual_token=actual_token, fallback=bundle.scene_decoder_fallback, recipe=recipe)
    row.update(status=status, reason=reason, actual_decoder=bundle.scene_decoder,
        actual_decoder_token=actual_token, scene_decoder_recipe=recipe,
        recipe_status="recorded" if (recipe or {}).get("configuration_id") else "no-apple-configuration-identity",
        actual_runtime=bundle.scene_decoder_runtime, fallback=bundle.scene_decoder_fallback,
        scene_scale_mode=bundle.scene_scale_mode, scene_align_factor=bundle.scene_align_factor,
        scene_align_error=bundle.scene_align_error, scene_scale=bundle.scene_scale,
        camera={"make": bundle.shot_make, "model": bundle.shot_model, "iso": bundle.shot_iso,
                "shutter": bundle.shot_shutter}, capture_readout=public_capture(bundle.capture_readout),
        source_codec={"suffix": path.suffix.lower(), "capture_descriptor": public_capture(bundle.capture_readout)},
        evidence={"provider": bundle.evidence_provider, "provider_version": bundle.evidence_provider_version,
            "status": getattr(bundle.evidence, "status", None), "raw_shape": list(bundle.raw_image.shape),
            "evidence_shape": bundle.evidence_shape, "scene_reliability_source": bundle.scene_reliability_source,
            "reference_error": bundle.scene_reference_error},
        noise_decode=bundle.noise_decode, lens_shading=bundle.lens_shading,
        scene_opcode_names=list(bundle.scene_opcode_names),
        decoder_output_scene=output_summary(bundle.scene_rec2020_render))
    analysis_started = time.monotonic()
    analysis, _, _ = analyze(bundle, 4, diagnostics=True, gamut_names=("P3",))
    auto = compute_auto_ev(bundle, analysis)
    bundle = with_intent_exposure(bundle, user_ev=auto.ev)
    plan = build_render_plan(bundle, analysis, "agx", "p3", chroma_nr=0.)
    hdr_plan = compile_hdr_agx_plan(plan, HdrDisplayTarget(peak_nits=800),
        analysis=analysis, scene_decoder=bundle.scene_decoder)
    row.update(analysis_plan_seconds=time.monotonic()-analysis_started, auto_ev=auto.ev,
        noise_model=dataclasses.asdict(analysis.noise_model), noise_hdr_factor=compile_tail_snr_gate(analysis),
        profiles=calibration_diagnostics(bundle.shot_make, bundle.shot_model, bundle.shot_shutter,
            bundle.shot_iso, bundle.capture_readout, analysis=analysis),
        hdr_peak_linear=hdr_plan.tone.peak_linear)
    regions = region_bounds(bundle.scene_rec2020_render)
    row["regions_yxxy"] = regions
    row["scene_regions"] = {name: output_summary(bundle.scene_rec2020_render[y0:y1,x0:x1])
        for name,(y0,x0,y1,x1) in regions.items()}
    scene = bundle.scene_rec2020_render
    if not math.isfinite(bundle.scene_scale) or bundle.scene_scale <= 0:
        raise RuntimeError("scene scale is not finite and positive")
    if case["size"] == "resized-full":
        preview, resize = resize_full(scene, case["resized_full_max_edge"])
        row["output_scene"] = output_summary(preview)
        row["scene_structure"] = scene_structure_summary(preview.astype(np.float32)/np.float32(bundle.scene_scale))
        row["preview_resampling"] = resize
        del preview
    else:
        row["output_scene"] = row["decoder_output_scene"]
        row["scene_structure"] = scene_structure_summary(scene.astype(np.float32)/np.float32(bundle.scene_scale))
    row["normalized_scene"] = output_summary(scene.astype(np.float32)/np.float32(bundle.scene_scale))
    row["size_semantics"] = {"full": "complete decoder output",
        "native-preview": "decoder native half/interactive path; not full resolution",
        "resized-full": "complete decode then deterministic area-block resize; not native preview"}[case["size"]]
    bundle = release_analysis_buffers(bundle)
    render_started = time.monotonic()
    sdr, hdr = render_ultrahdr_agx_pair(bundle, analysis, plan, hdr_plan, "p3", sdr_float=True)
    row.update(render_seconds=time.monotonic()-render_started, sdr=output_summary(sdr), hdr=output_summary(hdr),
        actual_processing={"chroma_cleanup": bundle.chroma_nr_status, "chroma_reason": bundle.chroma_nr_reason,
            "scene_transform": "none", "look": "none", "cfa_gated_colour": "not-requested",
            "decoder_controls": (recipe or {}).get("controls"),
            "personal_pixel_correction": "not-applied"})
    if not all(row[name]["finite_fraction"] == 1. for name in ("decoder_output_scene", "normalized_scene", "output_scene", "sdr", "hdr")):
        row.update(status="failed", reason="non-finite-output")
    row["source_sha256_after"] = digest(path)
    if row["source_sha256_after"] != row["sha256"]:
        row.update(status="failed", reason="source-changed-during-case")
    if _implementation_digest() != row["working_implementation_sha256"]:
        row.update(status="failed", reason="processing-code-changed-during-case")
    if calibration_fingerprint() != row["calibration_fingerprint"]:
        row.update(status="failed", reason="calibration-changed-during-case")
    row.update(elapsed_seconds=time.monotonic()-started, peak_process_rss_bytes=_rss_bytes())
    del sdr, hdr, scene, bundle, analysis, plan, hdr_plan
    gc.collect()
    return row


def _atomic_json(path, value, *, protected_sources=None):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    with DeliveryTransaction(path, protected_sources=protected_sources) as transaction:
        transaction.path.write_text(data, encoding="utf-8")
        transaction.commit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+")
    parser.add_argument("--matrix", choices=MATRIX, nargs="+", default=list(MATRIX))
    parser.add_argument("--sizes", choices=SIZES, nargs="+", default=list(SIZES))
    parser.add_argument("--scales", choices=("native", "aligned"), nargs="+", default=["native", "aligned"])
    parser.add_argument("--preview-max-edge", type=int, default=1280,
                        help="only resized-full area preview maximum edge; Apple native preview independently targets 1280")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=600.)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config:
        request = json.loads(args.worker_config)
        protection = ProtectedSources(request["path"]).check(args.worker_result)
        started = time.monotonic()
        try: result = run_case(Path(request["path"]), request["case"])
        except Exception as exc:
            result = {"case": request["case"], "file": Path(request["path"]).name,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic()-started, "peak_process_rss_bytes": _rss_bytes()}
        _atomic_json(args.worker_result, result, protected_sources=protection)
        return 0
    if not args.files or args.out is None: parser.error("--files and --out are required")
    protection = ProtectedSources(args.files).check(args.out)
    if args.preview_max_edge < 8 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("preview edge must be >=8 and timeout finite/positive")
    if args.out.exists(): parser.error("refusing to overwrite an existing report")
    for path in args.files:
        if not path.is_file(): parser.error(f"not a file: {path}")
    cases = build_cases(args.matrix, args.sizes, args.scales, args.preview_max_edge)
    report = {"schema": "agxraw-coreimage-raw-matrix-1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_changes": True, "implementation": implementation_identity(),
        "working_implementation_sha256": _implementation_digest(),
        "validation_tool_sha256": digest(Path(__file__).resolve()),
        "decoder_anchor_asset_sha256": decoder_anchor_fingerprint(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "rawpy": rawpy.__version__, "libraw_version": list(rawpy.libraw_version),
            "coreimage_runtime": coreimage_decode.decoder_runtime_id()},
        "calibration_fingerprint": calibration_fingerprint(), "public_library": library_status(),
        "comparison_scope": "actual decoder/configuration execution and finite SDR/HDR; no independent decoder accuracy or noise qualification",
        "missing_coverage": ["Nikon HE/HE* actual files", "independent body dark/flat/PTC repeated captures",
            "output residual/PSD calibration of adaptive Apple decoding", "full Apple gain-map container encoding matrix"],
        "rss_scope": "fresh worker process peak RSS includes imports/decode/analysis/render; no subtraction",
        "cases": cases, "samples": []}
    with tempfile.TemporaryDirectory(prefix="agxraw-coreimage-matrix-") as folder:
        for path in args.files:
            source_identity = {"file": path.name, "sha256": digest(path), "size_bytes": path.stat().st_size}
            for index, case in enumerate(cases):
                result_path = Path(folder)/f"result-{index}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker-config",
                    json.dumps({"path": str(path.resolve()), "case": case}), "--worker-result", str(result_path)]
                print(f"Checking {path.name}: {case['id']}", flush=True)
                started = time.monotonic()
                try:
                    process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    if process.returncode or not result_path.is_file():
                        row = {"case": case, "status": "failed", "reason": "worker-process-failed",
                            "exit_code": process.returncode, "stderr_tail": process.stderr[-4000:]}
                    else: row = json.loads(result_path.read_text())
                except subprocess.TimeoutExpired:
                    row = {"case": case, "status": "failed", "reason": "worker-timeout", "timeout_seconds": args.timeout}
                if row.get("sha256") not in (None, source_identity["sha256"]):
                    row.update(status="failed", reason="source-changed-between-cases",
                               expected_source_sha256=source_identity["sha256"])
                if row.get("working_implementation_sha256") not in (None, report["working_implementation_sha256"]):
                    row.update(status="failed", reason="processing-code-changed-between-cases")
                if row.get("calibration_fingerprint") not in (None, report["calibration_fingerprint"]):
                    row.update(status="failed", reason="calibration-changed-between-cases")
                for key,value in source_identity.items(): row.setdefault(key,value)
                row["worker_wall_seconds"] = time.monotonic()-started
                report["samples"].append(row)
                report["summary"] = {state: sum(r["status"] == state for r in report["samples"])
                                     for state in ("passed", "unavailable", "failed")}
                report["case_distribution_comparisons"]=case_comparisons(report["samples"])
                _atomic_json(args.out, report, protected_sources=protection)
                print(f"{row['status']}: {row.get('reason', row.get('error', ''))}", flush=True)
                result_path.unlink(missing_ok=True)
    return int(any(row["status"] == "failed" for row in report["samples"]))


if __name__ == "__main__": raise SystemExit(main())
