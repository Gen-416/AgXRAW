#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure exact-scope RAW9 anchors using verified flat target ROIs.

Coarse natural-image blocks are not registration. This initial tool requires
corresponding dedicated flat targets and independent held-out files. It writes
a candidate report; installing a qualified candidate is a separate command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dngscan.analysis import analyze  # noqa: E402
from dngscan.raw_io import load_raw  # noqa: E402
from dngscan.scene_transform import (  # noqa: E402
    DECODER_ANCHOR_SCHEMA, DECODER_ANCHOR_TRANSPORT_JSON, SCENE_TRANSFORMS,
    ANCHOR_RATIO_BOUNDS, ANCHOR_MAX_HOLDOUT_MEDIAN_ERROR, ANCHOR_MAX_HOLDOUT_P95_ERROR,
    _qualified_anchor_entry, _validated_anchor_scope, decoder_anchor_scope,
)
from dngscan.tone import scene_intent_rec2020  # noqa: E402
from dngscan.protected_output import ProtectedSources  # noqa: E402
from dngscan.delivery_transaction import DeliveryTransaction  # noqa: E402
from dngscan.source_metadata import source_metadata_session  # noqa: E402

MANIFEST_SCHEMA = "raw9-anchor-manifest-1"
CANDIDATE_SCHEMA = "raw9-anchor-candidate-2"
RATIO_BOUNDS = ANCHOR_RATIO_BOUNDS  # Measurement guard; never clamp into qualification.
MIN_ROI_PIXELS = 64
MIN_INDEPENDENT_FRAMES = 2  # In each of train and holdout.
MAX_FLAT_RELATIVE_MAD = .03
MAX_CHROMA_RELATIVE_MAD = .02
MAX_HOLDOUT_MEDIAN_ERROR = ANCHOR_MAX_HOLDOUT_MEDIAN_ERROR
MAX_HOLDOUT_P95_ERROR = ANCHOR_MAX_HOLDOUT_P95_ERROR


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_current_source(session, phase):
    if not session.assert_consistent(phase=phase):
        raise ValueError("anchor-source-identity-unavailable")


def _frame_paths(document, directory):
    """Discover declared inputs for preflight; measurement validates their contract."""
    frames = document.get("frames", []) if isinstance(document, dict) else []
    if not isinstance(frames, list):
        return []
    return [Path(directory) / frame["path"] for frame in frames
            if isinstance(frame, dict) and isinstance(frame.get("path"), str)
            and frame["path"]]


def _roi_pixels(bundle, bounds):
    image = np.asarray(bundle.scene_rec2020_render)
    if (image.ndim != 3 or image.shape[-1] < 3 or not isinstance(bounds, list)
            or len(bounds) != 4 or any(type(value) is not int for value in bounds)):
        raise ValueError("invalid-decoded-roi")
    y0, x0, y1, x1 = bounds
    height, width = image.shape[:2]
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("roi-outside-actual-decoded-geometry")
    # Separate coordinates in each decoder; no assumed registration/resampling.
    y0, x0, y1, x1 = y0 + 2, x0 + 2, y1 - 2, x1 - 2
    patch = image[y0:y1, x0:x1, :3]
    if min(patch.shape[:2], default=0) < 8 or patch.size // 3 < MIN_ROI_PIXELS:
        raise ValueError("flat-target-roi-insufficient-support")
    if not np.isfinite(patch).all():
        raise ValueError("nonfinite-decoded-roi")
    pixels = scene_intent_rec2020(patch.reshape(-1, 3), bundle, 1.)
    if not (np.isfinite(pixels).all() and np.all(pixels > 1e-5)
            and np.all(np.max(pixels, axis=1) > .02) and np.all(pixels < .90)):
        raise ValueError("roi-low-signal-negative-or-near-clipped")
    mask = getattr(bundle, "scene_reliability_exclusion", None)
    if mask is not None and np.asarray(mask).shape[:2] == image.shape[:2]:
        if np.any(np.asarray(mask)[y0:y1, x0:x1]):
            raise ValueError("roi-depends-on-unreliable-sensor-samples")
    median = np.median(pixels, axis=0)
    relative_mad = 1.4826 * np.median(np.abs(pixels - median), axis=0) / median
    chroma = pixels[:, (0, 2)] / pixels[:, 1, None]
    chroma_median = np.median(chroma, axis=0)
    chroma_mad = 1.4826 * np.median(np.abs(chroma - chroma_median), axis=0) / chroma_median
    if np.max(relative_mad) > MAX_FLAT_RELATIVE_MAD or np.max(chroma_mad) > MAX_CHROMA_RELATIVE_MAD:
        raise ValueError("roi-not-qualified-as-flat-target")
    grid = pixels.reshape(patch.shape)
    gradients = np.concatenate((np.abs(np.diff(grid, axis=0)).reshape(-1, 3),
                                np.abs(np.diff(grid, axis=1)).reshape(-1, 3))) / median
    gradient_p99 = float(np.percentile(gradients, 99))
    if gradient_p99 > .08:
        raise ValueError("roi-has-unqualified-spatial-structure")
    return median, {"bounds": [y0, x0, y1, x1], "decoded_shape": list(image.shape),
        "pixels": int(pixels.shape[0]), "relative_mad": relative_mad.tolist(),
        "chroma_relative_mad": chroma_mad.tolist(), "gradient_p99": gradient_p99}


def _fit_transport(samples, weights=None):
    weights = np.ones(len(samples)) if weights is None else np.asarray(weights)
    train = np.asarray([row["split"] == "train" for row in samples]) & (weights > .05)
    holdout = np.asarray([row["split"] == "holdout" for row in samples]) & (weights > .05)
    counts = {"train": len({row["input_sha256"] for row, used in zip(samples, train) if used}),
              "holdout": len({row["input_sha256"] for row, used in zip(samples, holdout) if used})}
    result = {"status": "unmeasured", "independent_frames": counts}
    if min(counts.values()) < MIN_INDEPENDENT_FRAMES:
        return {**result, "reason": "insufficient-independent-train-or-holdout-support"}
    ratios = np.asarray([row["ratio_rg_bg"] for row in samples], dtype=np.float64)
    fit, fit_weights = ratios[train], weights[train]
    ratio = np.asarray([fit[order, axis][np.searchsorted(np.cumsum(fit_weights[order]), fit_weights.sum() / 2)]
                        for axis in range(2) for order in [np.argsort(fit[:, axis])]])
    result["ratio_rg_bg"] = ratio.tolist()
    if not np.isfinite(ratio).all() or np.any(ratio < RATIO_BOUNDS[0]) or np.any(ratio > RATIO_BOUNDS[1]):
        return {**result, "status": "rejected", "reason": "measured-ratio-outside-guard-no-clamping"}
    residual = np.abs(ratios[holdout] / ratio - 1.)
    median_error, p95_error = np.median(residual, axis=0), np.percentile(residual, 95, axis=0)
    result.update(holdout_median_relative_error=median_error.tolist(),
        holdout_p95_relative_error=p95_error.tolist(), train_ratio_min=np.min(fit, axis=0).tolist(),
        train_ratio_max=np.max(fit, axis=0).tolist(),
        uncertainty="between-file spread; spatial samples are not independent replicates")
    if np.max(median_error) > MAX_HOLDOUT_MEDIAN_ERROR or np.max(p95_error) > MAX_HOLDOUT_P95_ERROR:
        return {**result, "status": "rejected", "reason": "independent-holdout-disagreement"}
    return {**result, "status": "qualified", "reason": "flat-target-independent-holdout-passed"}


def measure_candidate(manifest, directory, *, loader=load_raw, analyzer=analyze):
    report = {"schema": CANDIDATE_SCHEMA, "status": "unmeasured", "profile": None,
              "failures": [], "frames": []}
    try:
        report["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("unsupported-anchor-manifest")
        support = manifest.get("spatial_support")
        if (not isinstance(support, dict) or support.get("kind") != "flat-target-rois"
                or support.get("verified") is not True or not isinstance(support.get("verification_note"), str)
                or not support["verification_note"].strip()
                or manifest.get("target_kind") != "dedicated-flat-color-target"):
            raise ValueError("verified-flat-target-spatial-support-required")
        token = manifest.get("actual_decoder_token")
        if token not in ("9", "9.dng"):
            raise ValueError("explicit-exact-raw9-token-required")
        if manifest.get("scale_mode", "unity") != "unity":
            raise ValueError("anchor-tool-requires-explicit-unity-scale-policy")
        if "half_size" in manifest and type(manifest["half_size"]) is not bool:
            raise ValueError("half-size-must-be-explicit-boolean")
        frames = manifest.get("frames")
        if not isinstance(frames, list) or len(frames) < 4:
            raise ValueError("two-training-and-two-independent-holdout-files-required")
        samples, inputs, scopes, reference_versions, seen_hashes = [], [], [], [], set()
        source_checks = []
        for frame in frames:
            if not isinstance(frame, dict) or frame.get("split") not in ("train", "holdout"):
                raise ValueError("invalid-frame-or-independent-split")
            if not isinstance(frame.get("path"), str) or not frame["path"]:
                raise ValueError("frame-path-required")
            path = (Path(directory) / frame["path"]).resolve()
            # Both decoders, metadata and analysis share one immutable input
            # generation. A final hash alone would miss changed-then-restored
            # bytes and cannot bind already-produced pixels to their evidence.
            with source_metadata_session(path) as source_session:
                _require_current_source(source_session, "anchor-before-hash")
                digest = _sha256(path)
                _require_current_source(source_session, "anchor-after-hash")
                if digest in seen_hashes:
                    raise ValueError("duplicate-file-content-not-independent-support")
                seen_hashes.add(digest)
                if frame.get("sha256") is not None and frame["sha256"] != digest:
                    raise ValueError("input-file-hash-mismatch")
                bundles = {}
                for decoder in ("libraw", "coreimage"):
                    _require_current_source(source_session, "anchor-before-" + decoder)
                    bundle = loader(path, scene_half_size=manifest.get("half_size", False), wb_mode="5500k",
                        decoder=decoder, scene_highlight_mode="clip", coreimage_version=token, coreimage_scale="unity")
                    _require_current_source(source_session, "anchor-after-" + decoder)
                    if bundle.scene_decoder != decoder or getattr(bundle, "scene_decoder_fallback", None):
                        raise ValueError("actual-decoder-fallback-not-calibration-evidence")
                    if getattr(bundle, "wb_degradation", None):
                        raise ValueError("declared-white-balance-not-realized")
                    if decoder == "coreimage":
                        recorded_source = (getattr(bundle, "scene_decoder_recipe", None) or {}).get("source") or {}
                        if (recorded_source.get("status") != "verified-through-decode"
                                or recorded_source.get("sha256") != digest):
                            raise ValueError("decoder-source-does-not-match-measured-input")
                    analysis = analyzer(bundle, margin=4, diagnostics=False)[0]
                    _require_current_source(source_session, "anchor-after-analysis")
                    # Apple has no registered RAW loss mask. Require no aggregate
                    # loss rather than pretending that a copied RAW mask localizes it.
                    if (not np.isfinite(analysis.cell_union_pct) or analysis.cell_union_pct != 0
                            or getattr(bundle, "scene_processing_loss_pct", None) != 0
                            or getattr(bundle, "scene_loss_support_untrusted", False)):
                        raise ValueError("raw-or-decoder-loss-not-spatially-excluded")
                    bundles[decoder] = bundle
                reference_versions.append(str(getattr(bundles["libraw"], "scene_decoder_version", None)
                                              or getattr(bundles["libraw"], "evidence_provider_version", "")))
                scope = decoder_anchor_scope(bundles["coreimage"])
                if scope is None or scope["actual_decoder_token"] != token:
                    raise ValueError("actual-raw9-recipe-identity-not-qualified")
                if (bundles["libraw"].shot_make, bundles["libraw"].shot_model) != (scope["make"], scope["model"]):
                    raise ValueError("paired-camera-identity-mismatch")
                if scopes and scope != scopes[0]:
                    raise ValueError("mixed-camera-runtime-or-configuration")
                scopes.append(scope)
                rois = frame.get("rois")
                if not isinstance(rois, list) or not rois:
                    raise ValueError("corresponding-flat-target-rois-required")
                frame_report = {"path": str(path), "sha256": digest, "split": frame["split"], "scope": scope, "rois": []}
                names = set()
                for roi in rois:
                    if (not isinstance(roi, dict) or not isinstance(roi.get("id"), str)
                            or not roi["id"] or roi["id"] in names):
                        raise ValueError("unique-physical-target-roi-id-required")
                    names.add(roi["id"])
                    a, left = _roi_pixels(bundles["libraw"], roi.get("libraw"))
                    b, right = _roi_pixels(bundles["coreimage"], roi.get("coreimage"))
                    ca, cb = a[[0, 2]] / a[1], b[[0, 2]] / b[1]
                    samples.append({"split": frame["split"], "input_sha256": digest,
                        "roi": roi["id"], "chroma_libraw": ca.tolist(), "ratio_rg_bg": (cb / ca).tolist()})
                    frame_report["rois"].append({"id": roi["id"], "libraw": left, "coreimage": right})
                _require_current_source(source_session, "anchor-after-rois")
                report["frames"].append(frame_report)
                inputs.append({"name": path.name, "sha256": digest, "split": frame["split"]})
                source_checks.append((source_session, digest))
        if len(set(reference_versions)) != 1 or reference_versions[0] in ("", "None"):
            raise ValueError("mixed-or-unidentified-libraw-reference-version")
        global_fit = _fit_transport(samples)
        chroma = np.asarray([row["chroma_libraw"] for row in samples])
        class_weights = {}
        for preset in SCENE_TRANSFORMS.values():
            for region in preset.regions:
                for component in region.components or (region,):
                    delta = chroma - np.asarray(component.mu_rg_bg)
                    inverse = np.linalg.inv(np.asarray(component.cov_rg_bg) * region.scale**2)
                    weights = np.exp(-.5 * np.einsum("ni,ij,nj->n", delta, inverse, delta))
                    weights *= min(1., max(0., getattr(component, "weight", 1.)))
                    class_weights[region.name] = np.maximum(class_weights.get(region.name, np.zeros(len(samples))), weights)
        per_class = {name: _fit_transport(samples, weights) for name, weights in class_weights.items()}
        qualified = global_fit["status"] == "qualified" or any(row["status"] == "qualified" for row in per_class.values())
        identity = hashlib.sha256(_canonical({"scope": scopes[0], "inputs": inputs,
                                              "manifest_sha256": report["manifest_sha256"]})).hexdigest()
        profile = {"id": identity[:24], "status": "qualified" if qualified else "unmeasured", "scope": scopes[0],
            "transport": {"global": global_fit, "per_class": per_class},
            "validation": {"status": "passed" if qualified else "not-passed", "independent_holdout": True,
                "spatial_support": "flat-target-rois", "verification_note": support["verification_note"],
                "thresholds": {"ratio_bounds": list(RATIO_BOUNDS), "holdout_median_relative_error": MAX_HOLDOUT_MEDIAN_ERROR,
                               "holdout_p95_relative_error": MAX_HOLDOUT_P95_ERROR}},
            "source": {"method": "corresponding dedicated flat targets; independent-file holdout; no resampling",
                "inputs": inputs, "manifest_sha256": report["manifest_sha256"],
                "libraw_reference_version": reference_versions[0],
                "verification": "paired-decode-session-and-final-sha256-v1",
                "measured_utc": datetime.now(timezone.utc).isoformat()}}
        for session, digest in source_checks:
            _require_current_source(session, "anchor-measurement-commit")
            if _sha256(session.path) != digest:
                raise ValueError("anchor-input-changed-since-measurement")
            _require_current_source(session, "anchor-after-final-hash")
        report.update(status="qualified-candidate" if qualified else "unmeasured", profile=profile, samples=samples,
                      profile_sha256=hashlib.sha256(_canonical(profile)).hexdigest())
        if not qualified:
            report["failures"].append("no-global-or-class-transport-passed-independent-holdout")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, np.linalg.LinAlgError) as error:
        report["failures"].append(str(error))
    return report


def install_candidate(report, destination, *, protected_sources=None):
    protection = ProtectedSources(protected_sources, _frame_paths(report, Path.cwd())).check(destination)
    if (not isinstance(report, dict) or report.get("schema") != CANDIDATE_SCHEMA
            or report.get("status") != "qualified-candidate" or report.get("failures")):
        raise ValueError("only-qualified-candidate-reports-can-be-installed")
    profile = report.get("profile")
    if not _qualified_anchor_entry(profile) or _validated_anchor_scope(profile.get("scope")) is None:
        raise ValueError("candidate-profile-contract-invalid")
    if report.get("profile_sha256") != hashlib.sha256(_canonical(profile)).hexdigest():
        raise ValueError("candidate-profile-changed-after-measurement")
    evidence = report.get("frames")
    if not isinstance(evidence, list) or len(evidence) != len(profile["source"]["inputs"]):
        raise ValueError("candidate-input-evidence-incomplete")
    expected = {(row["sha256"], row["split"]) for row in profile["source"]["inputs"]}
    observed = set()
    for row in evidence:
        if (not isinstance(row, dict) or (row.get("sha256"), row.get("split")) not in expected
                or _sha256(row["path"]) != row["sha256"]):
            raise ValueError("candidate-input-changed-since-measurement")
        observed.add((row["sha256"], row["split"]))
    if observed != expected:
        raise ValueError("candidate-input-evidence-incomplete")
    try:
        asset = json.loads(destination.read_text())
    except (OSError, ValueError):
        asset = {"schema": DECODER_ANCHOR_SCHEMA, "version": 3, "profiles": []}
    if not isinstance(asset, dict) or asset.get("schema") != DECODER_ANCHOR_SCHEMA or asset.get("version") != 3 or not isinstance(asset.get("profiles"), list):
        raise ValueError("refuse-implicit-legacy-asset-migration")
    asset["profiles"] = [row for row in asset["profiles"] if isinstance(row, dict) and row.get("scope") != profile["scope"]] + [profile]
    with DeliveryTransaction(destination, protected_sources=protection) as transaction:
        transaction.path.write_text(json.dumps(asset, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
        transaction.commit()
    return {"status": "installed", "profile_id": profile["id"], "asset": str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--install-candidate", type=Path)
    parser.add_argument("--asset", type=Path, default=DECODER_ANCHOR_TRANSPORT_JSON)
    args = parser.parse_args(argv)
    if args.install_candidate:
        if args.manifest is not None or args.report is not None:
            parser.error("installation is separate from measurement")
        try:
            candidate = json.loads(args.install_candidate.read_text())
            protection = ProtectedSources(args.install_candidate,
                _frame_paths(candidate, Path.cwd())).check(args.asset)
            result = install_candidate(candidate, args.asset, protected_sources=protection)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(2, f"candidate not installed: {error}\n")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.manifest is None or args.report is None:
        parser.error("measurement needs a manifest and --report; no implicit image-folder corpus")
    protection = ProtectedSources(args.manifest).check(args.report)
    try:
        manifest = json.loads(args.manifest.read_text())
        protection = ProtectedSources(protection,
            _frame_paths(manifest, args.manifest.parent)).check(args.report)
        report = measure_candidate(manifest, args.manifest.parent)
    except (OSError, ValueError, TypeError) as error:
        report = {"schema": CANDIDATE_SCHEMA, "status": "unmeasured", "profile": None, "failures": [str(error)]}
    with DeliveryTransaction(args.report, protected_sources=protection) as transaction:
        transaction.path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        transaction.commit()
    print(json.dumps({"status": report["status"], "report": str(args.report), "failures": report["failures"]}, ensure_ascii=False))
    return 0 if report["status"] == "qualified-candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
