#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert the PhotonsToPhotos-derived bulk sensor table into dngscan priors.

Input: y-g-jiang's ``pdr_camera_data_14bit.js`` — a machine-generated second-
layer compilation of Bill Claff's PhotonsToPhotos chart data (RN_ADU +
Sensor_Characteristics) merged with the hletrd pixel-pitch database. The
file's own header records its sources, generation time and assumptions;
this converter adds a third layer: per-ISO read noise converted to
input-referred electrons through the unity-gain relation, and PDR derived
from the declared target-SNR model.

Licensing status is recorded, not laundered: PhotonsToPhotos is
"all rights reserved" with no published data-reuse policy. The owner's
2026-08-24 decision: import now for this open-source, non-commercial tool
that recomputes rather than republishes, WHILE permission is being sought
from the author; the output is one standalone data file
(dngscan/data/priors/p2p_bulk.json) so a licensing outcome can excise it
cleanly (same reversibility discipline as the CBLD precedent).

Derivations (documented in each entry):
    gain(iso)  = unityGainIso / iso                      [e-/DN]
    rn_e(iso)  = rndnADU14 * gain(iso)
    FW(iso)    = fwc_native * nativeIso / iso            [declared scaling]
    PDR(iso)   = log2( FW(iso) / S* ),
        S* solves S/sqrt(S + rn_e^2) = targetSNR
           = (t^2 + t*sqrt(t^2 + 4*rn_e^2)) / 2

Usage:
    python tools/import_p2p_pdr.py pdr_camera_data_14bit.js \\
        --out dngscan/data/priors/p2p_bulk.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def convert_camera(c: dict) -> dict | None:
    sc = c.get("sensorCharacteristics") or {}
    ug = sc.get("unityGainIso")
    fwc = c.get("fullWellElectronPerPixelNative")
    n_iso = c.get("nativeIso")
    t = c.get("targetSNR")
    rn = c.get("rn") or []
    if not (ug and fwc and n_iso and t and rn):
        return None
    ug, fwc, n_iso, t = float(ug), float(fwc), float(n_iso), float(t)
    if ug <= 0 or fwc <= 0 or n_iso <= 0 or t <= 0:
        return None
    rn_pts, pdr_pts, suspect = [], [], None
    for p in rn:
        iso = float(p["iso"])
        gain = ug / iso
        rn_e = float(p["rndnADU14"]) * gain
        if not (math.isfinite(rn_e) and 0.05 <= rn_e <= 200.0):
            return None
        fw = fwc * n_iso / iso
        s_star = (t * t + t * math.sqrt(t * t + 4 * rn_e * rn_e)) / 2.0
        pdr = math.log2(fw / s_star)
        if not (math.isfinite(pdr) and 1.0 <= pdr <= 17.0):
            return None
        x = round(math.log2(iso), 4)
        rn_pts.append((x, round(math.log2(rn_e), 4)))
        pdr_pts.append((x, round(pdr, 4)))
        # P2P hollow markers flag NR-affected points; record the first as the
        # suspect threshold, matching the curated entries' semantics.
        if p.get("markerSymbol") and suspect is None:
            suspect = iso
    return {
        "id": str(c["camera"]),
        "make_model": str(c["camera"]),
        "unity_gain_ev": round(math.log2(ug), 4),
        "fwc_e": fwc,
        "native_iso": n_iso,
        "pixel_pitch_um": c.get("pixelPitchUm"),
        "pdr_log2iso_ev": pdr_pts,
        "read_noise_log2iso_log2e": rn_pts,
        "suspect_iso_min": suspect,
        "derived": "PDR and rn_e derived per tools/import_p2p_pdr.py header",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw = args.input.read_text()
    data = json.loads(raw.split("=", 1)[1].rstrip().rstrip(";"))
    entries = []
    skipped = []
    for c in data.get("cameras", []):
        entry = convert_camera(c)
        if entry is None:
            skipped.append(c.get("camera", "?"))
        else:
            entries.append(entry)
    out = {
        "format": "dngscan-p2p-bulk-priors-1",
        "provenance": {
            "layer1": "PhotonsToPhotos (Bill Claff) chart data — "
                      "(c) William J. Claff, all rights reserved; no "
                      "published reuse policy; permission being sought",
            "layer2": {
                "compiler": "y-g-jiang pdr_camera_data_14bit.js",
                "generatedAt": data.get("generatedAt"),
                "sourcePages": data.get("sourcePages"),
                "assumptions": data.get("assumptions"),
            },
            "layer3": "dngscan derivation, tools/import_p2p_pdr.py",
            "decision": "owner 2026-08-24: imported pending permission; "
                        "this single file is the entire footprint",
        },
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(json.dumps({
        "entries": len(entries),
        "skipped": skipped[:8],
        "skipped_count": len(skipped),
        "bytes": args.out.stat().st_size,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
