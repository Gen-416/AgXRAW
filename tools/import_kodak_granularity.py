#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Digitize Kodak cine diffuse-rms-granularity charts into pinned assets.

Sources (Grain V2 / optics V2 P4 data pass):
  - KODAK VISION3 250D 5207 Technical Data, H-1-5207, Eastman Kodak,
    (c) 2022, Revised 3-22 — "Diffuse rms Granularity Curves" chart:
    per-channel characteristic density D(logE) (solid, Status M-like
    "Granularity" densitometry) plus granularity Sigma-D(logE) (dashed,
    48 um aperture microdensitometer, log right axis, rms = value*1000).
  - KODAK VISION Color Print Film 2383/3383, H-1-2383, (c) 2022,
    Revised 3-22 — same chart form (Status A, ECP-2D); the print stock's
    characteristic tables were scanned off the granularity chart itself
    by density-line crossings (its sensitometric chart uses a different
    per-channel exposure normalization and is not interchangeable).

Method (recorded per plan §15 digitization provenance rules): the chart
page is rendered at 4x via Quartz; the plot box, x ticks and the
log-scale sigma ticks are DETECTED programmatically (axis anchors:
x: logE0 @ px 1499.5, 116.4 px/logE; D: 0.0 @ py 1095.5,
193.83 px/D; sigma: 0.01 @ py 912.5, 183 px/decade — two detected
decades agree to <1 px, i.e. the right axis is a true log scale).
Curve anchors below were read MANUALLY against a calibrated grid
overlay rendered onto the chart (0.25 logE / 0.1 D / per-0.001 sigma
lines), zoomed 4-8x per region; zone-boundary double reads agreed
within 3-9%. Estimated uncertainty: ±0.03 D, ±5% sigma (upstream's own
caveat: the curve reflects "modified measuring techniques" and its
shape may vary between instruments). The original PDFs stay OUT of the
repo (Kodak copyright); only these read-off numbers with provenance.

The solid curves carry Status M base+mask density (B-channel base is
~1.0 because of the orange mask); sigma is diffuse rms Sigma-D at 48 um
against that same density coordinate, so sigma(D) pairs eliminate logE
per channel directly.

    python tools/import_kodak_granularity.py            # write all assets
    python tools/import_kodak_granularity.py --overlay 5207 OUT.png PAGE.png
        # render anchors over the (4x) chart page for visual verification
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dngscan" / "data" / "grain"

# Chart calibrations (pages rendered at 4x; H-1-5207 p.3, H-1-2383 p.4).
CAL_5207 = {
    "x0_px": 1499.5, "px_per_loge": 116.4,
    "d0_py": 1095.5, "px_per_density": 193.83,
    "sigma_ref_py": 912.5, "sigma_ref": 0.01, "px_per_decade": 183.0,
}
CAL_2383 = {
    "x0_px": 1407.5, "px_per_loge": 246.0,
    "d0_py": 1030.5, "px_per_density": 184.615,
    "sigma_ref_py": 683.0, "sigma_ref": 0.1, "px_per_decade": 175.0,
}

# Anchor tables: (logE, value). D anchors from the solid characteristic
# curves; sigma anchors from the dashed Sigma-D curves. Channel order
# follows the chart's own labels (B top, G middle, R bottom in density).
DATA_5207 = {
    "film": "Kodak VISION3 250D 5207",
    "process": "ECN-2",
    "densitometry": "granularity Sigma-D chart scale (Status M family, base+mask included)",
    "aperture_um": 48.0,
    "source": "H-1-5207, Eastman Kodak, (c) 2022, Revised 3-22, p.3 'Diffuse rms Granularity Curves'",
    "source_url": "https://www.kodak.com/content/products-brochures/Film/VISION3-250D-Technical-Data-EN.pdf",
    "method": "manual anchor read-off against a programmatically calibrated grid overlay; see tools/import_kodak_granularity.py docstring",
    "uncertainty": "±0.03 density, ±5% sigma; upstream notes curve shape varies with measuring equipment",
    "channels": {
        "B": {
            "density_loge": [
                [0.0, 1.02], [0.5, 1.02], [1.0, 1.07], [1.25, 1.14],
                [1.5, 1.24], [1.75, 1.33], [2.0, 1.50], [2.25, 1.67],
                [2.5, 1.82], [2.75, 1.99], [3.0, 2.14], [3.25, 2.27],
                [3.5, 2.42], [3.75, 2.56], [4.0, 2.70], [4.25, 2.81],
                [4.5, 2.90], [4.75, 2.97], [5.0, 3.00],
            ],
            "sigma_loge": [
                [0.0, 0.0118], [0.5, 0.0117], [1.0, 0.0140], [1.3, 0.0165],
                [1.5, 0.0160], [1.75, 0.0158], [2.0, 0.0140], [2.4, 0.0128],
                [2.75, 0.0137], [3.0, 0.0146], [3.3, 0.0140], [3.83, 0.0108],
                [4.26, 0.0091], [4.69, 0.0075], [5.0, 0.0062],
            ],
        },
        "G": {
            "density_loge": [
                [0.0, 0.66], [0.5, 0.66], [1.0, 0.68], [1.5, 0.76],
                [1.75, 0.90], [2.0, 1.03], [2.25, 1.19], [2.5, 1.35],
                [2.75, 1.50], [3.0, 1.66], [3.25, 1.85], [3.5, 2.00],
                [3.75, 2.14], [4.0, 2.28], [4.25, 2.43], [4.5, 2.54],
                [4.75, 2.64], [5.0, 2.72],
            ],
            "sigma_loge": [
                [0.0, 0.0053], [0.5, 0.0052], [1.0, 0.0060], [1.5, 0.0095],
                [1.75, 0.0084], [2.0, 0.0079], [2.27, 0.0077], [2.5, 0.0071],
                [2.75, 0.0066], [3.0, 0.0066], [3.4, 0.0069], [3.94, 0.0064],
                [4.37, 0.0056], [4.8, 0.0049], [5.0, 0.0047],
            ],
        },
        "R": {
            "density_loge": [
                [0.0, 0.23], [0.5, 0.23], [1.0, 0.23], [1.25, 0.24],
                [1.5, 0.28], [1.75, 0.47], [2.0, 0.60], [2.25, 0.74], [2.5, 0.90],
                [2.75, 1.06], [3.0, 1.21], [3.25, 1.33], [3.5, 1.44],
                [3.75, 1.58], [4.0, 1.70], [4.25, 1.81], [4.5, 1.89],
                [4.75, 1.96], [5.0, 2.01],
            ],
            "sigma_loge": [
                [0.0, 0.0050], [0.5, 0.0049], [1.0, 0.0058], [1.5, 0.0100],
                [1.75, 0.0091], [2.0, 0.0083], [2.27, 0.0077], [2.5, 0.0065],
                [2.75, 0.0056], [3.0, 0.0055], [3.4, 0.0053], [3.83, 0.0048],
                [4.26, 0.0047], [4.69, 0.0044], [5.0, 0.0043],
            ],
        },
    },
    "notes": [
        "dashed G/R cross near logE 2.27; assignment left of the crossing keeps R above G (they differ <10% there, per-chart labels only exist on the right)",
        "sigma(D) per channel = pair density_loge with sigma_loge on a common logE grid (both are functions of the same chart abscissa)",
    ],
}


# H-1-2383 (KODAK VISION Color Print Film 2383/3383), p.4 "Diffuse rms
# Granularity Curves": Status A, ECP-2D, 48 um. Same reading method and
# grid-overlay verification as 5207 (x ticks at 246 px/logE, D axis
# 184.6 px/D, sigma log axis 175 px/decade — the two detected decades
# agree exactly). Print film: the granularity curves stop near
# logE 1.95 where the characteristic curves reach D ~2.3-2.6, so the
# sigma(D) join only covers that range (the D 2.6-4.1 shoulder has no
# published granularity). Note R sits ABOVE G here (chart labels).
DATA_2383 = {
    "film": "Kodak VISION Color Print Film 2383",
    "process": "ECP-2D",
    "densitometry": "Status A (base included, clear base ~0.08)",
    "aperture_um": 48.0,
    "source": "H-1-2383, Eastman Kodak, (c) 2022, Revised 3-22, p.4 'Diffuse rms Granularity Curves'",
    "source_url": "https://www.kodak.com/content/products-brochures/Film/KODAK-VISION-Color-Print-Film-2383-3383-data-sheet.pdf",
    "method": "manual anchor read-off against a programmatically calibrated grid overlay; see module docstring",
    "uncertainty": "±0.03 logE horizontal on the steep print sigmoid (density-line crossing scan; G/R split by ordering where merged), ±6% sigma; upstream notes curve shape varies with measuring equipment",
    "channels": {
        "B": {
            "density_loge": [
                [0.0, 0.08], [0.5, 0.08], [0.878, 0.2], [1.018, 0.3],
                [1.16, 0.5], [1.285, 0.7], [1.325, 0.8], [1.404, 1.0],
                [1.461, 1.2], [1.537, 1.5], [1.608, 1.8], [1.687, 2.1],
                [1.77, 2.4], [1.85, 2.75], [1.892, 3.0], [1.931, 3.2],
            ],
            "sigma_loge": [
                [0.0, 0.0075], [0.25, 0.0085], [0.5, 0.0100], [0.75, 0.0141],
                [1.0, 0.0249], [1.25, 0.0400], [1.5, 0.0553], [1.75, 0.0605],
                [1.95, 0.0572],
            ],
        },
        "G": {
            "density_loge": [
                [0.0, 0.08], [0.55, 0.08], [0.945, 0.2], [1.055, 0.3],
                [1.20, 0.5], [1.31, 0.7], [1.35, 0.8], [1.465, 1.0],
                [1.508, 1.2], [1.573, 1.5], [1.646, 1.8], [1.724, 2.1],
                [1.801, 2.4], [1.855, 2.75], [1.923, 3.0], [1.963, 3.2],
            ],
            "sigma_loge": [
                [0.0, 0.0022], [0.25, 0.0022], [0.5, 0.0025], [0.75, 0.00285],
                [1.0, 0.0036], [1.25, 0.0052], [1.35, 0.0071], [1.5, 0.0092],
                [1.75, 0.0116], [1.95, 0.0146],
            ],
        },
        "R": {
            "density_loge": [
                [0.0, 0.08], [0.6, 0.08], [1.018, 0.2], [1.128, 0.3],
                [1.291, 0.5], [1.378, 0.7], [1.404, 0.8], [1.495, 1.0],
                [1.528, 1.2], [1.593, 1.5], [1.666, 1.8], [1.744, 2.1],
                [1.821, 2.4], [1.875, 2.75], [1.943, 3.0],
            ],
            "sigma_loge": [
                [0.0, 0.00245], [0.25, 0.0024], [0.5, 0.0026], [0.75, 0.0031],
                [1.0, 0.0040], [1.25, 0.0068], [1.35, 0.0090], [1.5, 0.0109],
                [1.75, 0.0140], [1.95, 0.0157],
            ],
        },
    },
    "notes": [
        "granularity curves end near logE 1.95 (D ~2.3-2.6); the print shoulder above has no published sigma",
        "R granularity sits above G on this stock (chart labels R over G at the right edge)",
        "characteristic tables come from the granularity chart itself (density-line crossing scan) — the sensitometric chart on the same page uses a different per-channel exposure normalization and is NOT interchangeable",
        "where B/G or G/R merge into one stroke the split follows the fixed B<G<R ordering; residual horizontal error <=0.03 logE -> <=2-3% sigma(D) error",
    ],
}

DATASETS = {
    "5207": (DATA_5207, CAL_5207, "granularity_5207.json", 5.0),
    "2383": (DATA_2383, CAL_2383, "granularity_2383.json", 1.95),
}


def sigma_of_density(channel: dict, loge_max: float, n: int = 61):
    """Parametric join: sample logE, return (D, sigma) rows."""
    import numpy as np

    d_tab = np.array(channel["density_loge"], dtype=float)
    s_tab = np.array(channel["sigma_loge"], dtype=float)
    loge = np.linspace(0.0, loge_max, n)
    dens = np.interp(loge, d_tab[:, 0], d_tab[:, 1])
    sig = np.interp(loge, s_tab[:, 0], s_tab[:, 1])
    # The chart's toe holds density FLAT while sigma still moves with
    # exposure, so the parametric join is not single-valued in D. The
    # render queries by density alone, and a pixel AT base density is the
    # unexposed state — so the fold keeps the FIRST (lowest-logE) row per
    # unique density and the emitted table is strictly increasing in D,
    # which the loader enforces again (review R1 item 5: np.interp on
    # duplicated x silently kept the LAST toe row instead).
    rows = []
    seen = -1.0
    for d, s in zip(dens, sig):
        d4 = round(float(d), 4)
        if d4 <= seen:
            continue
        seen = d4
        rows.append([d4, round(float(s), 6)])
    return rows


def build_asset(data: dict, cal: dict, loge_max: float) -> dict:
    payload = json.loads(json.dumps(data))  # deep copy
    payload["schema"] = 1
    payload["calibration_px"] = cal
    for name, ch in payload["channels"].items():
        ch["sigma_density"] = sigma_of_density(ch, loge_max)
    return payload


def overlay(dataset: str, page_png: str, out_png: str) -> None:
    """Render the anchor tables over the 4x chart page for verification."""
    import math

    from PIL import Image, ImageDraw

    data, cal, _, _ = DATASETS[dataset]
    im = Image.open(page_png).convert("RGB")
    dr = ImageDraw.Draw(im)
    colors = {"B": (0, 90, 255), "G": (0, 170, 0), "R": (255, 0, 0)}
    for name, ch in data["channels"].items():
        for loge, dv in ch["density_loge"]:
            x = cal["x0_px"] + loge * cal["px_per_loge"]
            y = cal["d0_py"] - dv * cal["px_per_density"]
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=colors[name], width=2)
        for loge, sv in ch["sigma_loge"]:
            x = cal["x0_px"] + loge * cal["px_per_loge"]
            y = cal["sigma_ref_py"] - (math.log10(sv) - math.log10(cal["sigma_ref"])) * cal["px_per_decade"]
            dr.rectangle([x - 3, y - 3, x + 3, y + 3], outline=colors[name], width=2)
    im.save(out_png)
    print(f"overlay written: {out_png}")


def main() -> int:
    if len(sys.argv) >= 5 and sys.argv[1] == "--overlay":
        overlay(sys.argv[2], sys.argv[4], sys.argv[3])
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, (data, cal, fname, loge_max) in DATASETS.items():
        out = OUT_DIR / fname
        out.write_text(
            json.dumps(build_asset(data, cal, loge_max), ensure_ascii=False,
                       indent=1) + "\n",
            encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
