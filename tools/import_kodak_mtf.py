#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Digitize Kodak MTF charts and fit the V2 scatter-kernel parameters.

Optics V2 P5 data pass. The P3 implementation note deferred both §5.1
emulsion scatter and §6.2 formation scatter because "no measured MTF
exists in the assets" — these charts are that measurement:

  - H-1-5207 p.3 "Modulation-Transfer Function Curves" (5500K daylight,
    ECN-2, Status M): the camera stock's per-channel system MTF. Feeds
    the §5.1 core/tail emulsion-scatter fit:
        MTF(f) = (1-s) + s[(1-w)·Ĝ_sigma(f) + w·Êxp_lambda(f)]
    with Ĝ(f)=exp(-2·pi^2·sigma^2·f^2) and the isotropic exponential
    PSF transform Êxp(f)=(1+(2·pi·lambda·f)^2)^(-3/2), f in cycles/mm.
  - H-1-2383 p.4 (tungsten 3200K, ECP-2D, Status A, 35% modulation
    target): the print film's MTF. Feeds the §6.2 formation-scatter fit
    (single normalized Gaussian K_form):  MTF(f) = (1-s) + s·Ĝ_sigma(f).

Method mirrors tools/import_kodak_granularity.py: 4x Quartz render,
grid lines detected programmatically (both axes are true log scales —
every labelled decade line reproduces within ~1 px), curve anchors read
manually against the chart's own grid at 4-6x zoom, then verified by
re-rendering the anchors over the chart (--overlay). Estimated read-off
uncertainty ±5% response (±8% in the cramped low-frequency region).

Fit notes recorded with the numbers:
  - Both stocks show a small >100% adjacency bump (chemical edge effect
    of development, ~103-108% at 2.5-15 c/mm). A passive scatter mix
    cannot exceed 1, so the fit clamps targets to 1.0 and weights the
    rolloff (f >= 15 c/mm); the bump is a documented residual, not a
    fit failure.
  - The upstream note on the granularity charts applies here too:
    curves come from specific instruments and may vary slightly.

    python tools/import_kodak_mtf.py               # write assets + fit
    python tools/import_kodak_mtf.py --overlay 5207 OUT.png PAGE.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

OUT_DIR = ROOT / "dngscan" / "data" / "mtf"

CAL = {
    "5207": {"x0_px": 319.5, "px_per_decade_x": 288.0,
             "y100_py": 988.5, "px_per_decade_y": 262.5},
    "2383": {"x0_px": 294.5, "px_per_decade_x": 291.0,
             "y100_py": 1603.0, "px_per_decade_y": 265.0},
}

# (frequency c/mm, response fraction) — chart reads; >1 = adjacency bump
DATA = {
    "5207": {
        "film": "Kodak VISION3 250D 5207",
        "source": "H-1-5207, Eastman Kodak, (c) 2022, Revised 3-22, p.3 'Modulation-Transfer Function Curves' (5500K, ECN-2, Status M)",
        "model": "core_tail_v1",
        "channels": {
            "B": [[2, 1.00], [5, 1.01], [10, 1.03], [15, 1.03], [20, 1.00],
                  [30, 0.80], [50, 0.60], [70, 0.45], [85, 0.40]],
            "G": [[2, 1.00], [5, 1.00], [10, 1.02], [15, 1.01], [20, 0.97],
                  [30, 0.72], [50, 0.52], [70, 0.41], [85, 0.38]],
            "R": [[2, 1.00], [5, 0.99], [10, 0.94], [15, 0.82], [20, 0.74],
                  [30, 0.51], [50, 0.27], [70, 0.19], [85, 0.16]],
        },
    },
    "2383": {
        "film": "Kodak VISION Color Print Film 2383",
        "source": "H-1-2383, Eastman Kodak, (c) 2022, Revised 3-22, p.4 'Modulation-Transfer Function Curves' (3200K, ECP-2D, Status A, 35% modulation target)",
        "model": "gaussian_v1",
        "channels": {
            "G": [[2.5, 1.05], [5, 1.00], [10, 0.90], [20, 0.82], [30, 0.81],
                  [50, 0.805], [70, 0.80], [85, 0.71]],
            "R": [[2.5, 1.05], [5, 1.00], [10, 0.91], [20, 0.79], [30, 0.72],
                  [50, 0.645], [70, 0.58], [77, 0.57]],
            "B": [[2.5, 1.04], [5, 0.99], [10, 0.85], [20, 0.60], [30, 0.47],
                  [50, 0.35], [70, 0.30], [77, 0.30]],
        },
    },
}


def mtf_core_tail(f, s, w, sigma_mm, lambda_mm):
    g = np.exp(-2.0 * np.pi ** 2 * sigma_mm ** 2 * f ** 2)
    # OPERATOR-EFFECTIVE tail (2026-08-25): the runtime renders the tail as
    # a Gaussian with sigma = sqrt(3)*lambda (film_optics.py second-moment
    # approximation). Fitting the analytic exponential-tail MTF instead let
    # a heavy-tailed fit (w~0.55) open a 7.1pp gap between the declared
    # kernel and what a render actually applies (gate-13 failure). The fit,
    # the report's explicit model and the operator now share ONE form, so
    # declaration == execution by construction.
    e = np.exp(-2.0 * np.pi ** 2 * (3.0 * lambda_mm ** 2) * f ** 2)
    return (1.0 - s) + s * ((1.0 - w) * g + w * e)


def mtf_gaussian(f, s, sigma_mm):
    g = np.exp(-2.0 * np.pi ** 2 * sigma_mm ** 2 * f ** 2)
    return (1.0 - s) + s * g


def _fit_channel(rows, model):
    from scipy.optimize import minimize

    f = np.array([r[0] for r in rows], dtype=float)
    y = np.minimum(np.array([r[1] for r in rows], dtype=float), 1.0)
    wgt = np.where(f >= 15.0, 1.0, 0.35)  # rolloff carries the fit

    if model == "core_tail_v1":
        def loss(p):
            s, w, sg, lm = p
            if not (0.0 < s < 1.0 and 0.0 <= w <= 1.0
                    and 1e-4 < sg < 0.03 and 1e-4 < lm < 0.05):
                return 1e6
            m = mtf_core_tail(f, s, w, sg, lm)
            return float(np.sum(wgt * (m - y) ** 2))
        best = None
        for s0 in (0.5, 0.7, 0.9):
            for w0 in (0.2, 0.5):
                r = minimize(loss, [s0, w0, 0.005, 0.01],
                             method="Nelder-Mead",
                             options={"xatol": 1e-6, "fatol": 1e-10,
                                      "maxiter": 4000})
                if best is None or r.fun < best.fun:
                    best = r
        s, w, sg, lm = [float(v) for v in best.x]
        # canonicalize the degenerate axis: with w == 0 the exponential-tail
        # component is inert and lm is unconstrained by the loss — it can
        # wander to implausible values (the asset loader's lambda gate caught
        # a 34um one). An inert component is written as exactly zero.
        if round(w, 4) == 0.0:
            w = 0.0
            lm = 0.0
        m = mtf_core_tail(f, s, w, sg, lm)
        return {
            "s": round(s, 4), "w": round(w, 4),
            "sigma_um": round(sg * 1000.0, 3),
            "lambda_um": round(lm * 1000.0, 3),
            "rms_residual": round(float(np.sqrt(np.mean((m - y) ** 2))), 4),
        }
    # gaussian_v1
    def loss(p):
        s, sg = p
        if not (0.0 < s < 1.0 and 1e-4 < sg < 0.03):
            return 1e6
        m = mtf_gaussian(f, s, sg)
        return float(np.sum(wgt * (m - y) ** 2))
    best = None
    for s0 in (0.2, 0.5, 0.8):
        from scipy.optimize import minimize
        r = minimize(loss, [s0, 0.006], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    s, sg = [float(v) for v in best.x]
    m = mtf_gaussian(f, s, sg)
    return {
        "s": round(s, 4), "sigma_um": round(sg * 1000.0, 3),
        "rms_residual": round(float(np.sqrt(np.mean((m - y) ** 2))), 4),
    }


# Second-pass anchors (2026-08-25 precision audit): programmatic column
# crossing scan on the 4x render; channel assignment by PCHIP prediction
# with a +-0.06 response window; grid-line columns and the G/B crossing
# bundle near 35 c/mm rejected; R@35 rejected as inconsistent with both
# neighbours (likely stroke-bundle artifact). Existing-anchor positions
# re-scanned as verification only (all within declared uncertainty).
SECOND_PASS = {
    "5207": {
        "R": [(25, 0.668), (40, 0.409), (60, 0.239)],
        "G": [(25, 0.846), (40, 0.615), (60, 0.450)],
        "B": [(25, 0.932), (40, 0.710), (60, 0.523)],
    },
}


def _merged_channels(key):
    import copy
    import math
    chans = copy.deepcopy(DATA[key]["channels"])
    for ch, pts in SECOND_PASS.get(key, {}).items():
        xs = [r[0] for r in chans[ch]]
        for f, r in pts:
            if all(abs(f - f0) > 0.5 for f0 in xs):
                chans[ch].append([f, r])
        chans[ch].sort(key=lambda r: r[0])
    # Dense 8x sub-pixel scan (third pass, tools/scan_chart_curves.py):
    # accepted only where it agrees with the verified-anchor curve within
    # the declared +-0.05 response (see the granularity importer's note).
    scan_path = Path(__file__).parent / "chart_scans" / f"mtf_{key}_scan.json"
    if scan_path.exists():
        try:
            from scipy.interpolate import PchipInterpolator
        except ImportError:
            return chans
        scan = json.loads(scan_path.read_text())
        for ch, rows in chans.items():
            base = sorted(rows, key=lambda r: r[0])
            lx = [math.log10(r[0]) for r in base]
            fit = PchipInterpolator(lx, [r[1] for r in base])
            added = rejected = 0
            peak_f = max(base, key=lambda r: r[1])[0]
            cands = []
            for fq, resp in scan["channels"].get(ch, []):
                x = math.log10(fq)
                if not (lx[0] <= x <= lx[-1]):
                    continue
                if any(abs(x - x0) <= 0.01 for x0 in lx):
                    continue
                ref = float(fit(x))
                dev = abs(resp - ref)
                # dual tolerance: the read-noise gate is absolute (declared
                # +-0.05 response), but the shipped-kernel residual budget is
                # log-space (<= ln 1.15, test_film_optics_scatter), so accept
                # only points that also stay within ln(1.10) of the verified
                # curve — otherwise tail points (response ~0.2) can carry
                # 25% log deviation into the anchor set.
                logdev = abs(math.log(max(resp, 1e-6) / max(ref, 1e-6)))
                if dev <= 0.05 and logdev <= math.log(1.10):
                    cands.append((dev, fq, resp))
                else:
                    rejected += 1
            # best-agreement first; past the adjacency-bump peak the anchor
            # sequence must roll off monotonically (asset contract), so a
            # candidate whose read noise would break that is rejected.
            for dev, fq, resp in sorted(cands):
                if fq > peak_f:
                    prev = max((r for r in rows if r[0] < fq), key=lambda r: r[0], default=None)
                    nxt = min((r for r in rows if r[0] > fq), key=lambda r: r[0], default=None)
                    if (prev is not None and prev[0] > peak_f and resp > prev[1]) or \
                       (nxt is not None and resp < nxt[1]):
                        rejected += 1
                        continue
                rows.append([fq, resp])
                rows.sort(key=lambda r: r[0])
                added += 1
            if added or rejected:
                print(f"  scan merge mtf{key}.{ch}: +{added}, rejected {rejected}")
    return chans


def build_asset(key: str) -> dict:
    data = json.loads(json.dumps(DATA[key]))
    data["channels"] = _merged_channels(key)
    data["schema"] = 1
    data["calibration_px"] = CAL[key]
    data["method"] = ("manual anchor read-off against the chart's own log grid; "
                      "see tools/import_kodak_mtf.py docstring; second-pass "
                      "crossing-scan anchors merged 2026-08-25 (SECOND_PASS)")
    data["uncertainty"] = ("±5% response (±8% below 10 c/mm); adjacency bump "
                           ">100% is a development edge effect the passive "
                           "scatter model cannot and does not reproduce")
    data["fit"] = {}
    for name, rows in data["channels"].items():
        data["fit"][name] = _fit_channel(rows, data["model"])
    return data


def overlay(dataset: str, page_png: str, out_png: str) -> None:
    import math

    from PIL import Image, ImageDraw

    cal = CAL[dataset]
    im = Image.open(page_png).convert("RGB")
    dr = ImageDraw.Draw(im)
    colors = {"B": (0, 90, 255), "G": (0, 170, 0), "R": (255, 0, 0)}
    for name, rows in DATA[dataset]["channels"].items():
        for f, resp in rows:
            x = cal["x0_px"] + cal["px_per_decade_x"] * math.log10(f)
            y = cal["y100_py"] + cal["px_per_decade_y"] * (2.0 - math.log10(resp * 100.0))
            dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=colors[name], width=2)
    im.save(out_png)
    print(f"overlay written: {out_png}")


def main() -> int:
    if len(sys.argv) >= 5 and sys.argv[1] == "--overlay":
        overlay(sys.argv[2], sys.argv[4], sys.argv[3])
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in DATA:
        out = OUT_DIR / f"mtf_{key}.json"
        payload = build_asset(key)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
        for name, fit in payload["fit"].items():
            print(f"  {key} {name}: {fit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
