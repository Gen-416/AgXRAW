#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""External cross-check: our density-domain 2383 chain vs a public PFE LUT.

Two independent implementation paths to the same physical claim (contract §8
外部互证): dngscan composes Vision3 negative + 2383 print from spektrafilm's
datasheet densitometry; the industry ships display-referred "film print emulation"
LUTs of the same stock (e.g. Resolve's Kodak 2383 cube). If both are faithful, the
neutral-ramp response shapes must converge up to the declared normalizations.

Chain A (ours)   : scene EV -> negative Status-M density -> contact print on 2383
                   -> per-channel transmittance (native / theatrical quotation,
                   no surround term — the LUT quotes the print too).
Chain B (theirs) : scene EV -> negative density -> Cineon code values
                   (CV = 95 + 500·(D - Dmin), the D-min-anchored printing-density
                   encoding) -> 3D LUT -> display RGB -> EOTF decode.

Declared normalizations, applied before comparison and reported:
  * a per-channel log2 offset (least squares) — printer-light / exposure-trim
    equivalence; PFE LUT authors chose their own trim, we solve ours per channel;
  * LUT output EOTF is selectable (--display-gamma 2.4 BT.1886 default; 2.2 and
    sRGB reported alongside) because cube files do not declare it;
  * LUT input encoding is selectable (--input cineon|logc3): "2383" cubes ship in
    both Cineon-log and LogC flavours and rarely say which. Run both; the one that
    converges identifies the LUT's actual input space — itself a result.

The LUT is validation-only and must NOT be committed (contract: LUT 不入库).
Usage:
  .venv/bin/python tools/crosscheck_2383.py --lut /path/to/kodak2383.cube \
      --stock vision3250d --input cineon
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.fit_film_curve import (  # noqa: E402
    LOG10_2,
    PROFILE_DIR,
    STOCKS,
    build_endtoend_target,
    print_density_curves,
    _load_curves,
)

EV_LO, EV_HI = -6.0, 5.5


def read_cube(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal .cube reader: (table[N,N,N,3], domain_min[3], domain_max[3])."""
    size = None
    dmin = np.zeros(3)
    dmax = np.ones(3)
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("TITLE"):
            continue
        parts = line.split()
        if parts[0] == "LUT_3D_SIZE":
            size = int(parts[1])
        elif parts[0] == "DOMAIN_MIN":
            dmin = np.array([float(v) for v in parts[1:4]])
        elif parts[0] == "DOMAIN_MAX":
            dmax = np.array([float(v) for v in parts[1:4]])
        elif parts[0] == "LUT_1D_SIZE":
            raise SystemExit("1D shaper cubes not supported; export a plain 3D cube")
        else:
            try:
                rows.append([float(v) for v in parts[:3]])
            except ValueError:
                continue
    if size is None or len(rows) != size ** 3:
        raise SystemExit(f"not a plain 3D cube: size={size}, rows={len(rows)}")
    # .cube row order: R fastest, then G, then B.
    table = np.asarray(rows, dtype=np.float64).reshape(size, size, size, 3)
    return table, dmin, dmax


def sample_cube(table: np.ndarray, rgb01: np.ndarray) -> np.ndarray:
    """Trilinear interpolation; rgb01 [N,3] in the cube's normalized domain."""
    n = table.shape[0]
    x = np.clip(rgb01, 0.0, 1.0) * (n - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    f = x - lo
    out = np.zeros_like(rgb01)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                ir = np.where(dr, hi[:, 0], lo[:, 0])
                ig = np.where(dg, hi[:, 1], lo[:, 1])
                ib = np.where(db, hi[:, 2], lo[:, 2])
                w = (
                    (f[:, 0] if dr else 1 - f[:, 0])
                    * (f[:, 1] if dg else 1 - f[:, 1])
                    * (f[:, 2] if db else 1 - f[:, 2])
                )
                # .cube indexing: table[b, g, r]
                out += w[:, None] * table[ib, ig, ir]
    return out


def cineon_encode(density_above_min: np.ndarray) -> np.ndarray:
    """Printing density -> Cineon 10-bit code01: CV = 95 + 500·D (0.002 D/code)."""
    return np.clip((95.0 + 500.0 * density_above_min) / 1023.0, 0.0, 1.0)


def logc3_encode(linear: np.ndarray) -> np.ndarray:
    """ARRI LogC3 (EI800) OETF for scene-linear reflectance (x=1 at 18%·5.555...)."""
    x = np.asarray(linear, dtype=np.float64) / 0.18 * 0.18  # explicit: scene linear
    cut, a, b, c, d, e, f = (
        0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809,
    )
    return np.where(x > cut, c * np.log10(a * x + b) + d, e * x + f)


def decode_display(rgb: np.ndarray, gamma: str) -> np.ndarray:
    v = np.clip(rgb, 0.0, 1.0)
    if gamma == "srgb":
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return v ** float(gamma)


DIVERE_LOG_RANGE = float(np.log10(65536.0))  # DiVERE math_ops._LOG65536


def divere_print_channels(curve_json: Path, ev: np.ndarray, d_neg: np.ndarray,
                          le_n: np.ndarray, ours: np.ndarray) -> np.ndarray:
    """Chain B via DiVERE's published Kodak 2383 curve (V7CN/DiVERE, config/curves).

    DiVERE applies per-channel curves in its density domain with the fixed
    normalization x = 1 - D/log10(65536) and y mapping back the same way
    (math_ops._apply_curves_merged_lut). Its 2383 preset's y-axis decodes to
    absolute print density (y_max ~0.98 -> D ~0.10 = 2383 Dmin; y_min ~0.152 ->
    D ~4.08 ~= Dmax), so T = 10^-(D - Dmin) matches our chain-A convention exactly.
    The curve's x is print-side exposure in density units; the per-channel exposure
    trim (DiVERE's dmax/pivot printer lights) is solved here as a scalar offset —
    the same printer-light equivalence the Cineon path declares.
    """
    import json as _json

    curves = _json.loads(curve_json.read_text(encoding="utf-8"))["curves"]
    out = np.zeros((ev.size, 3))
    for c, name in enumerate(("R", "G", "B")):
        pts = np.asarray(curves[name], dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        d_min_out = (1.0 - float(ys.max())) * DIVERE_LOG_RANGE
        dens_c = np.interp(ev * LOG10_2, le_n, d_neg[:, c])
        # Positive (print-exposure) density before the curve: negative flipped.
        d_pos = np.nanmax(d_neg[:, c]) - dens_c

        def channel_T(trim: float, scale: float,
                      d_pos=d_pos, xs=xs, ys=ys, d_min_out=d_min_out):
            x = 1.0 - np.clip((d_pos * scale + trim) / DIVERE_LOG_RANGE, 0.0, 1.0)
            y = np.interp(x, xs, ys)
            d_out = (1.0 - y) * DIVERE_LOG_RANGE
            return np.power(10.0, -(d_out - d_min_out))

        anchor = float(np.interp(0.0, ev, ours[:, c]))

        def solve_trim(scale: float) -> float:
            # Exposure trim (printer light): anchor DiVERE's mid-scale to chain
            # A's own EV-0 value in the SAME domain the comparison runs in.
            trims = np.linspace(-3.0, 4.0, 701)
            mid = np.array([np.interp(0.0, ev, channel_T(t, scale)) for t in trims])
            order = np.argsort(mid)
            return float(np.interp(anchor, mid[order], trims[order]))

        # Two declared per-channel freedoms, both native to DiVERE's own workflow:
        # exposure trim (printer light — their dmax/W/S keys) and density-contrast
        # scale (their R/F keys — the scan-inversion gamma convention). Scale is
        # solved against chain A and REPORTED: a fitted scale far from 1 means the
        # chains disagree about what one unit of negative density is, not about
        # the print curve's shape.
        ref = np.log2(np.maximum(ours[:, c], 1e-4))
        best_rms, best_scale, best_trim = np.inf, 1.0, solve_trim(1.0)
        for scale in np.linspace(0.6, 2.0, 71):
            trim = solve_trim(scale)
            t = channel_T(trim, scale)
            keep = (ours[:, c] > 1e-4) & (t > 1e-4)
            r = np.log2(np.maximum(t[keep], 1e-4)) - ref[keep]
            r -= float(np.mean(r))
            rms = float(np.sqrt(np.mean(r * r)))
            if rms < best_rms:
                best_rms, best_scale, best_trim = rms, float(scale), trim
        print(f"   divere ch {name}: fitted density scale {best_scale:.3f} "
              f"(1.0 = contact-print convention)")
        out[:, c] = channel_T(best_trim, best_scale)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lut", type=Path, required=True,
                    help=".cube 3D LUT, or a DiVERE curve .json with --input divere")
    ap.add_argument("--stock", default="vision3250d", choices=sorted(STOCKS))
    ap.add_argument("--input", default="cineon", choices=("cineon", "logc3", "divere"))
    ap.add_argument("--display-gamma", default="2.4")
    args = ap.parse_args()

    stock = STOCKS[args.stock]
    if stock.get("positive") or stock.get("print") not in ("kodak_2383", "kodak_2393"):
        raise SystemExit(f"{args.stock} is not a 2383/2393-printed cine stock")

    if args.input == "divere":
        # Density-domain oracle: DiVERE's curve is per-channel print density —
        # it has no colorimetric claim, so the comparison must run BEFORE our
        # viewing translation (whose CAT/basis legitimately mixes channels).
        # Chain A supplies per-channel print transmittance 10^-(D - Dmin) from
        # the solved dye amounts, the exact convention the curve decodes to.
        ev_a, amounts = print_density_curves(stock, surround_override="native")
        t_print = np.power(10.0, -(amounts - np.nanmin(amounts, axis=0)[None, :]))
        mask = (ev_a >= EV_LO) & (ev_a <= EV_HI)
        ev = ev_a[mask]
        ours = t_print[mask]
        print("comparison domain: per-channel print density (pre-colorimetry)")
    else:
        # Cube LUTs bake a display rendering; those stay a display comparison.
        ev_a, channels_a, _luma, _floor = build_endtoend_target(
            stock, surround_override="native"
        )
        mask = (ev_a >= EV_LO) & (ev_a <= EV_HI)
        ev = ev_a[mask]
        ours = channels_a[mask]

    # Chain B: negative density -> declared input encoding -> LUT -> display linear.
    le_n, d_neg = _load_curves(stock["negative"])
    if args.input == "divere":
        theirs = divere_print_channels(args.lut, ev, d_neg, le_n, ours)
    else:
        dens = np.stack(
            [np.interp(ev * LOG10_2, le_n, d_neg[:, c]) for c in range(3)], axis=1
        )
        if args.input == "cineon":
            lut_in = cineon_encode(dens - np.nanmin(d_neg, axis=0))
        else:
            # LogC-flavoured PFE cubes expect the *scene* in LogC, with the print
            # baked into the LUT; the negative stage must then be skipped.
            lut_in = np.repeat(logc3_encode(0.18 * np.exp2(ev))[:, None], 3, axis=1)
        table, dmin, dmax = read_cube(args.lut)
        lut_out = sample_cube(table, (lut_in - dmin) / np.maximum(dmax - dmin, 1e-9))
        theirs = decode_display(lut_out, args.display_gamma)

    print(f"stock={args.stock}  lut={args.lut.name}  input={args.input}  "
          f"eotf={args.display_gamma}  N={ev.size}")
    print(f"{'ch':3s} {'offset(stops)':>14s} {'rms(stops)':>11s} {'max(stops)':>11s}")
    floor = 1e-4
    for c, name in enumerate("RGB"):
        a = np.log2(np.maximum(ours[:, c], floor))
        b = np.log2(np.maximum(theirs[:, c], floor))
        keep = (ours[:, c] > floor) & (theirs[:, c] > floor)
        offset = float(np.mean(a[keep] - b[keep]))  # printer-light equivalence
        r = a[keep] - b[keep] - offset
        print(f"{name:3s} {offset:14.3f} {float(np.sqrt(np.mean(r * r))):11.3f} "
              f"{float(np.max(np.abs(r))):11.3f}")
    print("declared normalizations: per-channel log offset (printer light), "
          "selected EOTF, selected input encoding — see module docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
