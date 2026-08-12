#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-validate the dngscan decode/analysis chain against a synthetic
ideal-image pair with fully declared ground truth.

The pair comes from Jiangtherapee's ideal image module
(https://y-g-jiang.github.io/JO.html, by 知乎@姜尧耕): a browser-side
hyperspectral scene sampler that writes a synthetic DNG (Sony ILCE-7RM2
wrapper) whose black level, white level, ADC gain (kadc), full-well and
noise model are all KNOWN — the one thing a real photograph can never
give us. Variant A carries no noise; variant B carries the same
deterministic scene plus a PTC noise model (read + shot). B − A is
therefore a pure per-pixel noise realisation, which lets a single frame
pair recover gain and read noise by photon-transfer regression without
needing flat patches (the synthetic scene is texture-dense everywhere).

Checks:
  1. decode: LibRaw accepts the fake-wrapper DNG; analyze() completes.
  2. metadata black level reads exactly 512 on all four channels.
  3. resolved full-well falls back to metadata white_level (16383) —
     the scene barely clips, so no ceiling pile may override it.
  4. difference-frame PTC: kadc within 1% of declared truth; read
     noise within 10% after removing the difference-frame quantisation
     variance (1/6 ADU², see QUANT_VAR_ADU2) from the fit intercept.
     The derived full-well row is a CONSISTENCY check only — it is
     kadc x (WL - BL), algebraically tied to the gain row, so the
     check count overstates the number of independent evidences.

Run against the archived pair (see the README next to the assets):

    DNGSCAN_IDEAL_IMAGE_DIR=~/Pictures/AgXRAW样张/ideal-image \\
      python tools/validate_ideal_image.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dngscan.analysis import analyze  # noqa: E402
from dngscan.raw_io import load_raw  # noqa: E402

FILE_A = "ideal_A_none.dng"
FILE_B = "ideal_B_ptc.dng"
FILE_META = "ideal_A_metadata.json"
# Variant B's noise model (its metadata JSON was not exported; the
# generator UI declared these — recorded in the archive README).
READ_NOISE_E_B = 3.0

KADC_TOL = 0.01          # relative, on the PTC-recovered gain
FWC_TOL = 0.01           # relative, on kadc * (WL - BL)
READ_NOISE_TOL = 0.10    # relative, after quantisation-variance removal
# The difference frame carries TWO quantisation variances: the noisy
# frame's dither-like rounding error (1/12) plus the noiseless frame's
# deterministic per-pixel rounding error, which is again uniform across
# the pixels of a bin (another 1/12). Monte-Carlo confirms sigma^2 + 1/6
# to 4 digits even at sigma ~= 0.5 ADU.
QUANT_VAR_ADU2 = 1.0 / 6.0


def difference_ptc(a: np.ndarray, b: np.ndarray, black: float,
                   white: float) -> tuple[float, float]:
    """Recover (kadc e-/ADU, read noise e-) from a noiseless/noisy pair.

    Bins pixels by the noiseless frame's signal and regresses the
    difference-frame variance on signal: var_ADU = rn²/k² + 1/6 + m/k.
    """
    sig = (a - black).ravel()
    diff = (b - a).ravel()
    margin = 0.965 * (white - black)  # stay clear of clip on both frames
    keep = (sig >= 0) & (sig < margin) & ((b - black).ravel() < margin)
    sig, diff = sig[keep], diff[keep]
    # log-spaced bins: the intercept (read noise + quantisation, ~0.4 ADU²)
    # is tiny against the shot-noise slope, so it is only identifiable from
    # near-zero-signal bins — linear binning lumps those into one wide bin
    # whose mean signal already couples the intercept to slope error.
    edges = np.concatenate([[0.0], np.geomspace(1.0, margin, 80)])
    idx = np.digitize(sig, edges)
    xs, ys, ns = [], [], []
    for i in range(1, len(edges)):
        sel = idx == i
        n = int(sel.sum())
        if n < 500:
            continue
        xs.append(float(sig[sel].mean()))
        ys.append(float(diff[sel].var(ddof=1)))
        ns.append(n)
    if len(xs) < 5:
        raise RuntimeError("too few populated signal bins for a PTC fit")
    x, y = np.asarray(xs), np.asarray(ys)
    # GLS: the sd of a sample variance is ~ var*sqrt(2/n), so weight each
    # bin by sqrt(n)/y — otherwise high-signal bins drown the intercept.
    w = np.sqrt(np.asarray(ns, dtype=np.float64)) / y
    design = np.vstack([x, np.ones_like(x)]).T * w[:, None]
    (slope, intercept), *_ = np.linalg.lstsq(design, y * w, rcond=None)
    kadc = 1.0 / float(slope)
    rn_var_adu2 = max(float(intercept) - QUANT_VAR_ADU2, 0.0)
    read_e = float(np.sqrt(rn_var_adu2) * kadc)
    return kadc, read_e


def validate(asset_dir: Path) -> list[tuple[str, bool, str]]:
    """Run all checks; returns (name, passed, detail) rows."""
    truth = json.loads((asset_dir / FILE_META).read_text(encoding="utf-8"))
    bl_true = float(truth["dng"]["black_level"])
    wl_true = float(truth["dng"]["white_level"])
    kadc_true = float(truth["sampling"]["kadc_e_per_adu"])
    fwc_true = float(truth["sampling"]["full_well_e_per_pixel"])

    rows: list[tuple[str, bool, str]] = []
    bundles = {}
    for name in (FILE_A, FILE_B):
        bundle = load_raw(asset_dir / name, "clip")
        result, _, _ = analyze(bundle, 4, diagnostics=False)
        bundles[name] = bundle
        bl = tuple(float(v) for v in list(bundle.black_levels)[:4])
        rows.append((
            f"{name}: metadata black level",
            bl == (bl_true,) * 4,
            f"read {bl}, truth {bl_true}",
        ))
        rows.append((
            f"{name}: resolved full-well from metadata white_level",
            float(result.fullwell) == wl_true
            and "white_level" in result.fullwell_note,
            f"fullwell {result.fullwell} ({result.fullwell_note})",
        ))

    a = np.asarray(bundles[FILE_A].raw_image, np.float64)
    b = np.asarray(bundles[FILE_B].raw_image, np.float64)
    kadc, read_e = difference_ptc(a, b, bl_true, wl_true)
    fwc = kadc * (wl_true - bl_true)
    rows.append((
        "PTC gain kadc",
        abs(kadc - kadc_true) / kadc_true <= KADC_TOL,
        f"{kadc:.4f} e-/ADU, truth {kadc_true:.4f} "
        f"({abs(kadc - kadc_true) / kadc_true * 100:.2f}%)",
    ))
    rows.append((
        "PTC derived full-well consistency (kadc x (WL-BL), "
        "not independent of the gain check)",
        abs(fwc - fwc_true) / fwc_true <= FWC_TOL,
        f"{fwc:.0f} e-, truth {fwc_true:.0f} "
        f"({abs(fwc - fwc_true) / fwc_true * 100:.2f}%)",
    ))
    rows.append((
        "PTC read noise (dequantised)",
        abs(read_e - READ_NOISE_E_B) / READ_NOISE_E_B <= READ_NOISE_TOL,
        f"{read_e:.3f} e-, truth {READ_NOISE_E_B:.1f} "
        f"({abs(read_e - READ_NOISE_E_B) / READ_NOISE_E_B * 100:.1f}%)",
    ))
    return rows


def main() -> int:
    asset_dir = os.environ.get("DNGSCAN_IDEAL_IMAGE_DIR")
    if len(sys.argv) > 1:
        asset_dir = sys.argv[1]
    if not asset_dir:
        print("usage: DNGSCAN_IDEAL_IMAGE_DIR=<dir> python "
              "tools/validate_ideal_image.py  (or pass the dir as argv[1])")
        return 2
    rows = validate(Path(asset_dir).expanduser())
    failed = 0
    for name, ok, detail in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed += 0 if ok else 1
    print(f"{len(rows) - failed}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
