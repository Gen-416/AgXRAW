#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense sub-pixel digitization of the Kodak chart curves.

Third pass of the 2026-08-25 precision program (owner directive: push
digitization precision as high as computation allows). Supersedes the
handful of second-pass crossing-scan anchors with a dense, reproducible
scan:

  - pages rendered at 8x via Quartz (vector PDFs, so the 4x axis
    calibrations scale exactly by 2);
  - crossings extracted as darkness-weighted sub-pixel centroids
    (read precision ~±0.5 px at 8x ≈ ±0.001 logE / <0.5% response),
    instead of run midpoints at 4x (±2 px ≈ ±0.008 logE);
  - dense targets (0.02 D steps, 0.04 logE steps, log-spaced
    frequencies) with self-consistent channel assignment: predictions
    from a PCHIP fit of the accepted set, iterated to convergence;
  - acceptance requires a single candidate inside the prediction window
    AND cross-channel separation of at least 1.2 stroke widths — the
    stroke-merged zones (2383 toe D∈[0.35,0.9], R/G sigma above
    logE≈1.4, MTF G/B crossing near 35 c/mm) reject themselves, and the
    first-pass manual anchors remain authoritative there.

Output: tools/chart_scans/<dataset>.json, consumed by
import_kodak_granularity.py / import_kodak_mtf.py. The Kodak PDFs stay
out of the repo (copyright); this script re-derives the scan files from
the URLs recorded in the importers.

Usage:
    python tools/scan_chart_curves.py --pdf-5207 /path/5207.pdf \\
        --pdf-2383 /path/2383.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tools" / "chart_scans"
sys.path.insert(0, str(ROOT / "tools"))

RENDER_SCALE = 8.0          # 4x calibrations scale exactly by 2
CAL_SCALE = 2.0
DARK = 180.0                # capture threshold (0..255 gray)
MAX_RUN_PX = 28             # reject grid lines / text (stroke ~8-12 px at 8x)
SEP_FACTOR = 1.2            # min cross-channel separation, in stroke widths


def render_page(pdf: Path, pageno: int) -> np.ndarray:
    import Quartz
    from Foundation import NSURL

    doc = Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(str(pdf)))
    page = Quartz.CGPDFDocumentGetPage(doc, pageno)
    box = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFMediaBox)
    w, h = int(box.size.width * RENDER_SCALE), int(box.size.height * RENDER_SCALE)
    cs = Quartz.CGColorSpaceCreateDeviceGray()
    ctx = Quartz.CGBitmapContextCreate(None, w, h, 8, w, cs, Quartz.kCGImageAlphaNone)
    Quartz.CGContextSetGrayFillColor(ctx, 1.0, 1.0)
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, w, h))
    Quartz.CGContextScaleCTM(ctx, RENDER_SCALE, RENDER_SCALE)
    Quartz.CGContextDrawPDFPage(ctx, page)
    buf = Quartz.CGBitmapContextGetData(ctx).as_buffer(
        Quartz.CGBitmapContextGetBytesPerRow(ctx) * h)
    row_bytes = Quartz.CGBitmapContextGetBytesPerRow(ctx)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, row_bytes)[:, :w].copy()
    # no flip: the buffer is already top-down in the orientation the 4x
    # calibrations were measured in (verified against the first-pass renders)
    return arr


def subpixel_runs(profile: np.ndarray) -> list[tuple[float, int]]:
    """(darkness-weighted centroid, run length) for each dark run."""
    dark = profile < DARK
    out = []
    i = 0
    n = len(dark)
    while i < n:
        if dark[i]:
            j = i
            while j < n and dark[j]:
                j += 1
            lo, hi = max(0, i - 1), min(n, j + 1)
            w = np.clip(255.0 - profile[lo:hi].astype(float), 0.0, None)
            c = float((np.arange(lo, hi) * w).sum() / max(w.sum(), 1e-9))
            out.append((c, j - i))
            i = j
        else:
            i += 1
    return out


def scan_family(im, targets, target_to_px, px_to_value, predictors,
                window, max_run=MAX_RUN_PX, axis="row"):
    """Generic dense scan. predictors: {ch: callable(target)->value}.
    Returns {ch: [[target, value], ...]}, using single-candidate +
    cross-channel-separation acceptance."""
    accepted: dict[str, list] = {ch: [] for ch in predictors}
    stroke = []
    for t in targets:
        px = int(round(target_to_px(t)))
        if not (0 <= px < (im.shape[0] if axis == "row" else im.shape[1])):
            continue
        profile = im[px] if axis == "row" else im[:, px]
        runs = [(c, ln) for c, ln in subpixel_runs(profile) if ln <= max_run]
        if not runs:
            continue
        stroke.extend(ln for _, ln in runs)
        med_stroke = float(np.median([ln for _, ln in runs]))
        min_sep = SEP_FACTOR * med_stroke
        vals = [(px_to_value(c), c, ln) for c, ln in runs]
        preds_here = {ch: predictors[ch](t) for ch in predictors}
        for ch, p in preds_here.items():
            if p is None:
                continue
            near = [(v, c) for v, c, _ in vals if abs(_metric(v, p, window)) <= 1.0]
            if len(near) != 1:
                continue
            v, c = near[0]
            # stroke width expressed in value units at this position: a pair
            # of curves whose predictions sit closer than that is merged
            # within the vector stroke and cannot be split at any render
            # scale — reject rather than guess.
            sep_value = min_sep * abs(px_to_value(c + 0.5) - px_to_value(c - 0.5))
            if any(p2 is not None and ch2 != ch and abs(p2 - v) < sep_value
                   for ch2, p2 in preds_here.items()):
                continue
            accepted[ch].append([round(float(t), 4), float(v)])
    return accepted, (float(np.median(stroke)) if stroke else 0.0)


def _metric(v, p, window):
    kind, size = window
    if kind == "lin":
        return (v - p) / size
    return (math.log10(max(v, 1e-12)) - math.log10(max(p, 1e-12))) / size


def _pchip_or_none(pts, xcol=0, ycol=1, logy=False):
    from scipy.interpolate import PchipInterpolator

    a = np.asarray(sorted(pts, key=lambda r: r[xcol]), float)
    keep = [0]
    for i in range(1, len(a)):
        if a[i, xcol] > a[keep[-1], xcol] + 1e-9:
            keep.append(i)
    a = a[keep]
    y = np.log10(a[:, ycol]) if logy else a[:, ycol]
    f = PchipInterpolator(a[:, xcol], y)
    lo, hi = a[0, xcol], a[-1, xcol]

    def pred(x):
        if not (lo <= x <= hi):
            return None
        v = float(f(x))
        return 10 ** v if logy else v

    return pred


def iterate(scan_once, seeds, rounds=3):
    cur = seeds
    for _ in range(rounds):
        nxt = scan_once(cur)
        merged = {}
        for ch in seeds:
            pts = {round(x, 4): y for x, y in nxt.get(ch, [])}
            for x, y in seeds[ch]:
                pts.setdefault(round(float(x), 4), float(y))
            merged[ch] = sorted([[x, y] for x, y in pts.items()])
        cur = merged
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-5207", type=Path, required=True)
    ap.add_argument("--pdf-2383", type=Path, required=True)
    args = ap.parse_args()
    import import_kodak_granularity as G
    import import_kodak_mtf as M

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = {"5207": render_page(args.pdf_5207, 3),
             "2383": render_page(args.pdf_2383, 4)}

    for key, gcal, gdata in (("5207", G.CAL_5207, G.DATA_5207),
                             ("2383", G.CAL_2383, G.DATA_2383)):
        im = pages[key]
        cal = {k: v * CAL_SCALE if k.endswith(("px", "py")) or "px_per" in k else v
               for k, v in gcal.items()}
        loge_max = 5.0 if key == "5207" else 1.95
        out = {"format": "dngscan-chart-scan-1", "dataset": key,
               "render_scale": RENDER_SCALE, "channels": {}}

        # density(logE): horizontal density-line crossings, targets by D
        dmaxes = {ch: max(r[1] for r in gdata["channels"][ch]["density_loge"])
                  for ch in "RGB"}
        dmins = {ch: min(r[1] for r in gdata["channels"][ch]["density_loge"])
                 for ch in "RGB"}

        def scan_density(anchors_led):
            # anchors rows are [logE, D]; the predictor maps D -> logE
            preds = {ch: _pchip_or_none([[d, le] for le, d in anchors_led[ch]])
                     for ch in anchors_led}
            # predictor maps D -> logE; window 0.05 logE
            targets = np.arange(min(dmins.values()) + 0.04,
                                max(dmaxes.values()) - 0.02, 0.02)
            acc, _ = scan_family(
                im, targets,
                lambda D: cal["d0_py"] - D * cal["px_per_density"],
                lambda c: (c - cal["x0_px"]) / cal["px_per_loge"],
                preds, ("lin", 0.05), axis="row")
            # swap back to (logE, D) rows, clamp to plot box
            return {ch: [[le, D] for D, le in rows if 0.0 <= le <= loge_max]
                    for ch, rows in acc.items()}

        seeds_d = {ch: [[le, d] for le, d in gdata["channels"][ch]["density_loge"]]
                   for ch in "RGB"}
        dens = scan_density(seeds_d)
        # one refinement round with enriched predictions
        enriched = {ch: sorted(seeds_d[ch] + dens[ch], key=lambda r: r[0])
                    for ch in dens}
        dens = scan_density(enriched)

        # sigma(logE): vertical crossings on the log sigma axis
        def scan_sigma(anchors):
            preds = {ch: _pchip_or_none(anchors[ch], logy=True) for ch in anchors}
            targets = np.arange(0.04, loge_max - 0.02, 0.04)
            acc, _ = scan_family(
                im, targets,
                lambda x: cal["x0_px"] + x * cal["px_per_loge"],
                lambda c: cal["sigma_ref"] * 10 ** ((cal["sigma_ref_py"] - c)
                                                    / cal["px_per_decade"]),
                preds, ("log", 0.06), axis="col")
            return acc

        seeds_s = {ch: [[x, s] for x, s in gdata["channels"][ch]["sigma_loge"]]
                   for ch in "RGB"}
        sig = scan_sigma(seeds_s)
        sig = scan_sigma({ch: sorted(seeds_s[ch] + sig[ch]) for ch in sig})

        for ch in "RGB":
            out["channels"][ch] = {"density_loge": dens.get(ch, []),
                                   "sigma_loge": sig.get(ch, [])}
        path = OUT_DIR / f"granularity_{key}_scan.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)}: " + ", ".join(
            f"{ch} d+{len(out['channels'][ch]['density_loge'])}"
            f"/s+{len(out['channels'][ch]['sigma_loge'])}" for ch in "RGB"))

    # ---- MTF (both stocks) ----
    for key in ("5207", "2383"):
        im = pages[key]
        cal = {k: v * CAL_SCALE for k, v in M.CAL[key].items()}
        chans = M.DATA[key]["channels"]

        def scan_mtf(anchors):
            preds = {ch: _pchip_or_none([[math.log10(f), r] for f, r in anchors[ch]])
                     for ch in anchors}
            fmin = min(r[0] for ch in anchors for r in anchors[ch])
            fmax = max(r[0] for ch in anchors for r in anchors[ch])
            targets = np.linspace(math.log10(fmin) + 0.02,
                                  math.log10(fmax) - 0.02, 90)
            acc, _ = scan_family(
                im, targets,
                lambda lf: cal["x0_px"] + lf * cal["px_per_decade_x"],
                lambda c: 10 ** (-(c - cal["y100_py"]) / cal["px_per_decade_y"]),
                preds, ("lin", 0.05), axis="col")
            return {ch: [[round(10 ** lf, 2), r] for lf, r in rows]
                    for ch, rows in acc.items()}

        seeds = {ch: [list(r) for r in chans[ch]] for ch in chans}
        mtf = scan_mtf(seeds)
        mtf = scan_mtf({ch: sorted(seeds[ch] + mtf[ch]) for ch in mtf})
        out = {"format": "dngscan-chart-scan-1", "dataset": key,
               "render_scale": RENDER_SCALE,
               "channels": mtf}
        path = OUT_DIR / f"mtf_{key}_scan.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)}: " + ", ".join(
            f"{ch}+{len(mtf.get(ch, []))}" for ch in chans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
