#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""One fresh-process kernel/RSS probe; DNGSCAN_FAST=0 selects NumPy.

Use --repo to compare checkouts with their own compiled extensions. Peak RSS
includes inputs; extra_peak_mib subtracts the high-water mark after touching
inputs and loading the kernel. Time covers only the operation, not imports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--kernel", choices=("gain", "hdr", "base", "feather", "blur", "small-blur", "area", "scatter"), required=True)
    parser.add_argument("--height", type=int, default=4000)
    parser.add_argument("--width", type=int, default=6000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.7)
    args = parser.parse_args()
    if min(args.height, args.width) < 1:
        parser.error("dimensions must be positive")
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from dngscan import _fast, gainmap, raw_io, film_optics

    _fast.set_thread_budget(args.threads)
    h, w = args.height, args.width
    if args.kernel == "gain":
        img = np.full((h, w), 1000, np.uint16)
        colors = np.full((h, w), 0, np.uint8)
        raw = SimpleNamespace(raw_image_visible=img, raw_colors_visible=colors)
        op = SimpleNamespace(top=0, left=0, bottom=h, right=w, row_pitch=1, col_pitch=1,
                             origin_v=0., origin_h=0., spacing_v=1., spacing_h=1.,
                             points_v=2, points_h=2, gains=np.full((2, 2, 1), 1.1), plane=0, planes=1)
        run = lambda: raw_io._apply_gain_maps_mosaic(raw, [op], [512.], 16383, [16383.])
    elif args.kernel == "hdr":
        # Exactly the production readback boundary: strided RGB of RGBA.
        a = np.full((h, w, 4), 0.5, np.float16)
        e = np.full((h, w, 3), 0.5, np.float16)
        run = lambda: gainmap._roundtrip_error_arrays(a[..., :3], e)
    elif args.kernel == "base":
        a = np.full((h, w, 3), 128, np.uint8)
        e = np.full((h, w, 3), 127, np.uint8)
        run = lambda: gainmap._base_roundtrip_error_arrays(a, e)
    elif args.kernel == "feather":
        a = np.full((h, w, 3), 0.5, np.float32)
        run = lambda: raw_io._feather_masks_f16(a)
    elif args.kernel == "blur":
        a = np.full((h, w, 3), 0.5, np.float32)
        run = lambda: film_optics._blur_bounded(a, args.sigma)
    elif args.kernel == "small-blur":
        a = np.full((h, w, 3), 0.5, np.float32)
        run = lambda: film_optics._blur_small_sigma(a[..., 1], 0.7)
    elif args.kernel == "area":
        a = np.full((h, w, 3), 0.5, np.float32)
        run = lambda: film_optics.area_decimate(a, min(h, 384), min(w, 512))
    else:
        from dngscan.film_optics_assets import DEFAULT_STOCK_OPTICS, load_stock_optics

        kernel = load_stock_optics(DEFAULT_STOCK_OPTICS).emulsion_scatter
        a = np.full((h, w, 3), 0.5, np.float32)
        run = lambda: film_optics.apply_scatter_mix(a, 36.0 / w, kernel)

    def peak():
        scale = 2**20 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

    before = peak()
    start = time.perf_counter()
    result = run()
    seconds = time.perf_counter() - start
    after = peak()
    if args.kernel == "gain":
        assert np.all(img == 1048)
    elif args.kernel == "feather":
        assert result.dtype == np.float16 and np.all(result == 0.5)
    elif args.kernel in ("blur", "small-blur", "area", "scatter"):
        assert result.dtype == np.float32 and np.allclose(result, 0.5, rtol=0, atol=1e-5)
    print(json.dumps(dict(kernel=args.kernel, shape=[h, w], threads=args.threads,
                          native=_fast.kernel("feather_masks_f16") is not None,
                          seconds=seconds, peak_mib=after, extra_peak_mib=after - before)))


if __name__ == "__main__":
    main()
