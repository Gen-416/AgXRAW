#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export web-sized REAL HDR samples (ISO 21496-1 gain-map JPEG) for the README.

The README should let a reader with an HDR screen SEE the result, not a tone-mapped
picture of it. A full-size export is 11-27 MB, too heavy for a front page, so this tool
runs the normal export pipeline and only intercepts the finished pair right before
encoding: the SDR base and the HDR alternate are box-averaged in LINEAR light by the
same integer factor, then written and round-trip verified by the project's own
gain-map writer (share profile). Nothing about the rendering changes.

    python tools/make_hdr_showcase.py RAW OUT.jpg [--long-edge 2048] [-- extra dngscan args]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "dngscan").is_dir():
    sys.path.insert(0, str(ROOT))


def _box(a: np.ndarray, k: int) -> np.ndarray:
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    a = a[:h, :w].astype(np.float32)
    return a.reshape(h // k, k, w // k, k, a.shape[2]).mean(axis=(1, 3))


def _srgb_decode(u8: np.ndarray) -> np.ndarray:
    v = u8.astype(np.float32) / 255.0
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def _srgb_encode(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    v = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.rint(v * 255.0), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--long-edge", type=int, default=2048)
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]

    from dngscan import export as export_mod

    real_encode = export_mod.encode_finished_pair

    def downsized_encode(pair, out_path, delivery):
        h, w = pair.sdr_rgb_u8.shape[:2]
        k = max(1, -(-max(h, w) // int(args.long_edge)))
        if k > 1:
            base = _srgb_encode(_box(_srgb_decode(pair.sdr_rgb_u8), k))
            hdr = _box(np.asarray(pair.hdr_rgba_f16, dtype=np.float32), k).astype(np.float16)
            pair = replace(pair, sdr_rgb_u8=np.ascontiguousarray(base), hdr_rgba_f16=np.ascontiguousarray(hdr))
        info = real_encode(pair, out_path, delivery)
        print(f"showcase: {w}x{h} -> {pair.sdr_rgb_u8.shape[1]}x{pair.sdr_rgb_u8.shape[0]} (box {k}x{k}), "
              f"headroom {pair.display_headroom_ev:+.2f} EV")
        return info

    export_mod.encode_finished_pair = downsized_encode
    from dngscan.cli import main as cli_main

    return int(cli_main([str(args.raw), "--jpeg", str(args.out), "--output-format", "ultrahdr",
                         "--delivery-profile", "share", *extra]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
