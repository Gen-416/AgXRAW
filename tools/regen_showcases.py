# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate current RAW/HDR tutorial assets from explicit JSON manifests.

Use --list before rendering, --only to select jobs, and --install to replace
public documentation assets. Film research manifests moved to AgXFilm.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"
DEFAULT_SAMPLES = Path.home() / "Pictures" / "AgXRAW样张"
JPEG_QUALITY = 95


@dataclass
class RenderSpec:
    """One full-resolution render: source + CLI args -> scratch jpeg."""

    name: str
    source: str
    args: tuple[str, ...]


@dataclass
class AssetSpec:
    """One published asset: which render it comes from and how it is cut.

    crop_from: recover this asset's window inside the named render by NCC
    against the OLD published file, then cut the same window from the new
    render (None = whole frame resized to the published dimensions).
    """

    asset: str  # path under docs/assets
    render: str  # RenderSpec name
    crop_from_old: bool = False
    # Both halves of a comparison table share one crop window; the group
    # installs with the window recovered from its best-matching member.
    crop_group: str | None = None
    # 2026-08-28 refresh: the pristine published FULL frame the crop's window
    # is recovered against, as a path under docs/assets. Optional — the
    # crop_<name>.jpg next to <name>.jpg stays the
    # default; the editing tutorial names its crops differently.
    old_full: str | None = None
    # Post-cut delta guard (mean |new - pristine| over the crop). 25 codes
    # catches a mislocated window; a table whose published halves came from
    # an older, darker chain state legitimately sits near it and declares a
    # wider bound in its manifest instead of silently loosening the default.
    max_delta: float = 25.0


# Shared per-source view declarations. The park gallery was shot on a Sony
# ARW whose showcases declare daylight balance and highlight reconstruction
# (recorded in the session transcript that produced the originals); the
# Sigma DNGs use their as-shot balance.
PARK = ("--wb", "5500k", "--highlight-mode", "reconstruct")

RENDERS: list[RenderSpec] = []

ASSET_SPECS: list[AssetSpec] = []


@dataclass
class PlateSpec:
    """A composite plate: panels pasted into the OLD plate's measured grid.

    The old plate supplies gutters and the black caption strips verbatim
    (pixel-perfect labels, no font reproduction); only the image regions are
    replaced. Boxes are (x0, y0, x1, y1) in plate pixels, row-major panel
    order matching `panels` (RenderSpec names)."""

    asset: str
    panels: tuple[str, ...]
    boxes: tuple[tuple[int, int, int, int], ...]


def _grid_boxes(cols: tuple[tuple[int, int], ...], rows: tuple[tuple[int, int], ...]):
    return tuple(
        (x0, y0, x1, y1) for (y0, y1) in rows for (x0, x1) in cols
    )


# Measured from the published plates (flat-run gutter/caption detection).
# Plates drawn from scratch by tools/compose_plate.py (manifest key
# "composed_plates": {asset, cols:[{key,label}], rows:[{key,label,renders:{colkey: render name}}]}).
COMPOSED_PLATES: list[dict] = []

PLATES: list[PlateSpec] = []


def build_plate(spec: PlateSpec, scratch: Path) -> dict:
    old_path = ASSETS / spec.asset
    plate = Image.open(old_path).copy()
    corrs = []
    for name, box in zip(spec.panels, spec.boxes):
        panel = Image.open(scratch / f"{name}.jpg")
        x0, y0, x1, y1 = box
        resized = panel.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        # Source-identity guard: the new panel must correlate with the old
        # plate's same region (chain drift is a few codes; a wrong source or
        # wrong framing drops correlation off a cliff).
        a = _gray(plate.crop(box)).ravel()
        b = _gray(resized).ravel()
        corr = float(np.corrcoef(a, b)[0, 1])
        corrs.append(round(corr, 3))
        if corr < 0.90:
            raise RuntimeError(
                f"plate {spec.asset} panel {name}: correlation {corr:.3f} < 0.90 "
                "— wrong source or framing, refusing to install"
            )
        plate.paste(resized, (x0, y0))
    plate.save(old_path, quality=JPEG_QUALITY)
    return {"asset": spec.asset, "panels": len(spec.panels), "corr": corrs}


def render(spec: RenderSpec, samples: Path, scratch: Path, py: str) -> Path:
    out = scratch / f"{spec.name}.jpg"
    if out.exists():
        return out
    cmd = [
        py, "-m", "dngscan", str(samples / spec.source),
        "--jpeg", str(out), "--jpeg-quality", str(JPEG_QUALITY),
        "--output-format", "sdr", *spec.args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"render {spec.name} failed:\n{result.stderr[-800:]}")
    return out


def _gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32)


def _ncc_locate(needle: np.ndarray, hay: np.ndarray) -> tuple[float, int, int]:
    """Best normalized cross-correlation position of needle inside hay."""
    from numpy.fft import irfft2, rfft2

    nh, nw = needle.shape
    hh, hw = hay.shape
    if nh > hh or nw > hw:
        return -1.0, 0, 0
    n = needle - needle.mean()
    denom_n = float(np.sqrt((n * n).sum())) or 1.0
    shape = (hh + nh - 1, hw + nw - 1)
    corr = irfft2(rfft2(hay, shape) * rfft2(n[::-1, ::-1], shape), shape)
    corr = corr[nh - 1 : hh, nw - 1 : hw]
    ones = np.ones_like(n)
    hay_sum = irfft2(rfft2(hay, shape) * rfft2(ones[::-1, ::-1], shape), shape)[nh - 1 : hh, nw - 1 : hw]
    hay_sq = irfft2(rfft2(hay * hay, shape) * rfft2(ones[::-1, ::-1], shape), shape)[nh - 1 : hh, nw - 1 : hw]
    var = np.maximum(hay_sq - hay_sum * hay_sum / (nh * nw), 1e-6)
    # Flat hay windows (blank walls, sky) have a tiny denominator and mint
    # spurious NCC peaks; a real match needs comparable local texture.
    needle_std = float(needle.std()) or 1.0
    local_std = np.sqrt(var / (nh * nw))
    ncc = corr / (np.sqrt(var) * denom_n)
    ncc[local_std < 0.3 * needle_std] = -1.0
    idx = int(np.argmax(ncc))
    y, x = divmod(idx, ncc.shape[1])
    return float(ncc[y, x]), y, x


def recover_crop_box(old_crop: Image.Image, old_full: Image.Image) -> tuple[float, float, float, float]:
    """Normalized (x0, y0, x1, y1) of the published crop inside the published
    full-size frame, recovered by multi-scale NCC."""
    from PIL import ImageFilter

    old_crop = old_crop.filter(ImageFilter.GaussianBlur(2))
    old_full = old_full.filter(ImageFilter.GaussianBlur(2))
    crop_g = _gray(old_crop)
    best = (-1.0, None)
    for scale in np.linspace(0.25, 1.0, 32):
        w = int(round(old_full.width * scale * old_crop.width / max(old_crop.width, 1)))
        # search over needle sizes: resize the crop so it occupies `scale` of
        # the full frame's width
        needle_w = max(24, int(round(old_full.width * scale)))
        if needle_w >= old_full.width:
            continue
        needle_h = max(24, int(round(needle_w * old_crop.height / old_crop.width)))
        needle = np.asarray(
            Image.fromarray(crop_g.astype(np.uint8)).resize((needle_w, needle_h)),
            dtype=np.float32,
        )
        score, y, x = _ncc_locate(needle, _gray(old_full))
        if score > best[0]:
            best = (score, (x, y, x + needle_w, y + needle_h))
    score, box = best
    if box is None or score < 0.70:
        raise RuntimeError(f"crop recovery failed (best NCC {score:.3f})")
    x0, y0, x1, y1 = box
    return (
        x0 / old_full.width, y0 / old_full.height,
        x1 / old_full.width, y1 / old_full.height,
    )


def install(spec: AssetSpec, scratch: Path, shared_box=None, dry: bool = False) -> dict:
    old_path = ASSETS / spec.asset
    if spec.crop_from_old:
        import io as _io
        import subprocess as _sp

        blob = _sp.run(
            ["git", "show", f"HEAD:docs/assets/{spec.asset}"],
            capture_output=True, cwd=str(REPO),
        )
        if blob.returncode != 0:
            raise RuntimeError(f"cannot read pristine {spec.asset} from HEAD")
        old = Image.open(_io.BytesIO(blob.stdout))
    else:
        old = Image.open(old_path)
    new_full = Image.open(scratch / f"{spec.render}.jpg")
    if spec.crop_from_old:
        # The window is recovered against the PRISTINE published full-size
        # frame from git HEAD — the working-tree copy may already be the
        # regenerated render (install order), which polluted the NCC match.
        import io as _io
        import subprocess as _sp

        if spec.old_full:
            ref = spec.old_full
        else:
            base_name = spec.asset.replace("crop_", "").rsplit("/", 1)[-1]
            ref = str(Path(spec.asset).parent / base_name)
        blob = _sp.run(
            ["git", "show", f"HEAD:docs/assets/{ref}"],
            capture_output=True, cwd=str(REPO),
        )
        if blob.returncode != 0:
            raise RuntimeError(f"cannot read pristine {ref} from HEAD")
        old_full = Image.open(_io.BytesIO(blob.stdout))
        if shared_box is not None:
            nx0, ny0, nx1, ny1 = shared_box
        else:
            nx0, ny0, nx1, ny1 = recover_crop_box(old, old_full)
        box = (
            int(round(nx0 * new_full.width)), int(round(ny0 * new_full.height)),
            int(round(nx1 * new_full.width)), int(round(ny1 * new_full.height)),
        )
        region = new_full.crop(box)
        out = region.resize(old.size, Image.LANCZOS)
    else:
        out = new_full.resize(old.size, Image.LANCZOS)
    a = np.asarray(old, dtype=np.float32)
    b = np.asarray(out, dtype=np.float32)
    delta = float(np.abs(a - b).mean()) if a.shape == b.shape else float("nan")
    if spec.crop_from_old and not (delta < float(spec.max_delta)):
        raise RuntimeError(
            f"{spec.asset}: post-cut delta {delta:.1f} vs pristine crop "
            f"(bound {spec.max_delta:g}) — window mislocated, refusing to install"
        )
    if not dry:
        out.save(old_path, quality=JPEG_QUALITY)
    info = {"asset": spec.asset, "size": old.size, "mean_delta_vs_old": round(delta, 2)}
    if spec.crop_from_old:
        info["box"] = (nx0, ny0, nx1, ny1)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--install", action="store_true",
                    help="replace docs/assets files (otherwise render only)")
    ap.add_argument("--manifest", type=Path, nargs="*", default=None,
                    help="extra JSON manifest(s) {renders:[{name,source,args}], "
                         "assets:[{asset,render,crop_from_old,crop_group}]} "
                         "appended to the built-in specs (2026-08-28 refresh: "
                         "tutorial images that predate this script)")
    args = ap.parse_args()
    if not args.manifest:
        args.manifest = [REPO / "tools" / "showcase_manifests" / "editing_tutorial.json"]
    if args.manifest:
        import json as _json

        for mpath in args.manifest:
            data = _json.loads(Path(mpath).read_text(encoding="utf-8"))
            for rs in data.get("renders", []):
                RENDERS.append(RenderSpec(rs["name"], rs["source"], tuple(rs.get("args", []))))
            for a in data.get("assets", []):
                ASSET_SPECS.append(AssetSpec(a["asset"], a["render"],
                                             bool(a.get("crop_from_old", False)),
                                             a.get("crop_group"),
                                             old_full=a.get("old_full"),
                                             max_delta=float(a.get("max_delta", 25.0))))
            for pl in data.get("plates", []):
                PLATES.append(PlateSpec(pl["asset"], tuple(pl["panels"]),
                                        tuple(tuple(int(v) for v in box) for box in pl["boxes"])))
            for cp in data.get("composed_plates", []):
                COMPOSED_PLATES.append(cp)
    if args.list:
        for spec in RENDERS:
            print(f"{spec.name:32s} {spec.source:16s} {' '.join(spec.args)}")
        return 0
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="showcase-"))
    scratch.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    wanted = set(args.only) if args.only else None
    for spec in RENDERS:
        if wanted and spec.name not in wanted:
            continue
        print(f"render {spec.name} ...", flush=True)
        render(spec, args.samples, scratch, py)
    if args.install:
        groups = {}
        singles = []
        for aspec in ASSET_SPECS:
            if wanted and aspec.render not in wanted:
                continue
            if aspec.crop_group:
                groups.setdefault(aspec.crop_group, []).append(aspec)
            else:
                singles.append(aspec)
        for aspec in singles:
            print(install(aspec, scratch), flush=True)
        for gname, members in groups.items():
            best = None
            for m in members:
                try:
                    trial = install(m, scratch, dry=True)
                except RuntimeError:
                    continue
                if best is None or trial["mean_delta_vs_old"] < best["mean_delta_vs_old"]:
                    best = trial
            if best is None:
                raise RuntimeError(f"crop group {gname}: no member recovered a window")
            for m in members:
                print(install(m, scratch, shared_box=best["box"]), flush=True)
        for pspec in PLATES:
            if wanted and not any(p in wanted for p in pspec.panels):
                continue
            info = build_plate(pspec, scratch)
            print(info, flush=True)
        for cp in COMPOSED_PLATES:
            names = {r for row in cp["rows"] for r in row["renders"].values()}
            if wanted and not (names & wanted):
                continue
            import importlib.util as _ilu

            spec = _ilu.spec_from_file_location("compose_plate", REPO / "tools" / "compose_plate.py")
            mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
            plate_spec = {"cols": cp["cols"], "rows": [
                {"key": row["key"], "label": row["label"],
                 "files": {ck: str(scratch / f"{rn}.jpg") for ck, rn in row["renders"].items()}}
                for row in cp["rows"]]}
            mod.compose(str(ASSETS / cp["asset"]), plate_spec)
            print({"composed": cp["asset"]}, flush=True)
    print("scratch:", scratch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
