# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate the film showcase assets from their declared parameters.

Owner directive (2026-08-14): the full-mode showcases went stale when optics
V2 (R1 density-native grain, R3 default media scatter) changed default full
output. Doctrine for showcases: every table is rendered in ONE pipeline
generation (both halves together), dimensions are preserved exactly, and the
prose numbers are re-measured on the new renders. This manifest closes the
"no generation script" debt — P7b-era assets were rendered ad-hoc and the
exact chain state proved unrecoverable (renders from an in-flight working
tree; today's chain reproduces neither commit's output byte-for-byte, so
same-generation re-rendering of whole tables is the only honest refresh).

Usage:
    python tools/regen_showcases.py --list
    python tools/regen_showcases.py [--only NAME ...] [--samples DIR]
    python tools/regen_showcases.py --manifest tools/showcase_manifests/*.json --install

2026-08-28 refresh: the tutorial images that predate this script are declared
in tools/showcase_manifests/{readme,editing_tutorial,film_tutorial}.json
(same RenderSpec/AssetSpec/PlateSpec vocabulary, JSON) — README plates, the
editing tutorial (X100VI RAF + two iPhone ProRAW DNGs by absolute path) and
the film tutorial's curve/WB/strength/primaries/colour-head tables. Crops
whose published full frame has a different name declare `old_full`; a table
whose published halves came from an older chain state declares `max_delta`.

Renders land in a scratch directory first; --install resizes to each
published asset's exact dimensions and replaces docs/assets files. Crops are
recovered by NCC template matching against the OLD published asset so the
framing survives regeneration (multi-scale, normalized cross-correlation on
the green channel).
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
    # film-tutorial convention (crop_<name>.jpg next to <name>.jpg) stays the
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

RENDERS: list[RenderSpec] = [
    # --- park curve-preset gallery + full pair (DSC00225.ARW) ---
    RenderSpec("park_none", "DSC00225.ARW", PARK),
    RenderSpec("park_portra400", "DSC00225.ARW", ("--film", "portra400", *PARK)),
    RenderSpec("park_velvia100", "DSC00225.ARW", ("--film", "velvia100", *PARK)),
    RenderSpec(
        "park_velvia100_fullmode",
        "DSC00225.ARW",
        ("--film", "velvia100", "--film-mode", "full", *PARK),
    ),
    # --- HK hillside observe/full pair (Velvia 100) ---
    RenderSpec("fullmode_hk_observe", "_SDI0164.DNG", ("--film", "velvia100")),
    RenderSpec(
        "fullmode_hk_full",
        "_SDI0164.DNG",
        ("--film", "velvia100", "--film-mode", "full",
         "--film-neutralization", "native"),
    ),
    # --- Expo observe/full pair (Portra 400) ---
    RenderSpec("fullmode_expo_observe", "_SDI0231.DNG", ("--film", "portra400")),
    RenderSpec(
        "fullmode_expo_full",
        "_SDI0231.DNG",
        ("--film", "portra400", "--film-mode", "full",
         "--film-neutralization", "native"),
    ),
    # --- crossover pairs ---
    RenderSpec(
        "crossover_verita_off",
        "_SDI0115.DNG",
        ("--film", "verita200d", "--film-mode", "full",
         "--film-neutralization", "technical-neutral"),
    ),
    RenderSpec(
        "crossover_verita_datasheet",
        "_SDI0115.DNG",
        ("--film", "verita200d", "--film-mode", "full",
         "--film-neutralization", "native"),
    ),
    RenderSpec(
        "crossover_k64_off",
        "_SDI0165.DNG",
        ("--film", "kodachrome64", "--film-mode", "full",
         "--film-neutralization", "technical-neutral"),
    ),
    RenderSpec(
        "crossover_k64_datasheet",
        "_SDI0165.DNG",
        ("--film", "kodachrome64", "--film-mode", "full",
         "--film-neutralization", "native"),
    ),
    # --- §九 look grids: six declared readings per scene (doc-recorded CLI) ---
    RenderSpec("grid_hk_1", "_SDI0164.DNG", ()),
    RenderSpec("grid_hk_2", "_SDI0164.DNG", ("--film", "velvia100")),
    RenderSpec("grid_hk_3", "_SDI0164.DNG",
               ("--film", "velvia100", "--scene-transform-strength", "2.2", "--ev", "-0.3")),
    RenderSpec("grid_hk_4", "_SDI0164.DNG",
               ("--film", "velvia100", "--scene-transform-strength", "2.2", "--ev", "-0.3",
                "--toe-end-offset", "-1", "--midtone-contrast", "0.3")),
    RenderSpec("grid_hk_5", "_SDI0164.DNG",
               ("--film", "velvia100", "--film-mode", "full",
                "--film-neutralization", "datasheet", "--ev", "-0.3")),
    RenderSpec("grid_hk_6", "_SDI0164.DNG",
               ("--film", "velvia100", "--grade", "look:optic_warm_cyan",
                "--grade-strength", "0.5", "--scene-transform-strength", "2.0",
                "--ev", "-0.2")),
    RenderSpec("grid_park_1", "DSC00225.ARW", PARK),
    RenderSpec("grid_park_2", "DSC00225.ARW", ("--film", "portra400", *PARK)),
    RenderSpec("grid_park_3", "DSC00225.ARW",
               ("--film", "portra400", "--scene-transform-strength", "1.8",
                "--ev", "-0.2", "--midtone-contrast", "0.2", *PARK)),
    RenderSpec("grid_park_4", "DSC00225.ARW",
               ("--film", "portra400", "--scene-transform-strength", "1.8",
                "--ev", "-0.2", "--midtone-contrast", "0.2",
                "--toe-end-offset", "-1", *PARK)),
    RenderSpec("grid_park_5", "DSC00225.ARW",
               ("--film", "portra400", "--color-head-y", "15", *PARK)),
    RenderSpec("grid_park_6", "DSC00225.ARW",
               ("--film", "portra400", "--film-mode", "full",
                "--film-neutralization", "datasheet", "--ev", "-0.2", *PARK)),
    RenderSpec("grid_bs_1", "_SDI0173.DNG", ()),
    RenderSpec("grid_bs_2", "_SDI0173.DNG", ("--film", "velvia100")),
    RenderSpec("grid_bs_3", "_SDI0173.DNG",
               ("--film", "velvia100", "--scene-transform-strength", "2.2", "--ev", "-0.3")),
    RenderSpec("grid_bs_4", "_SDI0173.DNG",
               ("--film", "velvia100", "--scene-transform-strength", "2.2", "--ev", "-0.3",
                "--toe-end-offset", "-1", "--midtone-contrast", "0.3")),
    RenderSpec("grid_bs_5", "_SDI0173.DNG",
               ("--film", "velvia100", "--film-mode", "full",
                "--film-neutralization", "datasheet", "--ev", "-0.3")),
    RenderSpec("grid_bs_6", "_SDI0173.DNG",
               ("--film", "velvia100", "--grade", "look:optic_warm_cyan",
                "--grade-strength", "0.5", "--scene-transform-strength", "2.0",
                "--ev", "-0.2")),
    # --- README three-interpretations plate (temple, Vision3 250D) ---
    RenderSpec("interp_observe", "_SDI0094.DNG", ("--film", "vision3250d")),
    RenderSpec("interp_technical", "_SDI0094.DNG",
               ("--film", "vision3250d", "--film-mode", "full",
                "--film-appearance", "technical")),
    RenderSpec("interp_reference", "_SDI0094.DNG",
               ("--film", "vision3250d", "--film-mode", "full",
                "--film-appearance", "reference")),
]

ASSET_SPECS: list[AssetSpec] = [
    AssetSpec("film-tutorial/park_none.jpg", "park_none"),
    AssetSpec("film-tutorial/park_portra400.jpg", "park_portra400"),
    AssetSpec("film-tutorial/park_velvia100.jpg", "park_velvia100"),
    AssetSpec("film-tutorial/park_velvia100_fullmode.jpg", "park_velvia100_fullmode"),
    AssetSpec("film-tutorial/fullmode_hk_observe.jpg", "fullmode_hk_observe"),
    AssetSpec("film-tutorial/fullmode_hk_full.jpg", "fullmode_hk_full"),
    AssetSpec("film-tutorial/fullmode_expo_observe.jpg", "fullmode_expo_observe"),
    AssetSpec("film-tutorial/fullmode_expo_full.jpg", "fullmode_expo_full"),
    AssetSpec("film-tutorial/crossover_verita_off.jpg", "crossover_verita_off"),
    AssetSpec("film-tutorial/crossover_verita_datasheet.jpg", "crossover_verita_datasheet"),
    AssetSpec("film-tutorial/crossover_k64_off.jpg", "crossover_k64_off"),
    AssetSpec("film-tutorial/crossover_k64_datasheet.jpg", "crossover_k64_datasheet"),
    AssetSpec("film-tutorial/crop_crossover_verita_off.jpg", "crossover_verita_off", crop_from_old=True, crop_group="verita"),
    AssetSpec("film-tutorial/crop_crossover_verita_datasheet.jpg", "crossover_verita_datasheet", crop_from_old=True, crop_group="verita"),
    AssetSpec("film-tutorial/crop_crossover_k64_off.jpg", "crossover_k64_off", crop_from_old=True, crop_group="k64"),
    AssetSpec("film-tutorial/crop_crossover_k64_datasheet.jpg", "crossover_k64_datasheet", crop_from_old=True, crop_group="k64"),
]


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
PLATES: list[PlateSpec] = [
    PlateSpec(
        "film-tutorial/look_grid_hk.jpg",
        ("grid_hk_1", "grid_hk_2", "grid_hk_3", "grid_hk_4", "grid_hk_5", "grid_hk_6"),
        _grid_boxes(((0, 520), (526, 1046), (1052, 1572)), ((0, 780), (834, 1614))),
    ),
    PlateSpec(
        "film-tutorial/look_grid_park.jpg",
        ("grid_park_1", "grid_park_2", "grid_park_3", "grid_park_4", "grid_park_5", "grid_park_6"),
        _grid_boxes(((0, 1100), (1106, 2206)), ((0, 734), (788, 1522), (1576, 2310))),
    ),
    PlateSpec(
        "film-tutorial/look_grid_backstage.jpg",
        ("grid_bs_1", "grid_bs_2", "grid_bs_3", "grid_bs_4", "grid_bs_5", "grid_bs_6"),
        _grid_boxes(((0, 520), (526, 1046), (1052, 1572)), ((0, 780), (834, 1614))),
    ),
    PlateSpec(
        "film-three-interpretations.jpg",
        ("interp_observe", "interp_technical", "interp_reference"),
        _grid_boxes(((0, 1100), (1108, 2208), (2216, 3316)), ((0, 733),)),
    ),
]


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
            ref = f"film-tutorial/{base_name}"
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
    print("scratch:", scratch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
