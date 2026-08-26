#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the analog-optics operators on synthetic charts (FILM_OPTICS_V2 P0).

This is the tool that makes the V2 plan's exit gates checkable. It drives the
REAL production operators — `apply_film_core` with a full-frame spatial
context, the same code path a 60 MP export takes — over the §10.2 charts, and
reports for each of grain / halation / bloom:

    isolated delta, radial profile, half-energy radius, encircled energy,
    per-channel energy ratio, PSD, MTF50, granularity at a 48 um aperture

Each operator is measured ALONE. A combined render cannot answer "how wide is
the halo" because grain and bloom move the same pixels; the report always
renders a baseline with every amount at zero and subtracts.

    python tools/film_optics_report.py --json out.json
    python tools/film_optics_report.py --perf          # adds the 61 MP timing
    python tools/film_optics_report.py --charts DIR    # also writes PNG panels
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DNGSCAN_FAST", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from dngscan import film_optics_charts as charts
from dngscan import film_optics_diag as diag
from dngscan.film_develop import apply_film_core
from dngscan.film_optics import GATE_W_MM
from dngscan.film_optics_assets import (
    DEFAULT_PRINT_OPTICS,
    DEFAULT_STOCK_OPTICS,
    load_print_optics,
    load_stock_optics,
)

# P1: the report reads the same declared assets the renderer compiles, so a
# constant it prints can never be one the render path does not use.
STOCK_OPTICS = load_stock_optics(DEFAULT_STOCK_OPTICS)
PRINT_OPTICS = load_print_optics(DEFAULT_PRINT_OPTICS)
GRAIN = STOCK_OPTICS.grain
HALATION = STOCK_OPTICS.halation
FORMATION_SCATTER = PRINT_OPTICS.formation_scatter

# The GUI's declared tiers (gui/service.py). "standard" is the modelled
# default a user actually gets; "light" is the conservative tier. Amounts are
# (grain, halation, bloom).
TIERS: dict[str, tuple[float, float, float]] = {
    "light": (0.25, 0.20, 0.15),
    "standard": (0.50, 0.40, 0.30),
}

DEFAULT_STOCK = "portra400"

# Chart size. Large enough that a 0.55 mm halo is tens of pixels and a
# granularity estimate has thousands of independent samples, small enough to
# rerun the whole report in seconds.
CHART_H, CHART_W = 768, 1152


def make_plan(stock: str, **amounts) -> SimpleNamespace:
    base = dict(
        curve_preset=stock, film_mode="full", film_crossover="datasheet",
        film_exposure_ev=0.0, film_print_timing="fixed",
        film_print_medium="", film_print_exposure_ev=0.0,
        color_head_y=0.0, color_head_m=0.0,
        film_development="measured_default",
        film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
        film_compression=0.0, film_compression_knee=2.0,
        film_highlight_density=0.0,
        film_grain=0.0, film_halation=0.0, film_bloom=0.0,
        film_optics_seed=0,
        # Review R1 item 4: media-scatter enablement. The per-operator
        # measurements below switch it "off" on BOTH sides so the isolated
        # delta is the named operator alone — with it riding along, every
        # look slider silently dragged both scatter stages into the
        # measurement (and into gates certified against it).
        film_media_scatter="declared",
    )
    base.update(amounts)
    return SimpleNamespace(**base)


def develop(scene: np.ndarray, plan: SimpleNamespace) -> np.ndarray:
    """Full-frame oracle development of a scene-linear chart."""
    h, w = scene.shape[:2]
    flat = np.asarray(scene, dtype=np.float32).reshape(-1, 3)
    out = apply_film_core(flat, plan, spatial_shape=(h, w))
    return np.asarray(out, dtype=np.float64).reshape(h, w, 3)


def mm_per_px(width: int) -> float:
    return GATE_W_MM / float(width)


# --------------------------------------------------------------------------
# per-operator measurements
# --------------------------------------------------------------------------

def measure_spread(
    stock: str, amount_key: str, amount: float, *, color: str = "white"
) -> dict:
    """Radial characterisation of a spread operator on one isolated source.

    The source sits on a -6 EV field so the profile that comes back is the
    operator's kernel and not the neighbours' overlap.
    """
    # A 0.04 mm source: small against the 0.065-0.32 mm component radii, so
    # the profile that comes back is the KERNEL and not the source's own
    # footprint. The first version used a 1 mm disc, which was fine against a
    # 0.55 mm kernel and became the dominant term once P2 brought the halo
    # down to physical size.
    scene, (cy, cx) = charts.single_emitter(
        CHART_H, CHART_W, diameter_mm=0.04, exposure_ev=7.0,
        background_ev=-4.0, color=color,
    )
    base = develop(scene, make_plan(stock, film_media_scatter="off"))
    got = develop(
        scene,
        make_plan(stock, film_media_scatter="off", **{amount_key: amount}),
    )
    delta = diag.isolate(got, base)
    scale = mm_per_px(CHART_W)

    radii, prof, _ = diag.radial_profile(delta, (cy, cx), max_radius_px=CHART_H / 2)
    out: dict = {
        "half_energy_radius_mm": [],
        "encircled_at_declared_radius": [],
        "peak_delta": [],
    }
    # The widest component's radius: what "the declared radius" means once
    # halation is a component set rather than one kernel.
    declared_mm = max(c.radius_mm for c in HALATION.components)
    declared_px = declared_mm / scale
    for c in range(3):
        out["half_energy_radius_mm"].append(
            float(diag.half_energy_radius(radii, prof[:, c], baseline=0.0) * scale)
        )
        ee = diag.encircled_energy(radii, prof[:, c], baseline=0.0)
        j = int(np.searchsorted(radii, declared_px))
        out["encircled_at_declared_radius"].append(
            float(ee[min(j, ee.size - 1)]) if ee.size else float("nan")
        )
        out["peak_delta"].append(float(prof[:, c].max()))
    out["energy_ratio"] = [float(v) for v in diag.energy_ratio(delta, base)]
    out["delta_sum"] = [float(v) for v in delta.sum(axis=(0, 1))]
    out["base_sum"] = [float(v) for v in base.sum(axis=(0, 1))]
    # Halo colour, normalised to the strongest channel. §10.1 test 6: a BLUE
    # source must not hand back the same red halo a white one does. If this
    # vector is identical for every source colour, the operator has thrown the
    # source's spectrum away before spreading it.
    def _ring(lo_mm: float, hi_mm: float) -> list[float]:
        mm = radii * scale
        m = (mm > lo_mm) & (mm < hi_mm)
        e = (np.clip(prof[m], 0.0, None) * radii[m, None]).sum(axis=0)
        peak = float(np.max(np.abs(e)))
        return [float(v / peak) if peak > 0 else float("nan") for v in e]

    # Inner vs outer colour, which is the whole point of a component set:
    # a single kernel with a fixed weight vector returns the same hue at
    # every radius, so these two rows would be identical.
    out["halo_inner_ratio"] = _ring(0.03, 0.10)
    out["halo_outer_ratio"] = _ring(0.30, 1.00)
    out["halo_channel_ratio"] = out["halo_inner_ratio"]
    return out


def measure_grain(stock: str, amount: float) -> dict:
    """Granularity of the rendered print on a flat mid-tone patch, plus the
    intrinsic statistics of the film-space field that drives it.

    Two different questions live here. The field statistics say what KIND of
    process this is (correlation length, Selwyn slope) and are independent of
    any amount; the rendered patch says how much of it a viewer sees.
    """
    scene = charts.uniform_patch(CHART_H, CHART_W, 0.0)
    base = develop(scene, make_plan(stock, film_media_scatter="off"))
    got = develop(
        scene,
        make_plan(stock, film_grain=amount, film_media_scatter="off"),
    )
    delta = diag.isolate(got, base)
    lum = delta @ diag.LUMA_REC2020

    from dngscan.film_optics import _band_limited_field

    field = _band_limited_field(GRAIN, 0)
    pitch = GRAIN.pitch_um
    corr_cells = diag.correlation_length_cells(field)
    # The rendered patch is a print, so its "granularity" is reported in 8-bit
    # display code values — the unit a viewer can argue about. The datasheet
    # comparison lives on the density field below, where 48 um means something.
    return {
        "print_luma_rms_code8": float(np.sqrt(np.mean(lum ** 2)) * 255.0),
        "print_chroma_luma_ratio": float(diag.chroma_luma_ratio(delta)),
        "field_correlation_length_um": float(corr_cells * pitch),
        "field_blob_fwhm_um": float(2.0 * corr_cells * pitch),
        "field_selwyn_slope": float(diag.selwyn_slope(field)),
        "field_aperture_rms": {
            f"{int(round(n * pitch))}um": [float(v) for v in diag.aperture_rms(field, n)]
            for n in (1, 2, 4, 8)
        },
        # P4 (measured_sigma_v2): the field is scaled per channel exactly the
        # way the kernel scales it — chart sigma at mid chart density over the
        # field's own 48 um aperture RMS — and then measured back through the
        # 48 um aperture. The number is therefore the as-rendered datasheet
        # figure (x1000), and closing the calibration loop is the point.
        # (v1 profiles keep the historical sigma0*2 span quote.)
        "rms_granularity_48um_at_span2": _rms_granularity_quote(field, pitch),
    }


def _rms_granularity_quote(field, pitch):
    import numpy as np
    from dngscan import film_optics_diag as diag

    if GRAIN.model == "measured_sigma_v2":
        from dngscan.film_optics import _aperture_rms

        rms48 = max(_aperture_rms(GRAIN), 1e-9)
        scaled = np.array(field, dtype=np.float64, copy=True)
        for ch in range(3):
            base, dmax = GRAIN.chart_density[ch]
            tab = np.asarray(GRAIN.sigma_density[ch], dtype=np.float64)
            mid = 0.5 * (base + dmax)
            sig = float(np.interp(mid, tab[:, 0], tab[:, 1]))
            scaled[..., ch] *= sig / rms48
        return [float(v) for v in diag.rms_granularity(scaled, pitch)]
    return [
        float(v)
        for v in diag.rms_granularity(field * (GRAIN.sigma0 * 2.0), pitch)
    ]


def measure_pyramid_blockiness(stock: str, amount: float) -> dict:
    """Does the spread-map resample leave grid artefacts in the render?

    History, in full (ledger §11.1): P0 recorded real 8-px blocks from a
    nearest-neighbour pyramid expand; the runtime has since honoured the
    §11.1 contract (area-decimate + bilinear resample, no NN anywhere).
    P5b's "inverted" pass was scatter contamination (R1 item 4), and the
    R1/R4-era 4.73 "still-open defect" reading was a METRIC artefact: the
    chart (1152 px) sat under the spread cap so no resample ran at all,
    and the single edge-transition spike aliased against the 8-px modulus
    (the ratio grew monotonically with ANY probe modulus and the delta's
    autocorrelation showed no periodicity).

    The measurement now does what the gate always meant: render at a size
    where decimation genuinely engages, take many rows, EXCLUDE the edge
    transition's own neighbourhood, and read the |first-diff| ratio at the
    legacy 8-px modulus AND at the true bilinear knot pitch. A clean
    resample reads ~1.0 everywhere; NN blocks would read far above it at
    the knot pitch.
    """
    from dngscan.film_optics import spread_grid_shape

    bh, bw = 1400, 2100  # long side above every tier's spread cap
    grid = spread_grid_shape(bh, bw)
    assert grid != (bh, bw), "blockiness chart must engage decimation"
    scene = charts.edge_chart(bh, bw, tilt_deg=5.0)
    base = develop(scene, make_plan(stock, film_media_scatter="off"))
    got = develop(
        scene,
        make_plan(stock, film_bloom=amount, film_media_scatter="off"),
    )
    delta = diag.isolate(got, base)
    rows = delta[bh // 4 : 3 * bh // 4 : 7, :, 1]
    dif = np.abs(np.diff(rows, axis=1))
    edge_col = int(np.argmax(dif.mean(axis=0)))
    mask = np.ones(dif.shape[1], bool)
    mask[max(0, edge_col - 60) : edge_col + 60] = False
    idx = np.arange(dif.shape[1])
    out: dict = {"grid": list(grid), "chart": [bh, bw]}
    for factor in (2, 4, 8, 16):
        on = float(dif[:, (idx % factor == factor - 1) & mask].mean())
        off = float(dif[:, (idx % factor != factor - 1) & mask].mean())
        out[f"step_{factor}px_ratio"] = float(on / max(off, 1e-30))
    pitch = bw / grid[1]
    knots = np.round(np.arange(grid[1]) * pitch).astype(int)
    knots = knots[(knots > 0) & (knots < dif.shape[1])]
    on_k = float(dif[:, knots][:, mask[knots]].mean())
    off_mask = mask.copy()
    off_mask[knots] = False
    out["knot_pitch_px"] = float(pitch)
    out["knot_aligned_ratio"] = float(on_k / max(float(dif[:, off_mask].mean()), 1e-30))
    return out


def measure_mtf(stock: str, tier: str) -> dict:
    """Detail cutoff from the SPREAD operators, on the slanted edge.

    Grain is deliberately excluded. A slanted-edge MTF reconstructs the edge
    spread function from many sub-pixel phases; grain correlated at roughly a
    pixel survives that averaging and lands in the differentiated LSF, so a
    grain-on measurement reports the noise floor rather than any real loss of
    resolution. Grain's effect on detail belongs to the PSD and to the edge
    SNR below, not here.
    """
    scene = charts.edge_chart(CHART_H, CHART_W, tilt_deg=5.0)
    _, h, b = TIERS[tier]
    base = develop(scene, make_plan(stock))
    got = develop(scene, make_plan(stock, film_halation=h, film_bloom=b))
    out = {}
    for name, img in (("baseline", base), (f"{tier}_spread_only", got)):
        freq, mtf = diag.slanted_edge_mtf(img[:, :, 1], half_window_px=24.0)
        out[name] = {
            "mtf50_cycles_per_px": float(diag.mtf50(freq, mtf)),
            "mtf_at_0p25": float(np.interp(0.25, freq, mtf)),
        }
    # Edge SNR with grain on: the amplitude of the grain against the edge's
    # own contrast, which is the number that decides whether fine detail
    # survives the texture.
    g = TIERS[tier][0]
    grained = develop(scene, make_plan(stock, film_grain=g))
    noise = diag.isolate(grained, base) @ diag.LUMA_REC2020
    contrast = float(np.ptp(base @ diag.LUMA_REC2020))
    out["edge_snr_with_grain"] = float(
        contrast / max(float(np.sqrt(np.mean(noise ** 2))), 1e-12)
    )
    return out


def measure_mtf_residual() -> dict:
    """MTF transfer budget (review R1 item 2): measured / explicit.

    The explicit operators (§5.1 emulsion, §6.2 formation) implement fitted
    scatter mixes; the digitized charts are the measurement they were fitted
    to. The ratio MTF_measured / MTF_explicit at each chart frequency is the
    part of the medium's transfer the explicit model does NOT carry — the
    adjacency bump (chemical edge effect, >1) and the fit's own error. It is
    computed from the SHIPPED runtime assets, not the import tool's fit, so
    the number certifies what a render actually applies.
    """
    mtf_dir = ROOT / "dngscan" / "data" / "mtf"

    def _explicit(asset, ch_index: int, f: np.ndarray) -> np.ndarray:
        s = float(asset.s[ch_index])
        w = float(asset.w[ch_index])
        sg = float(asset.sigma_um[ch_index]) * 1e-3   # um -> mm
        ts = float(asset.tail_sigma_um[ch_index]) * 1e-3
        g = np.exp(-2.0 * np.pi ** 2 * sg ** 2 * f ** 2)
        # bi_gaussian_v1: the tail is a Gaussian at tail_sigma, the exact
        # form the runtime executes (R10 item 3)
        t = np.exp(-2.0 * np.pi ** 2 * ts ** 2 * f ** 2)
        return (1.0 - s) + s * ((1.0 - w) * g + w * t)

    out: dict = {}
    for name, asset in (
        ("5207_emulsion", STOCK_OPTICS.emulsion_scatter),
        ("2383_formation", PRINT_OPTICS.formation_scatter),
    ):
        if asset is None:
            out[name] = None
            continue
        chart = json.loads(
            (mtf_dir / f"mtf_{name.split('_')[0]}.json").read_text("utf-8")
        )
        rows: dict = {"source": chart["source"], "channels": {}}
        worst = 0.0
        for ch_name, ch_index in (("R", 0), ("G", 1), ("B", 2)):
            pts = np.asarray(chart["channels"][ch_name], dtype=np.float64)
            f, measured = pts[:, 0], pts[:, 1]
            explicit = _explicit(asset, ch_index, f)
            residual = measured / np.maximum(explicit, 1e-9)
            rows["channels"][ch_name] = [
                [float(a), float(b), float(c), float(d)]
                for a, b, c, d in zip(f, measured, explicit, residual)
            ]
            worst = max(worst, float(np.max(np.abs(np.log(residual)))))
        # One number for the gate: the worst |log residual| across every
        # chart point of this asset (log so 1.10 and 1/1.10 read equally).
        rows["max_abs_log_residual"] = worst
        out[name] = rows
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def build_report(stock: str = DEFAULT_STOCK, *, perf: bool = False) -> dict:
    report: dict = {
        "stock": stock,
        "chart_shape": [CHART_H, CHART_W],
        "gate_w_mm": GATE_W_MM,
        "mm_per_px": mm_per_px(CHART_W),
        "assets": {
            "stock": STOCK_OPTICS.asset_id,
            "print": PRINT_OPTICS.asset_id,
            "grain": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in vars(GRAIN).items()},
            "halation": {
                "provenance": HALATION.provenance,
                "model": HALATION.model,
                "dc_mode": HALATION.dc_mode,
                "anti_halation_class": HALATION.anti_halation_class,
                "components": [
                    {"name": c.name, "radius_mm": c.radius_mm,
                     "gate_ev": c.gate_ev.tolist(),
                     "transfer": c.transfer.tolist()}
                    for c in HALATION.components
                ],
                "total_return": HALATION.total_return().tolist(),
            },
            "formation_scatter": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in vars(FORMATION_SCATTER).items()
            } if FORMATION_SCATTER is not None else None,
        },
        "tiers": {k: list(v) for k, v in TIERS.items()},
        "grain": {},
        "halation": {},
        "bloom": {},
        "mtf": {},
    }
    for tier, (g, h, b) in TIERS.items():
        report["grain"][tier] = measure_grain(stock, g)
        report["halation"][tier] = measure_spread(stock, "film_halation", h)
        report["bloom"][tier] = measure_spread(stock, "film_bloom", b)
        report["mtf"][tier] = measure_mtf(stock, tier)
    report["mtf_residual"] = measure_mtf_residual()
    # P5e: the legacy display-threshold source gate this section measured
    # was deleted with its operator; the capture bloom reads SCENE-linear
    # exposure by construction (P3), which was the fix the P0 numbers
    # demanded.
    report["bloom"]["source_gate"] = "deleted_with_legacy_print_scatter"
    report["bloom"]["pyramid_blockiness"] = measure_pyramid_blockiness(
        stock, TIERS["standard"][2]
    )
    report["halation"]["blue_source"] = measure_spread(
        stock, "film_halation", TIERS["standard"][1], color="blue"
    )
    if perf:
        report["perf"] = measure_perf(stock)
    return report


def measure_perf(stock: str, megapixels: float = 61.0) -> dict:
    """Wall time and peak RSS of the banded production path at 61 MP.

    Driven through the renderer's own three-phase lifecycle — pass A for the
    halation maps, pass B for the bloom source, then the banded apply — not
    the full-frame oracle. Two reasons: the oracle holds the whole frame and
    reports a number no export can reproduce, and `apply_film_core` without a
    spatial context leaves the optics INERT by contract, so a naive timing
    loop measures the film curves twice and reports that the operators are
    free.

    The scene is generated per band rather than held: a 61 MP float32 RGB
    frame is 730 MiB on its own and would dominate the peak being measured.

    `ru_maxrss` is a monotone high-water mark, which is what makes
    `*_rss_growth_mib` the number §11.3 asks for: the optics run comes second,
    so its growth is exactly peak-with-optics minus peak-without.
    """
    import resource

    from dngscan.film_develop import prepare_film_spatial
    from dngscan.film_optics import area_decimate_rows, spread_grid_shape
    from dngscan.render import _optics_band_rows

    w = int(round(np.sqrt(megapixels * 1e6 * 3.0 / 2.0)))
    h = int(round(w * 2.0 / 3.0))
    band = _optics_band_rows(w)
    out: dict = {"height": h, "width": w, "band_rows": band}

    def rows_for(y0: int, y1: int) -> np.ndarray:
        return charts.uniform_patch(y1 - y0, w, 0.0).reshape(-1, 3)

    def rss_mib() -> float:
        # macOS reports maxrss in bytes, Linux in kibibytes.
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / (1 << 20) if sys.platform == "darwin" else raw / 1024.0

    for name, kw in (
        ("off", {}),
        ("standard", dict(zip(("film_grain", "film_halation", "film_bloom"),
                              TIERS["standard"]))),
    ):
        plan = make_plan(stock, **kw)
        base_rss = rss_mib()
        t0 = time.perf_counter()
        ctx = prepare_film_spatial(plan, h, w)
        if ctx is not None and (ctx.halation > 0.0 or ctx.bloom > 0.0):
            # P3: one scene pass drives both spatial operators. The separate
            # full-resolution pass B the old bloom needed — a whole
            # colorimetric + grain + halation walk just to threshold the
            # print — is gone with the operator that required it.
            dh, dw = spread_grid_shape(h, w)
            acc = np.zeros((dh, dw, 3), dtype=np.float64)
            ctx.begin_bloom_source()
            for y0 in range(0, h, band):
                y1 = min(y0 + band, h)
                rows = rows_for(y0, y1)
                ctx.accumulate_bloom_source(rows, y0, y1)
                area_decimate_rows(
                    rows.reshape(y1 - y0, w, 3), y0, h, w, dh, dw, acc
                )
            scene_dec = acc.astype(np.float32)
            del acc
            if ctx.bloom > 0.0:
                ctx.finish_bloom_map(scene_dec)
            if ctx.halation > 0.0:
                ctx.finish_maps(scene_dec, plan, stock)
            del scene_dec
        halo = ctx.scatter_halo_rows() if ctx is not None else 0
        band_eff = band * 3 if halo > 0 else band  # renderer's amortization
        for y0 in range(0, h, band_eff):
            y1 = min(y0 + band_eff, h)
            if ctx is not None:
                y0e, y1e = max(0, y0 - halo), min(h, y1 + halo)
                apply_film_core(
                    rows_for(y0e, y1e), plan,
                    spatial=(ctx, y0, y1, y0e, y1e),
                )
            else:
                apply_film_core(rows_for(y0, y1), plan, spatial=None)
        out[f"{name}_seconds"] = round(time.perf_counter() - t0, 3)
        out[f"{name}_peak_rss_mib"] = round(rss_mib(), 1)
        out[f"{name}_rss_growth_mib"] = round(rss_mib() - base_rss, 1)
        del ctx
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock", default=DEFAULT_STOCK)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--perf", action="store_true", help="add the 61 MP timing")
    args = ap.parse_args()

    report = build_report(args.stock, perf=args.perf)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
