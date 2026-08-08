# SPDX-License-Identifier: GPL-3.0-or-later
"""Film-takeover development core (film_mode="full") — the film v2
factorized chain (FILM_PRINT_RENDERING_PLAN §3/§5.4/§7.1).

Stage A runs ANALYTICALLY per pixel: optional Film Compression on the scene
(§8, before the emulsion), observer inverse -> per-layer log exposure ->
optional halation reinjection (§9.2, spatial context) -> three 1-D
characteristic curves (the film exposure state and the reversal anchor share
the layer-exposure slot; editorial developer recipes perturb the tables here,
§6) -> negative dye density -> optional density grain (§9.1).

Stage B is FACTORIZED, never one baked composite: B1 (negative density ->
log2 paper-layer exposure, 65^3, q-free) -> print timing tau (fixed = tau(0)
node; retimed = interpolated from the 0.25 EV table inside the pairing's
declared reachable span; custom = tau(0) + manual print exposure + the
modelled colour-head delta-tau, datasheet neutralization required) -> the
paper's 1-D development curves on the log2 axis -> B2 (positive-medium
density -> viewed Rec.2020, 65^3, keyed by print medium x viewing and reused
across stocks). Reversals skip B1/tau/paper into their direct B2. Medium
bloom (§9.2) applies after B2, before delivery gamut fit. Beyond SDR, the
Ultra HDR export runs this chain as "film print + scene HDR extension"
(hdr_agx.render_ultrahdr_film_pair).

Honesty labels: this is a TRISTIMULUS reconstruction CONSTRAINED by spectral
data — three numbers cannot recover a spectrum, and the observer inverse's
metamer residual is measured and stamped per stock. DIR couplers/interlayer
effects remain absent from the data and therefore from the chain. The
computational shaper domains (amount_lo/hi, dye_lo/hi) are baked WIDER than
the measured curve envelope to cover the declared editorial developer
envelope (review batch 13), so recipe-perturbed densities stay in-domain.

Grey-scale neutralization (plan.film_crossover storage; surface name
--film-neutralization): "datasheet" serves the chain verbatim; "bounded"
(off) divides the output per pixel by a BOUNDED neutral-cast curve indexed
at the pixel's LUMINANCE exposure EV_Y — the correction multiplier walks
h(t) = 1 + t*(1/cast - 1) toward full neutralization at the largest
t in [0,1] keeping every channel inside [0.25, 4], preserving neutral-axis
luminance exactly; deeper tint (Kodachrome's floor) stays medium character.
The division is at runtime, not a second baked volume: evaluated exactly per
pixel the quotient's visible-stop error equals the datasheet volume's own,
while a baked composite put the EV_Y kink diagonal to the grid (1.73 EV
worst off-axis, measured). For negatives the per-node cast tables ride the
print_state asset and interpolate in exposure state; both cast families are
solved against MEASURED development, which is why editorial recipes refuse
bounded neutralization.

Loading is fail-closed end to end: schema, kind, identity (stock/medium/
input space), shapes, monotone axes, finiteness and cast bounds all raise
instead of serving a stale or tampered asset; the published family is
additionally pinned by tests against dngscan/data/film_v2/MANIFEST.json.
"""
from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

from ._deps import np
from .color import EPS

REC2020_LUMA = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)
def _tetrahedral(lut: Any, u: Any, n: int) -> Any:
    """Vectorized tetrahedral interpolation on a cubic lattice. [N,3] -> [N,3]."""
    g = np.clip(u, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(g.astype(np.int32), n - 2)
    f = (g - i0).astype(np.float32)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    def at(dx: Any, dy: Any, dz: Any) -> Any:
        return lut[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]

    c000 = at(0, 0, 0)
    c111 = at(1, 1, 1)
    out = np.empty_like(c000)
    # Six tetrahedra of the unit cube, keyed by the ordering of (fx, fy, fz).
    orders = (
        (fx >= fy) & (fy >= fz),
        (fx >= fz) & (fz > fy),
        (fz > fx) & (fx >= fy),
        (fy > fx) & (fx >= fz),
        (fy >= fz) & (fz > fx),
        (fz > fy) & (fy > fx),
    )
    corners = (
        ((1, 0, 0), (1, 1, 0)),
        ((1, 0, 0), (1, 0, 1)),
        ((0, 0, 1), (1, 0, 1)),
        ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (0, 1, 1)),
    )
    axis_f = {"x": fx, "y": fy, "z": fz}
    weights = (
        ("x", "y", "z"),
        ("x", "z", "y"),
        ("z", "x", "y"),
        ("y", "x", "z"),
        ("y", "z", "x"),
        ("z", "y", "x"),
    )
    for mask, (cA, cB), (a1, a2, a3) in zip(orders, corners, weights):
        if not bool(np.any(mask)):
            continue
        f1, f2, f3 = axis_f[a1][mask], axis_f[a2][mask], axis_f[a3][mask]
        idx = np.nonzero(mask)[0]
        pA = lut[i0[idx, 0] + cA[0], i0[idx, 1] + cA[1], i0[idx, 2] + cA[2]]
        pB = lut[i0[idx, 0] + cB[0], i0[idx, 1] + cB[1], i0[idx, 2] + cB[2]]
        out[idx] = (
            (1.0 - f1)[:, None] * c000[idx]
            + (f1 - f2)[:, None] * pA
            + (f2 - f3)[:, None] * pB
            + f3[:, None] * c111[idx]
        )
    return out


_V2_DIR = _Path(__file__).with_name("data") / "film_v2"
_V2_CACHE: dict[str, tuple | None] = {}


def _npz(path):
    return np.load(path, allow_pickle=False)


def _check_common(z, kind: str, path) -> None:
    # schema 6 (review batch 14): the identity fields (stock/medium) became
    # part of the ABI, so pre-identity schema-5 files are refused by NUMBER
    # with a real reason instead of a KeyError swallowed into "unreadable".
    if int(z["schema"]) != 6:
        raise ValueError(
            f"{path.name}: schema {int(z['schema'])}, expected 6 — "
            "regenerate with tools/build_film_v2_assets.py"
        )
    if str(np.asarray(z["kind"])) != kind:
        raise ValueError(f"{path.name}: kind {np.asarray(z['kind'])}, expected {kind}")


def _load_v2(name: str):
    """Assemble the modular §7.1 asset family for one preset, fail closed.

    Returns (stock dict, {medium_id: (print_state dict | None, b2 dict)}).
    """
    key = str(name)
    if key in _V2_CACHE:
        return _V2_CACHE[key]
    stock_path = _V2_DIR / f"{key}.npz"
    entry = None
    try:
        with _npz(stock_path) as z:
            _check_common(z, "stock", stock_path)
            if str(np.asarray(z["stock"])) != key:
                raise ValueError(
                    f"stock asset identity {np.asarray(z['stock'])} != {key}"
                )
            observer = np.asarray(z["observer"], dtype=np.float64)
            char_le = np.asarray(z["char_le"], dtype=np.float64)
            char_amounts = np.asarray(z["char_amounts"], dtype=np.float64)
            lo = np.asarray(z["amount_lo"], dtype=np.float64)
            hi = np.asarray(z["amount_hi"], dtype=np.float64)
            if observer.shape != (3, 3) or not bool(np.isfinite(observer).all()):
                raise ValueError("observer mis-shaped")
            if char_amounts.shape != (char_le.size, 3) or char_le.size < 2:
                raise ValueError("characteristic tables mis-shaped")
            for _n, _a in (("char_le", char_le), ("char_amounts", char_amounts),
                           ("amount_lo", lo), ("amount_hi", hi)):
                if not bool(np.isfinite(_a).all()):
                    raise ValueError(f"{_n} non-finite")
            if not bool(np.all(np.diff(char_le) > 0)):
                raise ValueError("logE axis not strictly increasing")
            if not bool(np.all(hi > lo)):
                raise ValueError("amount domain degenerate")
            stock = {
                "observer": observer,
                "char_le": char_le,
                "char_amounts": char_amounts,
                "lo": lo,
                "hi": hi,
                "anchor": float(z["anchor_ev_offset"]),
                "exp_lo": float(z["exposure_ev_min"]),
                "exp_hi": float(z["exposure_ev_max"]),
                "reversal": bool(z["reversal"]),
                "default_medium": str(np.asarray(z["default_medium"])),
                "media": [str(m) for m in z["media"]],
            }
            if stock["reversal"]:
                cast_ev = np.asarray(z["cast_ev"], dtype=np.float32)
                cast = np.asarray(z["cast_bounded"], dtype=np.float32)
                if cast.shape != (cast_ev.size, 3) or cast_ev.size < 2 or \
                        not bool(np.all(np.diff(cast_ev) > 0)):
                    raise ValueError("reversal cast arrays mis-shaped")
                if not bool(np.isfinite(cast).all()) or \
                        float(cast.min()) < 0.25 - 1e-4 or float(cast.max()) > 4.0 + 1e-4:
                    raise ValueError("reversal cast outside its declared bound")
                stock["cast_ev"] = cast_ev
                stock["cast_bounded"] = cast
            if not stock["exp_hi"] > stock["exp_lo"]:
                raise ValueError("exposure domain degenerate")
        media = {}
        for medium in stock["media"]:
            b2_path = _V2_DIR / f"b2__{medium}.npz"
            with _npz(b2_path) as z:
                _check_common(z, "b2", b2_path)
                if str(np.asarray(z["medium"])) != medium:
                    raise ValueError(
                        f"{b2_path.name}: medium identity mismatch"
                    )
                if str(np.asarray(z["input_space"])) != "positive_density":
                    raise ValueError(f"{b2_path.name}: wrong input space")
                n = int(z["n"])
                vol = np.asarray(z["volume"], dtype=np.float32)
                if vol.shape != (n, n, n, 3) or n < 2:
                    raise ValueError(f"{b2_path.name}: volume mis-shaped")
                if not bool(np.isfinite(vol).all()) or float(vol.min()) < 0.0:
                    raise ValueError(f"{b2_path.name}: volume non-finite/negative")
                dye_lo = np.asarray(z["dye_lo"], dtype=np.float64)
                dye_hi = np.asarray(z["dye_hi"], dtype=np.float64)
                if not bool(np.all(dye_hi > dye_lo)):
                    raise ValueError(f"{b2_path.name}: dye domain degenerate")
                paper_le2 = np.asarray(z["paper_le2"], dtype=np.float64)
                paper_amounts = np.asarray(z["paper_amounts"], dtype=np.float64)
                if not bool(np.isfinite(paper_le2).all()) or                         not bool(np.isfinite(paper_amounts).all()):
                    raise ValueError(f"{b2_path.name}: paper tables non-finite")
                if not bool(np.isfinite(dye_lo).all()) or                         not bool(np.isfinite(dye_hi).all()):
                    raise ValueError(f"{b2_path.name}: dye domain non-finite")
                b2 = {
                    "volume": vol, "n": n,
                    "paper_le2": paper_le2,
                    "paper_amounts": paper_amounts,
                    "dye_lo": dye_lo, "dye_hi": dye_hi,
                }
            ps = None
            if not stock["reversal"]:
                ps_path = _V2_DIR / f"print__{key}__{medium}.npz"
                with _npz(ps_path) as z:
                    _check_common(z, "print_state", ps_path)
                    if str(np.asarray(z["stock"])) != key or \
                            str(np.asarray(z["medium"])) != medium:
                        raise ValueError(f"{ps_path.name}: identity mismatch")
                    if str(np.asarray(z["input_space"])) != "negative_density":
                        raise ValueError(f"{ps_path.name}: wrong input space")
                    n = int(z["n"])
                    b1 = np.asarray(z["b1_volume"], dtype=np.float32)
                    if b1.shape != (n, n, n, 3):
                        raise ValueError(f"{ps_path.name}: B1 mis-shaped")
                    if not bool(np.isfinite(b1).all()):
                        raise ValueError(f"{ps_path.name}: B1 non-finite")
                    tau_nodes = np.asarray(z["tau_nodes"], dtype=np.float64)
                    tau = np.asarray(z["tau"], dtype=np.float64)
                    if tau.shape != (tau_nodes.size, 3) or tau_nodes.size < 2 or \
                            not bool(np.all(np.diff(tau_nodes) > 0)):
                        raise ValueError(f"{ps_path.name}: tau table mis-shaped")
                    if not bool(np.isfinite(tau).all()) or \
                            not bool(np.isfinite(tau_nodes).all()):
                        raise ValueError(f"{ps_path.name}: tau non-finite")
                    _span = (float(z["retimed_ev_min"]), float(z["retimed_ev_max"]))
                    if not (np.isfinite(_span[0]) and np.isfinite(_span[1])
                            and _span[1] >= _span[0]):
                        raise ValueError(f"{ps_path.name}: retimed span invalid")
                    cast_ev = np.asarray(z["cast_ev"], dtype=np.float32)
                    casts = np.asarray(z["casts"], dtype=np.float32)
                    if casts.shape != (tau_nodes.size, cast_ev.size, 3):
                        raise ValueError(f"{ps_path.name}: casts mis-shaped")
                    if not bool(np.isfinite(casts).all()) or \
                            float(casts.min()) < 0.25 - 1e-4 or \
                            float(casts.max()) > 4.0 + 1e-4:
                        raise ValueError(f"{ps_path.name}: cast outside bound")
                    ps = {
                        "b1": b1, "n": n, "tau_nodes": tau_nodes, "tau": tau,
                        "cast_ev": cast_ev, "casts": casts,
                        "retimed_ev_min": float(z["retimed_ev_min"]),
                        "retimed_ev_max": float(z["retimed_ev_max"]),
                    }
            media[medium] = (ps, b2)
        entry = (stock, media)
    except (OSError, KeyError, ValueError) as exc:
        # Keep the underlying validation failure visible (review batch 14):
        # "missing or unreadable" hid schema and identity refusals.
        raise RuntimeError(
            f"film v2 asset family for '{key}' failed to load from "
            f"{_V2_DIR}: {type(exc).__name__}: {exc}"
        ) from exc
    if entry is None:
        raise RuntimeError(
            f"film v2 asset family for '{key}' is missing under {_V2_DIR}; "
            "regenerate with tools/build_film_v2_assets.py"
        )
    _V2_CACHE[key] = entry
    return entry


# Colour-head CC -> per-layer tau attenuation, valid ONLY inside the current
# paper-layer exposure model (§7.2, modelled): a Y filter of d = CC/100
# optical density attenuates the blue-sensitive layer's exposure by 10^-d,
# i.e. delta_tau = -d / log10(2) log2 units on that layer. Real filter
# spectra would change B1's integral density-dependently and must rebuild it.
_CC_LAYER = {"y": 2, "m": 1}
_LOG10_2 = 0.30102999566398119521


def _custom_delta_tau(color_head_y: float, color_head_m: float,
                      print_exposure_ev: float) -> Any:
    delta = np.full(3, float(print_exposure_ev), dtype=np.float64)
    delta[_CC_LAYER["y"]] -= (float(color_head_y) / 100.0) / _LOG10_2
    delta[_CC_LAYER["m"]] -= (float(color_head_m) / 100.0) / _LOG10_2
    return delta


class FilmSpatialContext:
    """Prepared per-render state for the §9 analog optics.

    Holds the decimated spread maps (halation from the pre-emulsion
    highlight exposure, bloom from the colorimetric developed image — both
    DEFINED on the decimated spread grid, see film_optics) plus the engaged
    amounts. The full-frame oracle and the renderer's sequential row-band
    path build the identical context, so band seams are exact.
    """

    __slots__ = (
        "height", "width", "profile", "grain", "halation", "bloom", "seed",
        "hal_map", "bloom_map", "geometry",
    )

    def __init__(self, height: int, width: int, plan: Any) -> None:
        from .film_optics import MODELLED_DEFAULT

        self.height = int(height)
        self.width = int(width)
        self.profile = MODELLED_DEFAULT
        self.grain = float(getattr(plan, "film_grain", 0.0) or 0.0)
        self.halation = float(getattr(plan, "film_halation", 0.0) or 0.0)
        self.bloom = float(getattr(plan, "film_bloom", 0.0) or 0.0)
        self.seed = int(getattr(plan, "film_optics_seed", 0) or 0)
        self.hal_map = None
        self.bloom_map = None
        # Orientation-aware centered gate mapping (review batch 13): portrait
        # images use the 24x36 rotated gate and non-3:2 aspects letterbox
        # inside it, so no row band ever samples outside the field.
        from .film_optics import FilmGeometry

        self.geometry = FilmGeometry.fit(self.height, self.width)

    @property
    def engaged(self) -> bool:
        return self.grain > 0.0 or self.halation > 0.0 or self.bloom > 0.0

    def band_geometry(self, y0: int, y1: int):
        return self.geometry.rows(y0, y1)

    def finish_maps(self, rgb_dec: Any, plan: Any, preset: str) -> None:
        """Build the spread maps from the decimated post-intent scene
        (linear Rec.2020, area-decimated). The bloom source is the
        COLORIMETRIC developed image of that decimated scene — the spatial
        operators themselves never enter the source definition."""
        from .film_optics import halation_spread_map, bloom_delta_map
        from .film_v2_math import film_compression_ev

        dh, dw = rgb_dec.shape[:2]
        if self.halation > 0.0:
            flat = rgb_dec.reshape(-1, 3).astype(np.float64)
            compression = float(getattr(plan, "film_compression", 0.0) or 0.0)
            if compression > 0.0:
                flat = film_compression_ev(
                    flat,
                    impact=compression,
                    knee_ev=float(getattr(plan, "film_compression_knee", 2.0) or 2.0),
                    highlight_color_density=float(
                        getattr(plan, "film_highlight_density", 0.0) or 0.0
                    ),
                )
            exposure_lin = (
                np.maximum(flat @ REC2020_LUMA, EPS) / 0.18
            ).reshape(dh, dw)
            _, _, geo_w_mm, _ = self.geometry.region()
            self.hal_map = halation_spread_map(
                exposure_lin, self.width, geo_w_mm, self.profile
            )
        if self.bloom > 0.0:
            flat_dec = np.maximum(rgb_dec.reshape(-1, 3).astype(np.float32), 0.0)
            developed = np.empty_like(flat_dec)
            # Chunked colorimetric develop of the decimated source: one shot
            # through the tetra gathers materialized ~550 MiB of transients
            # at the 2048-wide spread grid (review batch 13).
            step = 262_144
            for c0 in range(0, flat_dec.shape[0], step):
                c1 = min(c0 + step, flat_dec.shape[0])
                developed[c0:c1] = _apply_film_core_v2(
                    flat_dec[c0:c1], plan, preset, None
                )
            from .film_optics import _as_integral

            delta = bloom_delta_map(developed.reshape(dh, dw, 3), self.profile)
            # store the delta's integral image (float32) — the band applier
            # takes exact fractional-footprint area means from it
            self.bloom_map = _as_integral(delta).astype(np.float32)


def prepare_film_spatial(plan: Any, height: int, width: int) -> "FilmSpatialContext | None":
    """Renderer entry: a context when the plan engages any optics amount,
    else None (chunk-stream fast path). Call finish_maps with the decimated
    scene before applying bands."""
    ctx = FilmSpatialContext(height, width, plan)
    return ctx if ctx.engaged else None


def _apply_film_core_v2(
    rgb: Any, plan: Any, preset: str, spatial: tuple | None = None
) -> Any:
    from .film_v2_math import (
        amounts_to_unit,
        characteristic_amounts,
        developer_perturbation,
        film_compression_ev,
        layer_log_exposure,
    )

    stock, media = _load_v2(preset)
    # P4 §8: Film Compression happens BEFORE the emulsion — the film (and the
    # neutralization casts, which key on the emulsion's input luminance) sees
    # the compressed scene. impact 0 keeps the strict identity fast path.
    compression = float(getattr(plan, "film_compression", 0.0) or 0.0)
    if compression > 0.0:
        rgb = film_compression_ev(
            rgb,
            impact=compression,
            knee_ev=float(getattr(plan, "film_compression_knee", 2.0) or 2.0),
            highlight_color_density=float(
                getattr(plan, "film_highlight_density", 0.0) or 0.0
            ),
        ).astype(np.float32, copy=False)
    # P5 (§9): analog optics run through a prepared spatial context. spatial
    # is (ctx, y0, y1): this call's rows within the full image. Flat
    # colorimetric callers (probes, decimated map sources, tests) pass None
    # and see the amounts as inert by contract.
    ctx = y0 = y1 = None
    if spatial is not None:
        ctx, y0, y1 = spatial
        if ctx is None or not ctx.engaged:
            ctx = None
    exposure_ev = float(getattr(plan, "film_exposure_ev", 0.0) or 0.0)
    timing = str(getattr(plan, "film_print_timing", "fixed") or "fixed")
    medium = str(getattr(plan, "film_print_medium", "") or "") or stock["default_medium"]
    if medium not in media:
        raise ValueError(
            f"'{preset}' 未烘焙印相介质 '{medium}'（可用：{'/'.join(stock['media'])}）"
        )
    if not stock["exp_lo"] <= exposure_ev <= stock["exp_hi"]:
        raise ValueError(
            f"film_exposure_ev={exposure_ev} 超出 '{preset}' 资产声明域 "
            f"[{stock['exp_lo']}, {stock['exp_hi']}]"
        )
    ps, b2 = media[medium]
    char_amounts = stock["char_amounts"]
    if str(getattr(plan, "film_development", "measured_default")) == "editorial_custom":
        # P4 §6: the recipe perturbs the analytic characteristic tables — no
        # volume rebuild, because B1/B2 live in the (q-free) density domain.
        # validate_film_plans already refused retimed timing and bounded
        # neutralization (both are solved against measured development).
        char_amounts = developer_perturbation(
            stock["char_le"], char_amounts,
            contrast_delta=float(getattr(plan, "film_dev_contrast", 0.0) or 0.0),
            fog_delta=float(getattr(plan, "film_dev_fog", 0.0) or 0.0),
            color_density=float(getattr(plan, "film_dev_density", 0.0) or 0.0),
        )
    log_e = layer_log_exposure(rgb, stock["observer"])
    if ctx is not None and ctx.halation > 0.0:
        from .film_optics import halation_reinject_rows

        # §9.2: source is the PRE-emulsion highlight scene exposure (spread
        # map prepared on the decimated grid), reinjected red-heavy into
        # layer exposure before the curves.
        log_e = halation_reinject_rows(
            log_e, ctx.hal_map, y0, y1, ctx.height, ctx.width,
            ctx.profile, ctx.halation,
        )
    amounts = characteristic_amounts(
        log_e, stock["char_le"], char_amounts,
        ev_offset=exposure_ev + float(stock["anchor"]),
    )
    if ctx is not None and ctx.grain > 0.0:
        from .film_optics import apply_density_grain

        # §9.1: grain modulates DENSITY before printing. The span is the
        # branch's declared dye domain (negative: stock cube; reversal: the
        # direct B2 cube), so sigma peaks at mid density in that branch.
        _g_lo, _g_hi = (
            (stock["lo"], stock["hi"]) if not stock["reversal"]
            else (media[medium][1]["dye_lo"], media[medium][1]["dye_hi"])
        )
        amounts = apply_density_grain(
            amounts, _g_lo, _g_hi, ctx.band_geometry(y0, y1),
            ctx.profile, ctx.grain, ctx.seed,
        )
    bounded = str(getattr(plan, "film_crossover", "off")) != "datasheet"
    if stock["reversal"]:
        if timing != "fixed":
            raise ValueError("reversal_direct 无印相环节：timing 只能是 fixed")
        u = amounts_to_unit(amounts, b2["dye_lo"], b2["dye_hi"])
        developed = _tetrahedral(b2["volume"], u.astype(np.float32), b2["n"])
        if bounded:
            ev_y = np.log2(np.maximum(rgb @ REC2020_LUMA, EPS) / np.float32(0.18))
            for c in range(3):
                developed[:, c] /= np.interp(
                    ev_y, stock["cast_ev"], stock["cast_bounded"][:, c]
                )
        developed = np.maximum(developed, 0.0)
        if ctx is not None and ctx.bloom > 0.0:
            from .film_optics import bloom_apply_rows

            developed = bloom_apply_rows(
                developed, ctx.bloom_map, y0, y1, ctx.height, ctx.width,
                ctx.profile, ctx.bloom,
            )
        return developed.astype(np.float32, copy=False)
    # Negative: B1 -> +tau -> paper development -> B2 (ratified §5.4).
    u1 = amounts_to_unit(amounts, stock["lo"], stock["hi"])
    lep2 = _tetrahedral(ps["b1"], u1.astype(np.float32), ps["n"]).astype(np.float64)
    if timing == "retimed":
        if not ps["retimed_ev_min"] <= exposure_ev <= ps["retimed_ev_max"]:
            # The pairing's paper rails bound the re-timable span; beyond it
            # the mid-grey target is physically unreachable — declared, not
            # fabricated (§5.3 hard-reject convention).
            raise ValueError(
                f"'{preset}' × '{medium}' 的 retimed 可达域为 "
                f"[{ps['retimed_ev_min']:+.2f}, {ps['retimed_ev_max']:+.2f}] EV"
                f"（相纸轨道的物理界）；exposure={exposure_ev:+.2f} 超出。"
                "可改用 fixed timing 或缩小胶片曝光"
            )
        tau = np.array([
            np.interp(exposure_ev, ps["tau_nodes"], ps["tau"][:, c]) for c in range(3)
        ])
        cast_key_ev = exposure_ev
    elif timing == "custom":
        tau = ps["tau"][int(np.argmin(np.abs(ps["tau_nodes"])))].copy()
        tau = tau + _custom_delta_tau(
            float(getattr(plan, "color_head_y", 0.0) or 0.0),
            float(getattr(plan, "color_head_m", 0.0) or 0.0),
            float(getattr(plan, "film_print_exposure_ev", 0.0) or 0.0),
        )
        cast_key_ev = 0.0
    else:
        tau = ps["tau"][int(np.argmin(np.abs(ps["tau_nodes"])))]
        cast_key_ev = 0.0
    dye = np.stack([
        np.interp(lep2[:, c] + tau[c], b2["paper_le2"], b2["paper_amounts"][:, c])
        for c in range(3)
    ], axis=1)
    u2 = amounts_to_unit(dye, b2["dye_lo"], b2["dye_hi"])
    developed = _tetrahedral(b2["volume"], u2.astype(np.float32), b2["n"])
    if bounded:
        if timing == "custom":
            raise ValueError(
                "custom timing 与有界灰阶中性化互斥：手动印相的意义是保留"
                "印出的样子；请配 --film-neutralization datasheet"
            )
        ev_y = np.log2(np.maximum(rgb @ REC2020_LUMA, EPS) / np.float32(0.18))
        nodes = ps["tau_nodes"]
        i_hi = int(np.searchsorted(nodes, cast_key_ev, side="left").clip(1, nodes.size - 1))
        i_lo = i_hi - 1
        t = (cast_key_ev - nodes[i_lo]) / (nodes[i_hi] - nodes[i_lo])
        cast_e = (1.0 - t) * ps["casts"][i_lo] + t * ps["casts"][i_hi]
        for c in range(3):
            developed[:, c] /= np.interp(ev_y, ps["cast_ev"], cast_e[:, c])
    developed = np.maximum(developed, 0.0)
    if ctx is not None and ctx.bloom > 0.0:
        from .film_optics import bloom_apply_rows

        # §9.2: the positive medium's intrinsic scatter, after print
        # formation, before delivery gamut fit downstream.
        developed = bloom_apply_rows(
            developed, ctx.bloom_map, y0, y1, ctx.height, ctx.width,
            ctx.profile, ctx.bloom,
        )
    return developed.astype(np.float32, copy=False)


def apply_film_core(
    rgb_rec2020: Any,
    plan: Any,
    spatial_shape: tuple | None = None,
    spatial: tuple | None = None,
) -> Any:
    """Film-takeover development. [N,3] -> [N,3]; two-stage v2 by default.

    spatial_shape=(h, w) declares the flat array is the FULL image in
    row-major order — the full-frame oracle path: it builds the same
    FilmSpatialContext the renderer's row-band path builds (decimated
    spread maps, film-space grain), then applies it as one band. spatial=
    (ctx, y0, y1) is the renderer's banded form. Without either, the plan's
    optics amounts are inert by contract (probes, map sources, tests).
    """
    preset = str(getattr(plan, "curve_preset", "") or "")
    rgb = np.maximum(np.asarray(rgb_rec2020, dtype=np.float32), 0.0)
    if spatial is None and spatial_shape is not None:
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        ctx = prepare_film_spatial(plan, h, w)
        if ctx is not None:
            from .film_optics import area_decimate, spread_grid_shape

            if ctx.halation > 0.0 or ctx.bloom > 0.0:
                dh, dw = spread_grid_shape(h, w)
                ctx.finish_maps(
                    area_decimate(rgb.reshape(h, w, 3), dh, dw),
                    plan, preset,
                )
            spatial = (ctx, 0, h)
    return _apply_film_core_v2(rgb, plan, preset, spatial)


def film_reference_white_ev(plan: Any) -> float:
    """The scene EV where this plan's print reaches its paper-white plateau
    (90% of the TRUE plateau level) on a neutral ramp — the P6 HDR
    extension's join point: below it the SDR print IS the HDR body.

    The plateau is anchored at a scene EV far beyond every characteristic
    domain shifted by any legal film exposure / timing / manual print state
    (+24 EV; the curves clamp at their table ends, so this IS the chain's
    asymptotic upper bound — review batch 14: the old fixed 0..+6 scan
    declared +6 EV "the plateau" while e.g. Portra 400 at film exposure -2
    with a +2 custom print was still climbing at slope ~1.3 there). The
    search runs over [-6, +14] EV, so heavily re-timed states whose paper
    white arrives below scene mid-grey are found too; if 90% of the true
    plateau is not reached by +14 EV the join is pinned there — the gain
    field then never engages inside any real scene and the HDR export
    refuses honestly for lack of content."""
    evs = np.linspace(-6.0, 14.0, 401)
    probe = np.concatenate([evs, [24.0]])
    rgb = (0.18 * np.exp2(probe))[:, None].repeat(3, axis=1).astype(np.float32)
    out = apply_film_core(rgb, plan)
    luma = np.asarray(out, dtype=np.float64) @ REC2020_LUMA
    plateau = float(luma[-1])
    if plateau <= 0.0:
        return 2.0
    reached = np.nonzero(luma[:-1] >= 0.9 * plateau)[0]
    return float(evs[reached[0]]) if reached.size else float(evs[-1])
