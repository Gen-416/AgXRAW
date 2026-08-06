#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit AgX curve parameters to a film stock's published characteristic curves.

This is the formation leg of the film-observation contract
(docs/FILM_OBSERVATION_PLAN.zh-CN.md §4): AgX is not replaced — a film preset is a
*named coordinate* in AgX's existing parameter space, solved by least squares against
the end-to-end (negative + paired print) neutral response derived from spektrafilm's
datasheet-processed profiles. Every preset records its source and fit residual; the
runtime only ever consumes the fitted AgX parameters through the same compiled C1
machinery every render already uses.

End-to-end target construction (v2 spectral base; tools/spectral_base.py owns
the physical conventions):
    D_neg(lambda,e) = sum_dye amount_dye(e) * dyeSpec(lambda) + base(lambda)
    E_paper_c(e)    = trapz S_paper_c(lambda) * I_THKG3(lambda) * 10^-D_neg
                      (TH-KG3 = 3400K blackbody x Schott KG3, native 380-780/5nm)
    dye_paper_c(e)  = paper_curve_c(log10 E_paper_c + q_c)
    XYZ(e)          = relative colorimetry vs the medium white on the medium-grid
                      x CMF-support intersection; then flare in XYZ, the
                      luminance-only surround term, Bradford CAT medium-white ->
                      D65, and the Rec.2020 matrix (scene-linear D65 basis)
The three effective exposures q_c = k_c + t are solved DIRECTLY in the final
viewing domain — F(q) = log(RGB_view(EV0;q)/0.18) = 0 — so print finishing is
verification, not correction (no post-hoc anchor shift, no channel gains; the
recorded decomposition t = mean(q), k = q - t carries sum(k) = 0 exactly).
Reversals have no printer lights and keep their own declared anchor. Scene
EV = logE / log10(2).

Offline tool: writes dngscan/film_curve_presets.json entries and a comparison plot.
No scipy; a compact Nelder-Mead is included.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dngscan.drt import apply_c1_endpoints, curve_params_from_plan  # noqa: E402

PROFILE_DIR = PROJECT_ROOT / "dngscan_assets" / "spectral" / "spektrafilm"
PRESET_PATH = PROJECT_ROOT / "dngscan" / "film_curve_presets.json"
LOG10_2 = np.log10(2.0)

_KEY_OVERRIDES = {"fujifilm_xtra_400": "superia400"}  # the name people remember
_LABEL_OVERRIDES = {"fujifilm_xtra_400": "Fujifilm Superia X-TRA 400"}


def _short_key(profile_key: str) -> str:
    """Stable preset key from a profile name: vendor prefix dropped, joined."""
    if profile_key in _KEY_OVERRIDES:
        return _KEY_OVERRIDES[profile_key]
    trimmed = profile_key
    for prefix in ("kodak_", "fujifilm_"):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):]
            break
    return trimmed.replace("_", "")


def _default_wb(profile_key: str, info: dict) -> str:
    """Combo WB declaration from the stock's balance: tungsten cine stocks are
    calibrated at 3200K (that is what the T suffix means), everything else here is
    daylight film at 5500K."""
    if profile_key.endswith("t") and profile_key.split("_")[-1][:-1].isdigit():
        return "3200k"
    return "5500k"


def discover_stocks() -> dict[str, dict]:
    """Every filming-stage profile in the data directory, negatives and reversals.

    Data-driven on purpose: adding a stock is dropping its (CC BY-SA) profile into
    dngscan_assets/spectral/spektrafilm/ and re-running this tool. Negatives carry
    their declared target print; positives (slides) are their own display medium.
    """
    stocks: dict[str, dict] = {}
    for path in sorted(PROFILE_DIR.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        info = profile.get("info", {})
        if str(info.get("stage")) != "filming":
            continue
        profile_key = path.stem
        positive = str(info.get("type")) == "positive"
        target_print = info.get("target_print")
        if not positive and not target_print:
            continue
        name = _LABEL_OVERRIDES.get(profile_key, str(info.get("name", profile_key)))
        stocks[_short_key(profile_key)] = {
            "label": name if positive else f"{name}（负片+相纸）",
            "negative": profile_key,
            "print": None if positive else str(target_print),
            "positive": positive,
            "wb": _default_wb(profile_key, info),
        }
    return stocks


STOCKS = discover_stocks()

# Viewing-condition-complete translation (docs/FILM_OBSERVATION_PLAN §4): a medium's
# response is a report read in its native surround, and carrying the report to the
# delivery condition applies the classic surround term between the two conditions —
# and only that term. Perceived contrast drops as the surround darkens
# (Bartleson-Breneman equations; Fairchild 2013, Colour Appearance Models 3rd ed.),
# so media built for dark-surround viewing are ~1.5x contrastier than an
# average-surround rendering of the same scene (projected slides, theatrical prints)
# and dim-surround media ~1.2x (broadcast television convention;
# Giorgianni-Madden tradition). Delivery is declared average surround — the typical
# reading condition of a photograph; sRGB's stricter dim reference (IEC 61966-2-1,
# 64 lux) would add a <=1.2x term that is recorded as a known approximation, not
# modelled. All other colour-appearance phenomena (Hunt, Stevens, ...) need absolute
# luminance, which neither the print nor sRGB pins down: declared out of scope.
SURROUND_GAMMA = {"dark": 1.5, "dim": 1.2, "average": 1.0}
TARGET_SURROUND = "average"

# Native viewing surround of each *display medium* in the chain. Positives project in
# a dark room. These print stocks are theatrical projection films — the cine chain's
# display medium shares the slide's surround, which is why both take the same term.
# Reflection papers are read in bright/average light: their identity term is a
# coincidence of conditions, derived, not assumed.
PRINT_SURROUND = {"kodak_2383": "dark", "kodak_2393": "dark"}


def surround_exponent(native_surround: str) -> float:
    """Exponent translating a medium's transmittance to the delivery surround.

    T_delivery = T_native ** (gamma_target / gamma_native): a dark-surround medium
    (gamma 1.5) flattens by T^(1/1.5) on the average-surround delivery; a medium whose
    native condition matches the delivery passes through exactly.
    """
    return SURROUND_GAMMA[TARGET_SURROUND] / SURROUND_GAMMA[native_surround]


FIT_EV_LO, FIT_EV_HI = -6.5, 6.0
TARGET_POINTS_STORED = 192


def _load_curves(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.load(open(PROFILE_DIR / f"{name}.json"))["data"]
    log_e = np.asarray(data["log_exposure"], dtype=np.float64)
    density = np.asarray(data["density_curves"], dtype=np.float64)
    keep = np.all(np.isfinite(density), axis=1)
    return log_e[keep], density[keep]


# --- Spectral contact print -------------------------------------------------
#
# The channel shortcut this replaces treated the profile's density_curves as
# Status-M channel readings and printed them directly. Upstream semantics
# (agx-emulsion emulsion.py, verified numerically against each profile's
# midscale_neutral_density to rms 0.012-0.025) are richer: density_curves are
# PER-DYE amounts, and the spectral density of the developed stack is
#     D(lambda, e) = sum_dye amount_dye(e) * dye_spectrum(lambda) + base(lambda).
# Printing therefore becomes fully spectral: the paper sees the negative's true
# transmittance through its own spectral sensitivity under the enlarger lamp —
# retiring the "Status M as printing density" simplification whose cost the
# 2383 cross-check measured (wavelength-monotone scale ladder, blue worst).
# Fuji stocks with strong masking couplers were hit hardest: Superia printed
# ~40-60% contrastier than a real optical print, which is exactly the harshness
# the samples showed.
#
# Declared constants of this model:
#   * enlarger lamp: TH-KG3 — 3400 K blackbody through a Schott KG3 heat
#     filter (spectral_base.th_kg3_spd, the upstream spektrafilm convention;
#     an earlier 3200 K bare-blackbody stand-in is retired);
#   * viewing: the print profile's own viewing_illuminant (D50 for papers),
#     CIE 1931 2-degree observer, relative colorimetry against the medium's
#     clear/white point (per-channel normalization = von Kries to that white);
#   * display encoding of the target: linear Rec.709/sRGB primaries, so the
#     scalar target remains the CIE Y of the displayed print (the convention
#     the runtime consumes).



def _spd_tools():
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from calibrate_skin_matrix import blackbody_spd, cie_1931_cmf, illuminant_spd

    return blackbody_spd, cie_1931_cmf, illuminant_spd


def _spectral_base():
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import spectral_base

    return spectral_base


# Rec.2020 luminance weights (BT.2020 Y row) — the calibration basis luma.
_LUMA_2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)


def _load_spectral(name: str) -> dict[str, np.ndarray]:
    raw = json.load(open(PROFILE_DIR / f"{name}.json"))
    data = raw["data"]

    def arr(key):
        return np.array(
            [[np.nan if v is None else float(v) for v in row]
             if isinstance(row, list) else (np.nan if row is None else float(row))
             for row in data[key]], dtype=np.float64)

    def edge_hold(a: np.ndarray) -> np.ndarray:
        # Out-of-range convention for absorptive quantities (declared in
        # spectral_base's module docstring since stage 1, enforced here since
        # the Velvia-floor investigation): dye and base densities HOLD their
        # edge value outside the tabulated support. Zero-filling made the
        # developed stack TRANSPARENT out of band — Kodachrome's dyes are only
        # tabulated 425-695 nm, so its Dmax "black" leaked ~70% of the light
        # below 425 and above 695 nm into the CMF's blue and red lobes and
        # rendered as violet (measured blue channel 0.38 display-linear where
        # the in-band stopband sits at 1e-4; no viewing flare can mask a leak
        # 50x its own size). Sensitivities keep the ZERO convention — no
        # tabulated response means no response.
        out = np.atleast_2d(a.copy().T).T
        for c in range(out.shape[1]):
            col = out[:, c]
            fin = np.flatnonzero(np.isfinite(col))
            if fin.size == 0:
                col[:] = 0.0
                continue
            col[: fin[0]] = col[fin[0]]
            col[fin[-1] + 1 :] = col[fin[-1]]
            inner = ~np.isfinite(col)
            if np.any(inner):
                col[inner] = np.interp(
                    np.flatnonzero(inner), np.flatnonzero(~inner),
                    col[~inner],
                )
        return out.reshape(a.shape)

    wl = arr("wavelengths")
    dye = edge_hold(arr("channel_density"))                       # [81,3] C/M/Y
    base = edge_hold(arr("base_density"))                         # [81]
    le = np.asarray(data["log_exposure"], dtype=np.float64)
    amounts = np.asarray(data["density_curves"], dtype=np.float64)  # [n,3] per dye
    keep = np.all(np.isfinite(amounts), axis=1)
    sens = None
    if "log_sensitivity" in data:
        sens = np.power(10.0, np.nan_to_num(arr("log_sensitivity"), nan=-10.0))
    viewing = str(raw.get("info", {}).get("viewing_illuminant", "D50"))
    return {"wl": wl, "dye": dye, "base": base, "le": le[keep],
            "amounts": amounts[keep], "sens": sens, "viewing": viewing}


_VIEWING_CACHE: dict[tuple[bytes, str], tuple[np.ndarray, np.ndarray]] = {}


def _viewing_xyz(wl: np.ndarray, viewing: str):
    """(cmf [81,3], viewing SPD [81]) on the profile's wavelength grid. Cached —
    the printer-light solve evaluates the development hundreds of times."""
    key = (wl.tobytes(), viewing)
    hit = _VIEWING_CACHE.get(key)
    if hit is not None:
        return hit
    blackbody_spd, cie_1931_cmf, illuminant_spd = _spd_tools()
    import calibrate_skin_matrix as csm

    csm.WL = wl  # the helpers evaluate on their module grid; align it
    cmf = cie_1931_cmf(wl)
    # No silent stand-in: an unknown viewing illuminant is a data error, not a
    # 5000K shrug (review finding: fits depended on the environment's luck).
    spd = illuminant_spd(viewing, wl)
    result = (np.asarray(cmf, dtype=np.float64), np.asarray(spd, dtype=np.float64))
    _VIEWING_CACHE[key] = result
    return result


def _display_rec2020(reflect: np.ndarray, white: np.ndarray, wl: np.ndarray,
                     viewing: str, flare: float, surround_exp: float) -> np.ndarray:
    """Medium spectra -> scene-linear Rec.2020(D65) via the declared translation.

    Colorimetry runs on the intersection of the medium grid with the CMF support
    (trapezoidal), relative to the medium white; then flare in XYZ, the
    luminance-only surround term, Bradford CAT medium-white -> D65 and the
    Rec.2020 matrix — so printer lights never eat the D50->D65 difference. [n,3]
    """
    sb = _spectral_base()
    view_wl = sb.intersect_grid(wl)
    keep = np.isin(wl, view_wl)
    cmf, spd = _viewing_xyz(view_wl, viewing)
    weight = spd[:, None] * cmf                              # [m,3]
    xyz = sb.trapezoid(
        reflect[:, keep, None] * weight[None, :, :], view_wl, axis=1
    )
    xyz_w = sb.trapezoid(white[keep, None] * weight, view_wl, axis=0)
    y_w = max(float(xyz_w[1]), 1e-12)
    return sb.viewing_translation_rec2020(
        xyz / y_w, xyz_w / y_w, flare, surround_exp
    )


def _stack_reflectance(spec: dict, amounts: np.ndarray) -> np.ndarray:
    """10^-D for dye-amount rows through this medium's dye spectra + base. [n,81]"""
    dens = amounts @ spec["dye"].T + spec["base"][None, :]
    return np.power(10.0, -dens)


def _regrid(spec: dict, wl: np.ndarray) -> dict:
    """Resample a medium's spectral fields onto another wavelength grid.

    Profiles ship on different grids (film 380-780/5nm, papers coarser); the print
    integral needs one grid. Linear interpolation. Out-of-range semantics per the
    spectral-base declaration: dye and base densities HOLD their edge value (a dye
    stack does not clear at the table boundary), while sensitivities fill ZERO
    (no measured response is no response)."""
    if spec["wl"].shape == wl.shape and np.allclose(spec["wl"], wl):
        return spec
    out = dict(spec)
    out["wl"] = wl
    out["dye"] = np.stack(
        [np.interp(wl, spec["wl"], spec["dye"][:, k]) for k in range(3)], axis=1
    )
    out["base"] = np.interp(wl, spec["wl"], spec["base"])
    if spec["sens"] is not None:
        out["sens"] = np.stack(
            [
                np.interp(wl, spec["wl"], spec["sens"][:, k], left=0.0, right=0.0)
                for k in range(3)
            ],
            axis=1,
        )
    return out



# DISPLAY-ENVIRONMENT viewing flare — NOT part of film calibration.
#
# Three things used to be one: the negative's response, the paper/slide
# medium's physical response, and the viewing environment. The calibration
# chain now describes THE MEDIUM ONLY: paper black is the paper's true Dmax
# and slide black is the slide's Dmax through the declared appearance
# translation — with ZERO viewing flare. Baking IEC 61966-2-1's 1% reference
# flare (papers) or a 0.5% projection flare (slides) into the fitted targets
# lifted every film black into a haze that read as film character (Velvia's
# floor sat at 0.0307 linear, only 2.55 EV below mid-grey; Portra's at
# ~0.015): a display-room property masquerading as a medium property.
#
# If viewing-environment simulation returns, it belongs in the delivery /
# view-simulation layer: default OFF, SDR and HDR defined separately, SDR
# flare relative to reference white (never to file peak), and never written
# into the physical calibration these constants used to contaminate. The
# named values are kept for that future layer:
DISPLAY_VIEWING_FLARE = 0.01       # IEC 61966-2-1 sRGB reference room
DISPLAY_PROJECTION_FLARE = 0.005   # ISO 3664-class projection / lightbox

# What REMAINS in calibration is the appearance TRANSLATION (surround
# exponents): a dark-surround medium reported for an average-surround
# delivery is a translation of the medium's own appearance, not a room
# simulation — the untranslated report stays available as the *_theatrical
# quotation variants. (Bartleson-Breneman constants; contract §8.)

def _build_reversal_target(
    stock: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Slide film, spectrally: the developed dye stack viewed as its own display.

    Reversal has NO printing stage, so there are no printer lights to solve; its
    exposure anchor and mid-neutrality are their own declared normalization: a
    global exposure shift places viewed Y=0.18 at scene EV 0 (light-meter
    semantics) and per-channel mid gains pin exact neutrality (anti-hidden-WB:
    the exposure-DEPENDENT differential is what the neutral field carries).
    Colorimetry runs on the slide's native grid through the appearance
    translation only (dark-surround exponent, CAT to D65, Rec.2020) — ZERO
    viewing flare: slide black is the medium's Dmax through the translation,
    not a projection room's veiling glare (see DISPLAY_PROJECTION_FLARE)."""
    neg = _load_spectral(stock["negative"])
    exp = surround_exponent("dark")
    reflect = _stack_reflectance(neg, neg["amounts"])
    # White is the medium's own brightest state (Dmin INCLUDING residual dye), not
    # the bare base: slides never clear completely, and normalizing against the
    # bare base capped relative luminance at ~0.46 instead of ~1.
    white = _stack_reflectance(neg, np.nanmin(neg["amounts"], axis=0)[None, :])[0]
    flare = 0.0
    rgb = _display_rec2020(reflect, white, neg["wl"], neg["viewing"], flare, exp)
    deep = _stack_reflectance(neg, np.nanmax(neg["amounts"], axis=0)[None, :])
    floor_rgb = _display_rec2020(deep, white, neg["wl"], neg["viewing"], flare, exp)[0]
    ev = neg["le"] / LOG10_2
    y = rgb @ _LUMA_2020
    order = np.argsort(y)
    e0 = float(np.interp(0.18, y[order], ev[order]))
    if not np.isfinite(e0):
        raise RuntimeError(f"{stock['negative']}: no exposure reaches mid-gray")
    ev = ev - e0
    mid = np.array([float(np.interp(0.0, ev, rgb[:, c])) for c in range(3)])
    if np.any(mid <= 1e-6):
        raise RuntimeError(f"{stock['negative']}: degenerate mid-scale {mid}")
    gain = 0.18 / mid
    channels = np.maximum(rgb * gain[None, :], 1e-7)
    floor = float(np.maximum(floor_rgb * gain, 1e-7) @ _LUMA_2020)
    return ev, channels, channels @ _LUMA_2020, floor


def _finish_print(chain) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Print-chain finishing is verification, not correction: the q-solve already
    guarantees a neutral 0.18 at EV 0 in the final viewing domain, so there is no
    anchor shift and no per-channel gain left to apply."""
    channels = np.maximum(chain.develop(chain.q), 1e-7)
    t_neutral = channels @ _LUMA_2020
    t0 = float(np.interp(0.0, chain.ev, t_neutral))
    if abs(t0 - 0.18) > 5e-4:
        raise RuntimeError(f"{chain.label}: mid-gray anchor drifted: T(0)={t0:.5f}")
    floor = float(np.maximum(chain.floor_rgb, 1e-7) @ _LUMA_2020)
    return chain.ev, channels, t_neutral, floor


def _solved_print_chain(stock: dict, surround_override: str | None = None) -> SimpleNamespace:
    """The negative+paper printing chain with its printer lights solved.

    Everything build_endtoend_target needs before the finishing translation, exposed
    as one namespace so derived constructions (the colour-head field) can re-develop
    the SAME chain with perturbed printer lights instead of duplicating the model.
    """
    neg = _load_spectral(stock["negative"])
    # The printing integral runs on the negative's NATIVE grid (380-780/5nm);
    # the paper resamples onto it under the declared out-of-range semantics.
    wl = neg["wl"]
    paper = _regrid(_load_spectral(stock["print"]), wl)
    if paper["sens"] is None:
        raise RuntimeError(f"{stock['print']}: print profile has no log_sensitivity")
    # Surround term for the chain's display medium: theatrical projection prints share
    # the slide's dark surround; reflection papers already match the delivery
    # condition (exponent 1, exactly).
    if surround_override == "native":
        exp = 1.0
    else:
        exp = surround_exponent(PRINT_SURROUND.get(stock["print"], "average"))
    # Medium-native calibration: zero viewing flare in every print chain.
    # (The average-surround exponent is 1.0, so reflection papers carry no
    # appearance translation either — the chain below is the paper itself.)
    flare = 0.0

    sb = _spectral_base()
    enlarger = sb.th_kg3_spd(wl)
    # Paper-channel printing exposure through the negative's true spectral
    # transmittance: E_c(e) = trapz L_THKG3(l) T_neg(l,e) S_paper_c(l) dl.
    t_neg = _stack_reflectance(neg, neg["amounts"])                # [n,81]
    weight = paper["sens"] * enlarger[:, None]                     # [81,3]
    log_ep = np.log10(
        np.maximum(sb.trapezoid(t_neg[:, :, None] * weight[None, :, :], wl, axis=1), 1e-12)
    )                                                              # [n,3]

    white_amounts = np.nanmin(paper["amounts"], axis=0)[None, :]
    white = _stack_reflectance(paper, white_amounts)[0]
    dmax_amounts = np.nanmax(paper["amounts"], axis=0)[None, :]
    ev = neg["le"] / LOG10_2

    def develop(q: np.ndarray) -> np.ndarray:
        """Viewed Rec.2020 of the print developed at effective exposures q."""
        dye = np.stack([
            np.interp(log_ep[:, c] + q[c], paper["le"], paper["amounts"][:, c])
            for c in range(3)
        ], axis=1)
        reflect = _stack_reflectance(paper, dye)
        return _display_rec2020(reflect, white, wl, paper["viewing"], flare, exp)

    def mid_rgb(q: np.ndarray) -> np.ndarray:
        rgb0 = develop(q)
        return np.array([float(np.interp(0.0, ev, rgb0[:, c])) for c in range(3)])

    # Effective exposures q_c = k_c + t solved directly: the model only ever sees
    # the sum, so solving three lights PLUS a time has an exact null space. Newton
    # on F(q) = log(mid(q)/0.18); the Jacobian is well conditioned (measured worst
    # ~2.14 across the negative chains).
    q = np.array([
        float(np.interp(
            0.5 * (paper["amounts"][:, c].min() + paper["amounts"][:, c].max()),
            paper["amounts"][:, c], paper["le"],
        )) - float(np.interp(0.0, ev, log_ep[:, c]))
        for c in range(3)
    ])
    for _ in range(30):
        f = np.log(np.maximum(mid_rgb(q), 1e-9) / 0.18)
        if float(np.max(np.abs(f))) < 1e-11:
            break
        jac = np.empty((3, 3), dtype=np.float64)
        h = 1e-5
        for c in range(3):
            dq = q.copy()
            dq[c] += h
            jac[:, c] = (np.log(np.maximum(mid_rgb(dq), 1e-9) / 0.18) - f) / h
        q = q - np.linalg.solve(jac, f)
    residual = float(np.max(np.abs(np.log(np.maximum(mid_rgb(q), 1e-9) / 0.18))))
    if residual > 1e-8:
        raise RuntimeError(
            f"{stock['negative']}+{stock['print']}: printer solve residual {residual:.2e}"
        )
    # Recorded decomposition (reporting convention, not solver variables):
    # exposure time t = mean(q), printer lights k = q - t with sum(k) = 0.
    t_time = float(np.mean(q))
    k = q - t_time

    dye_deep = np.minimum(
        dmax_amounts,
        np.stack([
            np.interp(log_ep[:, c].max() + q[c], paper["le"], paper["amounts"][:, c])
            for c in range(3)
        ])[None, :],
    )
    floor_rgb = _display_rec2020(
        _stack_reflectance(paper, dye_deep), white, wl, paper["viewing"], flare, exp
    )[0]
    return SimpleNamespace(
        develop=develop,
        q=q,
        k=k,
        t=t_time,
        ev=ev,
        exp=exp,
        flare=flare,
        floor_rgb=floor_rgb,
        label=f"{stock['negative']}+{stock['print']}",
    )


def print_density_curves(stock: dict, surround_override: str | None = None):
    """(scene EV, per-channel print dye amounts[N,3]) of the solved chain.

    The density-domain view for external cross-checks: dye amounts in the paper
    profile's own density_curves units, straight off the solved effective
    exposures — before any colorimetry, CAT or display translation, which an
    external per-channel density oracle (DiVERE) has no claim over.
    """
    chain = _solved_print_chain(stock, surround_override)
    neg = _load_spectral(stock["negative"])
    paper = _regrid(_load_spectral(stock["print"]), neg["wl"])
    sb = _spectral_base()
    enlarger = sb.th_kg3_spd(neg["wl"])
    t_neg = _stack_reflectance(neg, neg["amounts"])
    weight = paper["sens"] * enlarger[:, None]
    log_ep = np.log10(
        np.maximum(sb.trapezoid(t_neg[:, :, None] * weight[None, :, :], neg["wl"], axis=1), 1e-12)
    )
    amounts = np.stack([
        np.interp(log_ep[:, c] + chain.q[c], paper["le"], paper["amounts"][:, c])
        for c in range(3)
    ], axis=1)
    return chain.ev, amounts


def build_endtoend_target(
    stock: dict, surround_override: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Neutral end-to-end response: EV grid, per-channel display linear, luma, floor.

    The floor is the paper's Dmax expressed as reflectance relative to paper white —
    the print medium's own display black. It is declared from data, not fitted: a
    print never reaches zero, and that lifted shadow floor is a structural part of
    the look AgX must reproduce through target_black_linear.

    surround_override="native" quotes the report verbatim (no surround term) — the
    theatrical variants, per the contract's quotation-vs-translation distinction.
    """
    if stock.get("positive"):
        return _build_reversal_target(stock)
    return _finish_print(_solved_print_chain(stock, surround_override))


# --- Enlarger colour head (Y/M subtractive filtration) ------------------------
#
# The printer lights k above are already the digital twin of the darkroom colour
# head: three per-channel log10-exposure trims on the paper. The user-facing
# colour head exposes exactly that machinery as a declared physical control in
# real darkroom units — CC filter density, where NN CC = 0.NN optical density of
# the named subtractive filter in the enlarger's light path:
#   * a Yellow filter absorbs blue light -> the paper's BLUE-sensitive layer
#     (which forms the print's yellow dye, channel 2) sees 10^-d less exposure;
#   * a Magenta filter absorbs green -> the GREEN-sensitive layer (magenta dye,
#     channel 1) sees 10^-d less.
# 30 CC = 0.30 density ~= one stop of that separation's printing exposure.
# The filter is modelled as ideal within-band attenuation of the named layer's
# exposure — the definition of CC filtration, and identical in kind to the k
# trims themselves. After each filtration change the print is RE-TIMED (a global
# exposure-time solve restoring viewed mid-gray luminance to 0.18), which is the
# darkroom convention: filtration decides colour, the test strip re-decides time.
# Direction therefore follows the darkroom mnemonic by construction: a print too
# yellow gets MORE Y filtration -> less blue exposure -> less yellow dye -> the
# finished print moves AWAY from yellow. Each build asserts this direction.
#
# The published field is g_c(EV; filter, cc) = T_c^filtered / T_c^base along the
# neutral ramp, finished under the BASE target's anchor (same flare, surround
# term, exposure anchor and per-channel mid gains — re-normalizing would erase
# the very cast the user dialled in). The runtime consumes it as an EV-dependent
# per-channel gain: spectrally derived, saturating exactly where the paper
# saturates (g -> 1 at both paper endpoints), and NOT a flat RGB multiplier.
# Y and M combine multiplicatively at runtime — a declared separable
# approximation; their physical coupling is second-order (only through the
# shared re-timing and dye-absorption overlap).

COLOR_HEAD_CC_GRID = (30.0, 60.0, 120.0, 200.0)
COLOR_HEAD_EV_POINTS = 64
# Paper layer a CC filter attenuates: Y absorbs blue (layer 2 -> yellow dye),
# M absorbs green (layer 1 -> magenta dye).
COLOR_HEAD_FILTER_LAYER = {"y": 2, "m": 1}


JOINT_HEAD_CC_STEP = 5.0
JOINT_HEAD_CC_MAX = 200.0


def _adaptive_ev_knots(
    full: np.ndarray, target_stop: float, max_knots: int
) -> np.ndarray:
    """Shared EV knot selection for the joint field. full: [Y, M, EV, 3].

    Starts from the endpoints and greedily inserts the full-resolution sample
    with the worst |log2| error of linear-in-gain interpolation over ALL
    detents and channels, until that worst error is below target_stop (or the
    knot budget runs out — the caller's audit then reports the shortfall).
    """
    n_ev = full.shape[2]
    log_full = np.log2(np.maximum(full, 1e-9))
    knots = [0, n_ev - 1]
    while len(knots) < min(max_knots, n_ev):
        ks = np.asarray(sorted(knots))
        pos = np.arange(n_ev)
        seg = np.clip(np.searchsorted(ks, pos, side="right") - 1, 0, ks.size - 2)
        k0, k1 = ks[seg], ks[seg + 1]
        w = ((pos - k0) / np.maximum(k1 - k0, 1)).astype(np.float32)
        approx = full[:, :, k0, :] * (1.0 - w)[None, None, :, None] \
            + full[:, :, k1, :] * w[None, None, :, None]
        err = np.abs(np.log2(np.maximum(approx, 1e-9)) - log_full)
        j = int(np.argmax(err.max(axis=(0, 1, 3))))
        if float(err.max()) < target_stop or j in knots:
            break
        knots.append(j)
    return np.asarray(sorted(knots))


def build_joint_color_head_field(
    stock: dict,
    theatrical: bool = False,
    ev_points: int = 192,
):
    """Stage-3 joint Y x M colour-head field, or None for reversal stocks.

    Every real detent (0..200 CC step 5, both filters) is solved JOINTLY: both
    filter densities perturb the effective exposures together and the print is
    re-timed ONCE — the single spectral solve the separable gY x gM product was
    refuted against (Portra 30Y+30M: ~0.35 stop RMS apart).

    The published gains are diagonal in Bradford LMS, not Rec.2020: extreme
    filtration drives print channels toward zero, where Rec.2020 components can
    cross zero and a ratio stops meaning anything; LMS of a physical (
    non-negative-XYZ) print stays positive, so the ratio field remains
    numerically meaningful over the whole hardware throw. The 0CC x 0CC entry
    is written as EXACT identity — the runtime's bit-exactness contract does
    not rest on solver residuals.

    Returns dict(ev[N], cc_grid[41], gains_lms[41,41,N,3] float32) with axes
    [Y, M, EV, channel].
    """
    if stock.get("positive"):
        return None
    sb = _spectral_base()
    chain = _solved_print_chain(stock, "native" if theatrical else None)
    ev, channels_base, t_neutral, _floor = _finish_print(chain)

    # LMS view of the chain's viewed Rec.2020 (D65) channels.
    rec2020_to_lms = sb._BRADFORD @ np.linalg.inv(sb.XYZ_TO_REC2020)

    def to_lms(rgb: np.ndarray) -> np.ndarray:
        return np.maximum(rgb @ rec2020_to_lms.T, 1e-9)

    def retimed(q_vec: np.ndarray) -> np.ndarray:
        lo, hi = -2.5, 2.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rgb0 = chain.develop(q_vec + mid)
            y0 = float(np.interp(0.0, ev, rgb0 @ _LUMA_2020))
            if y0 > 0.18:
                lo = mid
            else:
                hi = mid
        return np.maximum(chain.develop(q_vec + 0.5 * (lo + hi)), 1e-7)

    visible = t_neutral > 1e-3
    ev_vis = ev[visible]
    base_lms_vis = to_lms(channels_base[visible])

    # The detent solves are the expensive part (41 x 41 re-timings); the gain
    # curves along EV come out of each solve at the chain's full resolution
    # for free. Keep them all and pick the SHARED stored knots adaptively:
    # greedily insert the exposure sample whose linear-in-gain interpolation
    # is worst (in stops) across every Y x M x channel, until the whole field
    # is below the audit target — a hard guarantee at full chain resolution,
    # where a fixed 128-point axis measured up to 0.024 stop on random draws
    # and passed or failed by luck of the draw. ev_points caps the knot count.
    cc_grid = np.arange(0.0, JOINT_HEAD_CC_MAX + 0.1, JOINT_HEAD_CC_STEP)
    n_cc = cc_grid.size
    full = np.empty((n_cc, n_cc, ev_vis.size, 3), dtype=np.float32)
    for yi, y_cc in enumerate(cc_grid):
        for mi, m_cc in enumerate(cc_grid):
            if y_cc == 0.0 and m_cc == 0.0:
                full[yi, mi] = 1.0  # exact identity by construction
                continue
            q_f = chain.q.copy()
            q_f[COLOR_HEAD_FILTER_LAYER["y"]] -= y_cc / 100.0
            q_f[COLOR_HEAD_FILTER_LAYER["m"]] -= m_cc / 100.0
            filtered = retimed(q_f)[visible]
            full[yi, mi] = (to_lms(filtered) / base_lms_vis).astype(np.float32)

    knots = _adaptive_ev_knots(full, target_stop=0.015, max_knots=ev_points)
    ev_grid = ev_vis[knots]
    gains = np.ascontiguousarray(full[:, :, knots, :])

    # HARD GATE (review batch 7): the knot selection audits float32, but the
    # npz ships float16 — re-audit the FINAL bytes (f16-quantized knot gains,
    # linearly interpolated back onto the full-resolution chain grid) and
    # refuse to emit an asset whose worst error exceeds the acceptance gate.
    # Hitting the knot budget also lands here instead of shipping silently.
    gains_f16 = gains.astype(np.float16).astype(np.float32)
    pos = np.arange(ev_vis.size)
    seg = np.clip(np.searchsorted(knots, pos, side="right") - 1, 0, knots.size - 2)
    k0, k1 = knots[seg], knots[seg + 1]
    w = ((pos - k0) / np.maximum(k1 - k0, 1)).astype(np.float32)
    approx = gains_f16[:, :, seg, :] * (1.0 - w)[None, None, :, None] \
        + gains_f16[:, :, seg + 1, :] * w[None, None, :, None]
    audit_max_stop = float(np.abs(
        np.log2(np.maximum(approx, 1e-9) / np.maximum(full, 1e-9))
    ).max())
    if audit_max_stop > 0.02:
        raise RuntimeError(
            f"{chain.label}: joint colour-head field failed its own audit — "
            f"worst f16-quantized interpolation error {audit_max_stop:.4f} stop "
            f"> 0.02 gate ({knots.size} knots, budget {ev_points})"
        )

    # Oracle fixtures: random detents solved at the CHAIN's full EV resolution
    # (off-grid relative to the stored axis), so the runtime test measures the
    # deployed field + EV interpolation against the direct spectral solve — the
    # review's gate — rather than the field against itself.
    rng = np.random.default_rng(20260806)
    base_lms_full = to_lms(channels_base[visible])
    oracle = []
    for _ in range(8):
        yi = int(rng.integers(0, n_cc))
        mi = int(rng.integers(0, n_cc))
        q_f = chain.q.copy()
        q_f[COLOR_HEAD_FILTER_LAYER["y"]] -= float(cc_grid[yi]) / 100.0
        q_f[COLOR_HEAD_FILTER_LAYER["m"]] -= float(cc_grid[mi]) / 100.0
        truth = to_lms(retimed(q_f)[visible]) / base_lms_full
        pick = rng.integers(0, ev_vis.size, 6)
        for k in pick:
            oracle.append((float(yi), float(mi), float(ev_vis[k]), *truth[k]))
    return {
        "ev": ev_grid.astype(np.float32),
        "cc_grid": cc_grid.astype(np.float32),
        "gains_lms": gains,
        "oracle": np.asarray(oracle, dtype=np.float32),
        "basis": "bradford-lms",
        "label": chain.label,
        "schema": 3,
        "audit_max_stop": audit_max_stop,
    }


def _agx_curve(ev: np.ndarray, params_vec: np.ndarray, target_black: float = 0.0) -> np.ndarray:
    black_ev, white_ev, contrast, toe_p, shoulder_p, lat_lo, lat_hi = params_vec
    plan = SimpleNamespace(
        black_ev=float(black_ev),
        white_ev=float(white_ev),
        contrast=float(contrast),
        toe_power=float(toe_p),
        shoulder_power=float(shoulder_p),
        latitude_lo_ev=float(lat_lo),
        latitude_hi_ev=float(lat_hi),
        pivot_ev_offset=0.0,
        target_black_linear=float(target_black),
        target_white_linear=1.0,
        curve_gamma=2.2,
    )
    params = curve_params_from_plan(plan)
    return np.asarray(
        apply_c1_endpoints(ev.astype(np.float32), plan, params=params), dtype=np.float64
    )


PARAM_NAMES = (
    "black_ev", "white_ev", "contrast", "toe_power",
    "shoulder_power", "latitude_lo_ev", "latitude_hi_ev",
)

BOUNDS = np.array(
    [
        (-14.0, -2.5),   # black_ev (reversal media die shallow; -4 pinned pre-surround)
        (2.0, 8.5),      # white_ev
        (1.2, 5.5),      # contrast
        (0.8, 3.5),      # toe_power
        (1.2, 10.0),     # shoulder_power (slides clip highlights harder than any paper)
        (0.0, 2.5),      # latitude_lo_ev (cine negatives carry straight-line > 1.5 EV)
        (0.0, 2.5),      # latitude_hi_ev
    ]
)


def _pinned_params(vec: np.ndarray) -> list[str]:
    """Parameters sitting on a fit bound — the declared-extrapolation report.

    An endpoint pinned at a bound *outside the fit domain* (black_ev at -14 when the
    domain stops at -6.5, white_ev at 8.5 against +6.0) is not measured by the data:
    the toe/shoulder it describes lies beyond every sample, and the recorded value is
    an extrapolation the optimizer parked at the fence. Publishing the list keeps that
    honest instead of letting the JSON read as if every parameter were determined.
    Zero latitude is a legitimate interior solution (no linear mid-segment), not a pin.
    """
    pins = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS[i]
        if abs(vec[i] - hi) < 5e-3:
            pins.append(f"{name}@{hi:g}")
        elif abs(vec[i] - lo) < 5e-3 and not (name.startswith("latitude") and lo == 0.0):
            pins.append(f"{name}@{lo:g}")
    return pins


def _clip_bounds(vec: np.ndarray) -> np.ndarray:
    return np.clip(vec, BOUNDS[:, 0], BOUNDS[:, 1])


def _residual_stops(ev: np.ndarray, target: np.ndarray, vec: np.ndarray, floor_black: float = 0.0) -> np.ndarray:
    fitted = _agx_curve(ev, _clip_bounds(vec), target_black=floor_black)
    floor = 1e-4
    mask = target > floor
    return np.log2(np.maximum(fitted[mask], floor)) - np.log2(target[mask])


def _objective(ev: np.ndarray, target: np.ndarray, vec: np.ndarray, floor_black: float = 0.0) -> float:
    r = _residual_stops(ev, target, vec, floor_black)
    # RMS with a soft penalty on the worst point so the tail cannot be sacrificed
    # wholesale for the body.
    return float(np.sqrt(np.mean(r * r)) + 0.15 * np.max(np.abs(r)))


def nelder_mead(fn, x0: np.ndarray, steps: np.ndarray, iters: int = 900) -> np.ndarray:
    n = x0.size
    simplex = [x0.copy()]
    for i in range(n):
        v = x0.copy()
        v[i] += steps[i]
        simplex.append(v)
    values = [fn(v) for v in simplex]
    for _ in range(iters):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        f_r = fn(reflected)
        if f_r < values[0]:
            expanded = centroid + 2.0 * (centroid - worst)
            f_e = fn(expanded)
            simplex[-1], values[-1] = (
                (expanded, f_e) if f_e < f_r else (reflected, f_r)
            )
        elif f_r < values[-2]:
            simplex[-1], values[-1] = reflected, f_r
        else:
            contracted = centroid + 0.5 * (worst - centroid)
            f_c = fn(contracted)
            if f_c < values[-1]:
                simplex[-1], values[-1] = contracted, f_c
            else:
                best = simplex[0]
                simplex = [best] + [best + 0.5 * (v - best) for v in simplex[1:]]
                values = [values[0]] + [fn(v) for v in simplex[1:]]
    order = np.argsort(values)
    return simplex[order[0]]


def _model_note(stock: dict, theatrical: bool = False) -> str:
    """Provenance string: the chain model plus its viewing-condition translation."""
    if stock.get("positive"):
        g = SURROUND_GAMMA["dark"]
        return (
            "spectral reversal: dye-stack transmittance vs the medium's Dmin white "
            "under the declared viewing illuminant (relative colorimetry, CIE 1931), "
            f"dark-surround report to {TARGET_SURROUND} surround via T^(1/{g:g}), "
            "zero calibration flare (medium-native), global exposure anchor "
            "(light-meter semantics)"
        )
    surround = PRINT_SURROUND.get(str(stock.get("print")), "average")
    base = (
        "spectral contact print: paper spectral sensitivity x TH-KG3 enlarger "
        "(3400K x Schott KG3) over "
        "the negative's dye-stack transmittance, printer lights solved to a "
        "neutral 18% mid, viewed under the print's declared illuminant "
        "(relative colorimetry, CIE 1931)"
    )
    if theatrical:
        return (
            base
            + ", theatrical quotation: dark-surround report carried verbatim "
            "(no surround term — a quotation, not a translation; see contract §8.1)"
        )
    if surround == TARGET_SURROUND:
        return base + ", surround term identity (reflection paper matches delivery)"
    g = SURROUND_GAMMA[surround]
    return (
        base
        + f", {surround}-surround projection print to {TARGET_SURROUND} "
        f"surround via T^(1/{g:g})"
    )


def fit_stock(key: str, stock: dict, theatrical: bool = False) -> dict:
    ev_full, channels_full, target_full, floor_black = build_endtoend_target(
        stock, surround_override="native" if theatrical else None
    )
    mask = (ev_full >= FIT_EV_LO) & (ev_full <= FIT_EV_HI)
    ev, target = ev_full[mask], target_full[mask]
    channels = channels_full[mask]

    x0 = np.array([-8.0, 4.0, 3.0, 1.5, 2.9, 0.1, 0.2])
    steps = np.array([1.0, 0.6, 0.4, 0.25, 0.5, 0.15, 0.15])
    fn = lambda v: _objective(ev, target, v, floor_black)  # noqa: E731
    best = nelder_mead(fn, x0, steps)
    # Warm-restart refinement: a fresh small simplex around the incumbent escapes the
    # collapsed simplex of the first pass; two rounds measurably tighten the deep toe.
    for shrink in (0.25, 0.08):
        best = nelder_mead(fn, _clip_bounds(best), steps * shrink, iters=600)
    best = _clip_bounds(best)
    r = _residual_stops(ev, target, best, floor_black)
    rms, worst = float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))

    idx = np.linspace(0, ev.size - 1, TARGET_POINTS_STORED).round().astype(int)

    # Exposure-dependent colour, phase 1 (contract boundary #1 work package): the
    # per-channel ratio field r_c(EV) = T_c / T_neutral along the balanced neutral
    # ramp. This is the stock's own measured layer-saturation differential — the
    # first-order source of "highlights warm as the blue layer saturates" — with
    # provenance identical to the tone target (same channels, same balance, same
    # surround term). r_c(0) = 1 exactly by the per-channel mid-scale balance.
    # Runtime semantics (phase 2): out_c = C(EV_c) * r_c(EV_c) / r_c(EV_Y), which is
    # exactly 1 on the neutral axis (EV_c = EV_Y) — boundary #2 held by construction.
    # The ratio field is only defined where the display value is a measurement:
    # below ~1e-3 (about sRGB code 1) the channels sit on numerical clamps and a
    # ratio there is artifact, not dye differential. The runtime interpolation
    # clamps to the grid edges anyway, so trimming the grid to the visible domain
    # IS the declared out-of-domain semantics.
    visible = target > 1e-3
    ratio = channels[visible] / target[visible, None]
    # Deep saturated shadows of some media (Kodachrome above all) leave the
    # working gamut: a display channel there is clamped, and its ratio is a gamut fact,
    # not a dye differential. The stored field carries the same [0.25, 4] rail the
    # runtime applies (agx.channel_ratio_gain), so data and application agree on
    # where measurement ends and the rail begins.
    ratio = np.clip(ratio, 0.25, 4.0)
    ratio_ev = ev[visible]
    ridx = np.linspace(0, ratio_ev.size - 1, min(TARGET_POINTS_STORED, ratio_ev.size)
                       ).round().astype(int)
    r0 = np.array([float(np.interp(0.0, ratio_ev, ratio[:, c])) for c in range(3)])
    if np.max(np.abs(r0 - 1.0)) > 5e-3:
        raise RuntimeError(f"channel ratio mid-gray anchor drifted: r(0)={r0}")

    return {
        "label": stock["label"] + ("（影院放映外观）" if theatrical else ""),
        # Physical process classification, read straight from the chain model:
        # a stock whose profile declares a target print is a print-through
        # negative (C-41 stills / ECN-2 cine); a positive profile is reversal
        # film — its own display medium, physically without a printing stage.
        # The enlarger colour head exists only for the former.
        "process": "reversal" if stock.get("positive") else "negative",
        "color_head": (
            None
            if stock.get("positive")
            else {
                "format": "joint-lms-npz-v3",
                "file": f"color_head/{key}.npz",
                "cc_step": JOINT_HEAD_CC_STEP,
                "cc_max": JOINT_HEAD_CC_MAX,
                "model": (
                    "joint paper-layer exposure model: every real Y x M detent "
                    "solved together through the TH-KG3 printing chain with ONE "
                    "re-time (test-strip convention); gains diagonal in Bradford "
                    "LMS along the neutral ramp (stable at extreme filtration "
                    "where Rec.2020 components may cross zero); runtime "
                    "interpolates EV only — detents index directly. This is a "
                    "neutral-axis paper-exposure model, not a filter-spectra "
                    "oracle: no Y/M transmission spectra exist in the data yet."
                ),
            }
        ),
        "params": {
            "black_ev": round(float(best[0]), 4),
            "white_ev": round(float(best[1]), 4),
            "contrast": round(float(best[2]), 4),
            "toe_power": round(float(best[3]), 4),
            "shoulder_power": round(float(best[4]), 4),
            "latitude_lo_ev": round(float(best[5]), 4),
            "latitude_hi_ev": round(float(best[6]), 4),
            "target_black_linear": round(floor_black, 6),
        },
        # Schema v3 (medium-native black): the medium's own floor, stamped
        # explicitly, with the calibration's viewing flare REQUIRED to be zero.
        # target_black_linear must equal medium_floor_linear (loader-enforced);
        # delivery_viewing_flare is reserved for a future view-simulation layer
        # and stays zero until that layer exists.
        "black_policy": "medium-native",
        "medium_floor_linear": round(floor_black, 6),
        "calibration_viewing_flare": 0.0,
        "delivery_viewing_flare": 0.0,
        "fit": {
            "rms_stop": round(rms, 5),
            "max_stop": round(worst, 5),
            "domain_ev": [FIT_EV_LO, FIT_EV_HI],
            "pinned": _pinned_params(best),
        },
        "target_curve": {
            "ev": [round(float(v), 5) for v in ev[idx]],
            "display_linear": [round(float(v), 7) for v in target[idx]],
        },
        # v2 neutral field, Rec.2020 basis (stage C of the spectral rebuild).
        # neutral_rgb is the MEASUREMENT (the neutral scale's exposure-dependent
        # colour); neutral_gain = neutral_rgb / target_y is what a runtime that
        # wants to reproduce that colour multiplies by. Storing both keeps the
        # measurement from being mistaken for a multiplier.
        "neutral_curve": {
            "basis": "rec2020",
            "ev": [round(float(v), 5) for v in ratio_ev[ridx]],
            "target_y": [round(float(target[visible][i]), 7) for i in ridx],
            "neutral_rgb_rec2020": [
                [round(float(channels[visible][i, c]), 7) for c in range(3)]
                for i in ridx
            ],
            "neutral_gain_rec2020": [
                [round(float(ratio[i, c]), 5) for c in range(3)] for i in ridx
            ],
        },
        "combo": {
            # The film-observation expansion: declared WB (tungsten cine stocks are
            # 3200K by name), and the stock's spectral separation preset when the
            # prefeed calibrator has produced one.
            "wb": stock.get("wb", "5500k"),
            # Push processing changes development, not the emulsion, and the
            # theatrical quotation changes only the viewing translation: both share
            # the base stock's spectral separation preset.
            "scene_transform": (
                f"{key.removesuffix('_theatrical').split('push')[0]}_d55"
            ),
        },
        "source": {
            "film": f"spektrafilm/{stock['negative']}.json",
            "print": (
                f"spektrafilm/{stock['print']}.json"
                if stock.get("print")
                else "none (reversal: the slide is its own display medium)"
            ),
            "license": "CC BY-SA 4.0 (spektrafilm profiles, Andrea Volpato)",
            "model": _model_note(stock, theatrical=theatrical),
            "colorimetry": _colorimetry_note(),
            "printing_illuminant": _printing_illuminant_note(),
        },
    }


def _colorimetry_note() -> dict:
    import calibrate_skin_matrix as csm

    return csm.colorimetry_provenance()


def _printing_illuminant_note() -> dict:
    return _spectral_base().th_kg3_provenance()


def plot(presets: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(presets), figsize=(6.4 * len(presets), 4.6), dpi=150)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#101218")
    for ax, (key, p) in zip(axes, presets.items()):
        ax.set_facecolor("#101218")
        ev = np.array(p["target_curve"]["ev"])
        tgt = np.array(p["target_curve"]["display_linear"])
        vec = np.array([p["params"][k] for k in (
            "black_ev", "white_ev", "contrast", "toe_power",
            "shoulder_power", "latitude_lo_ev", "latitude_hi_ev")])
        fit = _agx_curve(ev, vec, target_black=p["params"].get("target_black_linear", 0.0))
        ax.plot(ev, np.log2(np.maximum(tgt, 1e-5)), color="#f0b35e", lw=2.4,
                label="datasheet end-to-end (neg + print)")
        ax.plot(ev, np.log2(np.maximum(fit, 1e-5)), color="#5ea8f0", lw=1.8, ls="--",
                label=f"AgX fit (rms {p['fit']['rms_stop']:.3f} stop)")
        ax.set_title(p["label"], color="#e6e9f0", fontsize=11)
        ax.set_xlabel("scene EV", color="#c9cfdb")
        ax.set_ylabel("output stops", color="#c9cfdb")
        ax.tick_params(colors="#9aa3b2")
        for s in ax.spines.values():
            s.set_color("#3a4152")
        leg = ax.legend(loc="lower right", framealpha=0.15, fontsize=8.5)
        for t in leg.get_texts():
            t.set_color("#e6e9f0")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stocks", nargs="*", default=list(STOCKS))
    ap.add_argument("--plot", type=Path,
                    default=PROJECT_ROOT / "docs" / "assets" / "film-curve-fits.png")
    args = ap.parse_args()

    presets = {}
    if PRESET_PATH.is_file():
        presets = json.load(open(PRESET_PATH)).get("presets", {})
    data_dir = PROJECT_ROOT / "dngscan" / "data" / "color_head"
    data_dir.mkdir(parents=True, exist_ok=True)

    def write_joint_head(key: str, stock: dict, theatrical: bool) -> None:
        field = build_joint_color_head_field(stock, theatrical=theatrical)
        if field is None:
            return
        out = data_dir / f"{key}.npz"
        np.savez_compressed(
            out,
            ev=field["ev"].astype(np.float32),
            cc_grid=field["cc_grid"].astype(np.float32),
            gains_lms=field["gains_lms"].astype(np.float16),
            oracle=field["oracle"],
            basis=np.asarray(field["basis"]),
            label=np.asarray(field["label"]),
            schema=np.int32(field["schema"]),
            audit_max_stop=np.float32(field["audit_max_stop"]),
        )
        print(f"  joint colour head -> {out.name} "
              f"({out.stat().st_size/1024:.0f} KiB)")

    for key in args.stocks:
        stock = STOCKS[key]
        preset = fit_stock(key, stock)
        presets[key] = preset
        write_joint_head(key, stock, theatrical=False)
        print(f"{key}: rms {preset['fit']['rms_stop']:.4f} stop, "
              f"max {preset['fit']['max_stop']:.4f} stop, params {preset['params']}")
        # Theatrical quotation variants for dark-surround projection chains: the
        # report carried verbatim (contract §8.1 — a quotation, not a translation).
        if PRINT_SURROUND.get(str(stock.get("print"))) == "dark":
            tkey = f"{key}_theatrical"
            tpreset = fit_stock(tkey, stock, theatrical=True)
            presets[tkey] = tpreset
            write_joint_head(tkey, stock, theatrical=True)
            print(f"{tkey}: rms {tpreset['fit']['rms_stop']:.4f} stop, "
                  f"max {tpreset['fit']['max_stop']:.4f} stop")
    PRESET_PATH.write_text(
        json.dumps({"version": 3, "presets": presets}, indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {PRESET_PATH}")
    if args.plot:
        plot(presets, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
