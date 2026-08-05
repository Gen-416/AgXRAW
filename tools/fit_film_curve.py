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

End-to-end target construction (SPECTRAL contact print; see the section comment at
_load_spectral for provenance and validation):
    D_neg(lambda,e) = sum_dye amount_dye(e) * dyeSpec(lambda) + base(lambda)
    E_paper_c(e)    = integral S_paper_c(lambda) * I_enlarger(lambda) * 10^-D_neg
    dye_paper_c(e)  = paper_curve_c(log10 E_paper_c + k_c)     (printer lights k)
    RGB(e)          = relative colorimetry of the print's dye stack vs its white
                      under the declared viewing illuminant (CIE 1931 -> linear sRGB)
    T(e)            = ((RGB + f)/(1 + f))^s                    (viewing flare f,
                      surround term s; both routed by the medium's viewing condition)
The printer lights k are solved so the print is neutral at mid; the exposure anchor
is then a GLOBAL shift solved so viewed Y = 0.18 (light-meter semantics — the
profiles' logE=0 is a normalization convention, not scene mid-gray), followed by
per-channel micro-gains that pin exact neutrality. Scene EV = logE / log10(2).

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
#   * enlarger lamp: 3200 K blackbody (tungsten-halogen; agx-emulsion's TH-KG3
#     convention, heat-filter shaping absorbed by the per-channel printer
#     lights k);
#   * viewing: the print profile's own viewing_illuminant (D50 for papers),
#     CIE 1931 2-degree observer, relative colorimetry against the medium's
#     clear/white point (per-channel normalization = von Kries to that white);
#   * display encoding of the target: linear Rec.709/sRGB primaries, so the
#     scalar target remains the CIE Y of the displayed print (the convention
#     the runtime consumes).

_XYZ_TO_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def _spd_tools():
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from calibrate_skin_matrix import blackbody_spd, cie_1931_cmf, illuminant_spd

    return blackbody_spd, cie_1931_cmf, illuminant_spd


def _load_spectral(name: str) -> dict[str, np.ndarray]:
    raw = json.load(open(PROFILE_DIR / f"{name}.json"))
    data = raw["data"]

    def arr(key):
        return np.array(
            [[np.nan if v is None else float(v) for v in row]
             if isinstance(row, list) else (np.nan if row is None else float(row))
             for row in data[key]], dtype=np.float64)

    wl = arr("wavelengths")
    dye = np.nan_to_num(arr("channel_density"), nan=0.0)          # [81,3] C/M/Y
    base = np.nan_to_num(arr("base_density"), nan=0.0)            # [81]
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
    try:
        spd = illuminant_spd(viewing)
    except Exception:
        spd = blackbody_spd(wl, 5000.0)
    result = (np.asarray(cmf, dtype=np.float64), np.asarray(spd, dtype=np.float64))
    _VIEWING_CACHE[key] = result
    return result


def _display_rgb(reflect: np.ndarray, white: np.ndarray, wl: np.ndarray,
                 viewing: str) -> np.ndarray:
    """Relative-colorimetric linear sRGB of spectra vs the medium white. [n,3]"""
    cmf, spd = _viewing_xyz(wl, viewing)
    weight = spd[:, None] * cmf                     # [81,3]
    xyz = reflect @ weight                          # [n,3]
    xyz_w = white @ weight                          # [3]
    xyz = xyz / max(float(xyz_w[1]), 1e-12)         # white Y -> 1
    rgb = xyz @ _XYZ_TO_SRGB.T
    # Von Kries in display primaries: the medium's own white renders neutral.
    rgb_w = (xyz_w / max(float(xyz_w[1]), 1e-12)) @ _XYZ_TO_SRGB.T
    return rgb / np.maximum(rgb_w[None, :], 1e-9)


def _stack_reflectance(spec: dict, amounts: np.ndarray) -> np.ndarray:
    """10^-D for dye-amount rows through this medium's dye spectra + base. [n,81]"""
    dens = amounts @ spec["dye"].T + spec["base"][None, :]
    return np.power(10.0, -dens)


def _working_grid() -> np.ndarray:
    """The colorimetry helpers' native 400-700/10nm grid; all media resample to it.

    Dye spectra and sensitivities are smooth at this resolution, and using the
    helpers' own grid keeps CMF/illuminant alignment exact instead of fighting it."""
    _spd_tools()
    import calibrate_skin_matrix as csm

    return np.asarray(csm.WL, dtype=np.float64)


def _regrid(spec: dict, wl: np.ndarray) -> dict:
    """Resample a medium's spectral fields onto another wavelength grid.

    Profiles ship on different grids (film 380-780/5nm, papers coarser); the print
    integral needs one grid. Linear interpolation, out-of-range clamped to the edge
    values (sensitivities there are already ~0)."""
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
            [np.interp(wl, spec["wl"], spec["sens"][:, k]) for k in range(3)], axis=1
        )
    return out


_LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Veiling glare of the reference viewing condition. The delivery contract already
# reads prints "as an sRGB-condition viewer would"; IEC 61966-2-1 specifies that
# reference environment with 1.0% viewing flare. Applied to the linear stimulus
# BEFORE the surround term (glare is physics in the room, surround is the
# appearance translation of what the room presents). The zero-flare shortcut was
# a declared missing piece of the contact-print model; with spectral printing
# honest about dye crosstalk, its absence became the binding source of crushed,
# abrupt shadows. Quotation (theatrical) presets skip flare with the rest of the
# viewing translation: a quotation carries the report, not the reading room.
VIEWING_FLARE = 0.01


def _view_transform(x: np.ndarray, flare: float, exp: float) -> np.ndarray:
    """Viewing flare then surround exponent — the medium-to-delivery translation."""
    flared = (np.maximum(x, 0.0) + flare) / (1.0 + flare)
    return np.power(np.maximum(flared, 1e-7), exp)


def _finish_target(ev: np.ndarray, rgb_linear: np.ndarray, floor_rgb: np.ndarray,
                   label: str, exp: float, flare: float,
                   return_norm: bool = False,
                   ):
    """Shared tail: viewing flare -> surround exponent -> per-channel mid balance.

    The per-channel balance at mid-scale is the declared neutrality contract of the
    curve layer (anti-hidden-WB: the preset must not smuggle a cast through the
    anchor); any residual mid cast of the medium is normalized out here and the
    channel_ratio field carries only the exposure-DEPENDENT differential.

    return_norm=True additionally returns the solved normalization {e0, gain} so a
    derived construction (the colour-head field) can re-finish PERTURBED chain states
    under the base target's exact anchor instead of re-normalizing the perturbation
    away.
    """
    def view(x: np.ndarray) -> np.ndarray:
        return _view_transform(x, flare, exp)

    rgb_view = view(rgb_linear)
    floor_view = view(floor_rgb)
    # Exposure anchor: "correct exposure" places the scene's 18% object where the
    # viewed medium reads 0.18 — solved as a global exposure shift, exactly what a
    # light meter does. The profiles' own logE=0 is a normalization convention, not
    # scene mid-gray (their spectral mid-scale reference sits at +0.16..+0.40 logE);
    # anchoring by gain at logE=0 parked slide mid-gray on the shoulder and warped
    # every translated reversal fit. Print paths already anchor Y through the
    # printer-light solve, so their shift comes out ~0.
    y_view = rgb_view @ _LUMA_709
    order = np.argsort(y_view)
    e0 = float(np.interp(0.18, y_view[order], ev[order]))
    if not np.isfinite(e0):
        raise RuntimeError(f"{label}: no exposure reaches mid-gray")
    ev = ev - e0
    mid = np.array([float(np.interp(0.0, ev, rgb_view[:, c])) for c in range(3)])
    if np.any(mid <= 1e-6):
        raise RuntimeError(f"{label}: degenerate mid-scale {mid}")
    gain = 0.18 / mid
    channels = np.maximum(rgb_view * gain[None, :], 1e-7)
    floor = float(np.maximum(floor_view * gain, 1e-7) @ _LUMA_709)
    t_neutral = channels @ _LUMA_709
    t0 = float(np.interp(0.0, ev, t_neutral))
    if abs(t0 - 0.18) > 5e-4:
        raise RuntimeError(f"{label}: mid-gray anchor drifted: T(0)={t0:.5f}")
    if return_norm:
        return ev, channels, t_neutral, floor, {"e0": e0, "gain": gain}
    return ev, channels, t_neutral, floor


def _build_reversal_target(
    stock: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Slide film, spectrally: the developed dye stack viewed as its own display.

    Reversal film is designed for dark-surround projection; the surround term
    translates the report to the delivery condition once (see SURROUND_GAMMA).
    Transmittance is computed from the per-dye amounts through the dye spectra and
    base, viewed against the clear base as white — full relative colorimetry rather
    than per-Status-channel shortcuts."""
    neg = _regrid(_load_spectral(stock["negative"]), _working_grid())
    exp = surround_exponent("dark")
    reflect = _stack_reflectance(neg, neg["amounts"])
    # White is the medium's own brightest state (Dmin INCLUDING residual dye), not
    # the bare base: slides never clear completely, and normalizing against the
    # bare base capped relative luminance at ~0.46 instead of ~1.
    white = _stack_reflectance(neg, np.nanmin(neg["amounts"], axis=0)[None, :])[0]
    rgb = _display_rgb(reflect, white, neg["wl"], neg["viewing"])
    deep = _stack_reflectance(neg, np.nanmax(neg["amounts"], axis=0)[None, :])
    floor_rgb = _display_rgb(deep, white, neg["wl"], neg["viewing"])[0]
    ev = neg["le"] / LOG10_2
    # Dark-surround media read in a darkened room: the IEC 1% veiling figure
    # describes bright/average viewing environments, and the dark-surround
    # appearance constant already carries that tradition's room assumptions.
    # Projection media therefore take zero declared flare.
    return _finish_target(ev, rgb, floor_rgb, stock["negative"], exp, 0.0)


def _solved_print_chain(stock: dict, surround_override: str | None = None) -> SimpleNamespace:
    """The negative+paper printing chain with its printer lights solved.

    Everything build_endtoend_target needs before the finishing translation, exposed
    as one namespace so derived constructions (the colour-head field) can re-develop
    the SAME chain with perturbed printer lights instead of duplicating the model.
    """
    wl = _working_grid()
    neg = _regrid(_load_spectral(stock["negative"]), wl)
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

    blackbody_spd, _cmf, _ill = _spd_tools()
    enlarger = blackbody_spd(neg["wl"], 3200.0)
    # Paper-channel exposure through the negative's true spectral transmittance.
    t_neg = _stack_reflectance(neg, neg["amounts"])                # [n,81]
    weight = paper["sens"] * enlarger[:, None]                     # [81,3]
    log_ep = np.log10(np.maximum(t_neg @ weight, 1e-12))           # [n,3]

    white_amounts = np.nanmin(paper["amounts"], axis=0)[None, :]
    white = _stack_reflectance(paper, white_amounts)[0]
    dmax_amounts = np.nanmax(paper["amounts"], axis=0)[None, :]
    ev = neg["le"] / LOG10_2

    def develop(k: np.ndarray) -> np.ndarray:
        dye = np.stack([
            np.interp(log_ep[:, c] + k[c], paper["le"], paper["amounts"][:, c])
            for c in range(3)
        ], axis=1)
        reflect = _stack_reflectance(paper, dye)
        return _display_rgb(reflect, white, paper["wl"], paper["viewing"])

    # Printer lights: three per-channel exposure trims solved so the scene's
    # mid-gray prints to a neutral 18% — the digital twin of the darkroom's
    # colour head. Initialized so each channel's mid negative lands mid-paper.
    k0 = np.array([
        float(np.interp(
            0.5 * (paper["amounts"][:, c].min() + paper["amounts"][:, c].max()),
            paper["amounts"][:, c], paper["le"],
        )) - float(np.interp(0.0, ev, log_ep[:, c]))
        for c in range(3)
    ])

    def objective(k: np.ndarray) -> float:
        rgb0 = develop(k)
        mid = np.array([float(np.interp(0.0, ev, rgb0[:, c])) for c in range(3)])
        return float(np.sum(np.log(np.maximum(mid, 1e-6) / 0.18) ** 2))

    k = nelder_mead(objective, k0, np.array([0.15, 0.15, 0.15]), iters=400)
    dye_deep = np.minimum(
        dmax_amounts,
        np.stack([
            np.interp(log_ep[:, c].max() + k[c], paper["le"], paper["amounts"][:, c])
            for c in range(3)
        ])[None, :],
    )
    floor_rgb = _display_rgb(
        _stack_reflectance(paper, dye_deep), white, paper["wl"], paper["viewing"]
    )[0]
    surround_kind = PRINT_SURROUND.get(stock["print"], "average")
    flare = (
        VIEWING_FLARE
        if surround_override != "native" and surround_kind == "average"
        else 0.0
    )
    return SimpleNamespace(
        develop=develop,
        k=k,
        ev=ev,
        exp=exp,
        flare=flare,
        floor_rgb=floor_rgb,
        label=f"{stock['negative']}+{stock['print']}",
    )


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
    chain = _solved_print_chain(stock, surround_override)
    return _finish_target(chain.ev, chain.develop(chain.k), chain.floor_rgb,
                          chain.label, chain.exp, chain.flare)


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


def build_color_head_field(stock: dict, theatrical: bool = False) -> dict | None:
    """Per-preset colour-head response grid, or None for reversal stocks.

    Reversal film has no printing stage — the slide is its own display medium —
    so there is physically no enlarger head to model and the field is None.
    """
    if stock.get("positive"):
        return None
    chain = _solved_print_chain(stock, "native" if theatrical else None)
    ev, channels_base, t_neutral, _floor, norm = _finish_target(
        chain.ev, chain.develop(chain.k), chain.floor_rgb,
        chain.label, chain.exp, chain.flare, return_norm=True,
    )
    gain = np.asarray(norm["gain"], dtype=np.float64)

    def finished(k_vec: np.ndarray) -> np.ndarray:
        """Perturbed chain finished under the base anchor (no re-normalization)."""
        v = _view_transform(chain.develop(k_vec), chain.flare, chain.exp)
        return np.maximum(v * gain[None, :], 1e-7)

    def retimed(k_vec: np.ndarray) -> np.ndarray:
        """Solve the print exposure time restoring viewed mid luminance to 0.18.

        A global shift t on all three printer lights IS exposure time in log10
        units; mid luminance is strictly decreasing in t (more exposure -> denser
        print), so bisection converges unconditionally.
        """
        lo, hi = -2.5, 2.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            y0 = float(np.interp(0.0, ev, finished(k_vec + mid) @ _LUMA_709))
            if y0 > 0.18:
                lo = mid
            else:
                hi = mid
        return finished(k_vec + 0.5 * (lo + hi))

    # Same visible-domain rule as the channel_ratio field: below ~1e-3 the display
    # channels sit on numerical clamps and a ratio is artifact, not filtration.
    visible = t_neutral > 1e-3
    ev_vis = ev[visible]
    idx = np.linspace(
        0, ev_vis.size - 1, min(COLOR_HEAD_EV_POINTS, ev_vis.size)
    ).round().astype(int)
    base_vis = channels_base[visible]

    field: dict[str, object] = {
        "unit": "cc",
        "cc_grid": [float(v) for v in COLOR_HEAD_CC_GRID],
        "ev": [round(float(v), 5) for v in ev_vis[idx]],
    }
    mid_gain_30cc: dict[str, list[float]] = {}
    for name, layer in COLOR_HEAD_FILTER_LAYER.items():
        per_cc = []
        for cc in COLOR_HEAD_CC_GRID:
            k_f = chain.k.copy()
            k_f[layer] -= cc / 100.0
            ratio = np.clip(retimed(k_f)[visible] / base_vis, 0.05, 20.0)
            if cc == 30.0:
                mid_gain_30cc[name] = [
                    round(float(np.interp(0.0, ev_vis, ratio[:, c])), 5)
                    for c in range(3)
                ]
            g_mid = float(np.interp(0.0, ev_vis, ratio[:, layer]))
            if g_mid <= 1.0:
                raise RuntimeError(
                    f"{chain.label}: colour-head direction violated for {name} "
                    f"{cc:g}CC: displayed channel {layer} mid gain {g_mid:.4f} <= 1 "
                    "(less dye must brighten its own band)"
                )
            per_cc.append([
                [round(float(ratio[i, c]), 5) for c in range(3)] for i in idx
            ])
        field[name] = per_cc
    field["mid_gain_30cc"] = mid_gain_30cc
    field["model"] = (
        "spectral print re-exposed: a Y (M) CC filter of density d attenuates the "
        "paper's blue-(green-)sensitive layer exposure by 10^-d through the same "
        "printer-light machinery the neutral solve uses; print re-timed to viewed "
        "mid-gray luminance 0.18 (darkroom test-strip convention); field = "
        "filtered/unfiltered display ratio along the neutral ramp under the base "
        "anchor; runtime interpolates in CC and EV and combines Y x M "
        "multiplicatively (declared separable approximation)"
    )
    return field


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
            "zero projection flare, global exposure anchor (light-meter semantics)"
        )
    surround = PRINT_SURROUND.get(str(stock.get("print")), "average")
    base = (
        "spectral contact print: paper spectral sensitivity x 3200K enlarger over "
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
    # Deep saturated shadows of some media (Kodachrome above all) leave the sRGB
    # gamut: a display channel there is clamped, and its ratio is a gamut fact,
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
        "color_head": build_color_head_field(stock, theatrical=theatrical),
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
        "channel_ratio_curve": {
            "ev": [round(float(v), 5) for v in ratio_ev[ridx]],
            "ratio_rgb": [
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
        },
    }


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


def augment_color_head() -> int:
    """Add process classification + colour-head fields to the published presets.

    Deliberately NOT a refit: tone params, fit residuals, target and ratio curves
    stay byte-identical (the golden freeze rests on them). The printer-light solve
    is deterministic, so the colour-head build reproduces the exact k of the
    original fit run before perturbing it.
    """
    raw = json.load(open(PRESET_PATH))
    presets = raw.get("presets", {})
    for key, preset in presets.items():
        theatrical = key.endswith("_theatrical")
        stock = STOCKS[key.removesuffix("_theatrical")]
        preset["process"] = "reversal" if stock.get("positive") else "negative"
        preset["color_head"] = build_color_head_field(stock, theatrical=theatrical)
        if preset["color_head"] is not None:
            mid = preset["color_head"]["mid_gain_30cc"]
            print(f"{key}: negative, 30CC mid gains Y={mid['y']} M={mid['m']}")
        else:
            print(f"{key}: reversal — no printing stage, no colour head")
    PRESET_PATH.write_text(
        json.dumps(raw, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {PRESET_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stocks", nargs="*", default=list(STOCKS))
    ap.add_argument("--plot", type=Path,
                    default=PROJECT_ROOT / "docs" / "assets" / "film-curve-fits.png")
    ap.add_argument(
        "--augment-color-head", action="store_true",
        help="only add process/colour-head fields to the existing presets (no refit)",
    )
    args = ap.parse_args()
    if args.augment_color_head:
        return augment_color_head()

    presets = {}
    if PRESET_PATH.is_file():
        presets = json.load(open(PRESET_PATH)).get("presets", {})
    for key in args.stocks:
        stock = STOCKS[key]
        preset = fit_stock(key, stock)
        presets[key] = preset
        print(f"{key}: rms {preset['fit']['rms_stop']:.4f} stop, "
              f"max {preset['fit']['max_stop']:.4f} stop, params {preset['params']}")
        # Theatrical quotation variants for dark-surround projection chains: the
        # report carried verbatim (contract §8.1 — a quotation, not a translation).
        if PRINT_SURROUND.get(str(stock.get("print"))) == "dark":
            tkey = f"{key}_theatrical"
            tpreset = fit_stock(tkey, stock, theatrical=True)
            presets[tkey] = tpreset
            print(f"{tkey}: rms {tpreset['fit']['rms_stop']:.4f} stop, "
                  f"max {tpreset['fit']['max_stop']:.4f} stop")
    PRESET_PATH.write_text(
        json.dumps({"version": 1, "presets": presets}, indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {PRESET_PATH}")
    if args.plot:
        plot(presets, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
