# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake the film-takeover 3D LUTs (spectral rebuild stage 4).

The v1 "full" mode fed Rec.2020 channels to per-channel characteristic curves —
an RGB heuristic the review correctly refused to call a reconstruction. This
bake runs the honest chain offline and samples it:

    plain scene Rec.2020 (D65, daylight-balanced; prefeed bypassed in full)
      -> constrained observer inverse (3x3, fitted on the rawtoaces training
         reflectances under D55: the film's spectral log-sensitivities against
         the same stimuli's Rec.2020 tristimulus; tristimulus input cannot
         recover spectra, so the metamer error of this step is a DECLARED
         residual, minimized over real materials rather than assumed away)
      -> per-layer log exposure (anchored so the neutral axis reproduces the
         fitter's neutral chain exactly)
      -> per-layer characteristic curves -> negative spectral density
      -> TH-KG3 print chain (negatives) or the slide viewed directly
         (reversals), with the same solved printer exposures / anchors as the
         published neutral targets
      -> XYZ -> CAT -> scene-referred Rec.2020 display-linear output.

Two crossover variants per stock, ONE baked volume: "datasheet" is the chain
verbatim (inter-layer crossover included as the data reports it), and
"neutralized" is a DIGITAL variant produced at runtime by dividing the
sampled datasheet output by a BOUNDED neutral-cast curve shipped alongside
the volume (a numeric convention, not a second physical process). The full
architecture argument — why the divisor must be bounded, why the bounded
composite must not be baked, and why bounded runtime division is exact in
the visible-stop metric — lives at the cast_b computation in bake().

The grid lives in per-channel log2 exposure (the reviewer's shaper):

    u_c = (log2(E_c / 0.18) - EV_MIN) / (EV_MAX - EV_MIN)

so shadow precision is not squandered on the linear top end. Outside the
domain the runtime clamps u — beyond EV_MAX the print sits on Dmin/Dmax where
the response is flat, and below EV_MIN it is film-base black.

LUT input is PLAIN scene Rec.2020 by declaration (schema 3): the observer
inverse is fitted on plain Rec.2020 stimuli, so the LUT owns the film's
spectral separation itself — the runtime BYPASSES the film scene-transform
(prefeed) in full mode. Feeding prefed pixels applied the film observer
twice (review batch 9's input-domain finding).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT))

import fit_film_curve as ff  # noqa: E402
import spectral_base as sb  # noqa: E402

TRAINING_CSV = PROJECT_ROOT / "dngscan_assets" / "spectral" / "rawtoaces_training_reflectance.csv"
OUT_DIR = PROJECT_ROOT / "dngscan" / "data" / "full_lut"

LUT_N = 65
EV_MIN, EV_MAX = -12.0, 8.0
SCENE_MID = 0.18
_LUMA_2020 = ff._LUMA_2020


def _d55_spd(wl: np.ndarray) -> np.ndarray:
    import calibrate_skin_matrix as csm

    return csm.illuminant_spd("D55", wl)


def _training_set(wl: np.ndarray) -> np.ndarray:
    raw = np.loadtxt(TRAINING_CSV, delimiter=",", skiprows=1)
    src_wl, refl = raw[:, 0], raw[:, 1:]
    out = np.stack(
        [np.interp(wl, src_wl, refl[:, i]) for i in range(refl.shape[1])], axis=1
    )
    return np.clip(out, 0.0, None)  # [len(wl), n_samples]


def observer_matrix(stock: dict) -> np.ndarray:
    """Constrained observer inverse A: scene Rec.2020 -> per-layer exposure.

    Least squares over the training reflectances under D55, with the exposure
    columns produced by the film's own spectral sensitivities and the stimulus
    columns by the CIE observer through the pipeline's daylight balance
    (von Kries D55 white -> Rec.2020 D65 via Bradford). Rows are scaled so the
    perfect reflector's exposure is exactly A @ [1,1,1] — the neutral anchor
    the LUT's exposure normalization relies on. Rows are true 3-variable
    active-set NNLS followed by the white anchor; the 5-fold CV refits with
    the IDENTICAL procedure (NNLS + anchor) so observer_cv_p99_stop measures
    the deployed algorithm, not an unanchored cousin of it.
    """
    import calibrate_skin_matrix as csm

    neg = ff._load_spectral(stock["negative"])
    wl = neg["wl"]
    d55 = _d55_spd(wl)
    refl = _training_set(wl)  # [W, S]
    refl = np.concatenate([np.ones((wl.size, 1)), refl], axis=1)  # 白板作锚
    stim = refl * d55[:, None]  # [W, S]
    exposures = sb.trapezoid(stim[:, :, None] * neg["sens"][:, None, :], wl, axis=0)
    cmf = csm.cie_1931_cmf(sb.intersect_grid(wl))
    keep = np.isin(wl, sb.intersect_grid(wl))
    xyz = sb.trapezoid(stim[keep][:, :, None] * cmf[:, None, :], wl[keep], axis=0)
    white_xyz = xyz[0]
    xyz = xyz / max(float(white_xyz[1]), 1e-12)
    m = sb.XYZ_TO_REC2020 @ sb.bradford_cat(white_xyz / max(float(white_xyz[1]), 1e-12))
    rgb = xyz @ m.T  # [S, 3]，白板 -> [1,1,1]

    def _fit_observer(rgb_fit: np.ndarray, exp_fit: np.ndarray) -> np.ndarray:
        """The production procedure, whole: per-layer NNLS + white anchor.

        rgb_fit/exp_fit must carry the white board at row 0 — it is both an
        NNLS training row and the anchor target.
        """
        a_fit = np.stack(
            [_nnls_3(rgb_fit, exp_fit[:, layer]) for layer in range(3)], axis=0
        )
        for layer in range(3):
            a_fit[layer] *= exp_fit[0, layer] / max(
                float(a_fit[layer] @ rgb_fit[0]), 1e-12
            )
        return a_fit

    a = _fit_observer(rgb, exposures)
    pred = rgb @ a.T
    resid = np.abs(np.log10(np.maximum(pred[1:], 1e-9) / np.maximum(exposures[1:], 1e-9)))
    # Leave-out cross validation on a deterministic 5-fold split: the fit must
    # not owe its residual to memorizing the training set. Each fold runs the
    # SAME NNLS + white-anchor procedure as the deployed matrix (the white
    # board stays in every fold — it is the anchor, not a held-out sample).
    n = rgb.shape[0] - 1
    cv = []
    for fold in range(5):
        mask = np.ones(n, dtype=bool)
        mask[fold::5] = False
        keep = np.concatenate(([True], mask))
        hold = np.concatenate(([False], ~mask))
        a_cv = _fit_observer(rgb[keep], exposures[keep])
        p_cv = rgb[hold] @ a_cv.T
        cv.append(np.abs(np.log10(
            np.maximum(p_cv, 1e-9) / np.maximum(exposures[hold], 1e-9)
        )))
    cv_p99 = float(np.percentile(np.concatenate(cv).ravel(), 99)) / np.log10(2.0)
    return a, float(np.percentile(resid, 99)) / np.log10(2.0), cv_p99


def _nnls_3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True non-negative least squares for a 3-variable row (active set).

    Clip-and-rescale of the unconstrained solution is NOT NNLS — the review
    called it, and with only three variables the exact active-set enumeration
    is eight subproblems.
    """
    best, best_cost = None, np.inf
    for mask in range(1, 8):
        idx = [i for i in range(3) if mask >> i & 1]
        sub, *_ = np.linalg.lstsq(x[:, idx], y, rcond=None)
        if np.any(sub < 0.0):
            continue
        row = np.zeros(3)
        row[idx] = sub
        cost = float(np.sum((x @ row - y) ** 2))
        if cost < best_cost:
            best, best_cost = row, cost
    if best is None:
        best = np.zeros(3)
    return best


class _Chain:
    """The stock's solved viewing chain, generalized to arbitrary dye stacks."""

    def __init__(self, stock: dict, theatrical: bool):
        self.stock = stock
        self.neg = ff._load_spectral(stock["negative"])
        self.wl = self.neg["wl"]
        self.reversal = bool(stock.get("positive"))
        if self.reversal:
            exp = ff.surround_exponent("dark")
            # Medium-native calibration: zero viewing flare (the display /
            # projection room is not a film property — see fit_film_curve's
            # DISPLAY_* constants for the future view-simulation layer).
            self.view = lambda reflect, white: ff._display_rec2020(
                reflect, white, self.wl, self.neg["viewing"], 0.0, exp
            )
            self.white = ff._stack_reflectance(
                self.neg, np.nanmin(self.neg["amounts"], axis=0)[None, :]
            )[0]
            # Reversal anchors (exposure shift + mid gains) from the published
            # neutral build — the LUT must sit in the same anchored frame.
            ev, channels, y, _floor = ff._build_reversal_target(stock)
            self._anchor_ev = ev
            base = self.view(
                ff._stack_reflectance(self.neg, self.neg["amounts"]), self.white
            )
            raw_y = base @ _LUMA_2020
            order = np.argsort(raw_y)
            self.e0 = float(np.interp(SCENE_MID, raw_y[order], (self.neg["le"] / ff.LOG10_2)[order]))
            mid = np.array([
                float(np.interp(self.e0, self.neg["le"] / ff.LOG10_2, base[:, c]))
                for c in range(3)
            ])
            self.gain = SCENE_MID / np.maximum(mid, 1e-9)
        else:
            self.chain = ff._solved_print_chain(
                stock, "native" if theatrical else None
            )
            paper = ff._regrid(ff._load_spectral(stock["print"]), self.wl)
            self.paper = paper
            enlarger = sb.th_kg3_spd(self.wl)
            self.print_weight = paper["sens"] * enlarger[:, None]
            self.paper_white = ff._stack_reflectance(
                paper, np.nanmin(paper["amounts"], axis=0)[None, :]
            )[0]

    def develop_amounts(self, neg_amounts: np.ndarray) -> np.ndarray:
        """Negative dye amounts [N,3] -> viewed Rec.2020 [N,3]."""
        if self.reversal:
            reflect = ff._stack_reflectance(self.neg, neg_amounts)
            rgb = self.view(reflect, self.white)
            return np.maximum(rgb * self.gain[None, :], 1e-7)
        t_neg = ff._stack_reflectance(self.neg, neg_amounts)
        log_ep = np.log10(np.maximum(
            sb.trapezoid(
                t_neg[:, :, None] * self.print_weight[None, :, :], self.wl, axis=1
            ),
            1e-12,
        ))
        dye = np.stack([
            np.interp(
                log_ep[:, c] + self.chain.q[c],
                self.paper["le"],
                self.paper["amounts"][:, c],
            )
            for c in range(3)
        ], axis=1)
        reflect = ff._stack_reflectance(self.paper, dye)
        return np.maximum(
            ff._display_rec2020(
                reflect, self.paper_white, self.wl, self.paper["viewing"],
                self.chain.flare, self.chain.exp,
            ),
            1e-7,
        )


def _layer_log_exposure(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-layer log10 exposure, neutral-anchored: gray ramps map exactly to
    the profile's logE axis (logE = scene EV * log10(2))."""
    mid = a @ np.full(3, SCENE_MID)
    e = np.maximum(rgb, 1e-9) @ a.T
    return np.log10(np.maximum(e, 1e-12) / np.maximum(mid, 1e-12)[None, :])


def _neg_amounts_at(neg: dict, log_e: np.ndarray, e_offset: float = 0.0) -> np.ndarray:
    return np.stack([
        np.interp(log_e[:, c] + e_offset * ff.LOG10_2, neg["le"], neg["amounts"][:, c])
        for c in range(3)
    ], axis=1)


def chain_eval(
    stock: dict, chain: _Chain, a: np.ndarray, rgb: np.ndarray, stage_a=None
) -> np.ndarray:
    """The full offline chain for arbitrary PLAIN scene Rec.2020 samples
    (prefeed bypassed in full mode; the observer inverse owns separation).
    ``stage_a`` (rgb -> per-layer log10 exposure) overrides the observer
    product so route-C field stocks bake oracles through the SAME Stage A
    the runtime dispatches to; None keeps the legacy observer path."""
    log_e = stage_a(rgb) if stage_a is not None else _layer_log_exposure(rgb, a)
    offset = chain.e0 if chain.reversal else 0.0
    amounts = _neg_amounts_at(chain.neg, log_e, offset)
    return chain.develop_amounts(amounts)


def bake(stock_key: str, theatrical: bool = False) -> dict:
    stock = ff.STOCKS[stock_key.removesuffix("_theatrical")]
    chain = _Chain(stock, theatrical)
    a, observer_p99, observer_cv_p99 = observer_matrix(stock)

    axis = EV_MIN + (EV_MAX - EV_MIN) * np.arange(LUT_N) / (LUT_N - 1)
    ev = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    rgb = SCENE_MID * np.exp2(ev)
    out = chain_eval(stock, chain, a, rgb)

    # Neutral ramp: the self-consistency gate and the neutralized division —
    # applied HERE, at full precision, before any quantization.
    ramp_ev = np.linspace(EV_MIN, EV_MAX, 257)
    ramp_rgb = SCENE_MID * np.exp2(ramp_ev)[:, None].repeat(3, axis=1)
    ramp_out = chain_eval(stock, chain, a, ramp_rgb)
    ramp_y = ramp_out @ _LUMA_2020
    cast = ramp_out / np.maximum(ramp_y, 1e-9)[:, None]

    # The neutralized variant is NOT a second baked volume. It is the
    # datasheet volume divided AT RUNTIME by a bounded neutral-cast curve
    # indexed at the sample's LUMINANCE exposure EV_Y (the same single-axis
    # declaration as the colour head; a per-channel-exposure divisor
    # re-imported the retired channels-as-layer-exposures reading and blew up
    # off-axis on hard reversals). Three measured facts force this shape:
    #   1. The divisor must be BOUNDED. The raw cast of a hard reversal is
    #      near-singular (Kodachrome's deep gray has ZERO green — its floor
    #      is magenta), and dividing by it blew off-axis samples up by whole
    #      stops no matter where the division happened (runtime: +4.3 EV
    #      Ektachrome / ~13 EV Kodachrome; baked at full precision: 4.4 /
    #      1.6 EV — both measured).
    #   2. The bounded composite out(x)/g(EV_Y(x)) must NOT be baked either:
    #      EV_Y is diagonal to the grid, and where g swings steeply (the
    #      Kodachrome green divisor moves 4x across ~2 EV in the shadow
    #      transition) the 65-grid cannot represent the kink — the baked
    #      composite measured 1.73 EV worst off-axis against its own oracle.
    #   3. With a bounded divisor, runtime division is EXACT in the visible-
    #      stop metric: the denominator is evaluated per pixel, so the
    #      quotient's log error equals the datasheet volume's log error
    #      (~0.03 stop). The original review refuted runtime division of the
    #      quantized LUT by the UNBOUNDED high-precision cast; the root sin
    #      there was the unbounded divisor, which is what the clip removes.
    # Bounding rule, in the CORRECTION-MULTIPLIER domain. The first cut —
    # clip the cast per channel then luma-renormalize — was measured to break
    # its own bound: the renormalizing scalar pushed channels back outside
    # (Ektachrome shipped a 0.081 divisor = +3.6 EV of gain). The honest
    # construction: let h* = 1/cast be the full neutralizing multiplier and
    # walk the straight line h(t) = 1 + t*(h* - 1) from identity toward full
    # correction, taking the largest t in [0,1] with every channel's h_i
    # inside [0.25, 4]. Because luma(cast) = 1 and luma(cast * h*) = 1, EVERY
    # point on that line preserves the neutral axis' luminance exactly — tone
    # never moves for any t — and the shipped divisor 1/h is truly bounded to
    # [0.25, 4] per channel. Where the medium's cast is inside the bound the
    # correction is complete (t = 1: strict neutrality); where a channel is
    # near-singular (Kodachrome's zero-green floor), t collapses toward 0 and
    # the floor is kept as medium character rather than half-chased.
    h_star = 1.0 / np.maximum(cast, 1e-9)
    t_hi = np.where(h_star > 4.0, 3.0 / np.maximum(h_star - 1.0, 1e-9), 1.0)
    t_lo = np.where(h_star < 0.25, 0.75 / np.maximum(1.0 - h_star, 1e-9), 1.0)
    t_ev = np.clip(np.min(np.minimum(t_hi, t_lo), axis=1), 0.0, 1.0)
    h = 1.0 + t_ev[:, None] * (h_star - 1.0)
    cast_b = 1.0 / np.clip(h, 0.25, 4.0)  # clip is a no-op by construction
    # The shipped curve is sampled on the LUT'S OWN 65-node axis and the
    # runtime interpolates it LINEARLY: on the neutral axis tetrahedral
    # interpolation degenerates to 1-D linear interpolation along the
    # diagonal, so the quotient of the two linear interpolants is a weighted
    # mediant of node-exact values — monotone between node tones, no
    # overshoot. A continuous 257-point denominator against the discretized
    # numerator measured a 0.55 EV tone spike in Kodachrome's cast
    # transition zone; node-matched sampling removes the mismatch class.
    assert (ramp_ev.size - 1) % (LUT_N - 1) == 0
    step = (ramp_ev.size - 1) // (LUT_N - 1)
    cast_ev = ramp_ev[::step].astype(np.float32)
    cast_b = cast_b[::step].astype(np.float32)

    # Oracle fixtures: random in-domain samples with float64-chain truth for
    # BOTH variants, shipped inside the npz so the runtime test compares the
    # exact deployed bytes against the offline chain — not against itself.
    rng = np.random.default_rng(20260806)
    oracle_ev = rng.uniform(-9.0, 5.0, (96, 3))
    oracle_rgb = SCENE_MID * np.exp2(oracle_ev)
    oracle_ds = chain_eval(stock, chain, a, oracle_rgb)
    oracle_nz = oracle_ds.copy()
    oracle_ev_y = np.log2(np.maximum(oracle_rgb @ _LUMA_2020, 1e-9) / SCENE_MID)
    for c in range(3):
        # Same f32 curve, same nodes, same linear interp the runtime uses.
        g = np.interp(oracle_ev_y, cast_ev, cast_b[:, c].astype(np.float64))
        oracle_nz[:, c] = oracle_nz[:, c] / g

    return {
        "datasheet": out.reshape(LUT_N, LUT_N, LUT_N, 3).astype(np.float32),
        "cast_ev": cast_ev,
        "cast_bounded": cast_b,
        "observer": a.astype(np.float64),
        "observer_p99_stop": observer_p99,
        "observer_cv_p99_stop": observer_cv_p99,
        "ramp_ev": ramp_ev.astype(np.float32),
        "ramp_y": ramp_y.astype(np.float32),
        "ramp_cast": cast.astype(np.float32),
        "oracle_ev": oracle_ev.astype(np.float32),
        "oracle_datasheet": oracle_ds.astype(np.float32),
        "oracle_neutralized": oracle_nz.astype(np.float32),
        "chain": chain,
        "a": a,
    }


def _quant_error_stop(volume_f32: np.ndarray) -> float:
    """Worst per-sample error introduced by the float16 quantization alone."""
    q = volume_f32.astype(np.float16).astype(np.float32)
    vis = volume_f32 > 5e-3
    if not np.any(vis):
        return 0.0
    return float(np.abs(np.log2(
        np.maximum(q[vis], 1e-9) / np.maximum(volume_f32[vis], 1e-9)
    )).max())


def write_lut(stock_key: str, theatrical: bool = False) -> None:
    """One baked datasheet volume + the bounded cast curve, oracles inside."""
    baked = bake(stock_key, theatrical)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{stock_key}_theatrical" if theatrical else stock_key
    quant = {"datasheet": _quant_error_stop(baked["datasheet"])}
    np.savez_compressed(
        OUT_DIR / f"{key}.npz",
        lut_datasheet=baked["datasheet"].astype(np.float16),
        cast_ev=baked["cast_ev"],
        cast_bounded=baked["cast_bounded"],
        ev_min=np.float32(EV_MIN),
        ev_max=np.float32(EV_MAX),
        n=np.int32(LUT_N),
        ramp_ev=baked["ramp_ev"],
        ramp_cast=baked["ramp_cast"],
        observer=baked["observer"],
        observer_p99_stop=np.float32(baked["observer_p99_stop"]),
        observer_cv_p99_stop=np.float32(baked["observer_cv_p99_stop"]),
        quant_err_datasheet_stop=np.float32(quant["datasheet"]),
        oracle_ev=baked["oracle_ev"],
        oracle_datasheet=baked["oracle_datasheet"],
        oracle_neutralized=baked["oracle_neutralized"],
        input_space=np.asarray("scene_rec2020"),
        schema=np.int32(3),
    )
    size = (OUT_DIR / f"{key}.npz").stat().st_size / 1024
    print(f"{key}: observer p99 {baked['observer_p99_stop']:.3f} "
          f"(cv {baked['observer_cv_p99_stop']:.3f}) stop; "
          f"quant ds {quant['datasheet']:.4f}; "
          f"{size:.0f} KiB")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*", default=None)
    args = ap.parse_args()
    keys = args.stocks or list(ff.STOCKS)
    for key in keys:
        stock = ff.STOCKS[key]
        write_lut(key, theatrical=False)
        if not stock.get("positive") and ff.PRINT_SURROUND.get(str(stock.get("print"))) == "dark":
            write_lut(key, theatrical=True)
