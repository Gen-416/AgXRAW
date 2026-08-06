# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake the film-takeover 3D LUTs (spectral rebuild stage 4).

The v1 "full" mode fed Rec.2020 channels to per-channel characteristic curves —
an RGB heuristic the review correctly refused to call a reconstruction. This
bake runs the honest chain offline and samples it:

    post-prefeed scene Rec.2020 (D65, daylight-balanced)
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

Two variants per stock: "datasheet" (the chain verbatim — inter-layer
crossover included as the data reports it) and "neutralized" (a DIGITAL
variant: each output channel divided by the neutral ramp's cast at that
channel's own input exposure, so grays stay strictly neutral; this is a
numeric convention, not a second physical process).

The grid lives in per-channel log2 exposure (the reviewer's shaper):

    u_c = (log2(E_c / 0.18) - EV_MIN) / (EV_MAX - EV_MIN)

so shadow precision is not squandered on the linear top end. Outside the
domain the runtime clamps u — beyond EV_MAX the print sits on Dmin/Dmax where
the response is flat, and below EV_MIN it is film-base black.

LUT input is POST-PREFEED Rec.2020 by declaration: the runtime applies the
material-aware prefeed as usual and the LUT must not embed it again.
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
    the LUT's exposure normalization relies on. Negative entries are clipped
    and the row rescaled (declared projection; measured residual reported).
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
    a_rows = []
    for layer in range(3):
        row, *_ = np.linalg.lstsq(rgb, exposures[:, layer], rcond=None)
        row = np.maximum(row, 0.0)
        scale = exposures[0, layer] / max(float(row @ rgb[0]), 1e-12)
        a_rows.append(row * scale)
    a = np.stack(a_rows, axis=0)
    pred = rgb @ a.T
    resid = np.abs(np.log10(np.maximum(pred[1:], 1e-9) / np.maximum(exposures[1:], 1e-9)))
    return a, float(np.percentile(resid, 99)) / np.log10(2.0)


class _Chain:
    """The stock's solved viewing chain, generalized to arbitrary dye stacks."""

    def __init__(self, stock: dict, theatrical: bool):
        self.stock = stock
        self.neg = ff._load_spectral(stock["negative"])
        self.wl = self.neg["wl"]
        self.reversal = bool(stock.get("positive"))
        if self.reversal:
            exp = ff.surround_exponent("dark")
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


def chain_eval(stock: dict, chain: _Chain, a: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """The full offline chain for arbitrary post-prefeed Rec.2020 samples."""
    log_e = _layer_log_exposure(rgb, a)
    offset = chain.e0 if chain.reversal else 0.0
    amounts = _neg_amounts_at(chain.neg, log_e, offset)
    return chain.develop_amounts(amounts)


def bake(stock_key: str, theatrical: bool = False) -> dict:
    stock = ff.STOCKS[stock_key.removesuffix("_theatrical")]
    chain = _Chain(stock, theatrical)
    a, observer_p99 = observer_matrix(stock)

    axis = EV_MIN + (EV_MAX - EV_MIN) * np.arange(LUT_N) / (LUT_N - 1)
    ev = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    rgb = SCENE_MID * np.exp2(ev)
    out = chain_eval(stock, chain, a, rgb).astype(np.float32)

    # Neutral ramp for the neutralized variant AND the self-consistency gate.
    ramp_ev = np.linspace(EV_MIN, EV_MAX, 257)
    ramp_rgb = SCENE_MID * np.exp2(ramp_ev)[:, None].repeat(3, axis=1)
    ramp_out = chain_eval(stock, chain, a, ramp_rgb)
    ramp_y = ramp_out @ _LUMA_2020
    cast = ramp_out / np.maximum(ramp_y, 1e-9)[:, None]

    return {
        "datasheet": out.reshape(LUT_N, LUT_N, LUT_N, 3),
        "observer": a.astype(np.float64),
        "observer_p99_stop": observer_p99,
        "ramp_ev": ramp_ev.astype(np.float32),
        "ramp_y": ramp_y.astype(np.float32),
        "ramp_cast": cast.astype(np.float32),
        "chain": chain,
        "a": a,
    }


def write_lut(stock_key: str, theatrical: bool = False) -> None:
    """One datasheet LUT + the neutral-ramp cast curve per stock.

    The "neutralized" crossover variant is DERIVED at runtime — each output
    channel divided by the neutral cast at that channel's own input exposure —
    so it costs a 257-point curve instead of a second 65^3 volume.
    """
    baked = bake(stock_key, theatrical)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{stock_key}_theatrical" if theatrical else stock_key
    np.savez_compressed(
        OUT_DIR / f"{key}.npz",
        lut=baked["datasheet"].astype(np.float16),
        ev_min=np.float32(EV_MIN),
        ev_max=np.float32(EV_MAX),
        n=np.int32(LUT_N),
        ramp_ev=baked["ramp_ev"],
        ramp_cast=baked["ramp_cast"],
        observer=baked["observer"],
        observer_p99_stop=np.float32(baked["observer_p99_stop"]),
        input_space=np.asarray("post-prefeed_rec2020"),
    )
    size = (OUT_DIR / f"{key}.npz").stat().st_size / 1024
    print(f"{key}: observer p99 {baked['observer_p99_stop']:.3f} stop; "
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
