# SPDX-License-Identifier: GPL-3.0-or-later
"""Display-side colour operators of the observe film engine.

Both run on the AgX core's display-linear Rec.2020 output.

Hue rendition (AgXFilm ``tools/hue_fit.py`` / ``observe_hue.py``): per stock,
fitted against the frozen full chain, in Oklab of the base output

    h' = h + dh(h, L, C),  C' = C * k(h, L, C),  L' = L + dL_per_C(h, L, C) * C

with design F(h) (x) [1, l, l^2, c] (F: Fourier order K in hue, l = L - L_MID,
c = C - C_MID). l and c are clamped to the fit domain and the outputs to
|dh| <= 30 deg, |log k| <= 0.5: no polynomial extrapolation.

Grey-scale crossover (AgXFilm ``tools/observe_crossover.py``): the full chain's
own per-channel cast of the grey scale, print-balanced (mid grey neutral, both
ends keep their tint), indexed by the SCENE pixel's luminance exposure
EV_Y = log2(Y/0.18); luminance is preserved so the tone curve stays exact.
"""
from __future__ import annotations

import numpy as np

REC2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)
# Rec.2020 (D65) -> XYZ, ITU-R BT.2020.
_REC2020_TO_XYZ = np.array([
    [0.636958048301291, 0.144616903586208, 0.168880975164172],
    [0.262700212011267, 0.677998071518871, 0.059301716469862],
    [0.000000000000000, 0.028072693049087, 1.060985057710791],
])
# Oklab (Ottosson 2020), XYZ-based M1 and M2.
_M1 = np.array([[0.8189330101, 0.3618667424, -0.1288597137],
                [0.0329845436, 0.9293118715, 0.0361456387],
                [0.0482003018, 0.2643662691, 0.6338517070]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_A = _M1 @ _REC2020_TO_XYZ
_A_INV = np.linalg.inv(_A)
_M2_INV = np.linalg.inv(_M2)
MAX_DH_DEG = 30.0
MAX_LOGK = 0.5


class HueRendition:
    def __init__(self, meta: dict, coef: dict):
        self.k = int(meta["K"])
        self.l_mid = float(meta["L_MID"])
        self.c_mid = float(meta["C_MID"])
        self.dom_l = tuple(meta["fit_domain"]["l"])
        self.dom_c = tuple(meta["fit_domain"]["c"])
        self.dh = np.asarray(coef["dh_deg"], dtype=np.float64)
        self.lk = np.asarray(coef["log_k"], dtype=np.float64)
        self.dl = np.asarray(coef["dL_per_C"], dtype=np.float64)
        n = (2 * self.k + 1) * 4
        if not (self.dh.size == self.lk.size == self.dl.size == n):
            raise ValueError("film hue asset: coefficient count does not match its basis")

    def _design(self, h, L, C):
        cols = [np.ones_like(h)]
        for i in range(1, self.k + 1):
            cols += [np.cos(i * h), np.sin(i * h)]
        f = np.stack(cols, axis=1)
        l = np.clip(L - self.l_mid, *self.dom_l)[:, None]
        c = np.clip(C - self.c_mid, *self.dom_c)[:, None]
        return np.concatenate([f, f * l, f * l * l, f * c], axis=1)

    def apply(self, rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
        x = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
        lab = np.cbrt(x @ _A.T) @ _M2.T
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
        C = np.hypot(a, b)
        h = np.arctan2(b, a)
        D = self._design(h, L, C)
        dh = np.clip(D @ self.dh, -MAX_DH_DEG, MAX_DH_DEG) * strength
        lk = np.clip(D @ self.lk, -MAX_LOGK, MAX_LOGK) * strength
        dl = (D @ self.dl) * C * strength
        h2 = h + np.deg2rad(dh)
        C2 = C * np.exp(lk)
        L2 = np.maximum(L + dl, 0.0)
        lab2 = np.stack([L2, C2 * np.cos(h2), C2 * np.sin(h2)], axis=1)
        out = ((lab2 @ _M2_INV.T) ** 3) @ _A_INV.T
        return np.maximum(out, 0.0).astype(np.float32)


class GreyCrossover:
    def __init__(self, table: dict):
        self.ev = np.asarray(table["ev"], dtype=np.float64)
        self.ratio = np.asarray(table["ratio"], dtype=np.float64)
        if self.ratio.shape != (self.ev.size, 3) or np.any(np.diff(self.ev) <= 0):
            raise ValueError("film crossover asset: malformed table")

    def apply(self, out: np.ndarray, scene: np.ndarray, strength: float = 1.0) -> np.ndarray:
        out = np.asarray(out, dtype=np.float32)
        y_s = np.maximum(np.asarray(scene, dtype=np.float64) @ REC2020_LUMA, 1e-8)
        ev_y = np.log2(y_s / 0.18)
        g = np.stack([np.interp(ev_y, self.ev, self.ratio[:, c]) for c in range(3)], axis=1)
        if strength != 1.0:
            g = np.power(g, strength)
        mapped = out * g.astype(np.float32)
        y0 = out.astype(np.float64) @ REC2020_LUMA
        y1 = mapped.astype(np.float64) @ REC2020_LUMA
        k = np.where(y1 > 1e-9, y0 / np.maximum(y1, 1e-9), 1.0)
        return (mapped * k[:, None].astype(np.float32)).astype(np.float32)
