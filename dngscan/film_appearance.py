# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference-print appearance layer: plan object and asset contract
(FILM_APPEARANCE_RECIPE_PLAN §9/§11, phase P1).

P1 ships the INFRASTRUCTURE and nothing visible: the immutable plan, the
fail-closed recipe loader, and the strict fast paths. The palette kernel
itself is P2; until it lands, only IDENTITY recipes (every field zero) are
accepted, and a non-identity asset refuses to load rather than silently
doing nothing — an asset that promises a look and delivers a no-op would be
the quiet version of that lie.

Contracts that hold from day one:

- `technical` mode never touches an asset, never allocates, and returns the
  developed frame object unchanged — the P1 exit gate is that every frozen
  byte in the repo stays identical.
- The compiled `FilmAppearancePlan` is the ONLY thing the runtime consults
  (the A3 lesson, inherited here rather than re-learned): recipe arrays ride
  the plan as immutable payload, resolved and hash-verified at compile time.
- Unknown schema, wrong stock/medium pairing, bad hash, non-finite or
  mis-shaped fields: ValueError at load, never a default.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

APPEARANCE_SCHEMA = 1
APPEARANCE_DIR = Path(__file__).with_name("data") / "film_appearance"
MANIFEST_PATH = APPEARANCE_DIR / "MANIFEST.json"

APPEARANCE_MODES = ("technical", "reference")

# §6.6 axes the P2 kernel will interpolate over. Declared here so the loader
# can validate shapes before any kernel exists to consume them.
EV_KNOTS = (-6.0, -3.0, 0.0, 3.0, 6.0)
HUE_KNOT_COUNT = 24
STRENGTH_MAX = 1.5


@dataclass(frozen=True)
class FilmAppearancePlan:
    """Immutable compiled appearance state (plan §11).

    `recipe` holds the loaded, validated field arrays for reference mode —
    resolved at compile so the runtime never touches the filesystem or any
    mutable registry. Arrays are wrapped read-only.
    """

    mode: str = "technical"
    recipe_id: str = ""
    strength: float = 1.0
    provenance: str = ""
    asset_sha256: str = ""
    recipe: dict | None = field(default=None, compare=False)


_RECIPE_CACHE: dict[tuple, dict] = {}


def _manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise ValueError(
            f"外观层资产清单缺失:{MANIFEST_PATH}(fail closed,不使用未钉扎资产)"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def recipe_path(recipe_id: str) -> Path:
    return APPEARANCE_DIR / f"{recipe_id}.npz"


def load_recipe(recipe_id: str, *, stock_id: str, medium_id: str) -> dict:
    """Load and validate one recipe asset, fail-closed on every axis.

    The (stock, medium) pairing is part of the asset's IDENTITY: a recipe
    authored against Portra-on-Endura silently applied to another stock
    would be exactly the unaccountable LUT the plan exists to avoid.
    """
    rid = str(recipe_id)
    path = recipe_path(rid)
    if not path.is_file():
        raise ValueError(f"外观 recipe '{rid}' 不存在:{path}")
    stat = path.stat()
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    # The pairing is part of the cache key: identity validation must run
    # for every DISTINCT claim, or a correct earlier load would let a wrong
    # stock ride the cache past the checks (caught by P1's own gate test).
    cache_key = (rid, stat.st_mtime_ns, sha, str(stock_id), str(medium_id))
    got = _RECIPE_CACHE.get(cache_key)
    if got is not None:
        return got

    pinned = _manifest().get("files", {}).get(path.name)
    if pinned != sha:
        raise ValueError(
            f"外观 recipe '{rid}' 哈希不匹配清单(资产被改动或清单未再生)"
        )

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(np.asarray(z["meta"])))
        if int(meta.get("schema", -1)) != APPEARANCE_SCHEMA:
            raise ValueError(
                f"外观 recipe schema {meta.get('schema')} != {APPEARANCE_SCHEMA}"
            )
        for key, expect in (
            ("recipe_id", rid), ("stock_id", str(stock_id)),
            ("medium_id", str(medium_id)),
        ):
            if str(meta.get(key)) != expect:
                raise ValueError(
                    f"外观 recipe 身份不符:{key}={meta.get(key)!r} != {expect!r}"
                )
        if str(meta.get("process_space")) != "display-linear-rec2020/oklab+scene-ev":
            raise ValueError(
                f"外观 recipe 处理空间未知:{meta.get('process_space')!r}"
            )
        provenance = str(meta.get("provenance", ""))
        if provenance not in ("editorial-authored", "empirical-own-target"):
            raise ValueError(f"外观 recipe provenance 未知:{provenance!r}")
        declared_neutral = str(meta.get("neutralization_policy", "print-balanced"))
        if declared_neutral not in (
            "technical-neutral", "print-balanced", "native",
        ):
            raise ValueError(
                f"外观 recipe 中性化声明未知:{declared_neutral!r}"
            )

        ev = np.asarray(z["ev_knots"], dtype=np.float64)
        hue = np.asarray(z["hue_knots_deg"], dtype=np.float64)
        if ev.shape != (len(EV_KNOTS),) or not np.allclose(ev, EV_KNOTS):
            raise ValueError("外观 recipe EV 轴与声明网格不符")
        if hue.shape != (HUE_KNOT_COUNT,) or not np.allclose(
            hue, np.arange(HUE_KNOT_COUNT) * (360.0 / HUE_KNOT_COUNT)
        ):
            raise ValueError("外观 recipe hue 轴与声明网格不符")

        fields: dict = {}
        for name, shape in (
            ("hue_delta_deg", (len(EV_KNOTS), HUE_KNOT_COUNT)),
            ("log_chroma_gain", (len(EV_KNOTS), HUE_KNOT_COUNT)),
            ("density_ev", (len(EV_KNOTS), HUE_KNOT_COUNT)),
            ("neutral_bias_ab", (len(EV_KNOTS), 2)),
        ):
            arr = np.asarray(z[name], dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"外观 recipe 字段 {name} 形状 {arr.shape} != {shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"外观 recipe 字段 {name} 含非有限值")
            arr.setflags(write=False)
            fields[name] = arr

        # Kernel scalars (§6.2/§6.3): richness shoulder knee/power and the
        # neutral-protection knee. Bounded fail-closed — an absurd knee is a
        # data error, not a style.
        chroma_knee = float(meta.get("chroma_knee", 0.18))
        chroma_power = float(meta.get("chroma_power", 2.0))
        neutral_c0 = float(meta.get("neutral_chroma_c0", 0.03))
        if not 0.05 <= chroma_knee <= 0.6:
            raise ValueError(f"chroma_knee={chroma_knee} 域为 [0.05, 0.6]")
        if not 1.0 <= chroma_power <= 4.0:
            raise ValueError(f"chroma_power={chroma_power} 域为 [1, 4]")
        if not 0.01 <= neutral_c0 <= 0.08:
            raise ValueError(f"neutral_chroma_c0={neutral_c0} 域为 [0.01, 0.08]")

        # Identity detection decides the strict fast path: an all-zero
        # recipe must stay BYTE-identical (the P1 exit gate), and the Oklab
        # round trip alone would cost that. Any non-zero field engages the
        # kernel.
        is_identity = all(float(np.abs(fields[k]).max()) == 0.0 for k in fields)

        out = {
            "meta": meta,
            "provenance": provenance,
            "sha256": sha,
            "chroma_knee": chroma_knee,
            "chroma_power": chroma_power,
            "neutral_c0": neutral_c0,
            "is_identity": is_identity,
            "neutralization_policy": declared_neutral,
            **fields,
            # PCHIP derivatives along the EV axis, precomputed at load so the
            # per-pixel evaluation is pure gathers (§6.6: monotone C1 on EV,
            # no overshoot past the knots).
            **{
                f"d_{k}": _pchip_derivatives(np.asarray(EV_KNOTS), fields[k])
                for k in ("hue_delta_deg", "log_chroma_gain", "density_ev")
            },
        }
    _RECIPE_CACHE.clear()
    _RECIPE_CACHE[cache_key] = out
    return out


def compile_appearance_plan(
    mode: str, strength: float, *, stock_id: str, medium_id: str
) -> FilmAppearancePlan:
    """Resolve the user's appearance selection into the immutable plan."""
    mode = str(mode or "technical")
    if mode not in APPEARANCE_MODES:
        raise ValueError(f"film_appearance={mode!r} 未知(可选 technical/reference)")
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= STRENGTH_MAX:
        raise ValueError(
            f"film_appearance_strength={strength!r} 域为 [0, {STRENGTH_MAX}]"
        )
    if mode == "technical":
        return FilmAppearancePlan(mode="technical")
    rid = f"{stock_id}__{medium_family(medium_id)}_reference_v1"
    recipe = load_recipe(rid, stock_id=stock_id, medium_id=medium_id)
    return FilmAppearancePlan(
        mode="reference",
        recipe_id=rid,
        strength=strength,
        provenance=recipe["provenance"],
        asset_sha256=recipe["sha256"],
        recipe=recipe,
    )


def medium_family(medium_id: str) -> str:
    """Collapse a concrete medium asset id to its recipe family name.

    Recipes are authored per stock x medium FAMILY (endura, 2383, direct):
    the translated/native variants of one paper share a palette intent, and
    a per-variant recipe would multiply authoring work without a measured
    reason. The concrete id still participates in load-time identity via
    meta.medium_id.
    """
    m = str(medium_id)
    if "endura" in m:
        return "endura"
    if "2383" in m or "2393" in m:
        return "print2383"
    if m.startswith("direct"):
        return "direct"
    return m or "unknown"


# --------------------------------------------------------------------------
# P2 palette kernel (plan §6)
# --------------------------------------------------------------------------

def _pchip_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone-cubic knot derivatives along axis 0.

    §6.6 requires the EV axis to be C1 with clamped overshoot; PCHIP gives
    both by construction — the interpolant never leaves the hull of its
    bracketing knots, so a recipe cannot manufacture a value its author
    never wrote. y is [K, ...]; returns d of the same shape.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = np.diff(x).reshape((-1,) + (1,) * (y.ndim - 1))
    delta = np.diff(y, axis=0) / h
    d = np.zeros_like(y)
    # interior: harmonic mean where slopes share a sign, else zero
    s_prev, s_next = delta[:-1], delta[1:]
    same = (s_prev * s_next) > 0.0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        harm = (w1 + w2) / (w1 / np.where(same, s_prev, 1.0)
                            + w2 / np.where(same, s_next, 1.0))
    d[1:-1] = np.where(same, harm, 0.0)
    # ends: one-sided, clamped to preserve monotonicity (Fritsch-Carlson)
    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    d0 = np.where(np.sign(d0) != np.sign(delta[0]), 0.0, d0)
    d0 = np.where(
        (np.sign(delta[0]) != np.sign(delta[1])) & (np.abs(d0) > 3.0 * np.abs(delta[0])),
        3.0 * delta[0], d0,
    )
    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    dn = np.where(np.sign(dn) != np.sign(delta[-1]), 0.0, dn)
    dn = np.where(
        (np.sign(delta[-1]) != np.sign(delta[-2])) & (np.abs(dn) > 3.0 * np.abs(delta[-1])),
        3.0 * delta[-1], dn,
    )
    d[0], d[-1] = d0, dn
    return d


def _grid_coefficients(e: np.ndarray, hue_deg: np.ndarray) -> tuple:
    """Shared per-pixel interpolation coefficients for every field.

    One hue (periodic Catmull-Rom over the 24 uniform knots — the 345->0
    seam is an ordinary interval) and one EV bracket (monotone Hermite) are
    computed ONCE per chunk; each field evaluation is then 16 gathers and
    fused multiply-adds. Computing these per field tripled the kernel cost.
    """
    H = HUE_KNOT_COUNT
    ev = np.asarray(EV_KNOTS, dtype=np.float32)
    step = np.float32(360.0 / H)
    hf = hue_deg / step
    base = np.floor(hf)
    t = (hf - base).astype(np.float32)
    j1 = base.astype(np.int64) % H
    j0, j2, j3 = (j1 - 1) % H, (j1 + 1) % H, (j1 + 2) % H
    t2 = t * t
    t3 = t2 * t
    w0 = np.float32(-0.5) * t3 + t2 - np.float32(0.5) * t
    w1 = np.float32(1.5) * t3 - np.float32(2.5) * t2 + np.float32(1.0)
    w2 = np.float32(-1.5) * t3 + np.float32(2.0) * t2 + np.float32(0.5) * t
    w3 = np.float32(0.5) * t3 - np.float32(0.5) * t2

    ec = np.clip(e, ev[0], ev[-1])
    i = np.clip(np.searchsorted(ev, ec, side="right") - 1, 0, ev.size - 2)
    dx = (ev[i + 1] - ev[i]).astype(np.float32)
    u = ((ec - ev[i]) / dx).astype(np.float32)
    u2 = u * u
    u3 = u2 * u
    h00 = np.float32(2.0) * u3 - np.float32(3.0) * u2 + np.float32(1.0)
    h10 = (u3 - np.float32(2.0) * u2 + u) * dx
    h01 = np.float32(-2.0) * u3 + np.float32(3.0) * u2
    h11 = (u3 - u2) * dx
    return (j0, j1, j2, j3, w0, w1, w2, w3, i, h00, h10, h01, h11)


def _sample_field(f: np.ndarray, d: np.ndarray, coef: tuple) -> np.ndarray:
    """Evaluate one [K, H] field at the shared per-pixel coefficients."""
    j0, j1, j2, j3, w0, w1, w2, w3, i, h00, h10, h01, h11 = coef
    i1 = i + 1

    def row(table, ridx):
        return (table[ridx, j0] * w0 + table[ridx, j1] * w1
                + table[ridx, j2] * w2 + table[ridx, j3] * w3)

    return (
        h00 * row(f, i) + h10 * row(d, i)
        + h01 * row(f, i1) + h11 * row(d, i1)
    )


def apply_film_appearance(developed: np.ndarray, plan: FilmAppearancePlan) -> np.ndarray:
    """Apply the palette kernel to mapped Rec.2020 (post-neutralization,
    pre-delivery). Plan §6, all of it pointwise — the streamed band path and
    the full-frame oracle share it with no full-frame temporary.

    Strict fast paths: technical, strength 0 and IDENTITY recipes return the
    same object (the P1 byte gate depends on it — the Oklab round trip alone
    would cost byte identity). The engaged path runs float32 end to end with
    the colour matrices pre-fused; measured at ~37% of the film core on a
    1 MP chunk (float64 first cut: 66%); the 10% target belongs to the P6
    native kernel — this NumPy path stays the correctness oracle.
    """
    if plan is None or plan.mode == "technical" or plan.strength == 0.0:
        return developed
    recipe = plan.recipe
    if recipe is None:
        raise ValueError("reference 模式的 plan 缺少已解析 recipe(编译期错误)")
    if recipe["is_identity"]:
        return developed

    rgb = np.asarray(developed, dtype=np.float32).reshape(-1, 3)
    strength = np.float32(plan.strength)
    m_fwd, m_inv = _fused_oklab_matrices()

    y = np.maximum(rgb @ _LUMA_REC2020, np.float32(1e-9))
    e = np.log2(y / np.float32(0.18))
    lab = np.cbrt(rgb @ m_fwd.T) @ _OKLAB_M2_F32.T
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    C = np.hypot(a, b)
    hdeg = np.degrees(np.arctan2(b, a)).astype(np.float32) % np.float32(360.0)

    c0 = np.float32(recipe["neutral_c0"])
    C2 = C * C
    w_c = C2 / (C2 + c0 * c0)
    ck = np.float32(recipe["chroma_knee"])
    cp = float(recipe["chroma_power"])
    cr = C / ck
    if cp == 2.0:      # the common case: float32 pow is ~10x a multiply
        shoulder = cr * cr
    elif cp == 1.0:
        shoulder = cr
    else:
        shoulder = cr ** np.float32(cp)
    r_sh = np.float32(1.0) / (np.float32(1.0) + shoulder)

    coef = _grid_coefficients(e, hdeg)
    dh = _sample_field(recipe["hue_delta_deg"], recipe["d_hue_delta_deg"], coef)
    gc = _sample_field(recipe["log_chroma_gain"], recipe["d_log_chroma_gain"], coef)
    dd = _sample_field(recipe["density_ev"], recipe["d_density_ev"], coef)

    sw = strength * w_c
    h_new = np.radians(hdeg + dh * sw)
    C_new = C * np.exp2(gc * r_sh * sw)
    # §6.4: darken LUMINANCE by dd EV holding the (a, b) direction — under
    # Oklab's Y^(1/3) homogeneity an exact L scale of 2^(-dd/3).
    L_new = L * np.exp2(dd * sw * np.float32(-1.0 / 3.0))

    a_new = C_new * np.cos(h_new)
    b_new = C_new * np.sin(h_new)
    nb = recipe["neutral_bias_ab"]
    if float(np.abs(nb).max()) > 0.0:
        ev_axis = np.asarray(EV_KNOTS, dtype=np.float32)
        ec = np.clip(e, ev_axis[0], ev_axis[-1])
        a_new += strength * np.interp(ec, ev_axis, nb[:, 0]).astype(np.float32)
        b_new += strength * np.interp(ec, ev_axis, nb[:, 1]).astype(np.float32)

    lms_ = np.stack([L_new, a_new, b_new], axis=1) @ _OKLAB_M2_INV_F32.T
    out = (lms_ * lms_ * lms_) @ m_inv.T
    # The chain contract downstream is non-negative mapped Rec.2020; the
    # opponent reconstruction can go negative for extreme recipes — clamped
    # here, and the authoring probes measure the pre-clamp share so a recipe
    # buying its look at the floor stays visible (§6.6).
    np.maximum(out, np.float32(0.0), out=out)
    return out.reshape(np.shape(developed))


_LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
_OKLAB_M2_F32 = None
_OKLAB_M2_INV_F32 = None
_FUSED = None


def _fused_oklab_matrices() -> tuple:
    """Rec.2020 -> LMS and LMS' -> Rec.2020 as single fused 3x3 float32
    matrices, built once — two of the four matmuls per call disappear."""
    global _FUSED, _OKLAB_M2_F32, _OKLAB_M2_INV_F32
    if _FUSED is None:
        from .constants import (
            OKLAB_M1, OKLAB_M1_INV, OKLAB_M2, OKLAB_M2_INV, RGB_TO_XYZ, XYZ_TO_RGB,
        )

        fwd = (np.asarray(OKLAB_M1) @ np.asarray(RGB_TO_XYZ["Rec2020"])).astype(np.float32)
        inv = (np.asarray(XYZ_TO_RGB["Rec2020"]) @ np.asarray(OKLAB_M1_INV)).astype(np.float32)
        _OKLAB_M2_F32 = np.asarray(OKLAB_M2, dtype=np.float32)
        _OKLAB_M2_INV_F32 = np.asarray(OKLAB_M2_INV, dtype=np.float32)
        _FUSED = (fwd, inv)
    return _FUSED
