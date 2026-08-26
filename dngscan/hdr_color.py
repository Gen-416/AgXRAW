# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR colour geometry around the native extended-white AgX formation.

The HDR curve is the sole brightness authority. Colour is chosen between two formations
of the same inset RGB: a reference-white AgX response supplies the conservative common
chroma path, while the extended-white AgX response supplies the native per-channel path.
Both candidates are aligned to the extended response's luminance before `rho` mixes them:

    common = reference * Y_native / Y_reference
    result = normalize_Y((1-rho)*common + rho*native, Y_native)

No smootherstep or second tone gain remains. `rho` changes only chromaticity, CFA masks
can withdraw per-channel freedom, and the native HDR curve always owns luminance.

The gamut fit is a separate function from the SDR one on purpose. The SDR fitter targets
[0,1]; an HDR rendition legitimately lives in [0, R_display], and reusing the SDR fitter
would clip away exactly the range this whole phase exists to produce.
"""
from __future__ import annotations

from typing import Any

from ._deps import np
from .constants import REC2020_LUMA, RGB_TO_XYZ

_EPS = np.float32(1e-6)


def output_luma_weights(output_gamut: str) -> Any:
    """Luminance row of the output space's RGB->XYZ matrix.

    Taken from the same matrix the render uses, so the projector's "Y is preserved"
    guarantee is about the Y the pipeline actually computes, not a nominally similar one.
    """
    space = {"srgb": "sRGB", "p3": "P3"}.get(str(output_gamut), "P3")
    # Copy before normalising. RGB_TO_XYZ is shared by the frozen SDR pipeline; using an
    # ndarray view here and dividing in place would silently rewrite its P3 Y row merely
    # because the HDR module was imported.
    weights = np.array(RGB_TO_XYZ[space][1], dtype=np.float64, copy=True)
    # The repository matrices are rounded display constants. A neutral RGB triplet must
    # nevertheless have Y equal to its component value, otherwise the neutral-axis
    # projector cannot preserve Y exactly by construction.
    weights /= np.sum(weights)
    return weights.astype(np.float32)


def formation_luma_weights(outset_matrix: Any) -> Any:
    """Rec.2020 luminance row for RGB immediately before the actual AgX outset.

    darktable's outset is intentionally not the inverse of its inset: purity restoration
    and unrotation are independent controls. Deriving this row from ``inverse(inset)``
    therefore normalises rho against a transform the pixels never take.
    """
    matrix = np.asarray(outset_matrix, dtype=np.float64)
    weights = np.asarray(REC2020_LUMA, dtype=np.float64) @ matrix
    weights /= np.sum(weights)
    return weights.astype(np.float32)


def raw_gated_channel_separation(
    rho: float,
    clip_masks_rgb: Any | None,
    *,
    y_native: Any | None = None,
    peak: float = 1.0,
) -> Any:
    """Turn global colour freedom into per-pixel/channel permission from CFA evidence.

    A single clipped CFA channel loses half of its independent path at a fully soft-clipped
    site while the two measured channels remain available. Once two channels clip, the
    second-largest mask continuously withdraws all independent separation; at full
    multi-channel clipping the pixel follows the common luminance path only.

    Peak-proximity convergence (two-route doctrine, 2026-08-26): when ``y_native``
    (formation-point luminance of the native path) and a ``peak`` above reference
    white are supplied, clip-compromised pixels additionally converge to the
    common path as their rendered luminance climbs from reference white (1.0 on
    the formation scale) to the content peak — luminance stays the native curve's,
    chroma authority shrinks to zero exactly where the RAW stopped measuring it.
    Unclipped pixels are untouched at any luminance, so fully reliable highlights
    keep their HDR chroma and material separation.
    """
    base = np.float32(np.clip(float(rho), 0.0, 1.0))
    if clip_masks_rgb is None:
        return base
    masks = np.clip(np.asarray(clip_masks_rgb, dtype=np.float32), 0.0, 1.0)
    second = np.partition(masks, 1, axis=-1)[..., 1]
    channel_permission = np.float32(1.0) - np.float32(0.5) * masks
    multi_permission = np.float32(1.0) - second
    permission = base * channel_permission * multi_permission[..., None]
    if y_native is None or float(peak) <= 1.0:
        return permission
    proximity = np.clip(
        (np.asarray(y_native, dtype=np.float32) - np.float32(1.0))
        / np.float32(float(peak) - 1.0),
        0.0,
        1.0,
    )
    clipness = np.max(masks, axis=-1)
    return permission * (
        np.float32(1.0) - proximity * clipness
    )[..., None]


def blend_native_hdr_paths(
    reference_formation_rgb: Any,
    native_formation_rgb: Any,
    rho: Any,
    luma_weights: Any,
) -> Any:
    """Mix conservative and native HDR chroma paths at one authoritative luminance.

    ``reference`` and ``native`` are both HDR-branch calculations from the same inset
    scene RGB. The former uses a 1.0 endpoint only to define path-to-white chromaticity;
    it never contributes a tone target. The latter uses the solved extended endpoint and
    defines Y. Scalar or CFA-gated per-channel rho is allowed; the final normalization
    prevents either form from becoming an implicit brightness control.
    """
    reference = np.asarray(reference_formation_rgb, dtype=np.float32)
    native = np.asarray(native_formation_rgb, dtype=np.float32)
    if reference.shape != native.shape:
        raise ValueError("reference and native HDR formation shapes must match")
    if bool(np.array_equal(reference, native)):
        return native

    w = np.asarray(luma_weights, dtype=np.float32)
    y_native = np.tensordot(native, w, axes=([-1], [0]))
    y_reference = np.tensordot(reference, w, axes=([-1], [0]))
    valid_reference = y_reference > _EPS
    common_scale = y_native / np.maximum(y_reference, _EPS)
    common = reference * common_scale[..., None]
    common = np.where(valid_reference[..., None], common, native)

    r = np.clip(np.asarray(rho, dtype=np.float32), 0.0, 1.0)
    if r.ndim > 0 and r.shape == y_native.shape:
        r = r[..., None]
    if bool(np.all(r <= 0.0)):
        return common.astype(np.float32, copy=False)
    if bool(np.all(r >= 1.0)):
        return native

    proposal = (np.float32(1.0) - r) * common + r * native
    y_proposal = np.tensordot(proposal, w, axes=([-1], [0]))
    scale = y_native / np.maximum(y_proposal, _EPS)
    mixed = proposal * scale[..., None]
    valid = (y_native > _EPS) & (y_proposal > _EPS)
    return np.where(valid[..., None], mixed, native).astype(np.float32, copy=False)


def neutral_axis_lambda(rgb: Any, peak: float, luma_weights: Any) -> Any:
    """Largest linear-RGB opponent scale keeping every channel inside [0, peak].

    Solves the bound directly rather than clipping per channel. Per-channel clipping moves
    a colour along whichever axis happened to overflow. Scaling the vector from the neutral
    axis keeps linear Y and the output-RGB opponent direction when Y is in range. That is a
    stable RGB geometry, but it is not a claim of constant perceptual hue: a CAM such as the
    JMh space used by ACES 2 is required for that stronger guarantee.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    w = np.asarray(luma_weights, dtype=np.float32)
    y_raw = np.tensordot(arr, w, axes=([-1], [0]))[..., None]
    limit = np.float32(peak)
    y = np.clip(y_raw, 0.0, limit)
    c = arr - y_raw

    big = np.float32(np.inf)
    neg = np.where(c < 0.0, y / np.maximum(-c, _EPS), big)
    pos = np.where(c > 0.0, (limit - y) / np.maximum(c, _EPS), big)
    lam = np.minimum(np.min(neg, axis=-1), np.min(pos, axis=-1))
    return np.clip(np.minimum(lam, np.float32(1.0)), 0.0, 1.0)


def fit_hdr_color_volume(rgb: Any, peak: float, output_gamut: str = "p3") -> Any:
    """Bring an HDR rendition inside [0, peak] along its linear-RGB neutral-axis ray.

    Y is preserved when it lies inside the display volume. If Y itself is below zero or
    above peak no in-volume colour can preserve it, so the neutral anchor is clipped to the
    nearest endpoint and the RGB opponent vector collapses as needed. Perceptual hue is only
    approximate here; this intentionally remains much simpler than ACES 2 JMh compression.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    limit = np.float32(peak)
    # Leave in-gamut pixels strictly untouched. Reconstructing them as y + 1.0*(arr - y)
    # is exact in real arithmetic but still introduces needless float32 roundoff.
    needs_fit = np.any((arr < 0.0) | (arr > limit), axis=-1)
    if not bool(np.any(needs_fit)):
        return arr

    w = output_luma_weights(output_gamut)
    y_raw = np.tensordot(arr, w, axes=([-1], [0]))[..., None]
    y = np.clip(y_raw, 0.0, limit)
    lam = neutral_axis_lambda(arr, limit, w)[..., None]
    fitted = y + lam * (arr - y_raw)
    # Exact equations can miss the boundary by a few float32 ulps. This clip is a numerical
    # guard after the neutral-axis solve, not the projector itself.
    fitted = np.clip(fitted, 0.0, limit)
    return np.where(needs_fit[..., None], fitted, arr)
