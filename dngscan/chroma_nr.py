# SPDX-License-Identifier: GPL-3.0-or-later
"""Chroma-only noise reduction: remove low-frequency colour mottle, keep
everything else (2026-08-31, owner-approved work item).

WHAT THIS IS. The LibRaw decode path applies no noise reduction at all and
the Core Image path is pinned to its least-smoothed calibrated end, so the
sensor's own noise flows through the pipeline as honest texture — the
retained luminance noise reads as grain, and that is deliberate. What does
NOT read as texture is the LOW-FREQUENCY chroma mottle: magenta/green
blotches tens of pixels across in high-ISO shadows, born from per-channel
noise integrated over the CFA at scales no demosaic can reach. This
operator removes exactly that band and nothing else. It is classified as
DIGITIZATION REPAIR (the same ethical slot as highlight reconstruction and
clip retreat): the mottle is a sampling artefact, not optical information.

WHAT IT MUST NOT TOUCH, by construction rather than by tuning:

- LUMINANCE, at the scene stage: the correction is projected to the
  zero-luma subspace (P = I - 1·w^T with the Rec.2020 luma row w), so the
  grain carried by scene Y is untouched to float precision. Bilinear
  upsampling is linear, so the projection survives the map's trip to full
  resolution. Every downstream tone core is per-channel nonlinear, so a
  zero-Y scene change still moves display luminance slightly — the
  promise is about what the operator touches, not the output.
- FINE CHROMA SPECKLE: the analysis runs on the DECIMATED spread grid
  (film_optics.spread_grid_shape, <= 1408/2048 on the long side), so all
  chroma structure finer than a decimated cell — including the film-like
  coloured graininess worth keeping — never enters the operator at all.
- LARGE-SCALE REAL COLOUR: the à-trous residual (everything coarser than
  the last detail level) passes through unshrunk. A plain lowpass
  subtraction would eat genuine colour gradients along with the mottle;
  the wavelet split is what makes "去斑不去色" a structural property.
- HIGH-AMPLITUDE CHROMA EDGES: within the detail levels the garrote
  removes the fraction 1/(1 + (d/T)²) of each coefficient d against the
  level's own robust noise floor (T = amount·3·σ_MAD): 94% at |d| = T/4,
  50% at |d| = T, 20% at 2T, and only ~T²/|d| of a strong edge; the
  absolute bite never exceeds T/2. A real colour boundary's coefficients
  sit far above T.

The realized band is octave-aligned to within √2 of [BAND_LO_PX,
BAND_HI_PX] (atrous_levels_for); on grids too small to reflect-pad a
level (min side < 2·2^k + 1) the cascade stops early and the band is
truncated at the top — only tiny renders, and silently.

Amount 0 is a strict identity — callers keep the no-context fast path and
never call in here.
"""
from __future__ import annotations

from ._deps import np

# Rec.2020 luma row — the projection axis. Kept local so the operator has
# no import-time dependency on the render modules that call it.
LUMA_W = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float32)

# The shrunk band, declared in FULL-RESOLUTION SENSOR PIXELS — the
# mottle's native coordinate (it is a CFA sampling artefact, so its scale
# rides the sensor grid, not the output size). Structure finer than
# BAND_LO_PX is the kept texture — pixel speckle and the film-like fine
# coloured graininess; structure coarser than BAND_HI_PX is treated as
# real colour and passes through in the residual. Both bounds are
# modelled choices, not measurements.
BAND_LO_PX = 8.0
BAND_HI_PX = 128.0
_SQRT2 = 2.0 ** 0.5
_B3 = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0


def atrous_levels_for(decimation_factor: float) -> tuple[int, ...]:
    """Which à-trous levels on the decimated grid fall inside the declared
    full-resolution band. Level k shrinks detail between hole spacings
    2^k and 2^(k+1) cells, i.e. factor·2^k .. factor·2^(k+1) sensor px —
    include it while the level's geometric centre factor·2^k·√2 lies in
    [BAND_LO_PX, BAND_HI_PX]. The realized band is octave-aligned, so it
    can only be declared to within a factor √2: it always lies INSIDE
    [BAND_LO_PX/√2, BAND_HI_PX·√2] and always COVERS [BAND_LO_PX·√2,
    BAND_HI_PX/√2] (the lower edge bounded below by the grid's own cell
    when the decimation factor exceeds BAND_LO_PX·√2). (Review batch 23: the earlier any-overlap rule let the
    top level reach 2·BAND_HI_PX at some decimation factors — a memory
    tier could silently widen the declared band.) At identity grids (small
    renders) the first levels fall below BAND_LO_PX and are skipped, which
    is what keeps pixel-scale speckle untouched there."""
    factor = max(float(decimation_factor), 1.0)
    levels = []
    for k in range(13):
        centre = factor * (2.0 ** k) * _SQRT2
        if centre > BAND_HI_PX:
            break
        if centre >= BAND_LO_PX:
            levels.append(k)
    return tuple(levels)


# Shrinkage scale: T_level = amount * K * sigma_level, sigma from the
# level's own median absolute deviation (MAD / 0.6745). The shrink is the
# Wiener-flavoured garrote removed = d·T²/(T²+d²): a noise-consistent
# coefficient (|d| ≲ T) is removed almost entirely, while a strong real
# structure loses only ~T²/|d| — unlike a hard/soft threshold's constant
# bite, which measurably desaturated a large uniform colour patch by tens
# of percent over the level cascade (probe 2026-08-31).
_THRESHOLD_K = 3.0
_MAD_TO_SIGMA = 1.0 / 0.6745


def _atrous_smooth(plane: np.ndarray, level: int) -> np.ndarray:
    """One à-trous B3 smoothing pass with hole spacing 2**level, separable,
    reflect-padded. plane is [h, w, c] float32."""
    step = 1 << level
    h, w = plane.shape[:2]
    out = plane
    for axis, n in ((0, h), (1, w)):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (2 * step, 2 * step)
        padded = np.pad(out, pad, mode="reflect")
        acc = np.zeros_like(out)
        for j, kv in enumerate(_B3):
            sl = [slice(None)] * out.ndim
            offset = j * step
            sl[axis] = slice(offset, offset + n)
            acc += np.float32(kv) * padded[tuple(sl)]
        out = acc
    return out


def chroma_correction_map(
    scene_dec: np.ndarray,
    amount: float,
    decimation_factor: float = 1.0,
) -> np.ndarray:
    """The zero-luma correction to ADD to the scene, on the decimated grid.

    scene_dec: [dh, dw, 3] area-decimated scene-linear Rec.2020 (signed —
    this is a colour-path repair, not a light-transport source, so it sees
    the same signed coordinates the colour path does).
    decimation_factor: full-resolution pixels per decimated cell (long-side
    ratio) — anchors the shrunk band in sensor pixels (atrous_levels_for).

    Returns [dh, dw, 3] float32 with w·map == 0 per pixel (float
    precision): callers upsample bilinearly and add.
    """
    if amount <= 0.0:
        raise ValueError("chroma_correction_map is only defined for amount > 0")
    dec = np.asarray(scene_dec, dtype=np.float32)
    y = dec @ LUMA_W
    chroma = dec - y[..., None]
    del dec, y

    # Multiplicative domain would be ill-defined at y <= 0; the additive
    # opponent form keeps the operator linear-in-signal and lets shadows —
    # where the mottle lives — carry proportionally small absolute
    # corrections bounded by their own chroma.
    total_removed = np.zeros_like(chroma)
    max_step = max((min(chroma.shape[:2]) - 1) // 2, 1)
    included = set(atrous_levels_for(decimation_factor))
    top = max(included) if included else -1
    # The cascade always runs from level 0 — an à-trous level's hole
    # spacing only avoids aliasing on the PROGRESSIVELY smoothed image —
    # but only the in-band levels shrink; protected fine levels pass
    # through untouched inside their detail coefficients.
    #
    # Review batch 23 (memory): the cascade runs ONE CHANNEL AT A TIME.
    # Every operation below is elementwise per channel (separable B3
    # smoothing, per-channel MAD, per-channel garrote), so the result is
    # byte-identical to the three-channel form while the transient working
    # set drops to about a third — measured 227 MiB -> ~70 MiB on the 1408
    # grid, which is what lets the pass-0 peak stay under the optics tier.
    for c in range(3):
        smooth = chroma[..., c]
        for level in range(top + 1):
            if (1 << level) > max_step:
                break  # grid too small to carry this scale's reflect pad
            coarser = _atrous_smooth(smooth, level)
            if level in included:
                detail = smooth - coarser
                # Per-level, per-channel robust noise floor. abs+median over
                # the whole grid: the mottle and noise dominate the
                # coefficient population at these scales; sparse real edges
                # do not move a MAD.
                mad = np.median(np.abs(detail))
                threshold = np.float32(
                    np.float32(float(amount) * _THRESHOLD_K * _MAD_TO_SIGMA)
                    * np.float32(mad)
                )
                t2 = np.square(threshold)
                total_removed[..., c] += detail * (
                    t2 / (t2 + np.square(detail) + np.float32(1e-30))
                )
                del detail
            smooth = coarser
        del smooth

    correction = -total_removed
    # Exact zero-luma projection: whatever numerical luma the per-channel
    # shrinkage introduced is subtracted here, so Y is preserved to float
    # precision at every pixel — the grain lives in Y and must not move.
    correction -= (correction @ LUMA_W)[..., None]
    return correction.astype(np.float32, copy=False)


def apply_chroma_correction_rows(
    rgb_rows: np.ndarray,
    correction_map: np.ndarray,
    y0: int,
    y1: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Add the upsampled correction to scene rows [y0, y1) — the streaming
    row-band form; the full-frame oracle passes (0, height). Bilinear
    upsampling is the same continuous surface everywhere, so band seams are
    zero by construction (film_optics.upsample_rows contract)."""
    from .film_optics import upsample_rows

    rows = np.asarray(rgb_rows, dtype=np.float32).reshape(y1 - y0, width, 3)
    return (
        rows + upsample_rows(correction_map, y0, y1, height, width)
    ).reshape(-1, 3)


def apply_chroma_correction_flat(
    rgb_flat: np.ndarray,
    correction_map: np.ndarray,
    start: int,
    end: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Row-band apply for a FLAT pixel chunk [start, end) that may cut rows
    mid-way (the 1M-pixel streaming chunks): upsample the covering rows and
    slice — the same continuous bilinear surface, so chunk boundaries are
    seam-free like band boundaries."""
    from .film_optics import upsample_rows

    y0 = start // width
    y1 = (end - 1) // width + 1
    up = upsample_rows(correction_map, y0, y1, height, width).reshape(-1, 3)
    flat = np.asarray(rgb_flat, dtype=np.float32).reshape(-1, 3)
    return flat + up[start - y0 * width:end - y0 * width]
