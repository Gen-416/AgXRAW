# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-band joint-chroma candidate; deliberately outside the render pipeline.

The supplied covariance describes independent pixels on this exact grid, or
the caller supplies already-propagated full band covariance. No RAW decoder
or two-dimensional PSD is inferred. Image structure only reduces shrinkage.
"""
from __future__ import annotations

import math
import numpy as np

from .chroma_nr import LUMA_W, _atrous_smooth, atrous_levels_for


def fixed_phase_psd_models(spectrum, iso, projection, *, phases,
                           levels=(0, 1, 2), assume_separable=False,
                           assume_independent_phases=False):
    """Bridge measured phase PSD to both same-band test operators.

    ``projection`` is a caller-proven constant map from the measured phase
    samples to this exact scene grid. It is not inferred from camera metadata
    and does not describe adaptive demosaicing. Complete normalized h/v PSD,
    one measured ISO, Parseval evidence and explicit independence/separability
    assumptions are required by the underlying numerical prototype.
    """
    from .spectral_variance import measured_phase_detail_variance
    if not assume_independent_phases:
        raise ValueError("phase independence must be explicit")
    names = tuple(phases)
    if not names or len(set(names)) != len(names):
        raise ValueError("distinct measured phases required")
    matrix = np.asarray(projection, dtype=np.float64)
    if matrix.shape != (3, len(names)) or not np.isfinite(matrix).all():
        raise ValueError("fixed phase projection must be finite 3-by-phase")
    bands = {phase: measured_phase_detail_variance(
        spectrum, phase, iso, levels=levels, assume_separable=assume_separable)
        for phase in names}
    p = np.eye(3) - np.ones((3, 1)) * LUMA_W.astype(float)[None, :] / LUMA_W.sum()
    covariance, diagonal = {}, {}
    for level in levels:
        phase_var = np.asarray([bands[phase][level] for phase in names])
        cov = (matrix * phase_var[None, :]) @ matrix.T
        if not np.isfinite(cov).all() or np.max(np.abs(cov)) > np.finfo(np.float32).max:
            raise ValueError("projected band covariance overflows")
        covariance[level] = cov
        diagonal[level] = np.diag(p @ cov @ p.T).astype(np.float32)
    return {"detail_covariance": covariance, "detail_variance": diagonal,
            "status": "experimental-fixed-linear-separable",
            "decoder_authorized": False,
            "assumptions": ("stationary-separable-PSD", "independent-phase-noise",
                            "caller-proven-constant-projection-on-identical-grid")}


def _chroma_model(covariance, max_condition):
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("joint candidate requires a finite constant 3x3 band covariance")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-16):
        raise ValueError("band covariance must be symmetric")
    eigen = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigen))), np.finfo(float).tiny)
    if eigen[0] < -1e-12 * scale:
        raise ValueError("band covariance must be positive semidefinite")
    if not np.any(matrix):
        return None
    w = LUMA_W.astype(np.float64)
    # Normalize the production float32 row to make the nullspace orthogonal
    # to exactly the same row, including its small sum-rounding error.
    projection = np.eye(3) - np.ones((3, 1)) * w[None, :] / w.sum()
    basis = np.linalg.svd(w[None, :], full_matrices=True)[2][1:].T
    c2 = basis.T @ projection @ matrix @ projection.T @ basis
    values, vectors = np.linalg.eigh(c2)
    if values[0] <= 0 or values[1] / values[0] > max_condition:
        raise ValueError("singular or ill-conditioned chroma covariance")
    axes = basis @ vectors
    return axes, values


def joint_multiscale_correction(scene, amount, decimation_factor=1., *,
                               noise_covariance=None, detail_covariance=None,
                               decimation_axis_factors=None, valid_mask=None,
                               max_condition=1e6, max_removed_fraction=.5):
    """Bounded joint garrote on exactly the production coarse B3 bands.

    Each coefficient's Mahalanobis energy uses full 2D chroma covariance.
    A B3 local energy excess only protects structure; it never raises the
    physical variance. The removed vector fraction is additionally bounded.
    Missing bands pass through. Singular nonzero covariance is rejected,
    rather than made invertible with invented diagonal variance.

    This candidate has no established superiority or real-camera preset and
    must not be selected by ``resolve_chroma_policy``.
    """
    dec = np.asarray(scene, dtype=np.float32)
    if dec.ndim != 3 or dec.shape[-1] != 3 or not np.isfinite(dec).all():
        raise ValueError("scene must be finite HxWx3")
    if (not math.isfinite(amount) or not 0 <= amount <= 1 or
            not math.isfinite(max_condition) or max_condition < 1 or
            not math.isfinite(max_removed_fraction) or not 0 < max_removed_fraction <= 1):
        raise ValueError("invalid joint shrinkage contract")
    if noise_covariance is not None and detail_covariance is not None:
        raise ValueError("provide independent-grid or propagated-band covariance, not both")
    if amount == 0 or (noise_covariance is None and detail_covariance is None):
        return np.zeros_like(dec)
    included = set(atrous_levels_for(decimation_factor, axis_factors=decimation_axis_factors))
    if not included:
        return np.zeros_like(dec)
    models = {}
    if noise_covariance is not None:
        model = _chroma_model(noise_covariance, max_condition)
        models = {level: model for level in included}
    else:
        if not isinstance(detail_covariance, dict):
            raise ValueError("band covariance must be keyed by B3 level")
        for level, covariance in detail_covariance.items():
            if type(level) is not int or not 0 <= level <= 12:
                raise ValueError("invalid band level")
            models[level] = _chroma_model(covariance, max_condition)
    if not any(model is not None for model in models.values()):
        return np.zeros_like(dec)
    valid = np.ones(dec.shape[:2], bool)
    if valid_mask is not None:
        source = np.asarray(valid_mask)
        if source.shape != valid.shape or not np.isfinite(source).all():
            raise ValueError("validity mask must match the scene grid")
        valid = source > .999999
    total_removed = np.zeros_like(dec)
    smooth = dec
    max_step = max((min(dec.shape[:2]) - 1) // 2, 1)
    for level in range(max(included) + 1):
        if (1 << level) > max_step:
            break
        coarser = _atrous_smooth(smooth, level)
        valid = _atrous_smooth((~valid).astype(np.float32), level) == 0
        if level in included and models.get(level) is not None:
            axes, values = models[level]
            detail = smooth - coarser
            coords = np.einsum("...i,ia->...a", detail, axes).astype(np.float32)
            if noise_covariance is not None:
                from .noise_propagation import atrous_detail_variance
                energy = atrous_detail_variance(np.ones(valid.shape), level)
            else:
                # Propagated constant band covariance already includes the
                # exact fixed operator. Boundary authority belongs to caller.
                energy = np.ones(valid.shape, np.float32)
            usable = valid & (energy > 0)
            norm2 = np.sum(np.square(coords) / values, axis=-1)
            norm2 = np.divide(norm2, energy, out=np.zeros_like(norm2), where=usable)
            local_energy = _atrous_smooth(norm2.astype(np.float32), 0)
            signal_energy = np.maximum(local_energy - 2., 0.) / 2.
            threshold2 = 9. * amount ** 2 / np.maximum(signal_energy, 1.)
            fraction = np.minimum(max_removed_fraction, threshold2 / (threshold2 + norm2 / 2. + 1e-30))
            fraction = np.where(usable, fraction, 0)
            total_removed += np.einsum("...a,ia->...i", coords * fraction[..., None], axes).astype(np.float32)
        smooth = coarser
    correction = -total_removed
    correction -= (correction @ LUMA_W)[..., None] / np.sum(LUMA_W)
    return correction
