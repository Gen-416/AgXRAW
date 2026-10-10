# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded numerical prototype for measured stationary separable noise.

This module is not the production decoder covariance model. Two line spectra
do not establish a two-dimensional covariance, and callers must explicitly
accept a separable, stationary, linear-kernel approximation.
"""
from __future__ import annotations

import math

import numpy as np


def one_sided_power_mass(power, transform_length, *, difference=False):
    """Convert collector per-bin power to variance mass, not PSD density.

    The collector stores |FFT|²/(N sum(window²)) without doubling interior
    rFFT bins. DC and even-N Nyquist appear once. A-B temporal power is twice
    one exposure's power only under the independent equal-noise-pair contract.
    """
    if type(transform_length) is not int or not 2 <= transform_length <= 1_048_576:
        raise ValueError("invalid spectrum transform length")
    value = np.asarray(power, dtype=np.float64)
    if value.ndim != 1 or value.size != transform_length // 2 + 1:
        raise ValueError("spectrum bins disagree with transform length")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("spectrum power must be finite and nonnegative")
    mass = value * (1. if difference else 2.)
    mass[0] *= .5
    if transform_length % 2 == 0:
        mass[-1] *= .5
    if not np.isfinite(mass).all():
        raise ValueError("spectrum variance mass overflows")
    return mass


def sensor_frequencies(channel_frequencies, *, sample_spacing=2.):
    """A Bayer phase sample advances two sensor pixels: f_sensor=f_plane/2."""
    spacing = float(sample_spacing)
    frequencies = np.asarray(channel_frequencies, dtype=np.float64)
    if (not math.isfinite(spacing) or spacing <= 0 or
            not np.isfinite(frequencies).all() or np.any(frequencies < 0) or
            np.any(frequencies > .5)):
        raise ValueError("invalid spectrum frequency or sample spacing")
    return frequencies / spacing


def separable_atrous_detail_variance(horizontal, vertical, *, length_h, length_v,
                                     levels=(0, 1, 2), difference=True,
                                     assume_separable=False, variance=None,
                                     variance_rtol=.1):
    """Variance of fixed B3-spline à-trous bands in the measured plane grid.

    Sxy=Sx*Sy/variance is an explicit hypothesis, never inferred from h/v
    marginal measurements. It cannot restore line-to-line/DC power removed by
    the collector or describe adaptive demosaicing, warps or nonlinear steps.
    No sensor→decoder scale is guessed. Work/memory are O((Nh+Nv)*levels).
    """
    if not assume_separable:
        raise ValueError("two line spectra require an explicit separable-noise hypothesis")
    tolerance = float(variance_rtol)
    if not math.isfinite(tolerance) or not 0 <= tolerance <= .25:
        raise ValueError("invalid spectral variance reconciliation tolerance")
    ph = one_sided_power_mass(horizontal, length_h, difference=difference)
    pv = one_sided_power_mass(vertical, length_v, difference=difference)
    vh, vv = float(ph.sum()), float(pv.sum())
    if vh == 0 and vv == 0:
        if variance not in (None, 0, 0.):
            raise ValueError("zero spectra disagree with declared variance")
        total = 0.
    else:
        total = .5 * (vh + vv) if variance is None else float(variance)
        if (not math.isfinite(total) or total <= 0 or min(vh, vv) <= 0 or
                max(abs(vh / total - 1), abs(vv / total - 1)) > tolerance):
            raise ValueError("axis or declared variances disagree; missing covariance power")
        ph /= vh
        pv /= vv
    requested = tuple(levels)
    if (not requested or any(type(level) is not int or not 0 <= level <= 12 for level in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("detail levels must be distinct integers in 0..12")
    fh = np.arange(ph.size) / float(length_h)
    fv = np.arange(pv.size) / float(length_v)
    previous_h, previous_v = np.ones(ph.size), np.ones(pv.size)
    result = {}
    for level in range(max(requested) + 1):
        # B3 spline [1,4,6,4,1]/16 has real frequency response cos(pi f)^4.
        next_h = previous_h * np.cos(np.pi * (2 ** level) * fh) ** 4
        next_v = previous_v * np.cos(np.pi * (2 ** level) * fv) ** 4
        if level in requested:
            # E[(Bh_prev Bv_prev - Bh_next Bv_next)^2], factored into
            # one-dimensional sums so no Nh×Nv covariance grid is allocated.
            response = (np.dot(ph, previous_h ** 2) * np.dot(pv, previous_v ** 2)
                        - 2 * np.dot(ph, previous_h * next_h) * np.dot(pv, previous_v * next_v)
                        + np.dot(ph, next_h ** 2) * np.dot(pv, next_v ** 2))
            result[level] = float(max(0., response) * total)
        previous_h, previous_v = next_h, next_v
    return result


def separable_linear_probe_variance(horizontal, vertical, kernel, *, length_h, length_v,
                                    gain_patch=None, difference=True, assume_separable=False,
                                    variance=None, variance_rtol=.1):
    """Variance of one explicitly known linear footprint, optionally after gain.

    Coordinates are in the measured channel-plane grid. ``kernel`` can describe
    a fixed interpolation, area average or detail band. A known positive gain
    patch is applied before that kernel: covariance becomes G C G, not a
    stationary PSD times the gain at the output pixel. At most 65x65 source
    samples are admitted. No adaptive decoder, extrapolation or border rule is
    inferred; the caller supplies an interior footprint and its exact weights.
    This research API is not called by the production RAW pipeline.
    """
    if not assume_separable:
        raise ValueError("two line spectra require an explicit separable-noise hypothesis")
    tolerance = float(variance_rtol)
    if not math.isfinite(tolerance) or not 0 <= tolerance <= .25:
        raise ValueError("invalid spectral variance reconciliation tolerance")
    weights = np.asarray(kernel, dtype=np.float64)
    if (weights.ndim != 2 or not min(weights.shape) >= 1 or max(weights.shape) > 65 or
            not np.isfinite(weights).all()):
        raise ValueError("linear probe requires a finite kernel with support at most 65x65")
    if gain_patch is not None:
        gain = np.asarray(gain_patch, dtype=np.float64)
        if gain.shape not in ((), weights.shape) or not np.isfinite(gain).all() or np.any(gain <= 0):
            raise ValueError("linear probe gain must be positive and match its footprint")
        with np.errstate(over="ignore", invalid="ignore"):
            weights = weights * gain
        if not np.isfinite(weights).all():
            raise ValueError("linear probe gain-weight product overflows")
    ph = one_sided_power_mass(horizontal, length_h, difference=difference)
    pv = one_sided_power_mass(vertical, length_v, difference=difference)
    vh, vv = float(ph.sum()), float(pv.sum())
    if vh == 0 and vv == 0:
        if variance not in (None, 0, 0.):
            raise ValueError("zero spectra disagree with declared variance")
        return 0.
    total = .5 * (vh + vv) if variance is None else float(variance)
    if (not math.isfinite(total) or total <= 0 or min(vh, vv) <= 0 or
            max(abs(vh / total - 1), abs(vv / total - 1)) > tolerance):
        raise ValueError("axis or declared variances disagree; missing covariance power")
    ph, pv = ph / vh, pv / vv

    def covariance(power, length, support):
        frequencies = np.arange(power.size) / float(length)
        # One lag at a time bounds scratch storage; no frequency x footprint²
        # or two-dimensional spectral grid is allocated.
        correlations = np.asarray([np.dot(power, np.cos(2 * np.pi * lag * frequencies))
                                   for lag in range(support)])
        indexes = np.arange(support)
        return correlations[np.abs(indexes[:, None] - indexes[None, :])]

    cy = covariance(pv, length_v, weights.shape[0])
    cx = covariance(ph, length_h, weights.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = float(total * np.sum(weights * (cy @ weights @ cx)))
    if not math.isfinite(predicted):
        raise ValueError("linear probe variance overflows")
    return max(0., predicted)


def measured_phase_detail_variance(spectrum, phase, iso, *, levels=(0, 1, 2),
                                   assume_separable=False, allow_hann=False,
                                   require_parseval=True, variance_rtol=.1):
    """Use retained h/v powers at one measured ISO, under explicit hypotheses.

    No extrapolation or interpolation of spectra, no guessed missing axis, and
    no implicit mapping through the adaptive RAW decoder are performed.
    """
    from .noise_spectrum import validate_spectrum

    value = validate_spectrum(spectrum)
    if value.get("mapping", {}).get("status") not in ("bayer", "single-colour"):
        raise ValueError("measured phase spectrum needs a verified position mapping")
    powers = []
    lengths = []
    for axis in ("h", "v"):
        record = value.get("axes", {}).get(axis, {})
        if record.get("normalization_status") != "declared" or record.get("transform_length") is None:
            raise ValueError("complete h/v spectrum normalization is unavailable")
        window = record.get("window")
        if window not in ("none", "rectangular") and not (window == "hann" and allow_hann):
            raise ValueError("windowed PSD needs an explicit window approximation")
        measurement = next((r for r in record.get("measurements", ()) if r["iso"] == iso), None)
        sample = (measurement or {}).get("phases", {}).get(phase, {})
        if not sample.get("diff_power"):
            raise ValueError("phase difference spectrum is unavailable at this measured ISO")
        status = sample.get("integration_status")
        if status in ("reference-mismatch", "incomplete-contract", "window-unavailable"):
            raise ValueError("phase spectrum integration is not qualified")
        if require_parseval and status not in ("verified-rectangular", "windowed-reference-comparison"):
            raise ValueError("phase spectrum has no corresponding within-line variance reference")
        if require_parseval and "parseval_ratio_diff" not in sample:
            raise ValueError("phase difference spectrum has no corresponding variance reference")
        if (window == "hann" and "parseval_ratio_diff" in sample and
                abs(sample["parseval_ratio_diff"] - 1.) > variance_rtol):
            raise ValueError("windowed spectrum disagrees with the within-line variance reference")
        powers.append(sample["diff_power"])
        lengths.append(record["transform_length"])
    return separable_atrous_detail_variance(
        powers[0], powers[1], length_h=lengths[0], length_v=lengths[1], levels=levels,
        difference=True, assume_separable=assume_separable, variance_rtol=variance_rtol)


def project_independent_phase_bands(phase_bands, projection, *, assume_independent=False):
    """Explicit fixed-linear bridge to chroma_nr's detail_variance input.

    Columns of ``projection`` correspond to sorted Cxx phase names. Callers
    own its units and sampling grid. This does not calibrate demosaicing.
    """
    if not assume_independent:
        raise ValueError("phase cross-covariance is unknown; independence must be explicit")
    if not phase_bands:
        raise ValueError("phase detail variances are unavailable")
    order = sorted(phase_bands)
    from .noise_spectrum import PHASES
    if any(phase not in PHASES for phase in order):
        raise ValueError("detail projection needs measured Cxx phase names")
    matrix = np.asarray(projection, dtype=np.float64)
    if matrix.shape != (3, len(order)) or not np.isfinite(matrix).all():
        raise ValueError("detail projection must be a finite 3-by-phase matrix")
    levels = set(phase_bands[order[0]])
    if (not levels or any(type(level) is not int or not 0 <= level <= 12 for level in levels) or
            any(set(phase_bands[p]) != levels for p in order)):
        raise ValueError("phase detail bands do not have the same levels")
    result = {}
    for level in sorted(levels):
        variances = np.asarray([phase_bands[p][level] for p in order], dtype=np.float64)
        if not np.isfinite(variances).all() or np.any(variances < 0):
            raise ValueError("phase detail variance must be finite and nonnegative")
        projected = np.square(matrix) @ variances
        if not np.isfinite(projected).all() or np.any(projected > np.finfo(np.float32).max):
            raise ValueError("projected detail variance exceeds the processing range")
        result[level] = projected.astype(np.float32)
    return result
