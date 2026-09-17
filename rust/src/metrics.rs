// SPDX-License-Identifier: GPL-3.0-or-later
//! Analysis / delivery metrics (dngscan/analysis.py compute_gamut_metrics,
//! dngscan/gainmap.py _roundtrip_error and _base_roundtrip_error).
use crate::stats::{
    block8_mean, median_f32, percentile_f32, retain_top_k, upper_percentile_from_top,
};
use half::f16;
use numpy::ndarray::ArrayView3;

#[inline(always)]
fn nan_to_num(v: f32, nan: f32, posinf: f32, neginf: f32) -> f32 {
    if v.is_nan() {
        nan
    } else if v.is_infinite() {
        if v > 0.0 {
            posinf
        } else {
            neginf
        }
    } else {
        v
    }
}

#[inline(always)]
fn max3(a: f32, b: f32, c: f32) -> f32 {
    // np.max over a finite triple
    let m = if b > a { b } else { a };
    if c > m {
        c
    } else {
        m
    }
}

/// analysis.compute_gamut_metrics: out-of-gamut counts among the bright half.
/// `scene` is (n, stride) float32 with the first three channels used, `y` (n,).
/// Returns (counts per matrix, bright_total). `matrices` are XYZ->RGB float64
/// row-major 3x3, `rec2020_to_xyz` likewise.
#[allow(clippy::too_many_arguments)]
pub fn gamut_counts(
    scene: &[f32],
    stride: usize,
    y: &[f32],
    inv_scale: f32,
    rec2020_to_xyz: &[f64; 9],
    matrices: &[[f64; 9]],
    eps: f32,
    gamut_eps: f32,
) -> (Vec<u64>, usize) {
    let n = y.len();
    let mut counts = vec![0u64; matrices.len()];
    if n == 0 {
        return (counts, 0);
    }
    let mut ycopy = y.to_vec();
    let median = median_f32(&mut ycopy);
    drop(ycopy);
    let mut bright: Vec<bool> = y.iter().map(|&v| v > median).collect();
    if !bright.iter().any(|&b| b) {
        bright = y.iter().map(|&v| v >= median && v > eps).collect();
    }
    let bright_total = bright.iter().filter(|&&b| b).count();
    if bright_total == 0 {
        return (counts, 0);
    }
    for (i, &is_bright) in bright.iter().enumerate() {
        if !is_bright {
            continue;
        }
        let px = &scene[i * stride..i * stride + 3];
        let lin = [px[0] * inv_scale, px[1] * inv_scale, px[2] * inv_scale];
        let m = rec2020_to_xyz;
        let (r, g, b) = (lin[0] as f64, lin[1] as f64, lin[2] as f64);
        let xyz = [
            nan_to_num((m[0] * r + m[1] * g + m[2] * b) as f32, 0.0, 1.0, 0.0),
            nan_to_num((m[3] * r + m[4] * g + m[5] * b) as f32, 0.0, 1.0, 0.0),
            nan_to_num((m[6] * r + m[7] * g + m[8] * b) as f32, 0.0, 1.0, 0.0),
        ];
        let (x, yy, z) = (xyz[0] as f64, xyz[1] as f64, xyz[2] as f64);
        for (k, mat) in matrices.iter().enumerate() {
            let rgb = [
                nan_to_num((mat[0] * x + mat[1] * yy + mat[2] * z) as f32, 0.0, 0.0, 0.0),
                nan_to_num((mat[3] * x + mat[4] * yy + mat[5] * z) as f32, 0.0, 0.0, 0.0),
                nan_to_num((mat[6] * x + mat[7] * yy + mat[8] * z) as f32, 0.0, 0.0, 0.0),
            ];
            let denom = max3(rgb[0], rgb[1], rgb[2]);
            if !(denom > eps) {
                continue;
            }
            if rgb[0] / denom < -gamut_eps || rgb[1] / denom < -gamut_eps || rgb[2] / denom < -gamut_eps {
                counts[k] += 1;
            }
        }
    }
    (counts, bright_total)
}

pub struct HdrRoundtrip {
    pub chroma_error: f64,
    pub relative_error: f64,
    pub median_relative_error: f64,
    pub p95_relative_error: f64,
    pub p99_relative_error: f64,
    pub p999_relative_error: f64,
    pub block_median_relative_error: f64,
    pub block_p95_relative_error: f64,
    pub block_p99_relative_error: f64,
    pub block_chroma_error: f64,
}

impl HdrRoundtrip {
    fn invalid() -> Self {
        Self {
            chroma_error: f64::INFINITY,
            relative_error: f64::INFINITY,
            median_relative_error: f64::INFINITY,
            p95_relative_error: f64::INFINITY,
            p99_relative_error: f64::INFINITY,
            p999_relative_error: f64::INFINITY,
            block_median_relative_error: f64::INFINITY,
            block_p95_relative_error: f64::INFINITY,
            block_p99_relative_error: f64::INFINITY,
            block_chroma_error: f64::INFINITY,
        }
    }
}

#[inline(always)]
fn chroma_terms(a: [f32; 3], e: [f32; 3]) -> [f32; 3] {
    // np.abs(ac / max(ac.sum(axis=1), 1e-6) - ec / max(ec.sum(axis=1), 1e-6)), sums sequential
    let sa = (a[0] + a[1]) + a[2];
    let se = (e[0] + e[1]) + e[2];
    let da = if sa > 1e-6 { sa } else { 1e-6 };
    let de = if se > 1e-6 { se } else { 1e-6 };
    [
        (a[0] / da - e[0] / de).abs(),
        (a[1] / da - e[1] / de).abs(),
        (a[2] / da - e[2] / de).abs(),
    ]
}

/// One row band's contribution: relative values (in place), chroma terms of
/// pixels with e_peak > 0.05, and the 8x8 block statistics of the band.
struct HdrBand {
    chroma: Vec<f32>,
    block_rel: Vec<f32>,
    block_chroma: Vec<f32>,
}

const BAND_ROWS: usize = 512; // multiple of 8, like gainmap._ROUNDTRIP_BAND_ROWS

fn hdr_band(
    expanded: ArrayView3<'_, f16>,
    intended: ArrayView3<'_, f16>,
    w: usize,
    w8: usize,
    row0: usize,
    row1: usize,
    relative: &mut [f32],
    band: &mut HdrBand,
) -> bool {
    let px = |buf: ArrayView3<'_, f16>, y: usize, x: usize| -> [f32; 3] {
        [buf[[y, x, 0]].to_f32(), buf[[y, x, 1]].to_f32(), buf[[y, x, 2]].to_f32()]
    };
    band.chroma.clear();
    band.block_rel.clear();
    band.block_chroma.clear();
    for y in row0..row1 {
        for x in 0..w {
            let a = px(expanded, y, x);
            let e = px(intended, y, x);
            // Validate every RGB component before max/quantiles can hide an
            // isolated invalid sample. Fused into the existing pixel scan.
            if !a.iter().chain(e.iter()).all(|v| v.is_finite()) {
                return false;
            }
            let e_peak = max3(e[0].abs(), e[1].abs(), e[2].abs());
            let d = max3((a[0] - e[0]).abs(), (a[1] - e[1]).abs(), (a[2] - e[2]).abs());
            relative[(y - row0) * w + x] = d / (if e_peak > 0.05 { e_peak } else { 0.05 });
            if e_peak > 0.05 {
                band.chroma.extend_from_slice(&chroma_terms(a, e));
            }
        }
    }
    let band_h8 = (row1 - row0) - (row1 - row0) % 8;
    if w8 > 0 && row0 % 8 == 0 {
        for by in 0..band_h8 / 8 {
            for bx in 0..w8 / 8 {
                let mut ma = [0.0f32; 3];
                let mut me = [0.0f32; 3];
                for c in 0..3 {
                    ma[c] = block8_mean(|i, j| expanded[[row0 + by * 8 + i, bx * 8 + j, c]].to_f32());
                    me[c] = block8_mean(|i, j| intended[[row0 + by * 8 + i, bx * 8 + j, c]].to_f32());
                }
                let e_peak = max3(me[0].abs(), me[1].abs(), me[2].abs());
                let d = max3((ma[0] - me[0]).abs(), (ma[1] - me[1]).abs(), (ma[2] - me[2]).abs());
                band.block_rel.push(d / (if e_peak > 0.05 { e_peak } else { 0.05 }));
                if e_peak > 0.05 {
                    band.block_chroma.extend_from_slice(&chroma_terms(ma, me));
                }
            }
        }
    }
    true
}

/// gainmap._roundtrip_error. Both float16 views have at least 3 channels;
/// retain their strides so an RGB slice of decoded RGBA needs no frame copy.
/// Row bands run in parallel
/// (every statistic is per pixel or per 8x8 block, and the top-K retention
/// is a multiset operation), so the result is independent of the split.
pub fn hdr_roundtrip(
    expanded: ArrayView3<'_, f16>,
    intended: ArrayView3<'_, f16>,
    h: usize,
    w: usize,
) -> Result<HdrRoundtrip, String> {
    let total = h * w;
    let h8 = h - h % 8;
    let w8 = w - w % 8;
    if total == 0 {
        return Ok(HdrRoundtrip::invalid());
    }
    let mut relative = vec![0.0f32; total];
    let top_k = ((0.01 * total as f64 * 3.0).ceil() as usize) + 8;
    let workers = crate::budget::budgeted_workers(8) as usize;
    // Share a 512-row scratch budget across workers. Every boundary remains
    // 8-row aligned so the block-mean operation order stays unchanged.
    let band_rows = (BAND_ROWS / workers / 8).max(1) * 8;
    let band_count = h.div_ceil(band_rows).min(workers);
    let band_chroma = band_rows.min(h) * w * 3;
    let mut scratch: Vec<HdrBand> = (0..band_count)
        .map(|_| HdrBand {
            chroma: Vec::with_capacity(band_chroma),
            block_rel: Vec::with_capacity(band_rows / 8 * (w8 / 8)),
            block_chroma: Vec::with_capacity(band_rows / 8 * (w8 / 8) * 3),
        })
        .collect();
    // Allocate once and reuse across batches, including the merge space.
    // Dropping/reallocating on successive worker threads can leave large
    // allocator caches resident even though the buffers are logically freed.
    let mut chroma_top = Vec::with_capacity(top_k + band_chroma);
    let mut chroma_count = 0usize;
    let mut block_rel = Vec::with_capacity(h8 / 8 * (w8 / 8));
    let mut block_chroma = Vec::with_capacity(h8 / 8 * (w8 / 8) * 3);
    let mut chunks = relative.chunks_mut(band_rows * w).enumerate();
    loop {
        let batch: Vec<(usize, &mut [f32])> = chunks.by_ref().take(band_count).collect();
        let active = batch.len();
        if active == 0 {
            break;
        }
        let valid = if workers == 1 {
            let (bi, chunk) = batch.into_iter().next().unwrap();
            let r0 = bi * band_rows;
            hdr_band(
                expanded, intended, w, w8, r0, (r0 + band_rows).min(h),
                chunk, &mut scratch[0],
            )
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = batch.into_iter()
                    .zip(scratch.iter_mut())
                    .map(|((bi, chunk), band)| {
                        let r0 = bi * band_rows;
                        s.spawn(move || hdr_band(
                            expanded, intended, w, w8, r0, (r0 + band_rows).min(h), chunk, band,
                        ))
                    })
                    .collect();
                // Join every worker, including when one band is invalid.
                handles.into_iter()
                    .fold(true, |valid, hnd| hnd.join().expect("band thread") && valid)
            })
        };
        if !valid {
            return Ok(HdrRoundtrip::invalid());
        }
        for band in &scratch[..active] {
            chroma_count += band.chroma.len();
            retain_top_k(&mut chroma_top, &band.chroma, top_k);
            block_rel.extend_from_slice(&band.block_rel);
            block_chroma.extend_from_slice(&band.block_chroma);
        }
    }
    drop(scratch);
    let chroma_p99 = if chroma_count > 0 {
        upper_percentile_from_top(&mut chroma_top, chroma_count, 99.0)? as f64
    } else {
        0.0
    };
    let median_relative = median_f32(&mut relative) as f64;
    let p95 = percentile_f32(&mut relative, 95.0) as f64;
    let p99 = percentile_f32(&mut relative, 99.0) as f64;
    let p999 = percentile_f32(&mut relative, 99.9) as f64;
    drop(relative);
    let (bm, bp95, bp99, bchroma) = if h8 > 0 && w8 > 0 {
        let bm = median_f32(&mut block_rel) as f64;
        let bp95 = percentile_f32(&mut block_rel, 95.0) as f64;
        let bp99 = percentile_f32(&mut block_rel, 99.0) as f64;
        let bchroma = if block_chroma.is_empty() {
            0.0
        } else {
            percentile_f32(&mut block_chroma, 99.0) as f64
        };
        (bm, bp95, bp99, bchroma)
    } else {
        (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY)
    };
    Ok(HdrRoundtrip {
        chroma_error: chroma_p99,
        relative_error: p99,
        median_relative_error: median_relative,
        p95_relative_error: p95,
        p99_relative_error: p99,
        p999_relative_error: p999,
        block_median_relative_error: bm,
        block_p95_relative_error: bp95,
        block_p99_relative_error: bp99,
        block_chroma_error: bchroma,
    })
}

pub struct BaseRoundtrip {
    pub mean_code_error: f64,
    pub p99_code_error: f64,
    pub max_code_error: f64,
    pub channel_bias_code_error: f64,
    pub block_p99_code_error: f64,
}

struct BaseBand {
    histogram: [usize; 256],
    abs_sum: f64,
    signed_sum: [f64; 3],
    max_err: f64,
    block_err: Vec<f32>,
}

fn base_band(decoded: &[u8], intended: &[u8], w: usize, w8: usize, row0: usize, row1: usize) -> BaseBand {
    let mut histogram = [0usize; 256];
    let mut abs_sum = 0.0f64;
    let mut signed_sum = [0.0f64; 3];
    let mut max_err = 0.0f64;
    for y in row0..row1 {
        for x in 0..w {
            let o = (y * w + x) * 3;
            let mut pe = 0.0f32;
            for c in 0..3 {
                let s = decoded[o + c] as f32 - intended[o + c] as f32;
                abs_sum += s.abs() as f64;
                signed_sum[c] += s as f64;
                if s.abs() > pe {
                    pe = s.abs();
                }
            }
            histogram[pe as usize] += 1;
            if pe as f64 > max_err {
                max_err = pe as f64;
            }
        }
    }
    let mut block_err = Vec::new();
    let band_h8 = (row1 - row0) - (row1 - row0) % 8;
    if w8 > 0 && row0 % 8 == 0 {
        for by in 0..band_h8 / 8 {
            for bx in 0..w8 / 8 {
                let mut m = [0.0f32; 3];
                for c in 0..3 {
                    m[c] = block8_mean(|i, j| {
                        let o = ((row0 + by * 8 + i) * w + bx * 8 + j) * 3 + c;
                        decoded[o] as f32 - intended[o] as f32
                    });
                }
                block_err.push(max3(m[0].abs(), m[1].abs(), m[2].abs()));
            }
        }
    }
    BaseBand { histogram, abs_sum, signed_sum, max_err, block_err }
}

/// gainmap._base_roundtrip_error. Sums accumulate in float64 per band and
/// are added band by band (the NumPy reference uses float64 pairwise sums —
/// a declared last-bits difference, far below any delivery tolerance);
/// everything else is exact.
pub fn base_roundtrip(decoded: &[u8], intended: &[u8], h: usize, w: usize) -> Result<BaseRoundtrip, String> {
    let total = h * w;
    let h8 = h - h % 8;
    let w8 = w - w % 8;
    let mut histogram = [0usize; 256];
    let mut abs_sum = 0.0f64;
    let mut signed_sum = [0.0f64; 3];
    let mut max_err = 0.0f64;
    let mut block_err: Vec<f32> = Vec::new();
    std::thread::scope(|s| {
        let workers = crate::budget::budgeted_workers(8) as usize;
        let mut ranges = (0..h).step_by(BAND_ROWS);
        loop {
            let handles: Vec<_> = ranges.by_ref()
                .take(workers)
                .map(|r0| s.spawn(move || base_band(
                    decoded, intended, w, w8, r0, (r0 + BAND_ROWS).min(h),
                )))
                .collect();
            if handles.is_empty() {
                break;
            }
            for hnd in handles {
                let band = hnd.join().expect("band thread");
                for (dst, count) in histogram.iter_mut().zip(band.histogram) {
                    *dst += count;
                }
                abs_sum += band.abs_sum;
                for c in 0..3 {
                    signed_sum[c] += band.signed_sum[c];
                }
                max_err = max_err.max(band.max_err);
                block_err.extend_from_slice(&band.block_err);
            }
        }
    });
    let block_p99 = if h8 > 0 && w8 > 0 {
        percentile_f32(&mut block_err, 99.0) as f64
    } else {
        f64::INFINITY
    };
    // Same float32 interpolation as the reference's exact upper percentile;
    // integer code errors permit exact rank lookup without O(H*W) storage.
    let p99 = if total == 0 {
        0.0
    } else {
        let pos = (total as f64 - 1.0) * 0.99;
        let rank_value = |rank: usize| -> f32 {
            let mut count = 0usize;
            for (value, &n) in histogram.iter().enumerate() {
                count += n;
                if count > rank {
                    return value as f32;
                }
            }
            unreachable!("histogram covers all pixels")
        };
        let lo = rank_value(pos.floor() as usize);
        let hi = rank_value(pos.ceil() as usize);
        let frac = pos - pos.floor();
        if frac >= 0.5 {
            (hi - (hi - lo) * ((1.0 - frac) as f32)) as f64
        } else {
            (lo + (hi - lo) * (frac as f32)) as f64
        }
    };
    let bias = signed_sum
        .iter()
        .map(|s| (s / total as f64).abs())
        .fold(0.0f64, f64::max);
    Ok(BaseRoundtrip {
        mean_code_error: if total > 0 { abs_sum / (total as f64 * 3.0) } else { 0.0 },
        p99_code_error: p99,
        max_code_error: max_err,
        channel_bias_code_error: if total > 0 { bias } else { 0.0 },
        block_p99_code_error: block_p99,
    })
}
