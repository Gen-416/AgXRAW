// SPDX-License-Identifier: GPL-3.0-or-later
//! Film v2 core per-pixel chain (dngscan/film_v2_math.py, film_develop.py):
//! Stage A layer exposure (signed 3x3 observer or the chromaticity-field
//! correction), the three characteristic curves, the rail-preserving
//! inter-image term, cube normalization, tetrahedral LUT interpolation, the
//! film compression bridge and the neutralization cast divisions.
//!
//! Existing fixtures pin exact NumPy parity; this is not a universal BLAS
//! guarantee. In particular, float32 scenes multiplied by float64 observer
//! coefficients need not have exact products: small/tail matrix batches can
//! differ at float64 rounding precision (tests/test_rust_stage3.py).
//! The native float64 (n,3)@(3,3) uses an FMA chain in k order;
//! float64 (n,3)@(3,) and (3,3)@(3,) are
//! sequential product-sums; np.interp is fma(slope, x - xp[j], fp[j]);
//! np.sum over 3 elements is sequential; float64 log10/log2/exp2/exp/pow
//! match the system libm.

use crate::spatial::np_interp;

pub const LOG10_2: f64 = 0.30102999566398119521;
pub const SCENE_MID: f64 = 0.18;

#[inline(always)]
fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else {
        b
    }
}
#[inline(always)]
fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else {
        b
    }
}
#[inline(always)]
fn clip64(v: f64, lo: f64, hi: f64) -> f64 {
    fmin64(fmax64(v, lo), hi)
}
#[inline(always)]
fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

/// (n,3) @ M^T for a float64 3x3, Accelerate order: acc = fma(v[k], m[c][k], acc).
#[inline(always)]
pub fn mat3_fma_chain_f64(v: [f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for c in 0..3 {
        let mut acc = 0.0f64;
        for k in 0..3 {
            acc = v[k].mul_add(m[c][k], acc);
        }
        out[c] = acc;
    }
    out
}

/// (3,3) @ v and (n,3) @ v for float64: sequential (a*w0 + b*w1) + c*w2.
#[inline(always)]
pub fn dot3_seq_f64(v: [f64; 3], w: [f64; 3]) -> f64 {
    (v[0] * w[0] + v[1] * w[1]) + v[2] * w[2]
}

/// Run a per-pixel (n,3) -> (n,3) map over thread-budgeted pixel chunks.
/// Every kernel here is a pure elementwise map, so the split is exact.
pub fn par_map3<I, O, F>(inp: &[I], out: &mut [O], f: F)
where
    I: Sync,
    O: Send,
    F: Fn(&[I], &mut [O]) + Sync,
{
    let n = out.len() / 3;
    debug_assert_eq!(inp.len(), out.len());
    let workers = crate::budget::workers_for(n).max(1) as usize;
    if workers <= 1 || n < 2 * workers {
        f(inp, out);
        return;
    }
    let chunk = (n + workers - 1) / workers * 3;
    std::thread::scope(|s| {
        for (i, o) in inp.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let fref = &f;
            s.spawn(move || fref(i, o));
        }
    });
}

/// film_v2_math.layer_log_exposure: signed observer, neutral-anchored log10.
pub fn layer_log_exposure<T: Copy + Into<f64>>(rgb: &[T], observer: &[[f64; 3]; 3], out: &mut [f64]) {
    let mid = [
        dot3_seq_f64([observer[0][0], observer[0][1], observer[0][2]], [SCENE_MID; 3]),
        dot3_seq_f64([observer[1][0], observer[1][1], observer[1][2]], [SCENE_MID; 3]),
        dot3_seq_f64([observer[2][0], observer[2][1], observer[2][2]], [SCENE_MID; 3]),
    ];
    let mid_f = [fmax64(mid[0], 1e-12), fmax64(mid[1], 1e-12), fmax64(mid[2], 1e-12)];
    for (px, o) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        let e = mat3_fma_chain_f64([px[0].into(), px[1].into(), px[2].into()], observer);
        for c in 0..3 {
            o[c] = (fmax64(e[c], 1e-12) / mid_f[c]).log10();
        }
    }
}

pub struct ChromaField<'a> {
    /// (n, n, 3) float64 per-layer correction table, row-major, indexed [ix, iy, c]
    pub table: &'a [f64],
    pub n: usize,
    pub domain: [f64; 4],
    pub xyz_from_rec2020: [[f64; 3]; 3],
    pub observer: [[f64; 3]; 3],
}

/// film_v2_math.chroma_field_log_exposure.
pub fn chroma_field_log_exposure<T: Copy + Into<f64>>(rgb: &[T], field: &ChromaField, out: &mut [f64]) {
    let obs = &field.observer;
    let mid = [
        dot3_seq_f64([obs[0][0], obs[0][1], obs[0][2]], [SCENE_MID; 3]),
        dot3_seq_f64([obs[1][0], obs[1][1], obs[1][2]], [SCENE_MID; 3]),
        dot3_seq_f64([obs[2][0], obs[2][1], obs[2][2]], [SCENE_MID; 3]),
    ];
    let mid_f = [fmax64(mid[0], 1e-12), fmax64(mid[1], 1e-12), fmax64(mid[2], 1e-12)];
    let [x0, x1, y0, y1] = field.domain;
    let n = field.n;
    let nm1 = (n - 1) as f64;
    for (px, o) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        let r = [px[0].into(), px[1].into(), px[2].into()];
        let e_obs = mat3_fma_chain_f64(r, obs);
        let finite = r.iter().all(|v| v.is_finite());
        let positive = finite && r.iter().all(|&v| v > 0.0);
        let neutral = r[0] == r[1] && r[1] == r[2];
        let safe = if positive { r } else { [1.0, 1.0, 1.0] };
        let xyz = mat3_fma_chain_f64(safe, &field.xyz_from_rec2020);
        let s = (xyz[0] + xyz[1]) + xyz[2];
        let lum = xyz[1];
        let cx = xyz[0] / s;
        let cy = lum / s;
        let ok = positive
            && !neutral
            && s.is_finite()
            && s > 1e-12
            && lum.is_finite()
            && lum > 1e-12
            && cx.is_finite()
            && cy.is_finite()
            && cx >= x0
            && cx <= x1
            && cy >= y0
            && cy <= y1;
        let fx = clip64(((if ok { cx } else { x0 }) - x0) / (x1 - x0), 0.0, 1.0) * nm1;
        let fy = clip64(((if ok { cy } else { y0 }) - y0) / (y1 - y0), 0.0, 1.0) * nm1;
        let ix = ((fx as i64).min(n as i64 - 2)) as usize; // astype(int64) truncates toward zero; fx >= 0
        let iy = ((fy as i64).min(n as i64 - 2)) as usize;
        let tx = fx - ix as f64;
        let ty = fy - iy as f64;
        let t = |a: usize, b: usize, c: usize| field.table[(a * n + b) * 3 + c];
        for c in 0..3 {
            // NumPy: ((A*(1-tx))*(1-ty) + (B*tx)*(1-ty)) + (C*(1-tx))*ty + (D*tx)*ty,
            // zeroed where the pixel is outside the field.
            let delta = if ok {
                t(ix, iy, c) * (1.0 - tx) * (1.0 - ty) + t(ix + 1, iy, c) * tx * (1.0 - ty)
                    + t(ix, iy + 1, c) * (1.0 - tx) * ty
                    + t(ix + 1, iy + 1, c) * tx * ty
            } else {
                0.0
            };
            let e = e_obs[c] * delta.exp2();
            o[c] = (fmax64(e, 1e-12) / mid_f[c]).log10();
        }
    }
}

/// film_v2_math.characteristic_amounts: three np.interp curves on
/// x = log_e + ev_offset * LOG10_2.
pub fn characteristic_amounts(log_e: &[f64], le: &[f64], table: &[f64], ev_offset: f64, out: &mut [f64]) {
    let shift = ev_offset * LOG10_2;
    let n = le.len();
    let cols: [Vec<f64>; 3] = [
        (0..n).map(|i| table[i * 3]).collect(),
        (0..n).map(|i| table[i * 3 + 1]).collect(),
        (0..n).map(|i| table[i * 3 + 2]).collect(),
    ];
    for (px, o) in log_e.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        for c in 0..3 {
            let x = px[c] + shift;
            o[c] = np_interp(x, le, &cols[c]);
        }
    }
}

/// film_develop._apply_film_core_v2 inter-image block (rail-preserving).
/// `neutral` is characteristic_amounts of the per-pixel mean logE repeated;
/// this function evaluates it too (np.mean over 3 = ((a+b)+c)/3).
pub fn interimage_amplify(
    amounts: &mut [f64],
    log_e: &[f64],
    le: &[f64],
    table: &[f64],
    rail_lo: [f64; 3],
    rail_hi: [f64; 3],
    beta: f64,
) {
    let n = le.len();
    let cols: [Vec<f64>; 3] = [
        (0..n).map(|i| table[i * 3]).collect(),
        (0..n).map(|i| table[i * 3 + 1]).collect(),
        (0..n).map(|i| table[i * 3 + 2]).collect(),
    ];
    for (a, l) in amounts.chunks_exact_mut(3).zip(log_e.chunks_exact(3)) {
        let le_mean = ((l[0] + l[1]) + l[2]) / 3.0;
        for c in 0..3 {
            let neutral = np_interp(le_mean, le, &cols[c]);
            let d = a[c] - neutral;
            let head = fmax64(if d >= 0.0 { rail_hi[c] - neutral } else { neutral - rail_lo[c] }, 1e-9);
            let mut t = fmin64(d.abs() / head, 1.0);
            t = (1.0 + beta) * t / (1.0 + beta * t);
            let sign = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            a[c] = neutral + sign * head * t;
        }
    }
}

/// film_develop._tetrahedral: cubic-lattice tetrahedral interpolation of a
/// (n, n, n, 3) float32 LUT at float32 unit coordinates.
pub fn tetrahedral(lut: &[f32], n: usize, u: &[f32], out: &mut [f32]) {
    let nm1 = (n - 1) as f32; // (n - 1) is a Python int: float32 multiply
    let at = |ix: usize, iy: usize, iz: usize, c: usize| lut[((ix * n + iy) * n + iz) * 3 + c];
    for (p, o) in u.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        let mut g = [0.0f32; 3];
        let mut i0 = [0usize; 3];
        let mut f = [0.0f32; 3];
        for c in 0..3 {
            let uc = if p[c].is_nan() {
                f32::NAN
            } else if p[c] < 0.0 {
                0.0
            } else if p[c] > 1.0 {
                1.0
            } else {
                p[c]
            };
            g[c] = uc * nm1;
            let gi = (g[c] as i32).min(n as i32 - 2); // astype(int32) truncation
            i0[c] = gi.max(0) as usize;
            // (g - i0) promotes to float64, then astype(float32)
            f[c] = (g[c] as f64 - gi as f64) as f32;
        }
        let (fx, fy, fz) = (f[0], f[1], f[2]);
        // six tetrahedra keyed by the ordering of (fx, fy, fz), same branch order
        let (ca, cb, f1, f2, f3): ([usize; 3], [usize; 3], f32, f32, f32) = if fx >= fy && fy >= fz {
            ([1, 0, 0], [1, 1, 0], fx, fy, fz)
        } else if fx >= fz && fz > fy {
            ([1, 0, 0], [1, 0, 1], fx, fz, fy)
        } else if fz > fx && fx >= fy {
            ([0, 0, 1], [1, 0, 1], fz, fx, fy)
        } else if fy > fx && fx >= fz {
            ([0, 1, 0], [1, 1, 0], fy, fx, fz)
        } else if fy >= fz && fz > fx {
            ([0, 1, 0], [0, 1, 1], fy, fz, fx)
        } else if fz > fy && fy > fx {
            ([0, 0, 1], [0, 1, 1], fz, fy, fx)
        } else {
            // NaN weights: NumPy leaves the row uninitialized (np.empty_like); never reached on finite input
            ([1, 0, 0], [1, 1, 0], fx, fy, fz)
        };
        for c in 0..3 {
            let c000 = at(i0[0], i0[1], i0[2], c);
            let c111 = at(i0[0] + 1, i0[1] + 1, i0[2] + 1, c);
            let pa = at(i0[0] + ca[0], i0[1] + ca[1], i0[2] + ca[2], c);
            let pb = at(i0[0] + cb[0], i0[1] + cb[1], i0[2] + cb[2], c);
            // (1 - f1) * c000 + (f1 - f2) * pA + (f2 - f3) * pB + f3 * c111, left-associative float32
            o[c] = ((1.0 - f1) * c000 + (f1 - f2) * pa) + (f2 - f3) * pb + f3 * c111;
        }
    }
}

/// film_v2_math.film_compression_ev on float64 Rec.2020 rows.
pub fn film_compression_ev(rgb: &[f64], impact: f64, knee_ev: f64, width_ev: f64, rho: f64, out: &mut [f64]) {
    let luma = [0.2627f64, 0.6780, 0.0593];
    let w = fmax64(width_ev, 1e-3);
    let k = knee_ev;
    for (px, o) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        let r = [px[0], px[1], px[2]];
        let y = fmax64(dot3_seq_f64(r, luma), 1e-9);
        let x = (y / SCENE_MID).log2();
        let x_f = if x > k { k + w * (1.0 - (-(x - k) / w).exp()) } else { x };
        let x_new = (1.0 - impact) * x + impact * x_f;
        let gain = (x_new - x).exp2();
        let mut v = [r[0] * gain, r[1] * gain, r[2] * gain];
        let d = fmax64(x - x_new, 0.0);
        if rho > 0.0 {
            let y_new = fmax64(dot3_seq_f64(v, luma), 1e-9);
            let cs = (-rho * d).exp();
            for c in 0..3 {
                v[c] = y_new * ((v[c] / y_new - 1.0) * cs + 1.0);
            }
        }
        for c in 0..3 {
            o[c] = fmax64(v[c], 0.0);
        }
    }
}

/// The per-pixel scene luminance EV the neutralization casts key on:
/// np.log2(np.maximum(rgb @ REC2020_LUMA, EPS) / np.float32(0.18)) + np.float32(offset)
/// with float32 rgb and the float32 luma row (sequential matvec).
pub fn scene_ev_luma_f32(rgb: &[f32], eps: f32, offset: f32, out: &mut [f32]) {
    let luma = [0.2627f32, 0.6780, 0.0593];
    for (px, o) in rgb.chunks_exact(3).zip(out.iter_mut()) {
        let y = (px[0] * luma[0] + px[1] * luma[1]) + px[2] * luma[2];
        *o = (fmax32(y, eps) / 0.18f32).log2() + offset;
    }
}

/// developed[:, c] /= np.interp(ev_y, cast_ev, cast[:, c]) — float32 developed,
/// float64 interp result (float32 /= float64 -> computed in float64, stored float32).
pub fn cast_divide_per_pixel(developed: &mut [f32], ev_y: &[f32], cast_ev: &[f64], cast: &[f64]) {
    let n = cast_ev.len();
    let cols: [Vec<f64>; 3] = [
        (0..n).map(|i| cast[i * 3]).collect(),
        (0..n).map(|i| cast[i * 3 + 1]).collect(),
        (0..n).map(|i| cast[i * 3 + 2]).collect(),
    ];
    for (px, &ev) in developed.chunks_exact_mut(3).zip(ev_y.iter()) {
        for c in 0..3 {
            let d = np_interp(ev as f64, cast_ev, &cols[c]);
            px[c] = (px[c] as f64 / d) as f32;
        }
    }
}
