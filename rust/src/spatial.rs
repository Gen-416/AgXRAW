// SPDX-License-Identifier: GPL-3.0-or-later
//! Film spatial operators (dngscan/film_optics.py): area decimation, bilinear
//! upsampling, the slabbed separable Gaussian, the frequency-matched 5-tap
//! small-sigma blur, the halation gate / pointwise return / component source,
//! the capture-bloom gate / source / apply, the emulsion/formation scatter
//! mix, film-space grain sampling, density grain, and the halation reinject.
//!
//! Every function reproduces the NumPy reference element for element. The
//! platform semantics it relies on are pinned by tests/test_rust_stage2.py:
//! Accelerate's (n,3)@(3,3) matmul is an FMA chain in k order; (n,3)@(3,)
//! is a plain sequential product-sum; einsum "cj,...j->...c" is sequential;
//! np.interp is fma(slope, x - xp[j], fp[j]); float64 sums use NumPy's
//! pairwise order; float32/float64 libm functions match the system libm.
use crate::budget::budgeted_workers;
use crate::numpy_sum::pairwise_sum_f64;
use numpy::ndarray::{ArrayView2, ArrayView3, Axis, s};

/// Run `f(row0, out_chunk)` over `out` split into row chunks of `row_len`
/// elements, on up to `budgeted_workers(8)` threads. Every kernel here is
/// per element or per row, so the split cannot change a result.
fn par_rows<F>(out: &mut [f32], row_len: usize, rows: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let workers = budgeted_workers(8).max(1) as usize;
    if workers <= 1 || rows < 2 * workers {
        f(0, out);
        return;
    }
    let chunk_rows = (rows + workers - 1) / workers;
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for (i, chunk) in out.chunks_mut(chunk_rows * row_len.max(1)).enumerate() {
            let fref = &f;
            handles.push(s.spawn(move || fref(i * chunk_rows, chunk)));
        }
        crate::budget::join_workers(handles);
    });
}

fn par_rows_f64<F>(out: &mut [f64], row_len: usize, rows: usize, f: F)
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    let workers = budgeted_workers(8).max(1) as usize;
    if workers <= 1 || rows < 2 * workers {
        f(0, out);
        return;
    }
    let chunk_rows = (rows + workers - 1) / workers;
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for (i, chunk) in out.chunks_mut(chunk_rows * row_len.max(1)).enumerate() {
            let fref = &f;
            handles.push(s.spawn(move || fref(i * chunk_rows, chunk)));
        }
        crate::budget::join_workers(handles);
    });
}

#[inline(always)]
fn fmax32(a: f32, b: f32) -> f32 {
    // np.maximum(a, b) for finite inputs (NaN propagates either way here)
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a > b {
        a
    } else {
        b
    }
}
#[inline(always)]
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a < b {
        a
    } else {
        b
    }
}
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
fn clip32(v: f32, lo: f32, hi: f32) -> f32 {
    fmin32(fmax32(v, lo), hi)
}
#[inline(always)]
fn clip64(v: f64, lo: f64, hi: f64) -> f64 {
    fmin64(fmax64(v, lo), hi)
}

/// `np.interp(x, xp, fp)` (float64, xp strictly increasing): NumPy computes
/// `fma(slope, x - xp[j], fp[j])` on the arm64 build (compiled with FP
/// contraction), clamping to the table ends.
pub fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= xp[0] {
        // numpy: x < xp[0] -> fp[0]; x == xp[0] falls into the first segment
        // whose slope*(0)+fp[0] is fp[0] too
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // j = searchsorted(xp, x, 'right') - 1
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let j = lo - 1;
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope.mul_add(x - xp[j], fp[j])
}

/// Accelerate sgemm/dgemm for (n,3)@(3,3)^T: acc = fma(u[k], t[c][k], acc), k = 0..3.
#[inline(always)]
pub fn mat3_fma_chain_f32(u: [f32; 3], t: &[[f32; 3]; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let mut acc = 0.0f32;
        for k in 0..3 {
            acc = u[k].mul_add(t[c][k], acc);
        }
        out[c] = acc;
    }
    out
}

/// (n,3)@(3,) float32 matvec: plain sequential (r*w0 + g*w1) + b*w2.
#[inline(always)]
pub fn dot3_seq_f32(v: [f32; 3], w: [f32; 3]) -> f32 {
    (v[0] * w[0] + v[1] * w[1]) + v[2] * w[2]
}

// ---------------------------------------------------------------------------
// area decimation / upsampling
// ---------------------------------------------------------------------------

pub enum Acc<'a> {
    F64(&'a mut [f64]),
    F32(&'a mut [f32]),
}

/// film_optics.area_decimate_rows: accumulate source rows [y0, y0+n) (float32
/// or float64 input, promoted to float64) into the decimated accumulator.
#[allow(clippy::too_many_arguments)]
pub fn area_decimate_rows<T: Copy + Into<f64>>(
    rows: ArrayView3<'_, T>,
    n: usize,
    y0: usize,
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    c: usize,
    mut acc: Acc,
) {
    if n == 0 || w == 0 || out_w == 0 || out_h == 0 {
        return;
    }
    // column edges
    let xe: Vec<f64> = (0..=out_w).map(|i| ((w * i) as f64) / (out_w as f64)).collect();
    let xi: Vec<usize> = xe
        .iter()
        .map(|&e| (e.floor() as i64).clamp(0, w as i64 - 1) as usize)
        .collect();
    let xf: Vec<f64> = xe.iter().zip(xi.iter()).map(|(&e, &i)| e - i as f64).collect();
    let dx: Vec<f64> = (0..out_w).map(|i| fmax64(xe[i + 1] - xe[i], 1e-12)).collect();
    // row edges
    let ye: Vec<f64> = (0..=out_h).map(|j| ((h * j) as f64) / (out_h as f64)).collect();
    let mut cs = vec![0.0f64; (w + 1) * c];
    let mut at = vec![0.0f64; (out_w + 1) * c];
    // per-row decimated columns (the band's `col`), then the two np.add.at
    // passes in NumPy's order: shift 0 over ALL rows, then shift 1 over all
    let mut cols = vec![0.0f64; n * out_w * c];
    let mut los = vec![0usize; n];
    for r in 0..n {
        for ch in 0..c {
            cs[ch] = 0.0;
            let mut run = 0.0f64;
            for x in 0..w {
                run += rows[[r, x, ch]].into();
                cs[(x + 1) * c + ch] = run;
            }
        }
        for i in 0..=out_w {
            let (a, b) = (xi[i], xi[i] + 1);
            for ch in 0..c {
                at[i * c + ch] = cs[a * c + ch] * (1.0 - xf[i]) + cs[b * c + ch] * xf[i];
            }
        }
        let col = &mut cols[r * out_w * c..(r + 1) * out_w * c];
        for i in 0..out_w {
            for ch in 0..c {
                col[i * c + ch] = (at[(i + 1) * c + ch] - at[i * c + ch]) / dx[i];
            }
        }
        let ys = (y0 + r) as f64;
        let mut count = 0usize; // searchsorted(ye, ys, 'right')
        while count < ye.len() && ye[count] <= ys {
            count += 1;
        }
        los[r] = (count as i64 - 1).clamp(0, out_h as i64 - 1) as usize;
    }
    for shift in 0..2usize {
        for r in 0..n {
            let lo = los[r];
            let ys = (y0 + r) as f64;
            let idx = (lo + shift).min(out_h - 1);
            let seg_lo = fmax64(ys, ye[idx]);
            let hi_edge = ye[(idx + 1).min(out_h)];
            let seg_hi = fmin64(ys + 1.0, hi_edge);
            let mut wgt = fmax64(seg_hi - seg_lo, 0.0) / fmax64(hi_edge - ye[idx], 1e-12);
            if shift == 1 && !(idx > lo) {
                wgt = 0.0;
            }
            let base = idx * out_w * c;
            let col = &cols[r * out_w * c..(r + 1) * out_w * c];
            for i in 0..out_w * c {
                let v = col[i] * wgt;
                match acc {
                    Acc::F64(ref mut a) => a[base + i] += v,
                    Acc::F32(ref mut a) => a[base + i] = (a[base + i] as f64 + v) as f32,
                }
            }
        }
    }
}

/// film_optics.upsample_rows: bilinear upsample of the decimated map for
/// output rows [y0, y1). Returns (y1 - y0, width, c) float32.
pub fn upsample_rows(
    map_dec: &[f32],
    dh: usize,
    dw: usize,
    c: usize,
    y0: usize,
    y1: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let n = y1 - y0;
    let mut out = vec![0.0f32; n * width * c];
    let xq: Vec<f64> = (0..width).map(|x| (x as f64 + 0.5) / width as f64 * dw as f64 - 0.5).collect();
    let xi: Vec<usize> = xq.iter().map(|&q| (q.floor() as i64).clamp(0, dw as i64 - 1) as usize).collect();
    let x1i: Vec<usize> = xi.iter().map(|&i| (i + 1).min(dw - 1)).collect();
    let xf: Vec<f64> = xq.iter().zip(xi.iter()).map(|(&q, &i)| clip64(q - i as f64, 0.0, 1.0)).collect();
    for r in 0..n {
        let yq = ((y0 + r) as f64 + 0.5) / height as f64 * dh as f64 - 0.5;
        let yi = (yq.floor() as i64).clamp(0, dh as i64 - 1) as usize;
        let y1i = (yi + 1).min(dh - 1);
        let yf = clip64(yq - yi as f64, 0.0, 1.0);
        for x in 0..width {
            for ch in 0..c {
                let a = map_dec[(yi * dw + xi[x]) * c + ch] as f64;
                let b = map_dec[(yi * dw + x1i[x]) * c + ch] as f64;
                let cc = map_dec[(y1i * dw + xi[x]) * c + ch] as f64;
                let d = map_dec[(y1i * dw + x1i[x]) * c + ch] as f64;
                let v = (a * (1.0 - xf[x]) + b * xf[x]) * (1.0 - yf) + (cc * (1.0 - xf[x]) + d * xf[x]) * yf;
                out[(r * width + x) * c + ch] = v as f32;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Gaussian blurs
// ---------------------------------------------------------------------------

#[inline(always)]
fn reflect_index(i: i64, n: i64) -> usize {
    // NumPy 'reflect' has no edge repeat; a singleton always maps to 0.
    // Folding a period also bounds work when the radius exceeds the image.
    if n <= 1 {
        return 0;
    }
    if i >= 0 && i < n {
        return i as usize;
    }
    let period = 2 * (n - 1);
    let k = i.rem_euclid(period);
    k.min(period - k) as usize
}
#[inline(always)]
fn wrap_index(i: i64, n: i64) -> usize {
    (((i % n) + n) % n) as usize
}

/// The float32 taps of film_optics._gaussian_blur_slabbed for `sigma`.
pub fn gaussian_taps(sigma: f64) -> Vec<f32> {
    let radius = ((3.0 * sigma).ceil() as i64).max(1);
    let mut k: Vec<f64> = (-radius..=radius)
        .map(|x| {
            let y = x as f64 / sigma;
            (-0.5 * (y * y)).exp()
        })
        .collect();
    let s = pairwise_sum_f64(&k);
    for v in k.iter_mut() {
        *v /= s;
    }
    k.iter().map(|&v| v as f32).collect()
}

/// One separable convolution row, vertical then horizontal in tap order.
/// Source stays borrowed, including strided channel/visible-window views.
/// Only `vertical` and the caller's output row are used as working storage.
fn convolve_row(
    img: ArrayView3<'_, f32>, y: usize, k: &[f32], periodic: bool,
    vertical: &mut [f32], dst: &mut [f32],
) {
    let (h, w, c) = img.dim();
    let radius = (k.len() / 2) as i64;
    let idx = |i: i64, n: usize| -> usize {
        if periodic { wrap_index(i, n as i64) } else { reflect_index(i, n as i64) }
    };
    vertical.fill(0.0);
    for (i, &kv) in k.iter().enumerate() {
        let sy = idx(y as i64 + i as i64 - radius, h);
        let row = img.index_axis(Axis(0), sy);
        if let Some(src) = row.as_slice() {
            for (acc, &v) in vertical.iter_mut().zip(src) {
                *acc += kv * v;
            }
        } else if c == 1 {
            // A channel view of interleaved RGB is a simple strided vector;
            // avoid a two-dimensional iterator carry for every sample.
            for (acc, &v) in vertical.iter_mut().zip(row.column(0).iter()) {
                *acc += kv * v;
            }
        } else {
            for (acc, &v) in vertical.iter_mut().zip(row.iter()) {
                *acc += kv * v;
            }
        }
    }
    dst.fill(0.0);
    // Tap-major contiguous inner loops vectorize without changing any
    // pixel's accumulation order. Only the borders need index folding.
    for (i, &kv) in k.iter().enumerate() {
        let offset = i as i64 - radius;
        let x0 = (-offset).clamp(0, w as i64) as usize;
        let x1 = (w as i64 - offset).clamp(x0 as i64, w as i64) as usize;
        if x1 > x0 {
            let sx = (x0 as i64 + offset) as usize;
            let src = &vertical[sx * c..(sx + x1 - x0) * c];
            for (acc, &v) in dst[x0 * c..x1 * c].iter_mut().zip(src) {
                *acc += kv * v;
            }
        }
        for x in (0..x0).chain(x1..w) {
            let sx = idx(x as i64 + offset, w);
            for ch in 0..c {
                dst[x * c + ch] += kv * vertical[sx * c + ch];
            }
        }
    }
}

fn separable_blur(img: ArrayView3<'_, f32>, k: &[f32], periodic: bool) -> Vec<f32> {
    let (h, w, c) = img.dim();
    let mut out = vec![0.0f32; h * w * c];
    if out.is_empty() {
        return out;
    }
    par_rows(&mut out, w * c, h, |row0, chunk| {
        let mut vertical = vec![0.0f32; w * c];
        for (r, dst) in chunk.chunks_exact_mut(w * c).enumerate() {
            convolve_row(img, row0 + r, k, periodic, &mut vertical, dst);
        }
    });
    out
}

/// film_optics._gaussian_blur_slabbed: out of place, with one temporary row
/// per worker instead of a copied input plus a full vertical-pass image.
pub fn gaussian_blur(img: ArrayView3<'_, f32>, sigma: f64, periodic: bool) -> Vec<f32> {
    if sigma <= 0.0 {
        return img.iter().copied().collect();
    }
    separable_blur(img, &gaussian_taps(sigma), periodic)
}

fn small_sigma_taps(sigma_px: f64) -> [f32; 5] {
    let pi = std::f64::consts::PI;
    let s2 = sigma_px * sigma_px;
    let g_half = (-0.5 * s2 * ((pi / 2.0) * (pi / 2.0))).exp();
    let g_nyq = (-0.5 * s2 * (pi * pi)).exp();
    let b = (1.0 - g_nyq) / 4.0;
    let cc = ((1.0 + g_nyq) / 2.0 - g_half) / 4.0;
    let a = 1.0 - 2.0 * b - 2.0 * cc;
    [cc as f32, b as f32, a as f32, b as f32, cc as f32]
}

/// film_optics._blur_small_sigma on a borrowed (possibly strided) plane.
pub fn blur_small_sigma(chan: ArrayView2<'_, f32>, sigma_px: f64) -> Vec<f32> {
    separable_blur(chan.insert_axis(Axis(2)), &small_sigma_taps(sigma_px), false)
}

// ---------------------------------------------------------------------------
// gates and sources
// ---------------------------------------------------------------------------

#[inline(always)]
fn smootherstep_poly(t: f32) -> f32 {
    // poly = t*6; -= 15; *= t; += 10; *= t; *= t; *= t
    let mut poly = t * 6.0f32;
    poly -= 15.0;
    poly *= t;
    poly += 10.0;
    poly *= t;
    poly *= t;
    poly *= t;
    poly
}

/// film_optics.halation_layer_gate on one pixel (per layer).
#[inline(always)]
pub fn halation_layer_gate_px(e: [f32; 3], e_ref: [f32; 3], gate_ev: &[[f32; 2]; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for l in 0..3 {
        let mut t = fmax32(e[l], 1e-20);
        t /= fmax32(e_ref[l], 1e-20);
        t = t.log2();
        t -= gate_ev[l][0];
        t /= fmax32(gate_ev[l][1] - gate_ev[l][0], 1e-6);
        t = clip32(t, 0.0, 1.0);
        out[l] = smootherstep_poly(t);
    }
    out
}

/// film_optics.halation_pointwise_return: sum over components of A_i @ (gate*e),
/// each product via einsum (sequential) and accumulated in component order.
pub fn halation_pointwise_return_px(e: [f32; 3], e_ref: [f32; 3], comps: &[HalationComponent]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for comp in comps {
        let g = halation_layer_gate_px(e, e_ref, &comp.gate_ev);
        let u = [g[0] * e[0], g[1] * e[1], g[2] * e[2]];
        for c in 0..3 {
            let t = &comp.transfer[c];
            out[c] += (t[0] * u[0] + t[1] * u[1]) + t[2] * u[2];
        }
    }
    out
}

pub struct HalationComponent {
    pub gate_ev: [[f32; 2]; 3],
    /// row-major 3x3, already float32
    pub transfer: [[f32; 3]; 3],
}

/// film_optics.halation_component_source on one pixel: (gate*e) @ transfer.T
/// through Accelerate's FMA chain.
#[inline(always)]
pub fn halation_component_source_px(e: [f32; 3], e_ref: [f32; 3], comp: &HalationComponent) -> [f32; 3] {
    let g = halation_layer_gate_px(e, e_ref, &comp.gate_ev);
    let u = [g[0] * e[0], g[1] * e[1], g[2] * e[2]];
    mat3_fma_chain_f32(u, &comp.transfer)
}

pub const REC2020_LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// film_optics.capture_bloom_gate on one value.
#[inline(always)]
pub fn capture_bloom_gate_px(y_over_grey: f32, t0: f64, t1: f64) -> f32 {
    // np.float32(t0) and np.float32(max(t1 - t0, 1e-6)): Python-float arithmetic first
    let t0_32 = t0 as f32;
    let span_32 = f64::max(t1 - t0, 1e-6) as f32;
    let mut t = fmax32(y_over_grey, 1e-20);
    t = t.log2();
    t -= t0_32;
    t /= span_32;
    t = clip32(t, 0.0, 1.0);
    smootherstep_poly(t)
}

/// film_optics.capture_bloom_source_rows on one pixel: rgb * gate(y/0.18).
#[inline(always)]
pub fn capture_bloom_source_px(rgb: [f32; 3], t0: f64, t1: f64) -> [f32; 3] {
    let mut y = dot3_seq_f32(rgb, REC2020_LUMA);
    y /= 0.18f32;
    let g = capture_bloom_gate_px(y, t0, t1);
    [rgb[0] * g, rgb[1] * g, rgb[2] * g]
}

/// film_optics.capture_bloom_apply_rows on one pixel given its upsampled glow.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn capture_bloom_apply_px(
    img: [f32; 3],
    glow: [f32; 3],
    t0: f64,
    t1: f64,
    core_ratio: [f64; 2],
    save_lights: f32,
    saturation: f32,
    amount: f32,
) -> [f32; 3] {
    let source = capture_bloom_source_px(img, t0, t1);
    let gy = dot3_seq_f32(glow, REC2020_LUMA);
    let sy = dot3_seq_f32(source, REC2020_LUMA);
    let ratio = sy / fmax32(gy, 1e-12);
    let cr0 = core_ratio[0] as f32;
    let cr_span = f64::max(core_ratio[1] - core_ratio[0], 1e-6) as f32;
    let t = clip32((ratio - cr0) / cr_span, 0.0, 1.0);
    // t * t * t * (t * (t * 6.0 - 15.0) + 10.0) with weak Python floats -> float32
    let smooth = ((t * t) * t) * ((t * (t * 6.0 - 15.0)) + 10.0);
    let w_core = 1.0 - save_lights * smooth;
    let mut delta = [glow[0] * w_core, glow[1] * w_core, glow[2] * w_core];
    if saturation != 1.0 {
        let dy = dot3_seq_f32(delta, REC2020_LUMA);
        for c in 0..3 {
            delta[c] = fmax32(dy + saturation * (delta[c] - dy), 0.0);
        }
    }
    [
        fmax32(img[0] + amount * delta[0], 0.0),
        fmax32(img[1] + amount * delta[1], 0.0),
        fmax32(img[2] + amount * delta[2], 0.0),
    ]
}

// ---------------------------------------------------------------------------
// scatter mix
// ---------------------------------------------------------------------------

/// One channel's ACTIVE blur components as film_optics._scatter_components
/// resolves them on the Python side: (sigma_px, weight).
pub struct ScatterChannel {
    pub s_mix: f64,
    pub comps: Vec<(f64, f64)>,
}

/// film_optics.apply_scatter_mix on an (h, w, 3) float32 slab.
pub fn apply_scatter_mix(img: ArrayView3<'_, f32>, chans: &[ScatterChannel; 3]) -> Vec<f32> {
    let (h, w, _) = img.dim();
    let mut out = vec![0.0f32; h * w * 3];
    if out.is_empty() {
        return out;
    }
    let prepared: Vec<(f32, Vec<(Vec<f32>, f32)>)> = chans.iter().map(|sc| {
        let mut wsum = 0.0f64;
        for &(_, weight) in &sc.comps {
            wsum += weight; // Python sum(): sequential
        }
        let inert = (1.0 - sc.s_mix * wsum) as f32;
        let comps = sc.comps.iter().map(|&(scale, weight)| {
            let taps = if scale < 1.0 { small_sigma_taps(scale).to_vec() } else { gaussian_taps(scale) };
            (taps, (sc.s_mix * weight) as f32)
        }).collect();
        (inert, comps)
    }).collect();
    // One row pool owns the budget. Channels/components stay inside each
    // worker, with two reused rows instead of whole-plane accumulators,
    // blurred images and nested per-channel Gaussian pools.
    par_rows(&mut out, w * 3, h, |row0, chunk| {
        let mut vertical = vec![0.0f32; w];
        let mut blurred = vec![0.0f32; w];
        for (r, dst) in chunk.chunks_exact_mut(w * 3).enumerate() {
            let y = row0 + r;
            for ch in 0..3 {
                let (inert, comps) = &prepared[ch];
                for x in 0..w {
                    dst[x * 3 + ch] = img[[y, x, ch]] * *inert;
                }
                let plane = img.slice(s![.., .., ch..ch + 1]);
                for (taps, weight) in comps {
                    convolve_row(plane, y, taps, false, &mut vertical, &mut blurred);
                    for (px, &v) in dst.chunks_exact_mut(3).zip(&blurred) {
                        px[ch] += *weight * v;
                    }
                }
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------
// film-space grain: integral-image sampling and density grain
// ---------------------------------------------------------------------------

/// The integral image of the master grain field: (gh+1, gw+1, c) float32.
pub struct IntegralImage<'a> {
    pub data: &'a [f32],
    /// logical (possibly transposed) grid: gh rows x gw cols of cells
    pub gh: usize,
    pub gw: usize,
    pub c: usize,
    /// the stored master is landscape (gw_store = gh when rotated): index transposed
    pub rotated: bool,
}

impl<'a> IntegralImage<'a> {
    #[inline(always)]
    fn at(&self, y: usize, x: usize, ch: usize) -> f64 {
        let (sy, sx, stride) = if self.rotated {
            (x, y, self.gh + 1)
        } else {
            (y, x, self.gw + 1)
        };
        self.data[(sy * stride + sx) * self.c + ch] as f64
    }
}

/// sample_field._ii_at on the edge grid: bilinear lookup of the integral image
/// at (yq, xq) for every channel.
fn ii_at(ii: &IntegralImage, yq: f64, xq: f64, out: &mut [f64]) {
    let yi = (yq.floor() as i64).clamp(0, ii.gh as i64 - 1) as usize;
    let xi = (xq.floor() as i64).clamp(0, ii.gw as i64 - 1) as usize;
    let yf = yq - yi as f64;
    let xf = xq - xi as f64;
    for ch in 0..ii.c {
        let top = ii.at(yi, xi, ch) * (1.0 - xf) + ii.at(yi, xi + 1, ch) * xf;
        let bot = ii.at(yi + 1, xi, ch) * (1.0 - xf) + ii.at(yi + 1, xi + 1, ch) * xf;
        out[ch] = top * (1.0 - yf) + bot * yf;
    }
}

fn interp1(ii: &IntegralImage, table_row: bool, q: f64, n: usize, out: &mut [f64]) {
    // table_row: row_tot = ii[gh] (indexed by x); else col_tot = ii[:, gw] (indexed by y)
    let qi = (q.floor() as i64).clamp(0, n as i64 - 1) as usize;
    let qf = q - qi as f64;
    for ch in 0..ii.c {
        let (a, b) = if table_row {
            (ii.at(ii.gh, qi, ch), ii.at(ii.gh, qi + 1, ch))
        } else {
            (ii.at(qi, ii.gw, ch), ii.at(qi + 1, ii.gw, ch))
        };
        out[ch] = a * (1.0 - qf) + b * qf;
    }
}

fn periodic_at(ii: &IntegralImage, yq: f64, xq: f64, out: &mut [f64]) {
    let gh = ii.gh as f64;
    let gw = ii.gw as f64;
    let ky = (yq / gh).floor();
    let kx = (xq / gw).floor();
    let ry = yq - ky * gh;
    let rx = xq - kx * gw;
    let c = ii.c;
    let mut base = vec![0.0f64; c];
    let mut wy = vec![0.0f64; c];
    let mut wx = vec![0.0f64; c];
    ii_at(ii, ry, rx, &mut base);
    interp1(ii, true, rx, ii.gw, &mut wy);
    interp1(ii, false, ry, ii.gh, &mut wx);
    for ch in 0..c {
        let total = ii.at(ii.gh, ii.gw, ch);
        out[ch] = base[ch] + ky * wy[ch] + kx * wx[ch] + (ky * kx) * total;
    }
}

/// The edge arrays of sample_field for one geometry and phase.
pub struct FieldEdges {
    /// query edges (mod-mapped when phased) and area edges, per axis
    pub ye_q: Vec<f64>,
    pub ye: Vec<f64>,
    pub xe_q: Vec<f64>,
    pub xe: Vec<f64>,
    pub straddle_row: Option<usize>,
    pub straddle_col: Option<usize>,
    pub phased: bool,
}

/// sample_field's edge construction. `rotated` swaps the gate sides and the
/// phase like the Python (the caller passes the already-transposed integral
/// image when rotated).
#[allow(clippy::too_many_arguments)]
pub fn field_edges(
    height: usize,
    width: usize,
    x0: f64,
    y0: f64,
    w_mm: f64,
    h_mm: f64,
    gate_w_mm: f64,
    gate_h_mm: f64,
    gh: usize,
    gw: usize,
    phase: (i64, i64),
) -> FieldEdges {
    let ye: Vec<f64> = (0..=height)
        .map(|i| clip64((y0 + h_mm * i as f64 / height as f64) / gate_h_mm * gh as f64, 0.0, gh as f64) + phase.0 as f64)
        .collect();
    let xe: Vec<f64> = (0..=width)
        .map(|i| clip64((x0 + w_mm * i as f64 / width as f64) / gate_w_mm * gw as f64, 0.0, gw as f64) + phase.1 as f64)
        .collect();
    if phase == (0, 0) {
        return FieldEdges { ye_q: ye.clone(), ye, xe_q: xe.clone(), xe, straddle_row: None, straddle_col: None, phased: false };
    }
    let g = |v: f64, n: usize| v - n as f64 * (v / n as f64).floor();
    let ye_q: Vec<f64> = ye.iter().map(|&v| g(v, gh)).collect();
    let xe_q: Vec<f64> = xe.iter().map(|&v| g(v, gw)).collect();
    let straddler = |edges: &[f64], n: usize| -> Option<usize> {
        for i in 0..edges.len() - 1 {
            if g(edges[i], n) > g(edges[i + 1], n) {
                return Some(i);
            }
        }
        None
    };
    FieldEdges {
        straddle_row: straddler(&ye, gh),
        straddle_col: straddler(&xe, gw),
        ye_q,
        ye,
        xe_q,
        xe,
        phased: true,
    }
}

/// sample_field: area-integrated sampling of the periodic master onto the
/// pixel grid; returns (height, width, c) float32.
pub fn sample_field(ii: &IntegralImage, edges: &FieldEdges, height: usize, width: usize) -> Vec<f32> {
    let c = ii.c;
    let mut out = vec![0.0f32; height * width * c];
    // main fill with the (possibly mod-mapped) single-lookup query
    fill(ii, edges, 0, height, 0, width, &edges.ye_q, &edges.xe_q, false, &mut out);
    if edges.phased {
        let gh = ii.gh as f64;
        let gw = ii.gw as f64;
        // shift into [0, 2G) once, by the first element's period
        let ph_ye: Vec<f64> = edges.ye.iter().map(|&v| v - gh * (edges.ye[0] / gh).floor()).collect();
        let ph_xe: Vec<f64> = edges.xe.iter().map(|&v| v - gw * (edges.xe[0] / gw).floor()).collect();
        if let Some(r) = edges.straddle_row {
            fill(ii, edges, r, r + 1, 0, width, &ph_ye, &ph_xe, true, &mut out);
        }
        if let Some(cidx) = edges.straddle_col {
            fill(ii, edges, 0, height, cidx, cidx + 1, &ph_ye, &ph_xe, true, &mut out);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn fill(
    ii: &IntegralImage,
    edges: &FieldEdges,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    yq_e: &[f64],
    xq_e: &[f64],
    periodic: bool,
    out: &mut [f32],
) {
    let c = ii.c;
    let width = edges.xe.len() - 1;
    let ncols = c1 - c0;
    if r1 <= r0 || ncols == 0 {
        return;
    }
    // e[row][col][ch] on the (rows+1) x (cols+1) edge grid, then difference
    let mut e = vec![0.0f64; (r1 - r0 + 1) * (ncols + 1) * c];
    let mut tmp = vec![0.0f64; c];
    for (ri, r) in (r0..=r1).enumerate() {
        for (ci, col) in (c0..=c1).enumerate() {
            if periodic {
                periodic_at(ii, yq_e[r], xq_e[col], &mut tmp);
            } else {
                ii_at(ii, yq_e[r], xq_e[col], &mut tmp);
            }
            e[(ri * (ncols + 1) + ci) * c..(ri * (ncols + 1) + ci + 1) * c].copy_from_slice(&tmp);
        }
    }
    for (ri, r) in (r0..r1).enumerate() {
        let dy = edges.ye[r + 1] - edges.ye[r];
        for (ci, col) in (c0..c1).enumerate() {
            let area_x = fmax64(edges.xe[col + 1] - edges.xe[col], 1e-12);
            let area = fmax64(dy * area_x, 1e-12);
            for ch in 0..c {
                let s11 = e[((ri + 1) * (ncols + 1) + ci + 1) * c + ch];
                let s10 = e[((ri + 1) * (ncols + 1) + ci) * c + ch];
                let s01 = e[(ri * (ncols + 1) + ci + 1) * c + ch];
                let s00 = e[(ri * (ncols + 1) + ci) * c + ch];
                out[(r * width + col) * c + ch] = ((s11 - s10 - s01 + s00) / area) as f32;
            }
        }
    }
}

/// film_optics.apply_density_grain, band_limited_gaussian_v1 branch.
/// `amounts` (n,3) float64 in, `field` (n,3) float32; returns (n,3) float64.
pub fn density_grain_v1(amounts: &[f64], lo: [f64; 3], hi: [f64; 3], field: &[f32], sigma_mul: f64) -> Vec<f64> {
    // sigma_mul = float(amount) * grain.sigma0 * 4.0 (Python scalar product first)
    let n = amounts.len() / 3;
    let mut out = vec![0.0f64; n * 3];
    let span = [fmax64(hi[0] - lo[0], 1e-9), fmax64(hi[1] - lo[1], 1e-9), fmax64(hi[2] - lo[2], 1e-9)];
    for i in 0..n {
        for ch in 0..3 {
            let a = amounts[i * 3 + ch];
            let dn = clip64((a - lo[ch]) / span[ch], 0.0, 1.0);
            let sigma = sigma_mul * dn * (1.0 - dn);
            out[i * 3 + ch] = a + sigma * span[ch] * field[i * 3 + ch] as f64;
        }
    }
    out
}

pub struct SigmaTable {
    pub base: f32,
    pub d: Vec<f64>,
    pub sigma: Vec<f64>,
}

/// film_optics.apply_density_grain, measured_sigma_v2 branch.
pub fn density_grain_v2(amounts: &[f64], field: &[f32], tables: &[SigmaTable; 3], amount_over_rms: f32) -> Vec<f64> {
    let n = amounts.len() / 3;
    let mut out = amounts.to_vec();
    let ln10 = (10.0f64).ln() as f32;
    let half_ln10 = ln10 * 0.5f32;
    par_rows_f64(&mut out, 3, n, |row0, chunk| {
        for (r, px) in chunk.chunks_exact_mut(3).enumerate() {
            let i = row0 + r;
            for ch in 0..3 {
                let tab = &tables[ch];
                let mut s = amounts[i * 3 + ch] as f32;
                s += tab.base;
                s = np_interp(s as f64, &tab.d, &tab.sigma) as f32;
                s *= amount_over_rms;
                px[ch] += (s * field[i * 3 + ch]) as f64;
                let mut s2 = s * s;
                s2 *= half_ln10;
                px[ch] += s2 as f64;
            }
        }
    });
    out
}

/// film_optics.halation_reinject_rows on rows [y0, y1) given the upsampled
/// spread for those rows. `log_e` (n*w, 3) float64; returns float64.
#[allow(clippy::too_many_arguments)]
pub fn halation_reinject(
    log_e: &[f64],
    give_lin: Option<&[f32]>,
    spread_up: &[f32],
    e_ref: [f32; 3],
    comps: &[HalationComponent],
    residual: bool,
    amount: f32,
) -> Vec<f64> {
    let n = log_e.len() / 3;
    let mut out = vec![0.0f64; n * 3];
    for i in 0..n {
        let lin = [
            (10.0f64).powf(log_e[i * 3]) as f32,
            (10.0f64).powf(log_e[i * 3 + 1]) as f32,
            (10.0f64).powf(log_e[i * 3 + 2]) as f32,
        ];
        let give = if residual {
            let src = match give_lin {
                Some(g) => [g[i * 3], g[i * 3 + 1], g[i * 3 + 2]],
                None => lin,
            };
            let g = halation_pointwise_return_px(src, e_ref, comps);
            [fmin32(amount * g[0], lin[0]), fmin32(amount * g[1], lin[1]), fmin32(amount * g[2], lin[2])]
        } else {
            [0.0, 0.0, 0.0]
        };
        for ch in 0..3 {
            let take = amount * spread_up[i * 3 + ch];
            let v = fmax32(lin[ch] + take - give[ch], 0.0);
            out[i * 3 + ch] = fmax64(v as f64, 1e-12).log10();
        }
    }
    out
}
