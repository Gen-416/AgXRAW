// SPDX-License-Identifier: GPL-3.0-or-later
//! Generic area decimation and bilinear upsampling, matching dngscan/spatial.py.
use numpy::ndarray::ArrayView3;

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


pub enum Acc<'a> {
    F64(&'a mut [f64]),
    F32(&'a mut [f32]),
}

/// spatial.area_decimate_rows: accumulate source rows [y0, y0+n) (float32
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

/// spatial.upsample_rows: bilinear upsample of the decimated map for
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
