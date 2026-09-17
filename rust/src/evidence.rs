// SPDX-License-Identifier: GPL-3.0-or-later
//! Decode-side evidence kernels (dngscan/raw_io.py): clip-mask feathering and
//! DNG GainMap opcode application. Both replicate the NumPy reference's
//! float32/float64 operation order element for element.
use half::f16;
use numpy::ndarray::{ArrayView2, ArrayViewMut2};

/// raw_io._feather_masks_f16: separable [1,4,6,4,1]/16 filter with edge
/// clamping, clipped to [0, 1] and stored as float16 (round to nearest even).
/// `mask` is (h, w, c) float32 contiguous; returns (h, w, c) float16.
pub fn feather_masks_f16(mask: &[f32], h: usize, w: usize, c: usize) -> Vec<f16> {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut out = vec![f16::ZERO; h * w * c];
    if h == 0 || w == 0 || c == 0 {
        return out;
    }
    // Each horizontal pass needs only one vertically filtered row. Write
    // directly into disjoint output rows: O(workers * w * c) scratch, and
    // obey the renderer's thread budget instead of spawning per channel.
    let row_len = w * c;
    let process = |row0: usize, chunk: &mut [f16]| {
        let mut vertical = vec![0.0f32; row_len];
        for (r, dst) in chunk.chunks_exact_mut(row_len).enumerate() {
            let y = row0 + r;
            vertical.fill(0.0);
            for (i, &weight) in K.iter().enumerate() {
                let sy = (y as isize + i as isize - 2).clamp(0, h as isize - 1) as usize;
                let src = &mask[sy * row_len..(sy + 1) * row_len];
                for (acc, &v) in vertical.iter_mut().zip(src) {
                    *acc += v * weight;
                }
            }
            for x in 0..w {
                for ch in 0..c {
                    let mut acc = 0.0f32;
                    for (i, &weight) in K.iter().enumerate() {
                        let sx = (x as isize + i as isize - 2).clamp(0, w as isize - 1) as usize;
                        acc += vertical[sx * c + ch] * weight;
                    }
                    dst[x * c + ch] = f16::from_f32(acc.clamp(0.0, 1.0));
                }
            }
        }
    };
    let workers = (crate::budget::workers_for(h * w) as usize).min(h);
    if workers == 1 {
        process(0, &mut out);
    } else {
        let rows = h.div_ceil(workers);
        std::thread::scope(|s| {
            for (i, chunk) in out.chunks_mut(rows * row_len).enumerate() {
                let process = &process;
                s.spawn(move || process(i * rows, chunk));
            }
        });
    }
    out
}

/// One DNG GainMap opcode (dng_opcode_GainMap) as the Python reader hands it over.
pub struct GainMapOp<'a> {
    pub top: i64,
    pub left: i64,
    pub bottom: i64,
    pub right: i64,
    pub row_pitch: i64,
    pub col_pitch: i64,
    pub origin_v: f64,
    pub origin_h: f64,
    pub spacing_v: f64,
    pub spacing_h: f64,
    pub points_v: i64,
    pub points_h: i64,
    /// map plane 0, row-major (points_v, points_h)
    pub gains: &'a [f64],
}

/// raw_io._apply_gain_maps_mosaic for one opcode: gains the mosaic in place.
/// `img`/`colors` are borrowed directly as the (possibly strided)
/// visible-area views; `blacks`/`whites` are the per-CFA-channel tables (a
/// single entry means the scalar path).
pub fn apply_gain_map_mosaic(
    mut img: ArrayViewMut2<'_, u16>,
    colors: ArrayView2<'_, u8>,
    op: &GainMapOp,
    blacks: &[f32],
    whites: &[f32],
) {
    let (h, w) = img.dim();
    let hi = h as i64;
    let wi = w as i64;
    let bottom = op.bottom.min(hi);
    let right = op.right.min(wi);
    if op.row_pitch <= 0 || op.col_pitch <= 0 || op.top >= bottom || op.left >= right {
        return;
    }
    let rows: Vec<i64> = (op.top..bottom).step_by(op.row_pitch as usize).collect();
    let cols: Vec<i64> = (op.left..right).step_by(op.col_pitch as usize).collect();
    if rows.is_empty() || cols.is_empty() {
        return;
    }
    let pv = op.points_v;
    let ph = op.points_h;
    let sv = if op.spacing_v > 1e-9 { op.spacing_v } else { 1e-9 };
    let sh = if op.spacing_h > 1e-9 { op.spacing_h } else { 1e-9 };
    // np.clip(x, 0, n-1) on float64, then floor -> int, clip to [0, n-2]
    let axis = |idx: i64, extent: i64, origin: f64, spacing: f64, points: i64| -> (usize, f64) {
        if points > 1 {
            let mut v = ((idx as f64 + 0.5) / extent as f64 - origin) / spacing;
            if v < 0.0 {
                v = 0.0;
            }
            if v > (points - 1) as f64 {
                v = (points - 1) as f64;
            }
            let i0 = (v.floor() as i64).clamp(0, points - 2) as usize;
            (i0, v - i0 as f64)
        } else {
            (0, 0.0)
        }
    };
    let (v0s, fvs): (Vec<usize>, Vec<f64>) = rows.iter().map(|&r| axis(r, hi, op.origin_v, sv, pv)).unzip();
    let (h0s, fhs): (Vec<usize>, Vec<f64>) = cols.iter().map(|&x| axis(x, wi, op.origin_h, sh, ph)).unzip();
    let phu = ph as usize;
    let blacks_scalar = blacks.len() <= 1;
    let whites_scalar = whites.len() <= 1;
    let b_scalar = blacks.first().copied().unwrap_or(0.0);
    let w_scalar = whites.first().copied().unwrap_or(0.0);
    for (ri, &r) in rows.iter().enumerate() {
        let v0 = v0s[ri];
        let v1 = (v0 + 1).min((pv - 1).max(0) as usize);
        let fv = fvs[ri];
        let row_lo = &op.gains[v0 * phu..v0 * phu + phu];
        let row_hi = &op.gains[v1 * phu..v1 * phu + phu];
        for (ci, &x) in cols.iter().enumerate() {
            let h0 = h0s[ci];
            let h1 = (h0 + 1).min((ph - 1).max(0) as usize);
            let fh = fhs[ci];
            let g00 = row_lo[h0];
            let g01 = row_lo[h1];
            let g10 = row_hi[h0];
            let g11 = row_hi[h1];
            // (g00 * (1 - fv) * (1 - fh) + g01 * (1 - fv) * fh + g10 * fv * (1 - fh) + g11 * fv * fh)
            let gains = g00 * (1.0 - fv) * (1.0 - fh) + g01 * (1.0 - fv) * fh
                + g10 * fv * (1.0 - fh)
                + g11 * fv * fh;
            let (y, xx) = (r as usize, x as usize);
            let sub = img[[y, xx]] as f32;
            let cid = colors[[y, xx]] as usize;
            let b = if blacks_scalar { b_scalar } else { blacks[cid.min(blacks.len() - 1)] };
            let wl = if whites_scalar { w_scalar } else { whites[cid.min(whites.len() - 1)] };
            // np.clip(b + (sub - b) * gains, 0.0, wl): float32 difference, float64 product and sum
            let corrected = b as f64 + ((sub - b) as f64) * gains;
            let mut v = corrected;
            if v < 0.0 {
                v = 0.0;
            }
            if v > wl as f64 {
                v = wl as f64;
            }
            img[[y, xx]] = v as u16;
        }
    }
}
