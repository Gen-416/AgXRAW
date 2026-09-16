// SPDX-License-Identifier: GPL-3.0-or-later
//! Decode-side evidence kernels (dngscan/raw_io.py): clip-mask feathering and
//! DNG GainMap opcode application. Both replicate the NumPy reference's
//! float32/float64 operation order element for element.
use half::f16;

/// raw_io._feather_masks_f16: separable [1,4,6,4,1]/16 filter with edge
/// clamping, clipped to [0, 1] and stored as float16 (round to nearest even).
/// `mask` is (h, w, c) float32 contiguous; returns (h, w, c) float16.
pub fn feather_masks_f16(mask: &[f32], h: usize, w: usize, c: usize) -> Vec<f16> {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut out = vec![f16::ZERO; h * w * c];
    if h == 0 || w == 0 || c == 0 {
        return out;
    }
    // channels are independent: one thread each (the C++/NumPy order of
    // operations is per element, so the split changes nothing)
    let planes: Vec<Vec<f16>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..c)
            .map(|ch| {
                s.spawn(move || {
                    let clamp_row = |r: isize| -> usize { r.clamp(0, h as isize - 1) as usize };
                    let clamp_col = |x: isize| -> usize { x.clamp(0, w as isize - 1) as usize };
                    let mut vbuf = vec![0.0f32; h * w];
                    for y in 0..h {
                        let dst = &mut vbuf[y * w..(y + 1) * w];
                        for (i, wgt) in K.iter().enumerate() {
                            let r = clamp_row(y as isize + i as isize - 2);
                            let row = &mask[(r * w) * c..(r * w + w) * c];
                            for x in 0..w {
                                // acc = acc + plane * weight, in that order
                                dst[x] += row[x * c + ch] * *wgt;
                            }
                        }
                    }
                    let mut plane = vec![f16::ZERO; h * w];
                    for y in 0..h {
                        let row = &vbuf[y * w..(y + 1) * w];
                        for x in 0..w {
                            let mut acc = 0.0f32;
                            for (i, wgt) in K.iter().enumerate() {
                                let xi = clamp_col(x as isize + i as isize - 2);
                                acc += row[xi] * *wgt;
                            }
                            let v = if acc < 0.0 {
                                0.0
                            } else if acc > 1.0 {
                                1.0
                            } else {
                                acc
                            };
                            plane[y * w + x] = f16::from_f32(v);
                        }
                    }
                    plane
                })
            })
            .collect();
        handles.into_iter().map(|hd| hd.join().expect("feather thread")).collect()
    });
    for (ch, plane) in planes.iter().enumerate() {
        for (i, v) in plane.iter().enumerate() {
            out[i * c + ch] = *v;
        }
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
/// `img`/`colors` are accessed through closures over the (possibly strided)
/// visible-area views; `blacks`/`whites` are the per-CFA-channel tables (a
/// single entry means the scalar path).
pub fn apply_gain_map_mosaic(
    h: usize,
    w: usize,
    op: &GainMapOp,
    blacks: &[f32],
    whites: &[f32],
    color_at: &dyn Fn(usize, usize) -> usize,
    get: &dyn Fn(usize, usize) -> u16,
    set: &mut dyn FnMut(usize, usize, u16),
) {
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
            let sub = get(y, xx) as f32;
            let cid = color_at(y, xx);
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
            set(y, xx, v as u16);
        }
    }
}
