// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared u8 pixel scan for the distinct absolute-base and coding gates.
use numpy::ndarray::ArrayView3;
use crate::metrics::BaseRoundtrip;
use crate::stats::percentile_f32;

pub struct BaseCoding {
    pub base: BaseRoundtrip,
    pub luma_rmse: f64,
    pub chroma_rmse: f64,
    pub local_luma_p99: f64,
}

// NumPy's contiguous float32 sum: eight lanes, then the same binary tree.
fn sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 { return a.iter().fold(-0.0_f32, |sum, &v| sum + v); }
    if n > 128 {
        let middle = (n / 2) & !7;
        return sum_f32(&a[..middle]) + sum_f32(&a[middle..]);
    }
    let mut lanes = [0.0; 8];
    lanes.copy_from_slice(&a[..8]);
    for i in (8..n - n % 8).step_by(8) {
        for j in 0..8 { lanes[j] += a[i + j]; }
    }
    let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for &v in &a[n - n % 8..] { sum += v; }
    sum
}

fn sum_f64(a: &[f32]) -> f64 {
    let n = a.len();
    if n < 8 { return a.iter().fold(-0.0_f64, |sum, &v| sum + v as f64); }
    if n > 128 {
        let middle = (n / 2) & !7;
        return sum_f64(&a[..middle]) + sum_f64(&a[middle..]);
    }
    let mut lanes = [0.0_f64; 8];
    for j in 0..8 { lanes[j] = a[j] as f64; }
    for i in (8..n - n % 8).step_by(8) {
        for j in 0..8 { lanes[j] += a[i + j] as f64; }
    }
    let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for &v in &a[n - n % 8..] { sum += v as f64; }
    sum
}

fn local_errors(dy: &[f32], h: usize, w: usize, out: &mut Vec<f32>) {
    let (h8, w8) = (h / 8 * 8, w / 8 * 8);
    if h8 == 0 || w8 == 0 {
        out.push(sum_f32(dy) / dy.len() as f32);
        return;
    }
    // np.mean(axis=(1,3)): pairwise sum along each contiguous eight-value
    // row, then sequentially add the rows in their original order.
    let row_blocks = |y0: usize, rows: usize, out: &mut Vec<f32>| {
        for bx in 0..w8 / 8 {
            if w8 == 8 {
                // A singleton block-column axis coalesces into one contiguous
                // reduction in NumPy, unlike the interleaved multi-block case.
                let mut block = [0.0_f32; 64];
                for y in 0..rows { block[y * 8..y * 8 + 8].copy_from_slice(&dy[(y0 + y) * w..(y0 + y) * w + 8]); }
                out.push(sum_f32(&block[..rows * 8]) / (rows * 8) as f32);
                continue;
            }
            let mut sum = 0.0_f32;
            for y in y0..y0 + rows { sum += sum_f32(&dy[y * w + bx * 8..y * w + bx * 8 + 8]); }
            out.push(sum / (rows * 8) as f32);
        }
    };
    for y in (0..h8).step_by(8) { row_blocks(y, 8, out); }
    if h8 < h { row_blocks(h8, h - h8, out); }
    if w8 < w {
        let mut block = [0.0_f32; 64];
        for y0 in (0..h8).step_by(8) {
            let mut count = 0;
            for y in y0..y0 + 8 {
                for x in w8..w { block[count] = dy[y * w + x]; count += 1; }
            }
            out.push(sum_f32(&block[..count]) / count as f32);
        }
        if h8 < h {
            let mut count = 0;
            for y in h8..h { for x in w8..w { block[count] = dy[y * w + x]; count += 1; } }
            out.push(sum_f32(&block[..count]) / count as f32);
        }
    }
}

pub fn base_and_coding(decoded: ArrayView3<'_, u8>, intended: ArrayView3<'_, u8>, sum_buffer_size: usize) -> BaseCoding {
    let (h, w, _) = decoded.dim();
    let total = h * w;
    let (h8, w8) = (h / 8 * 8, w / 8 * 8);
    let mut histogram = [0usize; 256];
    let mut abs_sum = 0.0_f64;
    let mut signed_sum = [0.0_f64; 3];
    let mut base_blocks = Vec::new();
    let mut local_blocks = Vec::new();
    let mut luma_sq = 0.0_f64;
    let mut chroma_sq = 0.0_f64;
    // Scratch is one original coding band, independent of image height.
    let mut dy = vec![0.0_f32; 128.min(h) * w];
    let mut squares = [dy.clone(), dy.clone(), dy.clone()];
    for y0 in (0..h).step_by(128) {
        let rows = (h - y0).min(128);
        let n = rows * w;
        let mut means = vec![[0.0_f32; 3]; rows / 8 * (w8 / 8)];
        for y in 0..rows {
            for x in 0..w {
                let mut diff = [0.0_f32; 3];
                for c in 0..3 {
                    diff[c] = decoded[[y0 + y, x, c]] as f32 - intended[[y0 + y, x, c]] as f32;
                    abs_sum += diff[c].abs() as f64;
                    signed_sum[c] += diff[c] as f64;
                    if y < rows / 8 * 8 && x < w8 { means[y / 8 * (w8 / 8) + x / 8][c] += diff[c]; }
                }
                histogram[diff[0].abs().max(diff[1].abs()).max(diff[2].abs()) as usize] += 1;
                let luma = (diff[0] * 0.299_f32 + diff[1] * 0.587_f32) + diff[2] * 0.114_f32;
                let i = y * w + x;
                dy[i] = luma.abs();
                squares[0][i] = luma * luma;
                let red = diff[0] - luma;
                let blue = diff[2] - luma;
                squares[1][i] = red * red;
                squares[2][i] = blue * blue;
            }
        }
        // float32 -> float64 dtype conversion uses NumPy's configured ufunc
        // buffer: pairwise within each buffer, then scalar sum across buffers.
        let sum = |values: &[f32]| values.chunks(sum_buffer_size).fold(0.0_f64, |acc, chunk| acc + sum_f64(chunk));
        luma_sq += sum(&squares[0][..n]);
        // Preserve per-band R then B addition, not a pooled chroma reduction.
        chroma_sq += sum(&squares[1][..n]);
        chroma_sq += sum(&squares[2][..n]);
        local_errors(&dy[..n], rows, w, &mut local_blocks);
        for mean in means { base_blocks.push((mean[0] / 64.0).abs().max((mean[1] / 64.0).abs()).max((mean[2] / 64.0).abs())); }
    }
    let rank = |rank: usize| {
        let mut accumulated = 0;
        for (value, count) in histogram.iter().enumerate() {
            accumulated += count;
            if accumulated > rank { return value as f32; }
        }
        0.0
    };
    let pos = (total as f64 - 1.0) * 0.99;
    let lo = rank(pos.floor() as usize);
    let hi = rank(pos.ceil() as usize);
    let frac = pos - pos.floor();
    let p99 = if frac >= 0.5 { hi - (hi - lo) * ((1.0 - frac) as f32) }
              else { lo + (hi - lo) * (frac as f32) };
    let max_code = histogram.iter().rposition(|&count| count != 0).unwrap_or(0) as f64;
    let local_count = local_blocks.len();
    // The Python coding oracle rounds (1-frac) after the float64 subtraction.
    // Existing HDR scalar percentile helpers retain their established ABI math.
    let local_luma_p99 = crate::stats::upper_percentile_selected(&mut local_blocks, local_count, 99.0)
        .expect("the complete local population was retained") as f64;
    BaseCoding {
        base: BaseRoundtrip {
            mean_code_error: abs_sum / (total as f64 * 3.0),
            p99_code_error: p99 as f64,
            max_code_error: max_code,
            channel_bias_code_error: signed_sum.iter().map(|v| (v / total as f64).abs()).fold(0.0, f64::max),
            block_p99_code_error: if h8 > 0 && w8 > 0 { percentile_f32(&mut base_blocks, 99.0) as f64 } else { f64::INFINITY },
        },
        luma_rmse: (luma_sq / total as f64).sqrt(),
        chroma_rmse: (chroma_sq / (2 * total) as f64).sqrt(),
        local_luma_p99,
    }
}
