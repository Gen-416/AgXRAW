// SPDX-License-Identifier: GPL-3.0-or-later
//! Order statistics with NumPy's exact float32 semantics (pinned by experiment
//! on NumPy 2.5, see tests/test_rust_stage1.py):
//!   * `np.median` of a float32 array = the middle element, or for even n the
//!     float32 mean of the two middle elements ((a + b) / 2 in float32);
//!   * `np.percentile(..., method="linear")` = `_lerp(v_lo, v_hi, gamma)` with
//!     gamma cast to float32, `diff = v_hi - v_lo` in float32 and the reversed
//!     form `v_hi - diff * (1 - gamma)` when gamma >= 0.5.

/// Ascending order statistic `rank` (0-based) of `values`, by partial selection.
pub fn order_stat(values: &mut [f32], rank: usize) -> f32 {
    let (_, v, _) = values.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
    *v
}

pub fn median_f32(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return f32::NAN;
    }
    if n % 2 == 1 {
        return order_stat(values, n / 2);
    }
    let hi = order_stat(values, n / 2);
    // everything left of the pivot is <= it; its maximum is the lower middle
    let lo = values[..n / 2]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |m, x| if x > m { x } else { m });
    (lo + hi) / 2.0f32
}

/// `np.percentile(values, q)` (linear) for a float32 array.
pub fn percentile_f32(values: &mut [f32], q: f64) -> f32 {
    let n = values.len();
    if n == 0 {
        return f32::NAN;
    }
    let pos = (n as f64 - 1.0) * (q / 100.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let v_hi = order_stat(values, hi);
    let v_lo = if hi == lo {
        v_hi
    } else {
        values[..hi]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |m, x| if x > m { x } else { m })
    };
    lerp_numpy(v_lo, v_hi, (pos - lo as f64) as f32)
}

/// NumPy `_lerp` on float32 operands with a float32 gamma.
#[inline]
pub fn lerp_numpy(v_lo: f32, v_hi: f32, gamma: f32) -> f32 {
    let diff = v_hi - v_lo;
    if gamma >= 0.5 {
        v_hi - diff * (1.0 - gamma)
    } else {
        v_lo + diff * gamma
    }
}

/// gainmap._exact_upper_percentile: an upper quantile from the retained top-K
/// values of a population of `total` samples.
pub fn upper_percentile_from_top(top: &mut [f32], total: usize, q: f64) -> Result<f32, String> {
    if total == 0 {
        return Ok(0.0);
    }
    let pos = (total as f64 - 1.0) * (q / 100.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let need = total - lo;
    if need > top.len() {
        return Err(format!(
            "top-K selection kept {} values but rank needs {}",
            top.len(),
            need
        ));
    }
    top.sort_unstable_by(|a, b| b.total_cmp(a)); // descending: top[j] is the j-th largest
    let v_lo = top[total - 1 - lo];
    let v_hi = top[total - 1 - hi];
    // the Python replica multiplies the float32 diff by a Python float (weak
    // under NEP 50), i.e. float32 arithmetic with the float64 frac rounded in
    let frac = pos - lo as f64;
    let diff = v_hi - v_lo;
    Ok(if frac >= 0.5 {
        v_hi - diff * ((1.0 - frac) as f32)
    } else {
        v_lo + diff * (frac as f32)
    })
}

/// Keep the `k` largest of `acc ++ more` (np.partition retention).
pub fn retain_top_k(acc: &mut Vec<f32>, more: &[f32], k: usize) {
    acc.extend_from_slice(more);
    if acc.len() > k {
        let cut = acc.len() - k;
        acc.select_nth_unstable_by(cut, |a, b| a.total_cmp(b));
        acc.drain(..cut);
    }
}

/// Sequential float32 mean of an 8x8 block in row-major (i, j) order —
/// NumPy's `x.reshape(bh, 8, bw, 8, c).mean(axis=(1, 3))` evaluates exactly
/// this order for float32 input (pinned by experiment).
#[inline]
pub fn block8_mean(get: impl Fn(usize, usize) -> f32) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..8 {
        for j in 0..8 {
            acc += get(i, j);
        }
    }
    acc / 64.0f32
}
