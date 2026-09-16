// SPDX-License-Identifier: GPL-3.0-or-later
//! NumPy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src
//! `pairwise_sum`) for float64 — the reduction order `np.sum`/`ndarray.sum`
//! use on a contiguous float64 axis. Needed wherever a kernel normalizes by
//! `k.sum()` and the result must match the NumPy reference bit for bit.
const PW_BLOCKSIZE: usize = 128;

pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &v in a {
            res += v;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i + 8 <= n - (n % 8) {
            // numpy: for (i = 8; i < n - (n % 8); i += 8)
            r[0] += a[i];
            r[1] += a[i + 1];
            r[2] += a[i + 2];
            r[3] += a[i + 3];
            r[4] += a[i + 4];
            r[5] += a[i + 5];
            r[6] += a[i + 6];
            r[7] += a[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        // remainder (n % 8 elements), sequential
        let mut j = n - (n % 8);
        while j < n {
            res += a[j];
            j += 1;
        }
        res
    } else {
        let n2 = (n / 2) - ((n / 2) % 8);
        pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
    }
}
