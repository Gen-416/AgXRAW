// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact RGB-group clipping counts, matching analysis.compute_color_clip_metrics.
//! Read only the supplied RAW/color views; scratch is four counters per worker.
use numpy::ndarray::{ArrayView2, ArrayView3};

#[derive(Clone, Copy)]
pub enum Samples<'a> {
    Mosaic(ArrayView2<'a, u16>, ArrayView2<'a, u8>, usize, usize),
    Linear(ArrayView3<'a, u16>, ArrayView3<'a, u8>),
}

impl Samples<'_> {
    fn rows(self) -> usize {
        match self {
            Self::Mosaic(raw, _, ph, _) => raw.shape()[0] / ph,
            Self::Linear(raw, _) => raw.shape()[0],
        }
    }

    fn count_rows(
        self,
        start: usize,
        end: usize,
        thresholds: &[i32; 256],
        groups: &[u8; 256],
    ) -> [u64; 4] {
        let mut counts = [0u64; 4];
        match self {
            Self::Mosaic(raw, colors, ph, pw) => {
                let cols = raw.shape()[1] / pw;
                for cy in start..end {
                    for cx in 0..cols {
                        let mut mask = 0u8;
                        for dy in 0..ph {
                            for dx in 0..pw {
                                let pos = [cy * ph + dy, cx * pw + dx];
                                let cid = colors[pos] as usize;
                                if i32::from(raw[pos]) >= thresholds[cid] {
                                    mask |= groups[cid];
                                }
                            }
                        }
                        counts[mask.count_ones() as usize] += 1;
                    }
                }
            }
            Self::Linear(raw, colors) => {
                let shape = raw.shape();
                for y in start..end {
                    for x in 0..shape[1] {
                        let mut mask = 0u8;
                        for c in 0..shape[2] {
                            let pos = [y, x, c];
                            let cid = colors[pos] as usize;
                            if i32::from(raw[pos]) >= thresholds[cid] {
                                mask |= groups[cid];
                            }
                        }
                        counts[mask.count_ones() as usize] += 1;
                    }
                }
            }
        }
        counts
    }
}

/// The binding validates the dimensions, cell period, LUTs and borrowed views.
/// Worker creation or panic returns an error, after explicitly joining every
/// started worker so no work outlives the caller's native-thread budget.
pub fn rgb_clip_counts(
    samples: Samples<'_>,
    thresholds: &[i32; 256],
    groups: &[u8; 256],
    sample_count: usize,
) -> std::io::Result<[u64; 4]> {
    let rows = samples.rows();
    let workers = (crate::budget::workers_for(sample_count) as usize).min(rows);
    if workers <= 1 {
        return Ok(samples.count_rows(0, rows, thresholds, groups));
    }
    let block = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let mut error = None;
        for start in (0..rows).step_by(block) {
            let end = start.saturating_add(block).min(rows);
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                samples.count_rows(start, end, thresholds, groups)
            }) {
                Ok(handle) => handles.push(handle),
                Err(exc) => {
                    error = Some(exc);
                    break;
                }
            }
        }
        let mut counts = [0u64; 4];
        for handle in handles {
            match handle.join() {
                Ok(part) => {
                    for (total, value) in counts.iter_mut().zip(part) {
                        *total += value;
                    }
                }
                Err(_) => {
                    if error.is_none() {
                        error = Some(std::io::Error::other("sensor worker panicked"));
                    }
                }
            }
        }
        error.map_or(Ok(counts), Err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, s, Array2, Array3};

    fn groups() -> [u8; 256] {
        let mut groups = [0u8; 256];
        groups[0] = 1;
        groups[1] = 2;
        groups[2] = 4;
        groups[3] = 2;
        groups
    }

    #[test]
    fn mosaic_uses_actual_colors_deduplicates_green_and_ignores_fringe() {
        let raw = array![
            [100, 100, 100, 100, 100],
            [100, 100, 0, 100, 100],
            [0, 0, 100, 100, 100],
            [0, 0, 100, 100, 100],
            [100, 100, 100, 100, 100],
        ];
        let colors = array![
            [1, 3, 0, 1, 0],
            [3, 1, 2, 1, 0],
            [0, 1, 0, 1, 0],
            [2, 3, 2, 255, 0],
            [0, 0, 0, 0, 0],
        ];
        let result = rgb_clip_counts(
            Samples::Mosaic(raw.view(), colors.view(), 2, 2),
            &[100; 256],
            &groups(),
            raw.len(),
        )
        .unwrap();
        assert_eq!(result, [1, 1, 1, 1]);
    }

    #[test]
    fn linear_preserves_signed_thresholds_reversed_and_zero_strides() {
        let raw = array![[[0, 2, 4], [1, 2, 3]], [[0, 0, 0], [9, 9, 9]]];
        let colors = array![[[0, 1, 2]]];
        let colors = colors.broadcast(raw.raw_dim()).unwrap();
        let mut thresholds = [65536; 256];
        thresholds[0] = -1;
        thresholds[1] = 2;
        thresholds[2] = 4;
        let result = rgb_clip_counts(
            Samples::Linear(
                raw.slice(s![..;-1, ..;-1, ..;-1]),
                colors.slice(s![..;-1, ..;-1, ..;-1]),
            ),
            &thresholds,
            &groups(),
            raw.len(),
        )
        .unwrap();
        assert_eq!(result, [0, 1, 1, 2]);
    }

    #[test]
    fn partitioned_transposed_mosaic_matches_serial_rows_and_empty_linear_counts() {
        let raw = Array2::from_shape_fn((513, 521), |(y, x)| ((y * 11 + x * 5) % 130) as u16);
        let colors = Array2::from_shape_fn((513, 521), |(y, x)| ((y + x) % 5) as u8);
        let samples = Samples::Mosaic(raw.t(), colors.t(), 6, 6);
        let thresholds = [100; 256];
        let groups = groups();
        let serial = samples.count_rows(0, samples.rows(), &thresholds, &groups);
        assert_eq!(
            rgb_clip_counts(samples, &thresholds, &groups, raw.len()).unwrap(),
            serial
        );
        let empty_raw = Array3::zeros((2, 3, 0));
        let empty_colors = Array3::zeros((2, 3, 0));
        assert_eq!(
            rgb_clip_counts(
                Samples::Linear(empty_raw.view(), empty_colors.view()),
                &thresholds,
                &groups,
                0,
            )
            .unwrap(),
            [6, 0, 0, 0]
        );
    }
}
