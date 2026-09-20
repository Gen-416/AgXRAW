// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact integer sensor scans over borrowed RAW/color views. Scratch is bounded
//! by the 256 possible channel IDs, with no full-frame maps or histograms.
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

/// Per-sensel scans include every row, column and channel; unlike RGB-group
/// counts they do not discard incomplete CFA cells.
#[derive(Clone, Copy)]
pub enum ChannelSamples<'a> {
    Mosaic(ArrayView2<'a, u16>, ArrayView2<'a, u8>),
    Linear(ArrayView3<'a, u16>, ArrayView3<'a, u8>),
}

impl ChannelSamples<'_> {
    fn rows(self) -> usize {
        match self {
            Self::Mosaic(raw, _) => raw.shape()[0],
            Self::Linear(raw, _) => raw.shape()[0],
        }
    }

    fn for_each_rows(self, start: usize, end: usize, mut visit: impl FnMut(u16, usize)) {
        match self {
            Self::Mosaic(raw, colors) => {
                for y in start..end {
                    for x in 0..raw.shape()[1] {
                        visit(raw[[y, x]], colors[[y, x]] as usize);
                    }
                }
            }
            Self::Linear(raw, colors) => {
                for y in start..end {
                    for x in 0..raw.shape()[1] {
                        for c in 0..raw.shape()[2] {
                            visit(raw[[y, x, c]], colors[[y, x, c]] as usize);
                        }
                    }
                }
            }
        }
    }
}

/// Each worker returns only its fixed-size counters. All successfully started
/// threads are joined even when another thread cannot start or has panicked.
fn reduce_channel_rows<T: Send, F: Fn(usize, usize) -> T + Sync>(
    rows: usize,
    sample_count: usize,
    scan: F,
    mut total: T,
    merge: impl Fn(&mut T, T),
) -> std::io::Result<T> {
    let workers = (crate::budget::workers_for(sample_count) as usize).min(rows);
    if workers <= 1 {
        return Ok(scan(0, rows));
    }
    let block = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let mut error = None;
        for start in (0..rows).step_by(block) {
            let end = start.saturating_add(block).min(rows);
            let scan = &scan;
            match std::thread::Builder::new().spawn_scoped(scope, move || scan(start, end)) {
                Ok(handle) => handles.push(handle),
                Err(exc) => {
                    error = Some(exc);
                    break;
                }
            }
        }
        for handle in handles {
            match handle.join() {
                Ok(part) => merge(&mut total, part),
                Err(_) => {
                    if error.is_none() {
                        error = Some(std::io::Error::other("sensor worker panicked"));
                    }
                }
            }
        }
        error.map_or(Ok(total), Err)
    })
}

#[derive(Debug, PartialEq, Eq)]
pub struct CeilingCounts {
    pub ceilings: [u16; 256],
    pub totals: [u64; 256],
    pub exact: [u64; 256],
    pub near: [u64; 256],
}

impl CeilingCounts {
    pub fn zero() -> Self {
        Self {
            ceilings: [0; 256],
            totals: [0; 256],
            exact: [0; 256],
            near: [0; 256],
        }
    }

    fn visit(&mut self, value: u16, cid: usize) {
        self.totals[cid] += 1;
        if value > self.ceilings[cid] {
            self.ceilings[cid] = value;
            self.exact[cid] = 1;
        } else if value == self.ceilings[cid] {
            self.exact[cid] += 1;
        }
    }

    fn merge(&mut self, other: Self) {
        for cid in 0..256 {
            self.totals[cid] += other.totals[cid];
            if other.ceilings[cid] > self.ceilings[cid] {
                self.ceilings[cid] = other.ceilings[cid];
                self.exact[cid] = other.exact[cid];
            } else if other.ceilings[cid] == self.ceilings[cid] {
                self.exact[cid] += other.exact[cid];
            }
        }
    }
}

/// Max/total/exact need one pass. Near counts use the final GLOBAL maximum,
/// requiring a second pass rather than a merge of worker-local near windows.
pub fn ceiling_counts(
    samples: ChannelSamples<'_>,
    windows: &[u16; 256],
    sample_count: usize,
) -> std::io::Result<CeilingCounts> {
    let mut result = reduce_channel_rows(
        samples.rows(),
        sample_count,
        |start, end| {
            let mut counts = CeilingCounts::zero();
            samples.for_each_rows(start, end, |value, cid| counts.visit(value, cid));
            counts
        },
        CeilingCounts::zero(),
        CeilingCounts::merge,
    )?;
    let lower: [u16; 256] =
        std::array::from_fn(|cid| result.ceilings[cid].saturating_sub(windows[cid]));
    result.near = reduce_channel_rows(
        samples.rows(),
        sample_count,
        |start, end| {
            let mut near = [0u64; 256];
            samples.for_each_rows(start, end, |value, cid| {
                near[cid] += u64::from(value >= lower[cid]);
            });
            near
        },
        [0u64; 256],
        |total, part| {
            for (total, part) in total.iter_mut().zip(part) {
                *total += part;
            }
        },
    )?;
    Ok(result)
}

/// Per-channel percentages use the caller's own threshold policy. Signed i32
/// comparison preserves negative thresholds and values above the uint16 range.
pub fn channel_clip_counts(
    samples: ChannelSamples<'_>,
    thresholds: &[i32; 256],
    sample_count: usize,
) -> std::io::Result<([u64; 256], [u64; 256])> {
    reduce_channel_rows(
        samples.rows(),
        sample_count,
        |start, end| {
            let mut totals = [0u64; 256];
            let mut hits = [0u64; 256];
            samples.for_each_rows(start, end, |value, cid| {
                totals[cid] += 1;
                hits[cid] += u64::from(i32::from(value) >= thresholds[cid]);
            });
            (totals, hits)
        },
        ([0u64; 256], [0u64; 256]),
        |total, part| {
            for cid in 0..256 {
                total.0[cid] += part.0[cid];
                total.1[cid] += part.1[cid];
            }
        },
    )
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

    #[test]
    fn ceiling_merges_global_maximum_exact_and_near_across_row_partitions() {
        let ids = [0u8, 7, 129, 255];
        let raw = Array2::from_shape_fn((512, 256), |(y, x)| match x % 4 {
            0 => [50, 100, 100, 99][y / 128],
            1 => [80, 90, 70, 90][y / 128],
            2 => 0,
            _ => [65535, 65534, 0, 65535][y / 128],
        });
        let colors = Array2::from_shape_fn(raw.raw_dim(), |(_, x)| ids[x % 4]);
        let mut windows = [0u16; 256];
        windows[0] = 1;
        windows[7] = 5;
        windows[129] = 65535;
        windows[255] = 1;
        for samples in [
            ChannelSamples::Mosaic(raw.view(), colors.view()),
            ChannelSamples::Mosaic(raw.t(), colors.t()),
        ] {
            let got = ceiling_counts(samples, &windows, raw.len()).unwrap();
            for (&cid, (max, exact, near)) in ids.iter().zip([
                (100, 16384, 24576),
                (90, 16384, 16384),
                (0, 32768, 32768),
                (65535, 16384, 24576),
            ]) {
                let cid = cid as usize;
                assert_eq!(got.ceilings[cid], max);
                assert_eq!(got.totals[cid], 32768);
                assert_eq!(got.exact[cid], exact);
                assert_eq!(got.near[cid], near);
            }
            for cid in 0..256 {
                if !ids.contains(&(cid as u8)) {
                    assert_eq!(
                        (
                            got.ceilings[cid],
                            got.totals[cid],
                            got.exact[cid],
                            got.near[cid]
                        ),
                        (0, 0, 0, 0)
                    );
                }
            }
        }
    }

    #[test]
    fn channel_scans_preserve_signed_thresholds_and_strided_linear_samples() {
        let raw = Array3::from_shape_fn((3, 5, 4), |(y, x, c)| {
            if c == 2 {
                if (y + x) % 2 == 0 {
                    65535
                } else {
                    65534
                }
            } else {
                0
            }
        });
        let colors = array![[[0u8, 7, 129, 255]]];
        let colors = colors.broadcast(raw.raw_dim()).unwrap();
        let mut thresholds = [0; 256];
        thresholds[0] = i32::MIN;
        thresholds[7] = 0;
        thresholds[129] = 65535;
        thresholds[255] = 65536;
        let samples = ChannelSamples::Linear(
            raw.view()
                .permuted_axes([1, 0, 2])
                .slice_move(s![..;-1, .., ..;-1]),
            colors
                .permuted_axes([1, 0, 2])
                .slice_move(s![..;-1, .., ..;-1]),
        );
        let (totals, hits) = channel_clip_counts(samples, &thresholds, raw.len()).unwrap();
        for (cid, hit) in [(0, 15), (7, 15), (129, 8), (255, 0)] {
            assert_eq!(totals[cid], 15);
            assert_eq!(hits[cid], hit);
        }
        assert_eq!(totals.iter().sum::<u64>(), 60);
        assert_eq!(hits.iter().sum::<u64>(), 38);
        let got = ceiling_counts(samples, &[1; 256], raw.len()).unwrap();
        assert_eq!(got.ceilings[129], 65535);
        assert_eq!(got.exact[129], 8);
        assert_eq!(got.near[129], 15);
        for cid in [0, 7, 255] {
            assert_eq!(got.ceilings[cid], 0);
            assert_eq!(got.exact[cid], 15);
            assert_eq!(got.near[cid], 15);
        }
    }

    #[test]
    fn channel_scans_empty_linear_samples_leave_all_channel_counts_zero() {
        let raw = Array3::zeros((2, 3, 0));
        let colors = Array3::zeros((2, 3, 0));
        let samples = ChannelSamples::Linear(raw.view(), colors.view());
        assert_eq!(
            ceiling_counts(samples, &[65535; 256], 0).unwrap(),
            CeilingCounts::zero()
        );
        assert_eq!(
            channel_clip_counts(samples, &[i32::MIN; 256], 0).unwrap(),
            ([0; 256], [0; 256])
        );
    }
}
