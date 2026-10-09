// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact spatial/tile support of local_detail.py, with fixed per-worker scratch.
use half::f16;
use numpy::ndarray::ArrayView3;
use std::sync::atomic::{AtomicBool, Ordering};

pub trait Sample: Copy + Sync {
    const ALWAYS_FINITE: bool;
    fn as_f32(self) -> f32;
}

impl Sample for u8 {
    const ALWAYS_FINITE: bool = true;
    fn as_f32(self) -> f32 { self as f32 }
}
impl Sample for f16 {
    const ALWAYS_FINITE: bool = false;
    fn as_f32(self) -> f32 { self.to_f32() }
}
impl Sample for f32 {
    const ALWAYS_FINITE: bool = false;
    fn as_f32(self) -> f32 { self }
}
impl Sample for f64 {
    const ALWAYS_FINITE: bool = false;
    fn as_f32(self) -> f32 { self as f32 }
}

const TILE: usize = 8;
const HALO: usize = 16; // 8x8 coefficient tile plus the largest forward distance.
const SCALES: [usize; 4] = [1, 2, 4, 8];

struct Scratch {
    decoded: [f32; HALO * HALO],
    intended: [f32; HALO * HALO],
    score: [f32; TILE * TILE],
}

impl Scratch {
    fn new() -> Self {
        Self { decoded: [0.0; HALO * HALO],
               intended: [0.0; HALO * HALO], score: [0.0; TILE * TILE] }
    }
}

fn coefficient(a0: f32, a1: f32, e0: f32, e1: f32, hdr: bool) -> Result<f32, ()> {
    // Keep every float32 materialization in the NumPy reference's order.
    let reference = e1 - e0;
    let error = ((a1 - a0) - reference).abs();
    let floor = if hdr {
        let sum = e0.abs() + e1.abs();
        let mean = sum * 0.5_f32;
        // Match the reference's float32 materializations. This translates the
        // SDR eight-code allowance into a conservative linear-light budget.
        let quant = mean.sqrt() * ((8.0_f64 / 255.0_f64) * (2.4_f64 / 1.055_f64)) as f32;
        if !sum.is_finite() || !quant.is_finite() { return Err(()); }
        (sum * 0.04_f32).max(0.01_f32).max(quant)
    } else { 8.0_f32 };
    if !reference.is_finite() || !error.is_finite() || !floor.is_finite() {
        return Err(());
    }
    let abs_reference = reference.abs();
    let denominator = abs_reference.max(floor);
    let activity = abs_reference / floor;
    let normalized = error / denominator;
    // Clipping an overflowing ratio would hide invalid derived arithmetic.
    if !activity.is_finite() || !normalized.is_finite() { return Err(()); }
    Ok(normalized.min(1.0_f32) * activity.min(1.0_f32))
}

fn top_mean(score: &mut [f32]) -> f32 {
    let split = score.len() - 16;
    score.select_nth_unstable_by(split, |a, b| a.partial_cmp(b).expect("finite detail loss"));
    let top = &score[split..];
    // NumPy's contiguous float32 pairwise sum uses eight lanes for n=16.
    // Partition may permute those lanes, so parity allows its final rounding bit.
    let lanes: [f32; 8] = std::array::from_fn(|i| top[i] + top[i + 8]);
    let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    sum / 16.0_f32
}

fn rgb<T: Sample>(source: ArrayView3<'_, T>, y: usize, x: usize) -> Result<[f32; 3], ()> {
    let values = std::array::from_fn(|c| source[[y,x,c]].as_f32());
    if !T::ALWAYS_FINITE && values.iter().any(|v| !v.is_finite()) { return Err(()); }
    Ok(values)
}

fn luma<T: Sample>(source: ArrayView3<'_, T>, y: usize, x: usize,
                   weights: [f32; 3]) -> Result<f32, ()> {
    let value = rgb(source,y,x)?;
    let luma = (value[0]*weights[0] + value[1]*weights[1]) + value[2]*weights[2];
    if !luma.is_finite() { return Err(()); }
    Ok(luma)
}

fn tile_range<A: Sample, E: Sample>(
    decoded: ArrayView3<'_, A>, intended: ArrayView3<'_, E>, hdr: bool,
    first: usize, stop: usize, tiles_x: usize, weights: [f32; 3],
) -> Result<f32, ()> {
    let (height,width,_) = intended.dim();
    let mut scratch = Scratch::new();
    let mut worst = 0.0_f32;
    for tile in first..stop {
        let row = tile / tiles_x * TILE;
        let col = tile % tiles_x * TILE;
        // No early score exit: all remaining RGB and derived coefficients must
        // be validated even after catastrophic loss. Alpha is never read.
        let rows = HALO.min(height - row);
        let cols = HALO.min(width - col);
        for y in 0..rows {
            for x in 0..cols {
                let index = y * HALO + x;
                scratch.decoded[index] = luma(decoded,row+y,col+x,weights)?;
                scratch.intended[index] = luma(intended,row+y,col+x,weights)?;
            }
        }
        let tile_h = TILE.min(height - row);
        let tile_w = TILE.min(width - col);
        for distance in SCALES {
            scratch.score.fill(0.0);
            for y in 0..tile_h {
                for x in 0..tile_w {
                    let index = y * HALO + x;
                    let mut score = 0.0_f32;
                    if col + x + distance < width {
                        let other = index + distance;
                        score = coefficient(scratch.decoded[index], scratch.decoded[other],
                            scratch.intended[index], scratch.intended[other], hdr)?;
                    }
                    if row + y + distance < height {
                        let other = index + distance * HALO;
                        score = score.max(coefficient(scratch.decoded[index], scratch.decoded[other],
                            scratch.intended[index], scratch.intended[other], hdr)?);
                    }
                    scratch.score[y*TILE+x] = score;
                }
            }
            if worst != 1.0 { worst = worst.max(top_mean(&mut scratch.score)); }
        }
    }
    Ok(worst)
}

pub fn local_detail<A: Sample, E: Sample>(
    decoded: ArrayView3<'_, A>, intended: ArrayView3<'_, E>, hdr: bool, weights: [f32; 3],
) -> Result<f32, &'static str> {
    let (height,width,_) = intended.dim();
    let tiles_x = width.div_ceil(TILE);
    let tile_count = height.div_ceil(TILE) * tiles_x;
    let workers = crate::budget::workers_for(height * width).min(tile_count.min(u32::MAX as usize) as u32).max(1);
    if workers == 1 {
        return tile_range(decoded, intended, hdr, 0, tile_count, tiles_x, weights)
            .map_err(|_| "local detail RGB and coefficient arithmetic must be finite");
    }
    let block = crate::budget::block_pixels(tile_count, workers);
    let mut results = vec![0.0_f32; workers as usize];
    let invalid = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (worker, result) in results.iter_mut().enumerate() {
            let first = worker * block;
            let stop = (first + block).min(tile_count);
            let invalid = &invalid;
            handles.push(scope.spawn(move || {
                match tile_range(decoded, intended, hdr, first, stop, tiles_x, weights) {
                    Ok(value) => *result = value,
                    Err(()) => invalid.store(true, Ordering::Relaxed),
                }
            }));
        }
        crate::budget::join_workers(handles);
    });
    if invalid.load(Ordering::Relaxed) {
        return Err("local detail RGB and coefficient arithmetic must be finite");
    }
    Ok(results.into_iter().fold(0.0_f32, f32::max))
}

pub fn default_weights(hdr: bool) -> [f32; 3] {
    if hdr {[0.22898022830486298_f32,0.6917159557342529_f32,0.07930383831262589_f32]}
    else {[0.299_f32,0.587_f32,0.114_f32]}
}

#[cfg(test)]
mod tests {
    use numpy::ndarray::Array3;
    #[test]
    fn supported_texture_and_partial_edges() {
        let intended = Array3::from_shape_fn((9, 17, 3), |(y,x,_)| if (x+y)%2==0 {0_u8} else {255});
        let flat = Array3::from_elem((9, 17, 3), 128_u8);
        assert_eq!(super::local_detail(intended.view(), intended.view(), false,super::default_weights(false)), Ok(0.0));
        assert_eq!(super::local_detail(flat.view(), intended.view(), false,super::default_weights(false)), Ok(1.0));
        let single = Array3::from_elem((1, 1, 3), 1_f32);
        assert_eq!(super::local_detail(single.view(), single.view(), true,super::default_weights(true)), Ok(0.0));
    }

    #[test]
    fn alpha_is_ignored_but_tail_rgb_nan_is_not() {
        let mut intended = Array3::from_shape_fn((48, 48, 4), |(y,x,c)| {
            if c==3 {f32::NAN} else if (x+y)%2==0 {0_f32} else {1_f32}
        });
        let flat = Array3::from_shape_fn((48, 48, 4), |(_,_,c)| if c==3 {f32::NAN} else {0.5_f32});
        assert_eq!(super::local_detail(flat.view(), intended.view(), true,super::default_weights(true)), Ok(1.0));
        intended[[47,47,1]]=f32::NAN;
        assert!(super::local_detail(flat.view(), intended.view(), true,super::default_weights(true)).is_err());
    }
}
