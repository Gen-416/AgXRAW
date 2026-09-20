// SPDX-License-Identifier: GPL-3.0-or-later
//! Loss evidence operations. Geometry and nearest-neighbour indices come from
//! the Python oracle; only scalar maxima and gathering move into native code.
use half::f16;
use numpy::ndarray::ArrayView3;
use std::sync::{atomic::{AtomicBool, Ordering}, Condvar, Mutex};

/// NumPy's half loop compares numerically but returns the original half bits.
/// In particular, equal values choose the left operand and the first NaN wins.
#[inline]
pub fn maximum_half(a: f16, b: f16) -> f16 {
    if a.is_nan() || (!b.is_nan() && a.to_f32() >= b.to_f32()) { a } else { b }
}

pub trait LossValue: Copy + Send + Sync {
    const ZERO: Self;
    const MIXED: bool;
    fn crop_supported(self) -> bool;
    fn merge_supported(self, mapped: bool) -> bool;
    fn maximum(self, other: Self) -> Self;
    fn merge(self, mask: f16, mapped: bool) -> f16;
}

impl LossValue for f16 {
    const ZERO: Self = f16::ZERO;
    const MIXED: bool = false;
    fn crop_supported(self) -> bool { true }
    fn merge_supported(self, mapped: bool) -> bool {
        // PIL's half -> float32 -> half path quiets signalling NaNs. Let the
        // original implementation decide their exact payload on this platform.
        !mapped || !self.is_nan()
    }
    fn maximum(self, other: Self) -> Self { maximum_half(self, other) }
    fn merge(self, mask: f16, _mapped: bool) -> f16 { maximum_half(mask, self) }
}

impl LossValue for f32 {
    const ZERO: Self = 0.0;
    const MIXED: bool = true;
    fn crop_supported(self) -> bool { !self.is_nan() && self.to_bits() != 0x8000_0000 }
    fn merge_supported(self, mapped: bool) -> bool {
        !self.is_nan() && (mapped || self.to_bits() != 0x8000_0000)
    }
    fn maximum(self, other: Self) -> Self {
        // Both operands have already passed crop_supported(). NumPy's CPU-
        // dependent NaN/negative-zero rules are deliberately left to its oracle.
        if self >= other { self } else { other }
    }
    fn merge(self, mask: f16, mapped: bool) -> f16 {
        if mapped {
            // Match the materialized half resize BEFORE taking the half maximum.
            maximum_half(mask, f16::from_f32(self))
        } else {
            // NumPy promotes half+float32 to float32, then casts into out=half.
            let a = mask.to_f32();
            f16::from_f32(if a >= self { a } else { self })
        }
    }
}

/// Start all workers before allowing any write. A thread-creation failure
/// therefore returns an error with the caller's in-place array untouched.
fn rows_mut<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    out: &mut [T], height: usize, width: usize, process: F,
) -> std::io::Result<()> {
    let workers = (crate::budget::workers_for(height * width) as usize).min(height);
    if workers <= 1 {
        process(0, out);
        return Ok(());
    }
    let rows = height.div_ceil(workers);
    let gate = (Mutex::new(0u8), Condvar::new());
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let mut error = None;
        for (i, chunk) in out.chunks_mut(rows * width * 3).enumerate() {
            let process = &process;
            let gate = &gate;
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                let mut state = gate.0.lock().expect("loss worker start lock");
                while *state == 0 {
                    state = gate.1.wait(state).expect("loss worker start wait");
                }
                let run = *state == 1;
                drop(state);
                if run { process(i * rows, chunk); }
            }) {
                Ok(handle) => handles.push(handle),
                Err(exc) => { error = Some(exc); break; }
            }
        }
        *gate.0.lock().expect("loss worker start lock") = if error.is_none() { 1 } else { 2 };
        gate.1.notify_all();
        crate::budget::join_workers(handles);
        error.map_or(Ok(()), Err)
    })
}

/// Preserve the reference's 128-row bands and dy-then-dx iteration, including
/// maximum(out, +0) for pixels outside each band's largest footprint.
pub fn crop<T: LossValue>(
    values: ArrayView3<'_, T>, ylo: &[usize], yhi: &[usize], xlo: &[usize], xhi: &[usize],
) -> std::io::Result<Option<Vec<T>>> {
    let (height, width) = (ylo.len(), xlo.len());
    let max_dx = xlo.iter().zip(xhi).map(|(lo, hi)| hi - lo).max().unwrap_or(0);
    let max_dy: Vec<usize> = ylo.chunks(128).zip(yhi.chunks(128))
        .map(|(lo, hi)| lo.iter().zip(hi).map(|(a, b)| b - a).max().unwrap_or(0))
        .collect();
    let mut out = vec![T::ZERO; height * width * 3];
    let unsupported = AtomicBool::new(false);
    rows_mut(&mut out, height, width, |row0, chunk| {
        for (row, target) in chunk.chunks_exact_mut(width * 3).enumerate() {
            let y = row0 + row;
            for x in 0..width {
                for dy in 0..max_dy[y / 128] {
                    let sy = ylo[y] + dy;
                    for dx in 0..max_dx {
                        let sx = xlo[x] + dx;
                        let valid = sy < yhi[y] && sx < xhi[x];
                        for c in 0..3 {
                            let value = if valid { values[[sy, sx, c]] } else { T::ZERO };
                            if !value.crop_supported() {
                                unsupported.store(true, Ordering::Relaxed);
                                return;
                            }
                            target[x * 3 + c] = target[x * 3 + c].maximum(value);
                        }
                    }
                }
            }
        }
    })?;
    Ok(if unsupported.load(Ordering::Relaxed) { None } else { Some(out) })
}

/// All dimensions/maps/aliasing are validated by the binding before borrowing
/// the mutable slice. Unsupported numerical cases also leave it untouched.
pub fn merge<T: LossValue>(
    masks: &mut [f16], height: usize, width: usize, processing: ArrayView3<'_, T>,
    maps: Option<(&[usize], &[usize])>,
) -> std::io::Result<bool> {
    let mapped = maps.is_some();
    if processing.iter().any(|&value| !value.merge_supported(mapped)) {
        return Ok(false);
    }
    if T::MIXED && !mapped
        && masks.iter().any(|value| value.is_nan() || value.to_bits() == 0x8000) {
        return Ok(false);
    }
    rows_mut(masks, height, width, |row0, chunk| {
        for (row, target) in chunk.chunks_exact_mut(width * 3).enumerate() {
            let y = row0 + row;
            let sy = maps.map_or(y, |(ys, _)| ys[y]);
            for x in 0..width {
                let sx = maps.map_or(x, |(_, xs)| xs[x]);
                for c in 0..3 {
                    target[x * 3 + c] = processing[[sy, sx, c]].merge(target[x * 3 + c], mapped);
                }
            }
        }
    })?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{Array3, s};

    #[test]
    fn half_maximum_keeps_nan_payload_and_left_zero() {
        let nan_a = f16::from_bits(0x7c13);
        let nan_b = f16::from_bits(0xfe33);
        assert_eq!(maximum_half(nan_a, nan_b).to_bits(), 0x7c13);
        assert_eq!(maximum_half(f16::ZERO, nan_b).to_bits(), 0xfe33);
        assert_eq!(maximum_half(f16::NEG_ZERO, f16::ZERO).to_bits(), 0x8000);
        assert_eq!(maximum_half(f16::ZERO, f16::NEG_ZERO).to_bits(), 0);
    }

    #[test]
    fn crop_keeps_half_payload_and_uses_zero_for_empty_footprints() {
        let mut values = Array3::from_elem((2, 2, 3), f16::NEG_ZERO);
        values[[0, 0, 1]] = f16::from_bits(0x7c13);
        let out = crop(values.view(), &[0, 1], &[2, 1], &[0, 1], &[2, 1]).unwrap().unwrap();
        assert_eq!(out[0].to_bits(), 0);
        assert_eq!(out[1].to_bits(), 0x7c13);
        assert!(out[3..].iter().all(|v| v.to_bits() == 0));
    }

    #[test]
    fn unsupported_merge_never_writes_and_strided_maps_round_first() {
        let mut processing = Array3::from_elem((3, 4, 3), 0.3333f32);
        processing[[2, 3, 2]] = f32::NAN;
        let mut masks = vec![f16::from_f32(0.1); 18];
        let before = masks.clone();
        assert!(!merge(&mut masks, 2, 3, processing.view(), Some((&[0, 2], &[0, 1, 3]))).unwrap());
        assert_eq!(masks, before);
        processing[[2, 3, 2]] = 0.3333;
        let source = processing.slice(s![..;-1, ..;-1, ..;-1]);
        assert!(merge(&mut masks, 2, 3, source, Some((&[0, 2], &[0, 1, 3]))).unwrap());
        assert!(masks.iter().all(|v| v.to_bits() == f16::from_f32(0.3333).to_bits()));
    }
}
