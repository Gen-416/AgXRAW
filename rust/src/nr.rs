// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact separable B3 passes. MAD and shrinkage remain with the NumPy oracle.
use numpy::ndarray::ArrayView2;

const WEIGHTS: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

fn reflected_indices(n: usize, step: usize) -> Vec<[usize; 5]> {
    (0..n).map(|i| {
        std::array::from_fn(|j| {
            if n == 1 { return 0; }
            let period = 2 * (n as i128 - 1);
            let index = (i as i128 + (j as i128 - 2) * step as i128).rem_euclid(period);
            if index < n as i128 { index as usize } else { (period - index) as usize }
        })
    }).collect()
}

fn pass<F: Fn(usize, &mut [f32]) + Sync>(
    out: &mut [f32], width: usize, workers: usize, read: F,
) {
    let rows = out.len() / width;
    let band = rows.div_ceil(workers.max(1)).max(1) * width;
    let compute = |offset: usize, dest: &mut [f32]| {
        for (row, dest_row) in dest.chunks_exact_mut(width).enumerate() {
            read(offset / width + row, dest_row);
        }
    };
    if workers <= 1 {
        compute(0, out);
    } else {
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (part, dest) in out.chunks_mut(band).enumerate() {
                let compute = &compute;
                handles.push(scope.spawn(move || compute(part * band, dest)));
            }
            crate::budget::join_workers(handles);
        });
    }
}

pub fn atrous_smooth(plane: ArrayView2<'_, f32>, step: usize) -> Vec<f32> {
    smooth_with_workers(plane, step, crate::budget::workers_for(plane.len()) as usize)
}

fn smooth_with_workers(plane: ArrayView2<'_, f32>, step: usize, workers: usize) -> Vec<f32> {
    let (height, width) = plane.dim();
    if height == 0 || width == 0 { return Vec::new(); }
    let yi = reflected_indices(height, step);
    let xi = reflected_indices(width, step);
    let mut temp = vec![0.0; plane.len()];
    let mut out = vec![0.0; plane.len()];
    pass(&mut temp, width, workers, |y, dest| {
        for j in 0..5 {
            let sy = yi[y][j];
            for (x, acc) in dest.iter_mut().enumerate() {
                // Both coordinates are bounded by the validated borrowed view;
                // ndarray's view also preserves negative and zero strides.
                *acc += WEIGHTS[j] * unsafe { *plane.uget([sy, x]) };
            }
        }
    });
    pass(&mut out, width, workers, |y, dest| {
        let source = &temp[y * width..(y + 1) * width];
        for j in 0..5 {
            for (x, acc) in dest.iter_mut().enumerate() {
                // Every precomputed reflected index lies in 0..width.
                *acc += WEIGHTS[j] * unsafe { *source.get_unchecked(xi[x][j]) };
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{Array2, s};

    #[test]
    fn reflected_large_holes_and_singletons() {
        assert_eq!(reflected_indices(1, 32), vec![[0; 5]]);
        assert_eq!(reflected_indices(3, 1)[0], [2, 1, 0, 1, 2]);
        assert_eq!(reflected_indices(3, 4)[1], [1; 5]);
        let single = Array2::from_elem((1, 1), 0.5);
        assert_eq!(atrous_smooth(single.view(), 32), vec![0.5]);
    }

    #[test]
    fn strided_and_parallel_passes_are_exact() {
        let data = Array2::from_shape_fn((19, 23), |(y, x)| (y as f32 - x as f32) * 0.17);
        let reversed = data.slice(s![..;-1, ..;2]);
        let contiguous = reversed.to_owned();
        for step in [1, 2, 8, 32] {
            let serial = smooth_with_workers(reversed, step, 1);
            let parallel = smooth_with_workers(contiguous.view(), step, 2);
            assert_eq!(serial.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                       parallel.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
    }
}
