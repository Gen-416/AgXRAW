// SPDX-License-Identifier: GPL-3.0-or-later
//! DNG camera-plane warp. No full-frame coordinate or floating RGB buffers.
use numpy::ndarray::ArrayView3;

pub fn warp<T: Copy + Sync, O: Copy + Default + Send>(
    src: ArrayView3<'_, T>, coefficients: &[[f64; 6]], cx: f64, cy: f64,
    aspect: f64, fisheye: bool, loss: bool,
    read: impl Fn(T) -> f64 + Sync, write: impl Fn(f64) -> O + Sync,
) -> Vec<O> {
    let (h, w, _) = src.dim();
    let (cx, cy) = (cx * w as f64, cy * h as f64);
    let radius = cx.max(w as f64 - cx).hypot(cy.max(h as f64 - cy) / aspect);
    let weights = |t: f64| [-0.5*t + t*t - 0.5*t*t*t, 1.0 - 2.5*t*t + 1.5*t*t*t,
                           0.5*t + 2.0*t*t - 1.5*t*t*t, -0.5*t*t + 0.5*t*t*t];
    let mut out = vec![O::default(); h*w*3];
    let rows = crate::budget::block_pixels(h, crate::budget::workers_for(h*w));
    std::thread::scope(|scope| {
        for (block, dst) in out.chunks_mut(rows*w*3).enumerate() {
            let read = &read;
            let write = &write;
            scope.spawn(move || {
                for (n, value) in dst.iter_mut().enumerate() {
                    let c = n % 3;
                    let xx = (n / 3) % w;
                    let yy = block * rows + n / (3*w);
                    let x = (xx as f64 - cx) / radius;
                    let y = (yy as f64 - cy) / (radius * aspect);
                    let rr = (x*x + y*y).min(1.0);
                    let [k0,k1,k2,k3,t0,t1] = coefficients[c.min(coefficients.len()-1)];
                    let ratio = if fisheye {
                        if rr < 1e-12 { 1.0 } else {
                            let r = rr.sqrt(); let t = r.atan(); let t2 = t*t;
                            t * (k0 + t2*(k1+t2*(k2+t2*k3))) / r
                        }
                    } else { k0 + rr*(k1+rr*(k2+rr*k3)) };
                    let sx = cx + radius*(x*ratio + t1*(rr+2.0*x*x) + 2.0*t0*x*y);
                    let sy = cy + radius*aspect*(y*ratio + t0*(rr+2.0*y*y) + 2.0*t1*x*y);
                    let outside = sx < 0.0 || sy < 0.0 || sx > (w-1) as f64 || sy > (h-1) as f64;
                    let sx = sx.clamp(0.0, (w-1) as f64);
                    let sy = sy.clamp(0.0, (h-1) as f64);
                    let ix = sx.floor() as isize; let iy = sy.floor() as isize;
                    let wx = weights(sx-ix as f64); let wy = weights(sy-iy as f64);
                    let mut acc: f64 = 0.0;
                    for j in 0..4 {
                        for i in 0..4 {
                            let row = (iy+j as isize-1).clamp(0,h as isize-1) as usize;
                            let col = (ix+i as isize-1).clamp(0,w as isize-1) as usize;
                            let sample = read(src[[row,col,c]]);
                            if loss { acc = acc.max(sample); }
                            else { acc += sample*wy[j]*wx[i]; }
                        }
                    }
                    *value = write(if loss { acc.max(if outside {1.0} else {0.0}) }
                                   else { acc.clamp(0.0,65535.0) });
                }
            });
        }
    });
    out
}
