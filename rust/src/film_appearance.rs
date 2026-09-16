// SPDX-License-Identifier: GPL-3.0-or-later
//! Native palette kernel for the film appearance layer (plan §16 P6 / E3).
//! The NumPy implementation in dngscan/film_appearance.py is the correctness
//! oracle; this port must match it elementwise (parity gate in
//! tests/test_film_appearance_p10.py). All tables, matrices and scalars come
//! from Python — the declaration source stays in one place.
use crate::budget::budgeted_workers;
use crate::pixel::{cmax, cmin};

pub struct FilmAppearanceParams<'a> {
    /// [K, H] recipe fields with their PCHIP EV-derivative tables (row-major).
    pub f_hue: &'a [f32],
    pub d_hue: &'a [f32],
    pub f_chroma: &'a [f32],
    pub d_chroma: &'a [f32],
    pub f_density: &'a [f32],
    pub d_density: &'a [f32],
    /// [K], strictly increasing
    pub ev_knots: &'a [f32],
    /// [K, 2] neutral-bias table (already x nb strength)
    pub nb_ab: &'a [f32],
    pub k_knots: usize,
    pub h_knots: usize,
    pub has_neutral_bias: bool,
    pub strength: f32,
    pub neutral_c0: f32,
    pub chroma_knee: f32,
    /// 2.0 and 1.0 fast paths, else powf
    pub chroma_power: f32,
    /// 1 + richness_delta (1.0 in reference)
    pub richness_mult: f32,
    /// 1 + color_density_delta
    pub density_mult: f32,
    /// Fused colour matrices, row-major 3x3 (from _fused_oklab_matrices):
    pub m_fwd: &'a [f32],
    pub m2: &'a [f32],
    pub m2_inv: &'a [f32],
    pub m_inv: &'a [f32],
}

const PI_F32: f32 = 3.14159265358979323846_f32;

#[inline(always)]
fn mat3_apply(m: &[f32], x: f32, y: f32, z: f32) -> [f32; 3] {
    [
        m[0] * x + m[1] * y + m[2] * z,
        m[3] * x + m[4] * y + m[5] * z,
        m[6] * x + m[7] * y + m[8] * z,
    ]
}

fn run_range(rgb: &[f32], scene_ev: &[f32], out: &mut [f32], p: &FilmAppearanceParams) -> i64 {
    let h_knots = p.h_knots;
    let k = p.k_knots;
    let hi = h_knots as i32;
    let step = 360.0f32 / (h_knots as f32);
    let c0sq = p.neutral_c0 * p.neutral_c0;
    let ev_lo = p.ev_knots[0];
    let ev_hi = p.ev_knots[k - 1];
    let mut neg_rows: i64 = 0;

    for (i, (px, o)) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)).enumerate() {
        let r = px[0];
        let g = px[1];
        let b_in = px[2];

        let lms = mat3_apply(p.m_fwd, r, g, b_in);
        let cl = lms[0].cbrt();
        let cm = lms[1].cbrt();
        let cs = lms[2].cbrt();
        let lab = mat3_apply(p.m2, cl, cm, cs);
        let l = lab[0];
        let a = lab[1];
        let b = lab[2];
        let c = a.hypot(b);
        let mut hdeg = b.atan2(a) * (180.0f32 / PI_F32);
        hdeg = hdeg % 360.0;
        if hdeg < 0.0 {
            hdeg += 360.0;
        }

        let s_ = c / cmax(l, 1e-6);
        let s2 = s_ * s_;
        let w_c = s2 / (s2 + c0sq);
        let cr = s_ / p.chroma_knee;
        let shoulder = if p.chroma_power == 2.0 {
            cr * cr
        } else if p.chroma_power == 1.0 {
            cr
        } else {
            cr.powf(p.chroma_power)
        };
        let r_sh = 1.0 / (1.0 + shoulder);

        // Periodic Catmull-Rom on the hue axis.
        let hf = hdeg / step;
        let base = hf.floor();
        let t = hf - base;
        let mut j1 = (base as i32) % hi;
        if j1 < 0 {
            j1 += hi;
        }
        let j0 = ((j1 - 1 + hi) % hi) as usize;
        let j2 = ((j1 + 1) % hi) as usize;
        let j3 = ((j1 + 2) % hi) as usize;
        let j1 = j1 as usize;
        let t2 = t * t;
        let t3 = t2 * t;
        let w0 = -0.5 * t3 + t2 - 0.5 * t;
        let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
        let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
        let w3 = 0.5 * t3 - 0.5 * t2;

        // Monotone Hermite bracket on the EV axis.
        let e_raw = scene_ev[i];
        let ec = cmin(ev_hi, cmax(ev_lo, e_raw));
        let mut seg = k - 2;
        for s in 0..(k - 1) {
            if ec < p.ev_knots[s + 1] {
                seg = s;
                break;
            }
        }
        let dx = p.ev_knots[seg + 1] - p.ev_knots[seg];
        let u = (ec - p.ev_knots[seg]) / dx;
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = (u3 - 2.0 * u2 + u) * dx;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = (u3 - u2) * dx;

        let r0 = seg * h_knots;
        let r1 = (seg + 1) * h_knots;
        let sample = |f: &[f32], d: &[f32]| -> f32 {
            let row_f0 = f[r0 + j0] * w0 + f[r0 + j1] * w1 + f[r0 + j2] * w2 + f[r0 + j3] * w3;
            let row_d0 = d[r0 + j0] * w0 + d[r0 + j1] * w1 + d[r0 + j2] * w2 + d[r0 + j3] * w3;
            let row_f1 = f[r1 + j0] * w0 + f[r1 + j1] * w1 + f[r1 + j2] * w2 + f[r1 + j3] * w3;
            let row_d1 = d[r1 + j0] * w0 + d[r1 + j1] * w1 + d[r1 + j2] * w2 + d[r1 + j3] * w3;
            h00 * row_f0 + h10 * row_d0 + h01 * row_f1 + h11 * row_d1
        };

        let dh = sample(p.f_hue, p.d_hue);
        let gc = sample(p.f_chroma, p.d_chroma) * p.richness_mult;
        let dd = sample(p.f_density, p.d_density) * p.density_mult;

        let sw = p.strength * w_c;
        let h_new = (hdeg + dh * sw) * (PI_F32 / 180.0f32);
        let k_dens = (dd * sw * (-1.0f32 / 3.0f32)).exp2();
        let c_new = c * (gc * r_sh * sw).exp2() * k_dens;
        let l_new = l * k_dens;

        let mut a_new = c_new * h_new.cos();
        let mut b_new = c_new * h_new.sin();
        if p.has_neutral_bias {
            // nb_ab is pre-multiplied by the neutral-bias strength; the kernel
            // adds strength * interp(ec) exactly like the oracle's np.interp.
            let ta = p.nb_ab[seg * 2];
            let tb = p.nb_ab[seg * 2 + 1];
            let na = p.nb_ab[(seg + 1) * 2];
            let nb = p.nb_ab[(seg + 1) * 2 + 1];
            a_new += p.strength * (ta + (na - ta) * u);
            b_new += p.strength * (tb + (nb - tb) * u);
        }

        let clms = mat3_apply(p.m2_inv, l_new, a_new, b_new);
        let xl = clms[0] * clms[0] * clms[0];
        let xm = clms[1] * clms[1] * clms[1];
        let xs = clms[2] * clms[2] * clms[2];
        let rgb_out = mat3_apply(p.m_inv, xl, xm, xs);
        let neg = rgb_out[0] < 0.0 || rgb_out[1] < 0.0 || rgb_out[2] < 0.0;
        neg_rows += if neg { 1 } else { 0 };
        o[0] = cmax(rgb_out[0], 0.0);
        o[1] = cmax(rgb_out[1], 0.0);
        o[2] = cmax(rgb_out[2], 0.0);
    }
    neg_rows
}

/// rgb: flat [n, 3] float32 (mapped Rec.2020), scene_ev: [n], out: [n, 3].
/// Returns the number of rows that had any negative component before the
/// final clamp (the Python side feeds the clamp_stats counters).
pub fn film_appearance_apply(
    rgb: &[f32],
    scene_ev: &[f32],
    out: &mut [f32],
    p: &FilmAppearanceParams,
) -> i64 {
    let n = rgb.len() / 3;
    let workers = budgeted_workers(((n / 65536).max(1)) as u32) as usize;
    if workers <= 1 {
        return run_range(rgb, scene_ev, out, p);
    }
    let chunk = (n + workers - 1) / workers;
    let counts: Vec<i64> = std::thread::scope(|s| {
        let handles: Vec<_> = rgb
            .chunks(chunk * 3)
            .zip(scene_ev.chunks(chunk))
            .zip(out.chunks_mut(chunk * 3))
            .map(|((r, e), o)| s.spawn(move || run_range(r, e, o, p)))
            .collect();
        handles.into_iter().map(|h| h.join().expect("kernel thread")).collect()
    });
    counts.iter().sum()
}
