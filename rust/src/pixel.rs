// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-pixel primitives shared by the SDR AgX kernel and the HDR formation
//! kernel. Behavioral reference: dngscan/agx.py, dngscan/punch.py. Each helper
//! replicates the float32 operation order of its NumPy counterpart (and the
//! C++ port it replaces, bit for bit); parity is enforced by
//! tests/test_fast_backend.py and tests/test_hdr_native.py.

pub const EPS: f32 = 1e-12;
pub const PIXEL_MID_GRAY: f32 = 0.18;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32) -> Rgb {
        Rgb { r, g, b }
    }
}

/// C++ `std::max(a, b)`: `(a < b) ? b : a` — a NaN in `a` stays, a NaN in `b` is lost.
#[inline(always)]
pub fn cmax(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

/// C++ `std::min(a, b)`: `(b < a) ? b : a`.
#[inline(always)]
pub fn cmin(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline(always)]
pub fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    cmin(hi, cmax(lo, v))
}

#[inline(always)]
pub fn max3(a: f32, b: f32, c: f32) -> f32 {
    cmax(a, cmax(b, c))
}

#[inline(always)]
pub fn min3(a: f32, b: f32, c: f32) -> f32 {
    cmin(a, cmin(b, c))
}

#[inline(always)]
pub fn dot3(w: &[f32; 3], v: Rgb) -> f32 {
    w[0] * v.r + w[1] * v.g + w[2] * v.b
}

/// Float32 matrix stage (the gamut fit's pre-merged matrices keep this form).
#[inline(always)]
pub fn mat3(m: &[f32; 9], v: Rgb) -> Rgb {
    Rgb::new(
        m[0] * v.r + m[1] * v.g + m[2] * v.b,
        m[3] * v.r + m[4] * v.g + m[5] * v.b,
        m[6] * v.r + m[7] * v.g + m[8] * v.b,
    )
}

/// R2 item 6 (ABI v8, SDR) / batch 25 (v10) / math review 2026-09-03 (v11):
/// one exact NumPy matrix stage — float64 matrix entries times the float32
/// value promoted to double, accumulated left-to-right in double, and
/// materialized to float32 exactly once (apply_rgb_matrix3's `out[:, r] =`
/// assignment). Rust never fuses the products into FMAs.
#[inline(always)]
pub fn mat3_exact_f64(m: &[f64; 9], v: Rgb) -> Rgb {
    let r = v.r as f64;
    let g = v.g as f64;
    let b = v.b as f64;
    Rgb::new(
        (m[0] * r + m[1] * g + m[2] * b) as f32,
        (m[3] * r + m[4] * g + m[5] * b) as f32,
        (m[6] * r + m[7] * g + m[8] * b) as f32,
    )
}

#[inline(always)]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if denom.abs() < 1e-9 {
        return 0.0;
    }
    let t = clampf((x - edge0) / denom, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn hue_in_arc(hue_deg: f32, lo: f32, hi: f32) -> f32 {
    let mut h = hue_deg % 360.0;
    if h < 0.0 {
        h += 360.0;
    }
    let inside;
    let edge;
    if lo <= hi {
        inside = (h >= lo) && (h <= hi);
        edge = cmin(h - lo, hi - h);
    } else {
        inside = (h >= lo) || (h <= hi);
        let d_lo = if h >= lo { h - lo } else { 360.0 - lo + h };
        let d_hi = if h <= hi { hi - h } else { 360.0 - h + hi };
        edge = cmin(d_lo, d_hi);
    }
    if inside {
        smoothstep(0.0, 6.0, edge)
    } else {
        0.0
    }
}

#[inline(always)]
pub fn nan_to_num(v: f32, nan_val: f32, posinf_val: f32, neginf_val: f32) -> f32 {
    if v.is_nan() {
        return nan_val;
    }
    if v.is_infinite() {
        return if v > 0.0 { posinf_val } else { neginf_val };
    }
    v
}

/// AgX opponent-luminance constants from the pinned darktable implementation.
pub const AGX_OPPONENT_Y: [f32; 3] = [0.2658180370250449, 0.59846986045365, 0.1357121025213052];

pub fn compress_into_gamut(rgb: Rgb) -> Rgb {
    let input_y = dot3(&AGX_OPPONENT_Y, rgb);
    let max_rgb = max3(rgb.r, rgb.g, rgb.b);
    let opponent = Rgb::new(max_rgb - rgb.r, max_rgb - rgb.g, max_rgb - rgb.b);
    let opponent_y = dot3(&AGX_OPPONENT_Y, opponent);
    let max_opponent = max3(opponent.r, opponent.g, opponent.b);
    let y_compensate_negative = max_opponent - opponent_y + input_y;

    let offset = cmax(-min3(rgb.r, rgb.g, rgb.b), 0.0);
    let rgb_offset = Rgb::new(rgb.r + offset, rgb.g + offset, rgb.b + offset);
    let max_offset = max3(rgb_offset.r, rgb_offset.g, rgb_offset.b);
    let opponent_offset = Rgb::new(
        max_offset - rgb_offset.r,
        max_offset - rgb_offset.g,
        max_offset - rgb_offset.b,
    );
    let max_inverse = max3(opponent_offset.r, opponent_offset.g, opponent_offset.b);
    let y_inverse = dot3(&AGX_OPPONENT_Y, opponent_offset);
    let mut y_new = dot3(&AGX_OPPONENT_Y, rgb_offset);
    y_new = max_inverse - y_inverse + y_new;

    let mut ratio = 1.0f32;
    if y_new > y_compensate_negative && y_new > EPS {
        ratio = y_compensate_negative / y_new;
    }
    Rgb::new(rgb_offset.r * ratio, rgb_offset.g * ratio, rgb_offset.b * ratio)
}

pub fn rgb_to_hue(rgb: Rgb) -> f32 {
    let maxc = max3(rgb.r, rgb.g, rgb.b);
    let minc = min3(rgb.r, rgb.g, rgb.b);
    let delta = maxc - minc;
    // NumPy semantics (agx._rgb_to_hsv): hue stays 0 unless delta > EPS, which also
    // keeps a NaN delta (e.g. inf - inf) on the neutral branch instead of poisoning it.
    if !(delta > EPS) {
        return 0.0;
    }
    let mut h;
    if maxc == rgb.r {
        h = ((rgb.g - rgb.b) / delta) % 6.0;
    } else if maxc == rgb.g {
        h = (rgb.b - rgb.r) / delta + 2.0;
    } else {
        h = (rgb.r - rgb.g) / delta + 4.0;
    }
    h = (h / 6.0) % 1.0;
    if h < 0.0 {
        h += 1.0;
    }
    h
}

pub fn hsv_to_rgb(h_in: f32, s_in: f32, v: f32) -> Rgb {
    let mut h = h_in % 1.0;
    if h < 0.0 {
        h += 1.0;
    }
    let s = cmax(s_in, 0.0);
    let hh = h * 6.0;
    let i = (hh.floor() as i32) % 6;
    let f = hh - hh.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => Rgb::new(v, t, p),
        1 => Rgb::new(q, v, p),
        2 => Rgb::new(p, v, t),
        3 => Rgb::new(p, q, v),
        4 => Rgb::new(t, p, v),
        _ => Rgb::new(v, p, q),
    }
}

pub fn mix_hue(rgb_linear: Rgb, pre_hue: f32, restore: f32) -> Rgb {
    let post_hue = rgb_to_hue(rgb_linear);
    let mut delta = post_hue - pre_hue;
    delta -= delta.round_ties_even(); // std::nearbyintf under the default rounding mode
    let restored = (pre_hue + (1.0 - restore) * delta) % 1.0;
    let maxc = max3(rgb_linear.r, rgb_linear.g, rgb_linear.b);
    let minc = min3(rgb_linear.r, rgb_linear.g, rgb_linear.b);
    let mut sat = 0.0f32;
    if maxc > EPS {
        sat = (maxc - minc) / maxc;
    }
    hsv_to_rgb(if restored < 0.0 { restored + 1.0 } else { restored }, sat, maxc)
}

/// Matrices needed by the scene-driven punch operator (dngscan/punch.py).
/// ABI v11: exact float64 stages (NumPy's apply_rgb_matrix3 on float64).
pub struct PunchMatrices<'a> {
    pub rec2020_to_xyz: &'a [f64; 9],
    pub xyz_to_rec2020: &'a [f64; 9],
    pub oklab_m1: &'a [f64; 9],
    pub oklab_m2: &'a [f64; 9],
    pub oklab_m1_inv: &'a [f64; 9],
    pub oklab_m2_inv: &'a [f64; 9],
}

pub const PUNCH_CHROMA_MAX: f32 = 1.5;
pub const PUNCH_SKIN_DAMP: f32 = 0.55;
pub const SKIN_HUE_LO: f32 = 20.0;
pub const SKIN_HUE_HI: f32 = 60.0;

pub fn apply_punch_rec2020_pixel(rgb_in: Rgb, strength: f32, m: &PunchMatrices) -> Rgb {
    if strength <= 1e-3 {
        return rgb_in;
    }
    let s = cmin(1.0, strength);
    let rgb = Rgb::new(
        nan_to_num(rgb_in.r, 0.0, 1e6, 0.0),
        nan_to_num(rgb_in.g, 0.0, 1e6, 0.0),
        nan_to_num(rgb_in.b, 0.0, 1e6, 0.0),
    );

    let xyz = mat3_exact_f64(m.rec2020_to_xyz, rgb);
    let mut lms = mat3_exact_f64(m.oklab_m1, xyz);
    lms.r = cmax(lms.r, 0.0).cbrt();
    lms.g = cmax(lms.g, 0.0).cbrt();
    lms.b = cmax(lms.b, 0.0).cbrt();
    let lab = mat3_exact_f64(m.oklab_m2, lms);

    let chroma = lab.g.hypot(lab.b);
    let mut hue = (lab.b.atan2(lab.g) * (180.0f32 / (std::f64::consts::PI as f32))) % 360.0;
    if hue < 0.0 {
        hue += 360.0;
    }

    let mut weight = smoothstep(0.005, 0.03, chroma);
    weight *= smoothstep(0.08, 0.22, lab.r);
    weight *= 1.0 - smoothstep(0.72, 0.92, lab.r);
    weight *= 1.0 - 0.35 * smoothstep(0.20, 0.42, chroma);
    weight *= 1.0 - (1.0 - PUNCH_SKIN_DAMP) * hue_in_arc(hue, SKIN_HUE_LO, SKIN_HUE_HI);
    let gain = 1.0 + (PUNCH_CHROMA_MAX - 1.0) * s * weight;

    let lab_out = Rgb::new(lab.r, lab.g * gain, lab.b * gain);
    let mut lms_ = mat3_exact_f64(m.oklab_m2_inv, lab_out);
    lms_.r = lms_.r * lms_.r * lms_.r;
    lms_.g = lms_.g * lms_.g * lms_.g;
    lms_.b = lms_.b * lms_.b * lms_.b;
    let xyz_out = mat3_exact_f64(m.oklab_m1_inv, lms_);
    let out = mat3_exact_f64(m.xyz_to_rec2020, xyz_out);
    Rgb::new(
        nan_to_num(out.r, 0.0, 1e6, -1e6),
        nan_to_num(out.g, 0.0, 1e6, -1e6),
        nan_to_num(out.b, 0.0, 1e6, -1e6),
    )
}
