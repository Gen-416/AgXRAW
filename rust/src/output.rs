// SPDX-License-Identifier: GPL-3.0-or-later
//! Fused SDR output kernel: matrix conversion, the authoritative 16-step Oklab
//! gamut fit, display transfer, deterministic injected TPDF dither, and uint8.
use crate::budget::{block_pixels, workers_for};
use crate::pixel::*;

pub const OUTPUT_GAMUT_FIT_ITERS: i32 = 16;
pub const OUTPUT_GAMUT_TOLERANCE: f32 = 1e-4;

/// R2 item 6 (ABI v8): the Rec.2020 -> output conversion is two exact
/// stages, each a float64-accumulated product materialized to float32 —
/// the NumPy graph's own operation order and rounding
/// (apply_rgb_matrix3(rec2020_to_xyz(rgb), XYZ_TO_RGB[space])). The gamut
/// fit's own matrices stay pre-merged float32: that path only runs on
/// out-of-gamut pixels and is gated by its documented 1e-4 tolerance.
#[derive(Clone, Debug)]
pub struct NativeOutputPlan {
    pub rec2020_to_xyz: [f64; 9],
    pub xyz_to_output: [f64; 9],
    pub output_to_lms: [f32; 9],
    pub lms_to_output: [f32; 9],
    pub oklab_m2: [f32; 9],
    pub oklab_m2_inv: [f32; 9],
    pub alpha: f32,
}

impl Default for NativeOutputPlan {
    fn default() -> Self {
        NativeOutputPlan {
            rec2020_to_xyz: [0.0; 9],
            xyz_to_output: [0.0; 9],
            output_to_lms: [0.0; 9],
            lms_to_output: [0.0; 9],
            oklab_m2: [0.0; 9],
            oklab_m2_inv: [0.0; 9],
            alpha: 0.0,
        }
    }
}

#[inline(always)]
fn sanitize(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    if value.is_infinite() {
        return if value > 0.0 { 1e6 } else { -1e6 };
    }
    value
}

#[inline(always)]
fn in_unit_gamut(value: Rgb) -> bool {
    let low = -OUTPUT_GAMUT_TOLERANCE;
    let high = 1.0 + OUTPUT_GAMUT_TOLERANCE;
    value.r >= low
        && value.r <= high
        && value.g >= low
        && value.g <= high
        && value.b >= low
        && value.b <= high
}

#[inline(always)]
fn oklab_to_output(lab_l: f32, lab_a: f32, lab_b: f32, plan: &NativeOutputPlan) -> Rgb {
    let lms_prime = mat3(&plan.oklab_m2_inv, Rgb::new(lab_l, lab_a, lab_b));
    let lms = Rgb::new(
        lms_prime.r * lms_prime.r * lms_prime.r,
        lms_prime.g * lms_prime.g * lms_prime.g,
        lms_prime.b * lms_prime.b * lms_prime.b,
    );
    mat3(&plan.lms_to_output, lms)
}

pub fn fit_output_pixel(input: Rgb, plan: &NativeOutputPlan) -> Rgb {
    let rgb = Rgb::new(sanitize(input.r), sanitize(input.g), sanitize(input.b));
    if in_unit_gamut(rgb) {
        return Rgb::new(
            clampf(rgb.r, 0.0, 1.0),
            clampf(rgb.g, 0.0, 1.0),
            clampf(rgb.b, 0.0, 1.0),
        );
    }

    let mut lms = mat3(&plan.output_to_lms, rgb);
    lms.r = lms.r.cbrt();
    lms.g = lms.g.cbrt();
    lms.b = lms.b.cbrt();
    let lab = mat3(&plan.oklab_m2, lms);

    let chroma = lab.g.hypot(lab.b);
    let ld = lab.r - 0.5;
    let abs_ld = ld.abs();
    let e1 = 0.5 + abs_ld + plan.alpha * chroma;
    let sign = if ld > 0.0 {
        1.0
    } else if ld < 0.0 {
        -1.0
    } else {
        0.0
    };
    let radicand = cmax(e1 * e1 - 2.0 * abs_ld, 0.0);
    let l0 = 0.5 * (1.0 + sign * (e1 - radicand.sqrt()));

    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _ in 0..OUTPUT_GAMUT_FIT_ITERS {
        let t = 0.5 * (lo + hi);
        let candidate = oklab_to_output(l0 * (1.0 - t) + t * lab.r, t * lab.g, t * lab.b, plan);
        if in_unit_gamut(candidate) {
            lo = t;
        } else {
            hi = t;
        }
    }

    let fitted = oklab_to_output(l0 * (1.0 - lo) + lo * lab.r, lo * lab.g, lo * lab.b, plan);
    Rgb::new(
        clampf(fitted.r, 0.0, 1.0),
        clampf(fitted.g, 0.0, 1.0),
        clampf(fitted.b, 0.0, 1.0),
    )
}

#[inline(always)]
fn display_encode(linear: f32) -> f32 {
    let value = clampf(linear, 0.0, 1.0);
    if value <= 0.0031308 {
        return value * 12.92;
    }
    1.055 * value.powf(1.0 / 2.4) - 0.055
}

#[inline(always)]
fn quantize(encoded: f32, noise_a: f32, noise_b: f32) -> u8 {
    let value = (encoded * 255.0 + 0.5 + noise_a - noise_b).floor();
    clampf(value, 0.0, 255.0) as u8
}

#[inline(always)]
fn quantize_noise(encoded: f32, noise: f32) -> u8 {
    let value = (encoded * 255.0 + 0.5 + noise).floor();
    clampf(value, 0.0, 255.0) as u8
}

fn finalize_range(
    input: &[f32],
    noise_a: &[f32],
    noise_b: Option<&[f32]>,
    output: &mut [u8],
    plan: &NativeOutputPlan,
    input_is_rec2020: bool,
) {
    for (i, (px, o)) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)).enumerate() {
        let mut rgb = Rgb::new(px[0], px[1], px[2]);
        if input_is_rec2020 {
            // Two exact stages with a float32 materialization between them —
            // the NumPy graph's own rounding (R2 item 6).
            rgb = mat3_exact_f64(&plan.rec2020_to_xyz, rgb);
            rgb = mat3_exact_f64(&plan.xyz_to_output, rgb);
        }
        let fitted = fit_output_pixel(rgb, plan);
        let na = &noise_a[i * 3..i * 3 + 3];
        match noise_b {
            Some(nb_all) => {
                let nb = &nb_all[i * 3..i * 3 + 3];
                o[0] = quantize(display_encode(fitted.r), na[0], nb[0]);
                o[1] = quantize(display_encode(fitted.g), na[1], nb[1]);
                o[2] = quantize(display_encode(fitted.b), na[2], nb[2]);
            }
            None => {
                o[0] = quantize_noise(display_encode(fitted.r), na[0]);
                o[1] = quantize_noise(display_encode(fitted.g), na[1]);
                o[2] = quantize_noise(display_encode(fitted.b), na[2]);
            }
        }
    }
}

pub fn finalize_u8(
    input: &[f32],
    noise_a: &[f32],
    noise_b: Option<&[f32]>,
    output: &mut [u8],
    plan: &NativeOutputPlan,
    input_is_rec2020: bool,
) {
    let n = input.len() / 3;
    let workers = workers_for(n);
    if workers <= 1 {
        finalize_range(input, noise_a, noise_b, output, plan, input_is_rec2020);
        return;
    }
    let block = block_pixels(n, workers) * 3;
    std::thread::scope(|s| {
        let mut nb_chunks = noise_b.map(|nb| nb.chunks(block));
        for ((i, na), o) in input
            .chunks(block)
            .zip(noise_a.chunks(block))
            .zip(output.chunks_mut(block))
        {
            let nb = nb_chunks.as_mut().and_then(|it| it.next());
            s.spawn(move || finalize_range(i, na, nb, o, plan, input_is_rec2020));
        }
    });
}

fn fit_range(input: &[f32], output: &mut [f32], plan: &NativeOutputPlan) {
    for (px, o) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        let fitted = fit_output_pixel(Rgb::new(px[0], px[1], px[2]), plan);
        o[0] = fitted.r;
        o[1] = fitted.g;
        o[2] = fitted.b;
    }
}

pub fn fit_output_gamut_f32(input: &[f32], output: &mut [f32], plan: &NativeOutputPlan) {
    let n = input.len() / 3;
    let workers = workers_for(n);
    if workers <= 1 {
        fit_range(input, output, plan);
        return;
    }
    let block = block_pixels(n, workers) * 3;
    std::thread::scope(|s| {
        for (i, o) in input.chunks(block).zip(output.chunks_mut(block)) {
            s.spawn(move || fit_range(i, o, plan));
        }
    });
}
