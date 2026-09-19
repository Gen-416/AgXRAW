// SPDX-License-Identifier: GPL-3.0-or-later
//! Scalar port of dngscan's darktable-derived AgX formation + C1 curve + hue
//! restoration + scene-driven punch. Behavioral reference:
//!   dngscan/agx.py, dngscan/drt.py, dngscan/punch.py
//! Original curve/formation derives from darktable AgX (GPL-3.0-or-later).
use crate::budget::{block_pixels, workers_for};
use crate::pixel::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct CurveParams {
    pub black_ev: f32,
    pub range_ev: f32,
    pub gamma: f32,
    pub target_black: f32,
    pub target_white: f32,

    pub toe_power: f32,
    pub toe_transition_x: f32,
    pub toe_transition_y: f32,
    pub toe_scale: f32,
    pub need_convex_toe: bool,
    pub toe_fallback_power: f32,
    pub toe_fallback_coefficient: f32,

    pub slope: f32,
    pub intercept: f32,

    pub shoulder_power: f32,
    pub shoulder_transition_x: f32,
    pub shoulder_transition_y: f32,
    pub shoulder_scale: f32,
    pub need_concave_shoulder: bool,
    pub shoulder_fallback_power: f32,
    pub shoulder_fallback_coefficient: f32,
}

/// ABI v11: every matrix stage NumPy evaluates with a float64 matrix
/// (agx._apply_matrix3 / apply_rgb_matrix3 on float64 constants: inset,
/// outset, the punch/Oklab excursion) is carried as float64 and applied
/// through mat3_exact_f64 — the exact-stage contract the output plan has
/// carried since v8.
#[derive(Clone, Debug)]
pub struct NativeAgxPlan {
    pub inset: [f64; 9],
    pub outset: [f64; 9],
    pub curve: CurveParams,
    pub hue_restore: f32,
    pub view_brightness: f32,
    pub punch_strength: f32,

    pub rec2020_to_xyz: [f64; 9],
    pub xyz_to_rec2020: [f64; 9],
    pub oklab_m1: [f64; 9],
    pub oklab_m2: [f64; 9],
    pub oklab_m1_inv: [f64; 9],
    pub oklab_m2_inv: [f64; 9],
}

impl Default for NativeAgxPlan {
    fn default() -> Self {
        NativeAgxPlan {
            inset: [0.0; 9],
            outset: [0.0; 9],
            curve: CurveParams::default(),
            hue_restore: 0.0,
            view_brightness: 0.0,
            punch_strength: 0.0,
            rec2020_to_xyz: [0.0; 9],
            xyz_to_rec2020: [0.0; 9],
            oklab_m1: [0.0; 9],
            oklab_m2: [0.0; 9],
            oklab_m1_inv: [0.0; 9],
            oklab_m2_inv: [0.0; 9],
        }
    }
}

fn sigmoid(x: f32, power: f32) -> f32 {
    let xp = cmax(x, 0.0).powf(power);
    x / (1.0 + xp).powf(1.0 / power)
}

fn scaled_sigmoid(
    x: f32,
    scale_value: f32,
    slope: f32,
    power: f32,
    transition_x: f32,
    transition_y: f32,
) -> f32 {
    if scale_value.abs() < EPS {
        return transition_y;
    }
    scale_value * sigmoid(slope * (x - transition_x) / scale_value, power) + transition_y
}

pub fn apply_curve_c1(x: f32, p: &CurveParams) -> f32 {
    let out;
    if x < p.toe_transition_x {
        if p.need_convex_toe {
            out = p.target_black
                + cmax(
                    0.0,
                    p.toe_fallback_coefficient * cmax(x, 0.0).powf(p.toe_fallback_power),
                );
        } else {
            out = scaled_sigmoid(
                x,
                p.toe_scale,
                p.slope,
                p.toe_power,
                p.toe_transition_x,
                p.toe_transition_y,
            );
        }
    } else if x > p.shoulder_transition_x {
        if p.need_concave_shoulder {
            out = p.target_white
                - cmax(
                    0.0,
                    p.shoulder_fallback_coefficient
                        * cmax(1.0 - x, 0.0).powf(p.shoulder_fallback_power),
                );
        } else {
            out = scaled_sigmoid(
                x,
                p.shoulder_scale,
                p.slope,
                p.shoulder_power,
                p.shoulder_transition_x,
                p.shoulder_transition_y,
            );
        }
    } else {
        out = p.slope * x + p.intercept;
    }
    clampf(out, p.target_black, p.target_white)
}

fn apply_c1_endpoints_rgb(inset: Rgb, curve: &CurveParams) -> Rgb {
    let channels = [inset.r, inset.g, inset.b];
    let mut out_channels = [0.0f32; 3];
    for c in 0..3 {
        let ev = cmax(channels[c] / PIXEL_MID_GRAY, EPS).log2();
        let mut x = (ev - curve.black_ev) / curve.range_ev;
        x = clampf(x, 0.0, 1.0);
        let encoded = apply_curve_c1(x, curve);
        out_channels[c] = cmax(encoded, 0.0).powf(curve.gamma);
    }
    Rgb::new(out_channels[0], out_channels[1], out_channels[2])
}

fn punch_matrices(plan: &NativeAgxPlan) -> PunchMatrices<'_> {
    PunchMatrices {
        rec2020_to_xyz: &plan.rec2020_to_xyz,
        xyz_to_rec2020: &plan.xyz_to_rec2020,
        oklab_m1: &plan.oklab_m1,
        oklab_m2: &plan.oklab_m2,
        oklab_m1_inv: &plan.oklab_m1_inv,
        oklab_m2_inv: &plan.oklab_m2_inv,
    }
}

pub fn process_pixel(input: Rgb, plan: &NativeAgxPlan) -> Rgb {
    let rgb = compress_into_gamut(input);
    let inset = mat3_exact_f64(&plan.inset, rgb);

    let restore_hue = plan.hue_restore > 1e-6;
    let mut pre_hue = 0.0f32;
    if restore_hue {
        let inset_nonneg = Rgb::new(cmax(inset.r, 0.0), cmax(inset.g, 0.0), cmax(inset.b, 0.0));
        pre_hue = rgb_to_hue(inset_nonneg);
    }

    let mut linear = apply_c1_endpoints_rgb(inset, &plan.curve);
    if (plan.view_brightness - 1.0).abs() > 1e-6 {
        let brightness = cmax(plan.view_brightness, EPS);
        let power = if brightness < 1.0 {
            1.0 / brightness.sqrt()
        } else {
            1.0 / brightness
        };
        linear.r = cmax(linear.r, 0.0).powf(power);
        linear.g = cmax(linear.g, 0.0).powf(power);
        linear.b = cmax(linear.b, 0.0).powf(power);
    }

    if restore_hue {
        linear = mix_hue(linear, pre_hue, plan.hue_restore);
    }

    let mapped = mat3_exact_f64(&plan.outset, linear);
    apply_punch_rec2020_pixel(mapped, plan.punch_strength, &punch_matrices(plan))
}

fn run_range(input: &[f32], output: &mut [f32], plan: &NativeAgxPlan) {
    for (i, o) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        let out = process_pixel(Rgb::new(i[0], i[1], i[2]), plan);
        o[0] = out.r;
        o[1] = out.g;
        o[2] = out.b;
    }
}

/// `input`/`output` are flat (N, 3) float32 buffers.
pub fn apply_agx_core_f32(input: &[f32], output: &mut [f32], plan: &NativeAgxPlan) {
    let n = input.len() / 3;
    let workers = workers_for(n);
    if workers <= 1 {
        run_range(input, output, plan);
        return;
    }
    let block = block_pixels(n, workers) * 3;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (i, o) in input.chunks(block).zip(output.chunks_mut(block)) {
            handles.push(s.spawn(move || run_range(i, o, plan)));
        }
        crate::budget::join_workers(handles);
    });
}
