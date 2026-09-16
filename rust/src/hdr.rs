// SPDX-License-Identifier: GPL-3.0-or-later
//! Scalar port of dngscan's HDR AgX formation chunk. Behavioral reference:
//!   dngscan/hdr_agx.py (_form_hdr_chunk), dngscan/hdr_curve.py (HdrCurveTable.apply),
//!   dngscan/hdr_color.py (raw_gated_channel_separation, blend_native_hdr_paths,
//!   fit_hdr_color_volume), dngscan/agx.py (prepare/finish formation), dngscan/punch.py.
//! Each step replicates the NumPy float32 operation order; the film takeover LUT
//! (full mode) and the colour-head gain field are excluded at dispatch time
//! (supports_hdr_formation) instead of being reimplemented here.
use crate::budget::{block_pixels, workers_for};
use crate::pixel::*;

/// One compiled scene-EV -> display-linear curve on a uniform grid
/// (dngscan/hdr_curve.py HdrCurveTable).
#[derive(Clone, Debug)]
pub struct HdrCurveTable {
    pub ev_start: f32,
    pub inv_step: f32,
    pub values: Vec<f32>,
}

/// Full HDR formation chain plan (dngscan/hdr_agx.py _form_hdr_chunk):
///   compress_into_gamut -> inset -> curve table (native + optional reference-white
///   chroma candidate blended by RAW-gated rho) -> hue restore -> outset -> punch ->
///   Rec.2020 -> XYZ -> output RGB -> nan_to_num -> HDR colour-volume fit.
/// ABI v11: every matrix stage is an exact float64 stage (see NativeAgxPlan).
#[derive(Clone, Debug)]
pub struct NativeHdrPlan {
    pub inset: [f64; 9],
    pub outset: [f64; 9],
    pub rec2020_to_xyz: [f64; 9],
    pub xyz_to_rec2020: [f64; 9],
    pub xyz_to_output: [f64; 9],
    pub oklab_m1: [f64; 9],
    pub oklab_m2: [f64; 9],
    pub oklab_m1_inv: [f64; 9],
    pub oklab_m2_inv: [f64; 9],
    /// Pre-outset formation luminance row (hdr_color.formation_luma_weights)
    /// for the native/reference chroma blend.
    pub formation_luma: [f32; 3],
    /// The output space's normalized luminance row (hdr_color.output_luma_weights)
    /// for the colour-volume fit.
    pub output_luma: [f32; 3],
    pub hue_restore: f32,
    pub punch_strength: f32,
    /// channel_separation * snr_gate; per-pixel rho is gated by the CFA clip masks.
    pub global_rho: f32,
    /// Scene-authorized content peak, 2^H_rendered (_pack_peak); no margin.
    pub peak: f32,
    pub native_table: HdrCurveTable,
    pub reference_table: HdrCurveTable,
    pub has_reference: bool,
}

// hdr_curve._EPS: floor before the scene-EV log.
const TABLE_EPS: f32 = 1e-12;
// hdr_color._EPS: luminance guard in the blend and the colour-volume fit.
const BLEND_EPS: f32 = 1e-6;

#[inline(always)]
fn table_lookup(table: &HdrCurveTable, ev: f32) -> f32 {
    let size = table.values.len() as i32;
    let mut u = (ev - table.ev_start) * table.inv_step;
    if u.is_nan() {
        // np.clip keeps NaN and the NumPy interpolation then returns NaN as well.
        return u;
    }
    u = clampf(u, 0.0, (size - 1) as f32);
    let mut idx = u as i32;
    idx = idx.min(size - 2);
    let frac = u - idx as f32;
    let lo = table.values[idx as usize];
    let hi = table.values[idx as usize + 1];
    lo + (hi - lo) * frac
}

#[inline(always)]
fn scene_ev(value: f32) -> f32 {
    // np.log2(np.maximum(rgb, _EPS) / SCENE_MIDGRAY); cmax(value, eps) keeps NaN
    // in the first argument exactly like np.maximum does.
    (cmax(value, TABLE_EPS) / PIXEL_MID_GRAY).log2()
}

#[inline(always)]
fn apply_table(table: &HdrCurveTable, inset: Rgb) -> Rgb {
    Rgb::new(
        table_lookup(table, scene_ev(inset.r)),
        table_lookup(table, scene_ev(inset.g)),
        table_lookup(table, scene_ev(inset.b)),
    )
}

fn gated_rho(plan: &NativeHdrPlan, mask: Option<&[f32]>, y_native: f32) -> [f32; 3] {
    let base = clampf(plan.global_rho, 0.0, 1.0);
    let mask = match mask {
        None => return [base, base, base],
        Some(m) => m,
    };
    let m0 = clampf(mask[0], 0.0, 1.0);
    let m1 = clampf(mask[1], 0.0, 1.0);
    let m2 = clampf(mask[2], 0.0, 1.0);
    // np.partition(masks, 1)[..., 1]: the median of three, i.e. the second-largest.
    let second = cmax(cmin(m0, m1), cmin(cmax(m0, m1), m2));
    let multi_permission = 1.0 - second;
    // Peak-proximity convergence (v9): clip-compromised pixels lose their
    // remaining chroma authority continuously as the native formation luminance
    // climbs from reference white to the content peak.
    let mut converge = 1.0f32;
    if plan.peak > 1.0 {
        let proximity = clampf((y_native - 1.0) / (plan.peak - 1.0), 0.0, 1.0);
        let clipness = cmax(m0, cmax(m1, m2));
        converge = 1.0 - proximity * clipness;
    }
    [
        base * (1.0 - 0.5 * m0) * multi_permission * converge,
        base * (1.0 - 0.5 * m1) * multi_permission * converge,
        base * (1.0 - 0.5 * m2) * multi_permission * converge,
    ]
}

fn blend_native_paths(reference: Rgb, native: Rgb, rho: &[f32; 3], w: &[f32; 3]) -> Rgb {
    let y_native = dot3(w, native);
    let y_reference = dot3(w, reference);
    let common_scale = y_native / cmax(y_reference, BLEND_EPS);
    let mut common = Rgb::new(
        reference.r * common_scale,
        reference.g * common_scale,
        reference.b * common_scale,
    );
    if !(y_reference > BLEND_EPS) {
        common = native;
    }
    let r0 = clampf(rho[0], 0.0, 1.0);
    let r1 = clampf(rho[1], 0.0, 1.0);
    let r2 = clampf(rho[2], 0.0, 1.0);
    if r0 <= 0.0 && r1 <= 0.0 && r2 <= 0.0 {
        return common;
    }
    if r0 >= 1.0 && r1 >= 1.0 && r2 >= 1.0 {
        return native;
    }
    let proposal = Rgb::new(
        (1.0 - r0) * common.r + r0 * native.r,
        (1.0 - r1) * common.g + r1 * native.g,
        (1.0 - r2) * common.b + r2 * native.b,
    );
    let y_proposal = dot3(w, proposal);
    let scale = y_native / cmax(y_proposal, BLEND_EPS);
    if y_native > BLEND_EPS && y_proposal > BLEND_EPS {
        return Rgb::new(proposal.r * scale, proposal.g * scale, proposal.b * scale);
    }
    native
}

fn fit_hdr_pixel(inp: Rgb, limit: f32, w: &[f32; 3]) -> Rgb {
    let needs_fit = inp.r < 0.0
        || inp.r > limit
        || inp.g < 0.0
        || inp.g > limit
        || inp.b < 0.0
        || inp.b > limit;
    if !needs_fit {
        return inp;
    }
    let y_raw = dot3(w, inp);
    let y = clampf(y_raw, 0.0, limit);
    let c = [inp.r - y_raw, inp.g - y_raw, inp.b - y_raw];
    let mut lam = f32::INFINITY;
    for ci in c.iter() {
        if *ci < 0.0 {
            lam = cmin(lam, y / cmax(-*ci, BLEND_EPS));
        }
        if *ci > 0.0 {
            lam = cmin(lam, (limit - y) / cmax(*ci, BLEND_EPS));
        }
    }
    lam = clampf(cmin(lam, 1.0), 0.0, 1.0);
    let mut fitted = Rgb::new(
        y + lam * (inp.r - y_raw),
        y + lam * (inp.g - y_raw),
        y + lam * (inp.b - y_raw),
    );
    fitted.r = clampf(fitted.r, 0.0, limit);
    fitted.g = clampf(fitted.g, 0.0, limit);
    fitted.b = clampf(fitted.b, 0.0, limit);
    fitted
}

pub fn process_pixel(input: Rgb, mask: Option<&[f32]>, plan: &NativeHdrPlan) -> Rgb {
    let mut inp = input;
    if inp.r.is_nan() || inp.g.is_nan() || inp.b.is_nan() {
        // NumPy's compress_into_gamut poisons the whole pixel through its max/min
        // reductions; std::max/std::min only propagate a NaN in the first argument,
        // so replicate the poisoning explicitly.
        inp = Rgb::new(f32::NAN, f32::NAN, f32::NAN);
    }
    let rgb = compress_into_gamut(inp);
    let inset = mat3_exact_f64(&plan.inset, rgb);

    let restore_hue = plan.hue_restore > 1e-6;
    let mut pre_hue = 0.0f32;
    if restore_hue {
        let inset_nonneg = Rgb::new(cmax(inset.r, 0.0), cmax(inset.g, 0.0), cmax(inset.b, 0.0));
        pre_hue = rgb_to_hue(inset_nonneg);
    }

    let native = apply_table(&plan.native_table, inset);

    let punch = PunchMatrices {
        rec2020_to_xyz: &plan.rec2020_to_xyz,
        xyz_to_rec2020: &plan.xyz_to_rec2020,
        oklab_m1: &plan.oklab_m1,
        oklab_m2: &plan.oklab_m2,
        oklab_m1_inv: &plan.oklab_m1_inv,
        oklab_m2_inv: &plan.oklab_m2_inv,
    };
    // finish_formation + punch + rec2020_to_output + nan_to_num, mirroring the
    // NumPy formation_tail closure.
    let formation_tail = |formation_in: Rgb| -> Rgb {
        let mut formation = formation_in;
        if restore_hue {
            formation = mix_hue(formation, pre_hue, plan.hue_restore);
        }
        let mut mapped = mat3_exact_f64(&plan.outset, formation);
        mapped = apply_punch_rec2020_pixel(mapped, plan.punch_strength, &punch);
        // ABI v10: exact float64 stages, materialized to float32 per stage
        let xyz = mat3_exact_f64(&plan.rec2020_to_xyz, mapped);
        let output_linear = mat3_exact_f64(&plan.xyz_to_output, xyz);
        Rgb::new(
            nan_to_num(output_linear.r, 0.0, 1e6, -1e6),
            nan_to_num(output_linear.g, 0.0, 1e6, -1e6),
            nan_to_num(output_linear.b, 0.0, 1e6, -1e6),
        )
    };

    let output_linear;
    if plan.has_reference {
        let reference = apply_table(&plan.reference_table, inset);
        let rho = gated_rho(plan, mask, dot3(&plan.formation_luma, native));
        let blended = blend_native_paths(reference, native, &rho, &plan.formation_luma);
        // The blend equalizes Y at the formation point, but hue restore and punch
        // are not Y-preserving. The native branch is the sole Y authority
        // end-to-end: run both candidates through the same tail and re-anchor the
        // blend to the native branch's final Y (see the NumPy body).
        let final_native = formation_tail(native);
        let final_blend = formation_tail(blended);
        let y_native = dot3(&plan.output_luma, final_native);
        let y_blend = dot3(&plan.output_luma, final_blend);
        if y_native > 1e-9 && y_blend > 1e-9 {
            let scale = y_native / cmax(y_blend, 1e-9);
            output_linear = Rgb::new(
                final_blend.r * scale,
                final_blend.g * scale,
                final_blend.b * scale,
            );
        } else {
            output_linear = final_native;
        }
    } else {
        output_linear = formation_tail(native);
    }
    fit_hdr_pixel(output_linear, plan.peak, &plan.output_luma)
}

fn run_range(input: &[f32], masks: Option<&[f32]>, output: &mut [f32], plan: &NativeHdrPlan) {
    for (i, (px, o)) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)).enumerate() {
        let mask = masks.map(|m| &m[i * 3..i * 3 + 3]);
        let out = process_pixel(Rgb::new(px[0], px[1], px[2]), mask, plan);
        o[0] = out.r;
        o[1] = out.g;
        o[2] = out.b;
    }
}

/// `clip_masks` may be None (no CFA evidence: rho stays the clamped global value).
pub fn apply_hdr_formation_f32(
    input: &[f32],
    clip_masks: Option<&[f32]>,
    output: &mut [f32],
    plan: &NativeHdrPlan,
) {
    let n = input.len() / 3;
    // Same bounded fan-out as the SDR kernels: the caller may already run several
    // chunk workers, so the per-call parallelism stays capped at 8 threads.
    let workers = workers_for(n);
    if workers <= 1 {
        run_range(input, clip_masks, output, plan);
        return;
    }
    let block = block_pixels(n, workers) * 3;
    std::thread::scope(|s| {
        let mut mask_chunks = clip_masks.map(|m| m.chunks(block));
        for (i, o) in input.chunks(block).zip(output.chunks_mut(block)) {
            let m = mask_chunks.as_mut().and_then(|it| it.next());
            s.spawn(move || run_range(i, m, o, plan));
        }
    });
}
