// SPDX-License-Identifier: GPL-3.0-or-later
//! `dngscan._dngscan_fast` — the optional native kernels, ported from the
//! pybind11 C++ extension (2026-09) with the same module API and ABI: the
//! Python side (dngscan/_fast.py, dngscan/fast_plan.py) is unchanged. Every
//! kernel replicates the NumPy reference's float32 operation order; the parity
//! gates are tests/test_fast_backend.py, tests/test_hdr_native.py and
//! tests/test_film_appearance_p10.py.
mod agx;
mod budget;
mod evidence;
mod film_appearance;
mod hdr;
mod metrics;
mod output;
mod pixel;
mod stats;

use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::sync::atomic::Ordering;

/// Native ABI version. v8: exact float64 two-stage output matrices; v9
/// (#136): HDR per-pixel peak-proximity confidence; v10 (batch 25): HDR
/// output stage float64; v11 (math review 2026-09-03): inset/outset and the
/// punch/Oklab matrices of both kernels exact float64. The Rust port keeps
/// v11 — same plan attributes, same results.
pub const NATIVE_ABI_VERSION: i32 = 11;

fn read_f32(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f32> {
    obj.getattr(name)?.extract::<f32>()
}

fn read_bool(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    obj.getattr(name)?.extract::<bool>()
}

fn read_mat9_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<[f64; 9]> {
    let v: Vec<f64> = obj.getattr(name)?.extract()?;
    v.as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("matrix must have 9 elements"))
}

fn read_mat9_f32(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<[f32; 9]> {
    let v: Vec<f32> = obj.getattr(name)?.extract()?;
    v.as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("matrix must have 9 elements"))
}

fn read_vec3(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<[f32; 3]> {
    let v: Vec<f32> = obj.getattr(name)?.extract()?;
    v.as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("vector must have 3 elements"))
}

/// pybind11's `array_t<float, c_style | forcecast>`: any array-like becomes a
/// C-contiguous float32 array (a no-op view when it already is one).
fn as_f32_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float32")?)?;
    let arr = np
        .getattr("ascontiguousarray")?
        .call((obj,), Some(&kwargs))?;
    Ok(arr.cast_into::<PyArrayDyn<f32>>()?)
}

fn require_rgb(arr: &Bound<'_, PyArrayDyn<f32>>, name: &str) -> PyResult<usize> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!("{name} must be (N, 3)")));
    }
    Ok(shape[0])
}

fn require_same_shape(n: usize, arr: &Bound<'_, PyArrayDyn<f32>>, name: &str) -> PyResult<()> {
    if require_rgb(arr, name)? != n {
        return Err(PyValueError::new_err(format!("{name} must match rgb shape")));
    }
    Ok(())
}

fn rows3_f32<'py>(py: Python<'py>, data: Vec<f32>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_vec(py, data).reshape([n, 3])?.into_any())
}

fn rows3_u8<'py>(py: Python<'py>, data: Vec<u8>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_vec(py, data).reshape([n, 3])?.into_any())
}

fn curve_from_py(obj: &Bound<'_, PyAny>) -> PyResult<agx::CurveParams> {
    Ok(agx::CurveParams {
        black_ev: read_f32(obj, "black_ev")?,
        range_ev: read_f32(obj, "range_ev")?,
        gamma: read_f32(obj, "gamma")?,
        target_black: read_f32(obj, "target_black")?,
        target_white: read_f32(obj, "target_white")?,
        toe_power: read_f32(obj, "toe_power")?,
        toe_transition_x: read_f32(obj, "toe_transition_x")?,
        toe_transition_y: read_f32(obj, "toe_transition_y")?,
        toe_scale: read_f32(obj, "toe_scale")?,
        need_convex_toe: read_bool(obj, "need_convex_toe")?,
        toe_fallback_power: read_f32(obj, "toe_fallback_power")?,
        toe_fallback_coefficient: read_f32(obj, "toe_fallback_coefficient")?,
        slope: read_f32(obj, "slope")?,
        intercept: read_f32(obj, "intercept")?,
        shoulder_power: read_f32(obj, "shoulder_power")?,
        shoulder_transition_x: read_f32(obj, "shoulder_transition_x")?,
        shoulder_transition_y: read_f32(obj, "shoulder_transition_y")?,
        shoulder_scale: read_f32(obj, "shoulder_scale")?,
        need_concave_shoulder: read_bool(obj, "need_concave_shoulder")?,
        shoulder_fallback_power: read_f32(obj, "shoulder_fallback_power")?,
        shoulder_fallback_coefficient: read_f32(obj, "shoulder_fallback_coefficient")?,
    })
}

fn agx_plan_from_py(obj: &Bound<'_, PyAny>) -> PyResult<agx::NativeAgxPlan> {
    Ok(agx::NativeAgxPlan {
        inset: read_mat9_f64(obj, "inset")?,
        outset: read_mat9_f64(obj, "outset")?,
        curve: curve_from_py(&obj.getattr("curve")?)?,
        hue_restore: read_f32(obj, "hue_restore")?,
        view_brightness: read_f32(obj, "view_brightness")?,
        punch_strength: read_f32(obj, "punch_strength")?,
        rec2020_to_xyz: read_mat9_f64(obj, "rec2020_to_xyz")?,
        xyz_to_rec2020: read_mat9_f64(obj, "xyz_to_rec2020")?,
        oklab_m1: read_mat9_f64(obj, "oklab_m1")?,
        oklab_m2: read_mat9_f64(obj, "oklab_m2")?,
        oklab_m1_inv: read_mat9_f64(obj, "oklab_m1_inv")?,
        oklab_m2_inv: read_mat9_f64(obj, "oklab_m2_inv")?,
    })
}

fn output_plan_from_py(obj: &Bound<'_, PyAny>) -> PyResult<output::NativeOutputPlan> {
    Ok(output::NativeOutputPlan {
        rec2020_to_xyz: read_mat9_f64(obj, "rec2020_to_xyz")?,
        xyz_to_output: read_mat9_f64(obj, "xyz_to_output")?,
        output_to_lms: read_mat9_f32(obj, "output_to_lms")?,
        lms_to_output: read_mat9_f32(obj, "lms_to_output")?,
        oklab_m2: read_mat9_f32(obj, "oklab_m2")?,
        oklab_m2_inv: read_mat9_f32(obj, "oklab_m2_inv")?,
        alpha: read_f32(obj, "alpha")?,
    })
}

fn table_from_py<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<hdr::HdrCurveTable> {
    let values = as_f32_array(py, &obj.getattr("values")?)?;
    let shape = values.shape().to_vec();
    if shape.len() != 1 || shape[0] < 2 {
        return Err(PyValueError::new_err(
            "curve table values must be a 1-D array of >= 2 samples",
        ));
    }
    let values = values.readonly().as_slice()?.to_vec();
    Ok(hdr::HdrCurveTable {
        ev_start: read_f32(obj, "ev_start")?,
        inv_step: read_f32(obj, "inv_step")?,
        values,
    })
}

fn hdr_plan_from_py<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<hdr::NativeHdrPlan> {
    let has_reference = read_bool(obj, "has_reference")?;
    let native_table = table_from_py(py, &obj.getattr("native_table")?)?;
    let reference_table = if has_reference {
        table_from_py(py, &obj.getattr("reference_table")?)?
    } else {
        native_table.clone()
    };
    Ok(hdr::NativeHdrPlan {
        inset: read_mat9_f64(obj, "inset")?,
        outset: read_mat9_f64(obj, "outset")?,
        rec2020_to_xyz: read_mat9_f64(obj, "rec2020_to_xyz")?,
        xyz_to_rec2020: read_mat9_f64(obj, "xyz_to_rec2020")?,
        xyz_to_output: read_mat9_f64(obj, "xyz_to_output")?,
        oklab_m1: read_mat9_f64(obj, "oklab_m1")?,
        oklab_m2: read_mat9_f64(obj, "oklab_m2")?,
        oklab_m1_inv: read_mat9_f64(obj, "oklab_m1_inv")?,
        oklab_m2_inv: read_mat9_f64(obj, "oklab_m2_inv")?,
        formation_luma: read_vec3(obj, "formation_luma")?,
        output_luma: read_vec3(obj, "output_luma")?,
        hue_restore: read_f32(obj, "hue_restore")?,
        punch_strength: read_f32(obj, "punch_strength")?,
        global_rho: read_f32(obj, "global_rho")?,
        peak: read_f32(obj, "peak")?,
        native_table,
        reference_table,
        has_reference,
    })
}

/// Per-process native thread budget; 0 = hardware concurrency (S3).
#[pyfunction]
fn set_thread_budget(budget: u32) {
    budget::THREAD_BUDGET.store(budget, Ordering::Relaxed);
}

#[pyfunction]
fn native_abi_version() -> i32 {
    NATIVE_ABI_VERSION
}

#[pyfunction]
fn apply_agx_core_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    plan: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rgb = as_f32_array(py, rgb)?;
    let n = require_rgb(&rgb, "rgb")?;
    let plan = agx_plan_from_py(plan)?;
    let rgb_ro = rgb.readonly();
    let input = rgb_ro.as_slice()?;
    let mut out = vec![0f32; n * 3];
    py.detach(|| agx::apply_agx_core_f32(input, &mut out, &plan));
    rows3_f32(py, out, n)
}

#[pyfunction]
fn apply_hdr_formation_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    clip_masks: &Bound<'py, PyAny>,
    plan: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rgb = as_f32_array(py, rgb)?;
    let n = require_rgb(&rgb, "rgb")?;
    let masks = if clip_masks.is_none() {
        None
    } else {
        let m = as_f32_array(py, clip_masks)?;
        require_same_shape(n, &m, "clip_masks")?;
        Some(m)
    };
    let plan = hdr_plan_from_py(py, plan)?;
    let rgb_ro = rgb.readonly();
    let input = rgb_ro.as_slice()?;
    let masks_ro = masks.as_ref().map(|m| m.readonly());
    let masks_slice = match masks_ro.as_ref() {
        Some(m) => Some(m.as_slice()?),
        None => None,
    };
    let mut out = vec![0f32; n * 3];
    py.detach(|| hdr::apply_hdr_formation_f32(input, masks_slice, &mut out, &plan));
    rows3_f32(py, out, n)
}

#[pyfunction]
fn fit_output_gamut_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    plan: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rgb = as_f32_array(py, rgb)?;
    let n = require_rgb(&rgb, "rgb")?;
    let plan = output_plan_from_py(plan)?;
    let rgb_ro = rgb.readonly();
    let input = rgb_ro.as_slice()?;
    let mut out = vec![0f32; n * 3];
    py.detach(|| output::fit_output_gamut_f32(input, &mut out, &plan));
    rows3_f32(py, out, n)
}

fn finalize_common<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    noise_a: &Bound<'py, PyAny>,
    noise_b: Option<&Bound<'py, PyAny>>,
    plan: &Bound<'py, PyAny>,
    input_is_rec2020: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let rgb = as_f32_array(py, rgb)?;
    let n = require_rgb(&rgb, "rgb")?;
    let na = as_f32_array(py, noise_a)?;
    require_same_shape(n, &na, if noise_b.is_some() { "noise_a" } else { "noise" })?;
    let nb = match noise_b {
        Some(b) => {
            let b = as_f32_array(py, b)?;
            require_same_shape(n, &b, "noise_b")?;
            Some(b)
        }
        None => None,
    };
    let plan = output_plan_from_py(plan)?;
    let rgb_ro = rgb.readonly();
    let na_ro = na.readonly();
    let nb_ro = nb.as_ref().map(|b| b.readonly());
    let input = rgb_ro.as_slice()?;
    let na_s = na_ro.as_slice()?;
    let nb_s = match nb_ro.as_ref() {
        Some(b) => Some(b.as_slice()?),
        None => None,
    };
    let mut out = vec![0u8; n * 3];
    py.detach(|| output::finalize_u8(input, na_s, nb_s, &mut out, &plan, input_is_rec2020));
    rows3_u8(py, out, n)
}

#[pyfunction]
fn finalize_rec2020_u8_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    noise_a: &Bound<'py, PyAny>,
    noise_b: &Bound<'py, PyAny>,
    plan: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    finalize_common(py, rgb, noise_a, Some(noise_b), plan, true)
}

#[pyfunction]
fn finalize_output_u8_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    noise_a: &Bound<'py, PyAny>,
    noise_b: &Bound<'py, PyAny>,
    plan: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    finalize_common(py, rgb, noise_a, Some(noise_b), plan, false)
}

#[pyfunction]
fn finalize_rec2020_u8_noise_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    noise: &Bound<'py, PyAny>,
    plan: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    finalize_common(py, rgb, noise, None, plan, true)
}

#[pyfunction]
fn finalize_output_u8_noise_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    noise: &Bound<'py, PyAny>,
    plan: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    finalize_common(py, rgb, noise, None, plan, false)
}

/// Film appearance palette kernel (E3); returns (out, pre-clamp rows).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn film_appearance_apply_f32<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    scene_ev: &Bound<'py, PyAny>,
    f_hue: &Bound<'py, PyAny>,
    d_hue: &Bound<'py, PyAny>,
    f_chroma: &Bound<'py, PyAny>,
    d_chroma: &Bound<'py, PyAny>,
    f_density: &Bound<'py, PyAny>,
    d_density: &Bound<'py, PyAny>,
    ev_knots: &Bound<'py, PyAny>,
    nb_ab: &Bound<'py, PyAny>,
    has_neutral_bias: bool,
    strength: f32,
    neutral_c0: f32,
    chroma_knee: f32,
    chroma_power: f32,
    richness_mult: f32,
    density_mult: f32,
    m_fwd: &Bound<'py, PyAny>,
    m2: &Bound<'py, PyAny>,
    m2_inv: &Bound<'py, PyAny>,
    m_inv: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, i64)> {
    let rgb = as_f32_array(py, rgb)?;
    let shape = rgb.shape().to_vec();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err("rgb must be (N, 3) float32"));
    }
    let n = shape[0];
    let scene_ev = as_f32_array(py, scene_ev)?;
    if scene_ev.shape().len() != 1 || scene_ev.shape()[0] != n {
        return Err(PyValueError::new_err("scene_ev must be (N,)"));
    }
    let ev_knots = as_f32_array(py, ev_knots)?;
    let k = ev_knots.shape()[0];
    let fields = [
        as_f32_array(py, f_hue)?,
        as_f32_array(py, d_hue)?,
        as_f32_array(py, f_chroma)?,
        as_f32_array(py, d_chroma)?,
        as_f32_array(py, f_density)?,
        as_f32_array(py, d_density)?,
    ];
    let h = fields[0].shape().get(1).copied().unwrap_or(0);
    for arr in fields.iter() {
        let s = arr.shape();
        if s.len() != 2 || s[0] != k || s[1] != h {
            return Err(PyValueError::new_err("field tables must be (K, H)"));
        }
    }
    let nb_ab = as_f32_array(py, nb_ab)?;
    {
        let s = nb_ab.shape();
        if s.len() != 2 || s[0] != k || s[1] != 2 {
            return Err(PyValueError::new_err("nb_ab must be (K, 2)"));
        }
    }
    let mats = [
        as_f32_array(py, m_fwd)?,
        as_f32_array(py, m2)?,
        as_f32_array(py, m2_inv)?,
        as_f32_array(py, m_inv)?,
    ];
    for m in mats.iter() {
        if m.len() != 9 {
            return Err(PyValueError::new_err("matrices must have 9 elements"));
        }
    }
    let rgb_ro = rgb.readonly();
    let ev_ro = scene_ev.readonly();
    let knots_ro = ev_knots.readonly();
    let f_ro: Vec<_> = fields.iter().map(|a| a.readonly()).collect();
    let nb_ro = nb_ab.readonly();
    let m_ro: Vec<_> = mats.iter().map(|a| a.readonly()).collect();
    let params = film_appearance::FilmAppearanceParams {
        f_hue: f_ro[0].as_slice()?,
        d_hue: f_ro[1].as_slice()?,
        f_chroma: f_ro[2].as_slice()?,
        d_chroma: f_ro[3].as_slice()?,
        f_density: f_ro[4].as_slice()?,
        d_density: f_ro[5].as_slice()?,
        ev_knots: knots_ro.as_slice()?,
        nb_ab: nb_ro.as_slice()?,
        k_knots: k,
        h_knots: h,
        has_neutral_bias,
        strength,
        neutral_c0,
        chroma_knee,
        chroma_power,
        richness_mult,
        density_mult,
        m_fwd: m_ro[0].as_slice()?,
        m2: m_ro[1].as_slice()?,
        m2_inv: m_ro[2].as_slice()?,
        m_inv: m_ro[3].as_slice()?,
    };
    let input = rgb_ro.as_slice()?;
    let ev = ev_ro.as_slice()?;
    let mut out = vec![0f32; n * 3];
    let neg = py.detach(|| film_appearance::film_appearance_apply(input, ev, &mut out, &params));
    Ok((rows3_f32(py, out, n)?, neg))
}

/// The C++ extension's self-test, ported verbatim: one grey pixel through the
/// three kernels with identity-ish plans.
#[pyfunction]
fn self_test() -> bool {
    let mut plan = agx::NativeAgxPlan::default();
    plan.curve.black_ev = -10.0;
    plan.curve.range_ev = 16.5;
    plan.curve.gamma = 2.2;
    plan.curve.target_white = 1.0;
    plan.curve.slope = 0.1;
    plan.curve.toe_transition_x = 0.1;
    plan.curve.shoulder_transition_x = 0.9;
    plan.curve.toe_transition_y = 0.01;
    plan.curve.shoulder_transition_y = 0.99;
    plan.curve.toe_power = 1.5;
    plan.curve.shoulder_power = 3.3;
    plan.curve.toe_scale = 1.0;
    plan.curve.shoulder_scale = 1.0;
    for i in [0, 4, 8] {
        plan.inset[i] = 1.0;
        plan.outset[i] = 1.0;
        plan.oklab_m1[i] = 1.0;
        plan.oklab_m2[i] = 1.0;
        plan.oklab_m1_inv[i] = 1.0;
        plan.oklab_m2_inv[i] = 1.0;
    }
    plan.rec2020_to_xyz[0] = 0.637_f32 as f64;
    plan.rec2020_to_xyz[4] = 1.0;
    plan.rec2020_to_xyz[8] = 1.0;
    plan.xyz_to_rec2020[0] = 1.7167_f32 as f64;
    plan.xyz_to_rec2020[4] = 1.6165_f32 as f64;
    plan.xyz_to_rec2020[8] = 0.9421_f32 as f64;
    let input = [0.18f32, 0.18, 0.18];
    let mut out = [0f32; 3];
    agx::apply_agx_core_f32(&input, &mut out, &plan);

    let mut output_plan = output::NativeOutputPlan::default();
    for i in [0, 4, 8] {
        output_plan.rec2020_to_xyz[i] = 1.0;
        output_plan.xyz_to_output[i] = 1.0;
        output_plan.output_to_lms[i] = 1.0;
        output_plan.lms_to_output[i] = 1.0;
        output_plan.oklab_m2[i] = 1.0;
        output_plan.oklab_m2_inv[i] = 1.0;
    }
    output_plan.alpha = 0.05;
    let noise = [0f32; 3];
    let mut encoded = [0u8; 3];
    output::finalize_u8(&input, &noise, Some(&noise), &mut encoded, &output_plan, false);

    let table = hdr::HdrCurveTable {
        ev_start: -8.0,
        inv_step: 1.0 / 13.0,
        values: vec![0.0, 1.0],
    };
    let mut hdr_plan = hdr::NativeHdrPlan {
        inset: [0.0; 9],
        outset: [0.0; 9],
        rec2020_to_xyz: [0.0; 9],
        xyz_to_rec2020: [0.0; 9],
        xyz_to_output: [0.0; 9],
        oklab_m1: [0.0; 9],
        oklab_m2: [0.0; 9],
        oklab_m1_inv: [0.0; 9],
        oklab_m2_inv: [0.0; 9],
        formation_luma: [1.0 / 3.0; 3],
        output_luma: [1.0 / 3.0; 3],
        hue_restore: 0.6,
        punch_strength: 0.0,
        global_rho: 0.0,
        peak: 4.0,
        native_table: table.clone(),
        reference_table: table,
        has_reference: false,
    };
    for i in [0, 4, 8] {
        hdr_plan.inset[i] = 1.0;
        hdr_plan.outset[i] = 1.0;
        hdr_plan.rec2020_to_xyz[i] = 1.0;
        hdr_plan.xyz_to_rec2020[i] = 1.0;
        hdr_plan.xyz_to_output[i] = 1.0;
        hdr_plan.oklab_m1[i] = 1.0;
        hdr_plan.oklab_m2[i] = 1.0;
        hdr_plan.oklab_m1_inv[i] = 1.0;
        hdr_plan.oklab_m2_inv[i] = 1.0;
    }
    let mut hdr_out = [0f32; 3];
    hdr::apply_hdr_formation_f32(&input, None, &mut hdr_out, &hdr_plan);

    out[0] >= 0.0
        && out[1] >= 0.0
        && out[2] >= 0.0
        && encoded[0] > 0
        && encoded[1] > 0
        && encoded[2] > 0
        && hdr_out[0] > 0.0
        && hdr_out[0] <= 4.0
        && hdr_out[1] == hdr_out[0]
        && hdr_out[2] == hdr_out[0]
}

// ---------------------------------------------------------------------------
// Stage 1 (2026-09-15): decode-side evidence and analysis/delivery metrics.
// NumPy stays the reference implementation (dngscan/raw_io.py, analysis.py,
// gainmap.py); tests/test_rust_stage1.py pins bit-identity.

fn as_f16_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDyn<half::f16>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float16")?)?;
    let arr = np.getattr("ascontiguousarray")?.call((obj,), Some(&kwargs))?;
    Ok(arr.cast_into::<PyArrayDyn<half::f16>>()?)
}

fn as_u8_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("uint8")?)?;
    let arr = np.getattr("ascontiguousarray")?.call((obj,), Some(&kwargs))?;
    Ok(arr.cast_into::<PyArrayDyn<u8>>()?)
}

fn as_f64_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    let arr = np.getattr("ascontiguousarray")?.call((obj,), Some(&kwargs))?;
    Ok(arr.cast_into::<PyArrayDyn<f64>>()?)
}

/// raw_io._feather_masks_f16: (h, w, c) float32 -> (h, w, c) float16.
#[pyfunction]
fn feather_masks_f16<'py>(py: Python<'py>, mask: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mask = as_f32_array(py, mask)?;
    let shape = mask.shape().to_vec();
    if shape.len() != 3 {
        return Err(PyValueError::new_err("mask must be (H, W, C)"));
    }
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let ro = mask.readonly();
    let data = ro.as_slice()?;
    let out = py.detach(|| evidence::feather_masks_f16(data, h, w, c));
    Ok(PyArray1::from_vec(py, out).reshape([h, w, c])?.into_any())
}

/// raw_io._apply_gain_maps_mosaic for ONE opcode, in place on the uint16
/// visible mosaic view. `op` carries the DNG GainMap attributes (top, left,
/// bottom, right, row_pitch, col_pitch, origin_v/h, spacing_v/h, points_v/h,
/// gains); `colors` is the uint8 CFA index view of the same window.
#[pyfunction]
fn apply_gain_map_mosaic<'py>(
    py: Python<'py>,
    mut img: PyReadwriteArray2<'py, u16>,
    colors: PyReadonlyArray2<'py, u8>,
    op: &Bound<'py, PyAny>,
    blacks: Vec<f32>,
    whites: Vec<f32>,
) -> PyResult<()> {
    let gains_obj = as_f64_array(py, &op.getattr("gains")?)?;
    let gshape = gains_obj.shape().to_vec();
    if gshape.len() != 3 {
        return Err(PyValueError::new_err("gains must be (points_v, points_h, planes)"));
    }
    let planes = gshape[2];
    let gains_ro = gains_obj.readonly();
    let gains_all = gains_ro.as_slice()?;
    // map plane 0 (dng_gain_map::Interpolate for the single mosaic image plane)
    let gains: Vec<f64> = gains_all.iter().step_by(planes.max(1)).copied().collect();
    let geti = |name: &str| -> PyResult<i64> { op.getattr(name)?.extract::<i64>() };
    let getf = |name: &str| -> PyResult<f64> { op.getattr(name)?.extract::<f64>() };
    let gop = evidence::GainMapOp {
        top: geti("top")?,
        left: geti("left")?,
        bottom: geti("bottom")?,
        right: geti("right")?,
        row_pitch: geti("row_pitch")?,
        col_pitch: geti("col_pitch")?,
        origin_v: getf("origin_v")?,
        origin_h: getf("origin_h")?,
        spacing_v: getf("spacing_v")?,
        spacing_h: getf("spacing_h")?,
        points_v: geti("points_v")?,
        points_h: geti("points_h")?,
        gains: &gains,
    };
    if (gshape[0] as i64) != gop.points_v || (gshape[1] as i64) != gop.points_h {
        return Err(PyValueError::new_err("gains grid does not match points_v/points_h"));
    }
    let mut img_view = img.as_array_mut();
    let colors_view = colors.as_array();
    let (h, w) = (img_view.shape()[0], img_view.shape()[1]);
    if colors_view.shape() != [h, w] {
        return Err(PyValueError::new_err("colors must match img shape"));
    }
    let color_at = |y: usize, x: usize| -> usize { colors_view[[y, x]] as usize };
    // the view may be strided (visible window): index through ndarray
    let mut writes: Vec<(usize, usize, u16)> = Vec::new();
    {
        let get = |y: usize, x: usize| -> u16 { img_view[[y, x]] };
        let mut set = |y: usize, x: usize, v: u16| writes.push((y, x, v));
        evidence::apply_gain_map_mosaic(h, w, &gop, &blacks, &whites, &color_at, &get, &mut set);
    }
    for (y, x, v) in writes {
        img_view[[y, x]] = v;
    }
    Ok(())
}

/// analysis.compute_gamut_metrics core: (counts per matrix, bright_total).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gamut_counts<'py>(
    py: Python<'py>,
    scene: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    inv_scale: f32,
    rec2020_to_xyz: Vec<f64>,
    matrices: Vec<Vec<f64>>,
    eps: f32,
    gamut_eps: f32,
) -> PyResult<(Vec<u64>, usize)> {
    let scene = as_f32_array(py, scene)?;
    let sshape = scene.shape().to_vec();
    if sshape.len() != 2 || sshape[1] < 3 {
        return Err(PyValueError::new_err("scene must be (N, >=3)"));
    }
    let y = as_f32_array(py, y)?;
    if y.shape().len() != 1 || y.shape()[0] != sshape[0] {
        return Err(PyValueError::new_err("y must be (N,)"));
    }
    let m: [f64; 9] = rec2020_to_xyz
        .as_slice()
        .try_into()
        .map_err(|_| PyValueError::new_err("rec2020_to_xyz must have 9 elements"))?;
    let mats: Vec<[f64; 9]> = matrices
        .iter()
        .map(|v| v.as_slice().try_into().map_err(|_| PyValueError::new_err("matrix must have 9 elements")))
        .collect::<PyResult<_>>()?;
    let sro = scene.readonly();
    let yro = y.readonly();
    let sdata = sro.as_slice()?;
    let ydata = yro.as_slice()?;
    let stride = sshape[1];
    Ok(py.detach(|| metrics::gamut_counts(sdata, stride, ydata, inv_scale, &m, &mats, eps, gamut_eps)))
}

/// gainmap._roundtrip_error on the decoded (h, w, >=3) float16 rendition and
/// the intended (h, w, >=3) float16 one. Returns the metrics dict.
#[pyfunction]
fn hdr_roundtrip_metrics<'py>(
    py: Python<'py>,
    expanded: &Bound<'py, PyAny>,
    intended: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = as_f16_array(py, expanded)?;
    let e = as_f16_array(py, intended)?;
    let (sa, se) = (a.shape().to_vec(), e.shape().to_vec());
    if sa.len() != 3 || se.len() != 3 || sa[2] < 3 || se[2] < 3 {
        return Err(PyValueError::new_err("renditions must be (H, W, >=3)"));
    }
    if sa[0] != se[0] || sa[1] != se[1] {
        return Err(PyValueError::new_err("rendition shapes differ"));
    }
    let aro = a.readonly();
    let ero = e.readonly();
    let ad = aro.as_slice()?;
    let ed = ero.as_slice()?;
    let r = py
        .detach(|| metrics::hdr_roundtrip(ad, sa[2], ed, se[2], sa[0], sa[1]))
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("chroma_error", r.chroma_error)?;
    d.set_item("relative_error", r.relative_error)?;
    d.set_item("median_relative_error", r.median_relative_error)?;
    d.set_item("p95_relative_error", r.p95_relative_error)?;
    d.set_item("p99_relative_error", r.p99_relative_error)?;
    d.set_item("p999_relative_error", r.p999_relative_error)?;
    d.set_item("block_median_relative_error", r.block_median_relative_error)?;
    d.set_item("block_p95_relative_error", r.block_p95_relative_error)?;
    d.set_item("block_p99_relative_error", r.block_p99_relative_error)?;
    d.set_item("block_chroma_error", r.block_chroma_error)?;
    Ok(d)
}

/// gainmap._base_roundtrip_error on (h, w, 3) uint8 renditions.
#[pyfunction]
fn base_roundtrip_metrics<'py>(
    py: Python<'py>,
    decoded: &Bound<'py, PyAny>,
    intended: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = as_u8_array(py, decoded)?;
    let e = as_u8_array(py, intended)?;
    let (sa, se) = (a.shape().to_vec(), e.shape().to_vec());
    if sa.len() != 3 || sa[2] != 3 || sa != se {
        return Err(PyValueError::new_err("renditions must be identical (H, W, 3) uint8"));
    }
    let aro = a.readonly();
    let ero = e.readonly();
    let ad = aro.as_slice()?;
    let ed = ero.as_slice()?;
    let r = py
        .detach(|| metrics::base_roundtrip(ad, ed, sa[0], sa[1]))
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("base_mean_code_error", r.mean_code_error)?;
    d.set_item("base_p99_code_error", r.p99_code_error)?;
    d.set_item("base_max_code_error", r.max_code_error)?;
    d.set_item("base_channel_bias_code_error", r.channel_bias_code_error)?;
    d.set_item("base_block_p99_code_error", r.block_p99_code_error)?;
    Ok(d)
}

#[pymodule]
fn _dngscan_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "dngscan optional native kernels (Rust)")?;
    m.add_function(wrap_pyfunction!(set_thread_budget, m)?)?;
    m.add_function(wrap_pyfunction!(native_abi_version, m)?)?;
    m.add_function(wrap_pyfunction!(film_appearance_apply_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apply_agx_core_f32, m)?)?;
    m.add_function(wrap_pyfunction!(apply_hdr_formation_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fit_output_gamut_f32, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_rec2020_u8_f32, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_output_u8_f32, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_rec2020_u8_noise_f32, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_output_u8_noise_f32, m)?)?;
    m.add_function(wrap_pyfunction!(self_test, m)?)?;
    m.add_function(wrap_pyfunction!(feather_masks_f16, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gain_map_mosaic, m)?)?;
    m.add_function(wrap_pyfunction!(gamut_counts, m)?)?;
    m.add_function(wrap_pyfunction!(hdr_roundtrip_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(base_roundtrip_metrics, m)?)?;
    Ok(())
}
