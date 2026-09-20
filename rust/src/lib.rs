// SPDX-License-Identifier: GPL-3.0-or-later
//! `dngscan._dngscan_fast` — the optional native kernels, ported from the
//! pybind11 C++ extension (2026-09). The versioned API is checked by the
//! Python side (dngscan/_fast.py, dngscan/fast_plan.py). Every
//! kernel replicates the NumPy reference's float32 operation order; the parity
//! gates are tests/test_fast_backend.py, tests/test_hdr_native.py and
//! tests/test_film_appearance_p10.py.
mod agx;
mod budget;
mod evidence;
mod lens;
mod loss;
mod film_appearance;
mod film_core;
mod hdr;
mod metrics;
mod numpy_sum;
mod output;
mod pixel;
mod sensor;
mod spatial;
mod stats;

use numpy::{PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::sync::atomic::Ordering;

/// Native ABI version. v8: exact float64 two-stage output matrices; v9
/// (#136): HDR per-pixel peak-proximity confidence; v10 (batch 25): HDR
/// output stage float64; v11 (math review 2026-09-03): inset/outset and the
/// punch/Oklab matrices of both kernels exact float64. v12 adds the P3
/// luminance row and local luminance metrics to the HDR delivery scanner.
/// v13 tracks in-place GainMap clipping and adds camera-plane DNG warps.
/// v14 adds WarpRectilinear2 and embedded lens radial splines.
/// v15 adds crop-footprint and in-place processing-loss maxima.
/// v16 adds exact sensor RGB-group clipping counts.
pub const NATIVE_ABI_VERSION: i32 = 16;

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

/// Preserve strides when the kernel supports borrowed ndarray views.
fn as_view_array<'py, T: numpy::Element>(
    py: Python<'py>, obj: &Bound<'py, PyAny>, dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    Ok(np.getattr("asarray")?.call((obj,), Some(&kwargs))?.cast_into::<PyArrayDyn<T>>()?)
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

/// ndarray's borrowed view requires native endian/aligned typed pointers and
/// dimensions/byte-address spans representable by isize, including strided views.
fn sensor_view_shape<T: numpy::Element>(
    array: &Bound<'_, PyArrayDyn<T>>, name: &str,
) -> PyResult<usize> {
    let shape = array.shape();
    if !(shape.len() == 2 || shape.len() == 3)
        || shape.iter().any(|&dim| dim > isize::MAX as usize) {
        return Err(PyValueError::new_err(format!("{name} must be a 2-D or 3-D array")));
    }
    if array.dtype().is_native_byteorder() == Some(false) {
        return Err(PyValueError::new_err(format!("{name} must have native byte order")));
    }
    let align = std::mem::align_of::<T>();
    if (array.data() as usize) % align != 0
        || array.strides().iter().any(|&stride| stride % align as isize != 0) {
        return Err(PyValueError::new_err(format!("{name} must be aligned")));
    }
    let len = shape.iter().try_fold(1usize, |n, &dim| n.checked_mul(dim));
    let nonzero_len = shape.iter().try_fold(1usize, |n, &dim| n.checked_mul(dim.max(1)));
    let span = shape.iter().zip(array.strides()).try_fold(
        std::mem::size_of::<T>(), |span, (&dim, &stride)| {
            span.checked_add(dim.saturating_sub(1).checked_mul(stride.unsigned_abs())?)
        },
    );
    if len.and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
        .is_none_or(|bytes| bytes > isize::MAX as usize)
        || nonzero_len.is_none_or(|n| n > isize::MAX as usize)
        || array.strides().contains(&isize::MIN)
        || span.is_none_or(|bytes| bytes > isize::MAX as usize) {
        return Err(PyValueError::new_err(format!("{name} dimensions or strides are too large")));
    }
    Ok(len.expect("validated sensor array length"))
}

#[pyfunction]
fn sensor_rgb_clip_counts_u16(
    py: Python<'_>, raw: &Bound<'_, PyAny>, colors: &Bound<'_, PyAny>,
    thresholds: Vec<i32>, groups: Vec<u8>, period_h: usize, period_w: usize,
) -> PyResult<Vec<u64>> {
    let raw = raw.cast::<PyArrayDyn<u16>>()
        .map_err(|_| PyValueError::new_err("raw must be a uint16 array"))?;
    let colors = colors.cast::<PyArrayDyn<u8>>()
        .map_err(|_| PyValueError::new_err("colors must be a uint8 array"))?;
    let sample_count = sensor_view_shape(raw, "raw")?;
    sensor_view_shape(colors, "colors")?;
    if raw.shape() != colors.shape() {
        return Err(PyValueError::new_err("colors must match raw shape"));
    }
    if period_h == 0 || period_w == 0
        || period_h.checked_mul(period_w).is_none_or(|n| n > isize::MAX as usize)
        || (raw.ndim() == 3 && (period_h != 1 || period_w != 1)) {
        return Err(PyValueError::new_err("invalid sensor cell period"));
    }
    let thresholds: [i32; 256] = thresholds.try_into()
        .map_err(|_| PyValueError::new_err("thresholds must have 256 entries"))?;
    let groups: [u8; 256] = groups.try_into()
        .map_err(|_| PyValueError::new_err("groups must have 256 entries"))?;
    if groups.iter().any(|&group| !matches!(group, 0 | 1 | 2 | 4)) {
        return Err(PyValueError::new_err("groups entries must be 0, 1, 2 or 4"));
    }
    if sample_count == 0 {
        // A zero-channel linear pixel has zero clipped groups. Avoid creating
        // empty ndarray views: rust-numpy normalizes negative strides through
        // pointer arithmetic even when the associated axis has length zero.
        let cells = if raw.ndim() == 3 { raw.shape()[0] * raw.shape()[1] } else { 0 };
        return Ok(vec![cells as u64, 0, 0, 0]);
    }
    let raw_read = raw.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let colors_read = colors.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let samples = if raw.ndim() == 2 {
        sensor::Samples::Mosaic(
            raw_read.as_array().into_dimensionality::<numpy::ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("raw must be 2-D"))?,
            colors_read.as_array().into_dimensionality::<numpy::ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("colors must be 2-D"))?,
            period_h, period_w,
        )
    } else {
        sensor::Samples::Linear(
            raw_read.as_array().into_dimensionality::<numpy::ndarray::Ix3>()
                .map_err(|_| PyValueError::new_err("raw must be 3-D"))?,
            colors_read.as_array().into_dimensionality::<numpy::ndarray::Ix3>()
                .map_err(|_| PyValueError::new_err("colors must be 3-D"))?,
        )
    };
    py.detach(|| sensor::rgb_clip_counts(samples, &thresholds, &groups, sample_count))
        .map(|counts| counts.to_vec())
        .map_err(|e| PyRuntimeError::new_err(format!("sensor workers: {e}")))
}

fn as_f16_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDyn<half::f16>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float16")?)?;
    let arr = np.getattr("asarray")?.call((obj,), Some(&kwargs))?;
    Ok(arr.cast_into::<PyArrayDyn<half::f16>>()?)
}

fn loss_rgb_shape<T: numpy::Element>(array: &Bound<'_, PyArrayDyn<T>>, name: &str) -> PyResult<(usize, usize)> {
    let shape = array.shape();
    if shape.len() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] != 3
        || shape.iter().try_fold(std::mem::size_of::<T>(), |n, &dim| n.checked_mul(dim))
            .is_none_or(|n| n > isize::MAX as usize) {
        return Err(PyValueError::new_err(format!("{name} must be nonempty H,W,3")));
    }
    let align = std::mem::align_of::<T>();
    if (array.data() as usize) % align != 0
        || array.strides().iter().any(|&stride| stride % align as isize != 0) {
        return Err(PyValueError::new_err(format!("{name} must be aligned")));
    }
    Ok((shape[0], shape[1]))
}

fn loss_indices(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<usize>> {
    let array = obj.cast::<PyArray1<isize>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 1-D intp array")))?;
    if array.is_empty() || (array.data() as usize) % std::mem::align_of::<isize>() != 0
        || array.strides()[0] % std::mem::align_of::<isize>() as isize != 0 {
        return Err(PyValueError::new_err(format!("{name} must be nonempty and aligned")));
    }
    let read = array.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    read.as_array().iter().map(|&value| usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be nonnegative")))).collect()
}

/// Compare actual byte-address envelopes, not just NumPy base-object identity.
/// This also catches aliases manufactured through independent buffer wrappers.
fn loss_byte_range<T: numpy::Element>(array: &Bound<'_, PyArrayDyn<T>>) -> (i128, i128) {
    let mut lo = array.data() as usize as i128;
    let mut hi = lo + std::mem::size_of::<T>() as i128;
    for (&dim, &stride) in array.shape().iter().zip(array.strides()) {
        let offset = (dim - 1) as i128 * stride as i128;
        lo += offset.min(0);
        hi += offset.max(0);
    }
    (lo, hi)
}

fn crop_loss_typed<'py, T: numpy::Element + loss::LossValue>(
    py: Python<'py>, values: &Bound<'py, PyArrayDyn<T>>,
    ylo: &[usize], yhi: &[usize], xlo: &[usize], xhi: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let (height, width) = loss_rgb_shape(values, "values")?;
    if ylo.len() != yhi.len() || xlo.len() != xhi.len()
        || ylo.iter().zip(yhi).any(|(&lo, &hi)| lo > hi || hi > height)
        || xlo.iter().zip(xhi).any(|(&lo, &hi)| lo > hi || hi > width)
        || ylo.len().checked_mul(xlo.len()).and_then(|n| n.checked_mul(3 * std::mem::size_of::<T>()))
            .is_none_or(|n| n > isize::MAX as usize) {
        return Err(PyValueError::new_err("invalid crop footprint bounds"));
    }
    let read = values.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let view = read.as_array().into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| PyValueError::new_err("values must be H,W,3"))?;
    let result = py.detach(|| loss::crop(view, ylo, yhi, xlo, xhi))
        .map_err(|e| PyRuntimeError::new_err(format!("loss workers: {e}")))?;
    match result {
        Some(out) => Ok(PyArray1::from_vec(py, out).reshape([ylo.len(), xlo.len(), 3])?.into_any()),
        None => Ok(py.None().into_bound(py)),
    }
}

#[pyfunction]
fn crop_loss_footprint<'py>(
    py: Python<'py>, values: &Bound<'py, PyAny>, ylo: &Bound<'py, PyAny>,
    yhi: &Bound<'py, PyAny>, xlo: &Bound<'py, PyAny>, xhi: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let ylo = loss_indices(ylo, "ylo")?;
    let yhi = loss_indices(yhi, "yhi")?;
    let xlo = loss_indices(xlo, "xlo")?;
    let xhi = loss_indices(xhi, "xhi")?;
    if let Ok(array) = values.cast::<PyArrayDyn<half::f16>>() {
        crop_loss_typed(py, array, &ylo, &yhi, &xlo, &xhi)
    } else if let Ok(array) = values.cast::<PyArrayDyn<f32>>() {
        crop_loss_typed(py, array, &ylo, &yhi, &xlo, &xhi)
    } else {
        Err(PyValueError::new_err("values must be a float16 or float32 array"))
    }
}

fn merge_loss_typed<'py, T: numpy::Element + loss::LossValue>(
    py: Python<'py>, masks: &Bound<'py, PyArrayDyn<half::f16>>,
    processing: &Bound<'py, PyArrayDyn<T>>, maps: Option<(&[usize], &[usize])>,
) -> PyResult<Bound<'py, PyAny>> {
    let (height, width) = loss_rgb_shape(masks, "masks")?;
    let (source_h, source_w) = loss_rgb_shape(processing, "processing")?;
    if !masks.is_c_contiguous() {
        return Err(PyValueError::new_err("masks must be C contiguous"));
    }
    match maps {
        None if (height, width) != (source_h, source_w) =>
            return Err(PyValueError::new_err("processing must match masks shape without maps")),
        Some((ys, xs)) if ys.len() != height || xs.len() != width
            || ys.iter().any(|&y| y >= source_h) || xs.iter().any(|&x| x >= source_w) =>
            return Err(PyValueError::new_err("invalid nearest-neighbour index maps")),
        _ => (),
    }
    let (mlo, mhi) = loss_byte_range(masks);
    let (plo, phi) = loss_byte_range(processing);
    if mlo < phi && plo < mhi {
        return Err(PyValueError::new_err("masks and processing must not overlap"));
    }
    let read = processing.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let view = read.as_array().into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| PyValueError::new_err("processing must be H,W,3"))?;
    let mut write = masks.try_readwrite().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let target = write.as_slice_mut()?;
    let supported = py.detach(|| loss::merge(target, height, width, view, maps))
        .map_err(|e| PyRuntimeError::new_err(format!("loss workers: {e}")))?;
    Ok(if supported { masks.clone().into_any() } else { py.None().into_bound(py) })
}

#[pyfunction(signature = (masks, processing, y_indices=None, x_indices=None))]
fn merge_processing_loss_f16_inplace<'py>(
    py: Python<'py>, masks: &Bound<'py, PyAny>, processing: &Bound<'py, PyAny>,
    y_indices: Option<&Bound<'py, PyAny>>, x_indices: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let masks = masks.cast::<PyArrayDyn<half::f16>>()
        .map_err(|_| PyValueError::new_err("masks must be a float16 array"))?;
    let indices = match (y_indices, x_indices) {
        (Some(ys), Some(xs)) => Some((loss_indices(ys, "y_indices")?, loss_indices(xs, "x_indices")?)),
        (None, None) => None,
        _ => return Err(PyValueError::new_err("provide both nearest-neighbour index maps")),
    };
    let maps = indices.as_ref().map(|(ys, xs)| (ys.as_slice(), xs.as_slice()));
    if let Ok(array) = processing.cast::<PyArrayDyn<half::f16>>() {
        merge_loss_typed(py, masks, array, maps)
    } else if let Ok(array) = processing.cast::<PyArrayDyn<f32>>() {
        merge_loss_typed(py, masks, array, maps)
    } else {
        Err(PyValueError::new_err("processing must be a float16 or float32 array"))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(image, coefficients, cx, cy, aspect, fisheye, loss, knots=None, scales=None, scale=1.0))]
fn warp_dng<'py>(py: Python<'py>, image: &Bound<'py, PyAny>, coefficients: Vec<Vec<f64>>,
    cx: f64, cy: f64, aspect: f64, fisheye: bool, loss: bool,
    knots: Option<Vec<f64>>, scales: Option<Vec<Vec<f64>>>, scale: f64,
) -> PyResult<Bound<'py, PyAny>> {
    if ![1,3].contains(&coefficients.len()) || coefficients.iter().any(|r| ![6,20].contains(&r.len()) || r.iter().any(|x| !x.is_finite()))
        || !cx.is_finite() || !cy.is_finite() || !(0.0..=1.0).contains(&cx) || !(0.0..=1.0).contains(&cy)
        || !aspect.is_finite() || aspect <= 0.0 {
        return Err(PyValueError::new_err("invalid DNG warp parameters"));
    }
    let knots = knots.unwrap_or_default();
    let scales = scales.unwrap_or_default();
    if !scale.is_finite() || scale <= 0.0 || (!knots.is_empty() &&
        (knots.len()<2 || knots.iter().any(|x| !x.is_finite()) ||
         knots.windows(2).any(|a| a[1]<=a[0]) || scales.len()!=3 ||
         scales.iter().any(|r| r.len()!=knots.len() || r.iter().any(|x| !x.is_finite() || *x<=0.0)))) {
        return Err(PyValueError::new_err("invalid radial spline"));
    }
    let coeff: Vec<[f64;6]> = coefficients.iter().map(|r| if r.len()==6 {
        r.as_slice().try_into().unwrap()
    } else {[1.0,0.0,0.0,0.0,r[15],r[16]]}).collect();
    for r in &coefficients {
        if r.len()==20 && (!(0.0..=1.0).contains(&r[17]) || r[18]<=r[17] || r[18]>1.0 || ![0.0,1.0].contains(&r[19])) {
            return Err(PyValueError::new_err("invalid extended DNG radial parameters"));
        }
    }
    if loss {
        let a = as_f16_array(py, image)?;
        let ro = a.readonly();
        let src = ro.as_array().into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| PyValueError::new_err("warp image must be H,W,3"))?;
        let (h,w,c) = src.dim();
        if h==0 || w==0 || c!=3 { return Err(PyValueError::new_err("warp image must be nonempty H,W,3")); }
        let out = py.detach(|| lens::warp(src,&coeff,&coefficients,cx,cy,aspect,fisheye,true,&knots,&scales,scale,
            |v| v.to_f64(), half::f16::from_f64));
        Ok(PyArray1::from_vec(py,out).reshape([h,w,3])?.into_any())
    } else {
        let a = as_view_array::<u16>(py, image, "uint16")?;
        let ro = a.readonly();
        let src = ro.as_array().into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| PyValueError::new_err("warp image must be H,W,3"))?;
        let (h,w,c) = src.dim();
        if h==0 || w==0 || c!=3 { return Err(PyValueError::new_err("warp image must be nonempty H,W,3")); }
        let out = py.detach(|| lens::warp(src,&coeff,&coefficients,cx,cy,aspect,fisheye,false,&knots,&scales,scale,
            |v| v as f64, |v| v as u16));
        Ok(PyArray1::from_vec(py,out).reshape([h,w,3])?.into_any())
    }
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
#[pyfunction(signature = (img, colors, op, blacks, whites, loss=None))]
fn apply_gain_map_mosaic<'py>(
    py: Python<'py>,
    mut img: PyReadwriteArray2<'py, u16>,
    colors: PyReadonlyArray2<'py, u8>,
    op: &Bound<'py, PyAny>,
    blacks: Vec<f32>,
    whites: Vec<f32>,
    mut loss: Option<PyReadwriteArray2<'py, u8>>,
) -> PyResult<()> {
    let gains_obj = as_f64_array(py, &op.getattr("gains")?)?;
    let gshape = gains_obj.shape().to_vec();
    if gshape.len() != 3 || gshape.contains(&0) {
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
    if gop.top < 0 || gop.left < 0 || gop.row_pitch <= 0 || gop.col_pitch <= 0 {
        return Err(PyValueError::new_err("GainMap origin must be nonnegative and pitches positive"));
    }
    let img_view = img.as_array_mut();
    let colors_view = colors.as_array();
    let (h, w) = (img_view.shape()[0], img_view.shape()[1]);
    if colors_view.shape() != [h, w] {
        return Err(PyValueError::new_err("colors must match img shape"));
    }
    // Keep the NumPy borrows alive while native code updates the actual
    // visible window. No per-pixel write list or contiguous mosaic copy.
    let loss_view = loss.as_mut().map(|m| m.as_array_mut());
    if loss_view.as_ref().is_some_and(|m| m.shape() != [h, w]) {
        return Err(PyValueError::new_err("loss must match img shape"));
    }
    py.detach(move || evidence::apply_gain_map_mosaic(img_view, colors_view, &gop, &blacks, &whites, loss_view));
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
    luma_weights: [f32; 3],
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
    let ad = aro.as_array().into_dimensionality::<numpy::ndarray::Ix3>().unwrap();
    let ed = ero.as_array().into_dimensionality::<numpy::ndarray::Ix3>().unwrap();
    let r = py
        .detach(|| metrics::hdr_roundtrip(ad, ed, sa[0], sa[1], &luma_weights))
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
    d.set_item("block_p95_luma_error", r.block_p95_luma_error)?;
    d.set_item("highlight_max_luma_error", r.highlight_max_luma_error)?;
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

// ---------------------------------------------------------------------------
// Stage 2 (2026-09-15): film spatial operators (dngscan/film_optics.py).

fn vec3_f32(v: Vec<f32>, what: &str) -> PyResult<[f32; 3]> {
    v.as_slice().try_into().map_err(|_| PyValueError::new_err(format!("{what} must have 3 elements")))
}

fn halation_comps_from_py(py: Python<'_>, comps: &Bound<'_, PyAny>) -> PyResult<Vec<spatial::HalationComponent>> {
    let mut out = Vec::new();
    for item in comps.try_iter()? {
        let item = item?;
        let gate = as_f32_array(py, &item.get_item(0)?)?;
        let transfer = as_f32_array(py, &item.get_item(1)?)?;
        if gate.shape() != [3, 2] || transfer.shape() != [3, 3] {
            return Err(PyValueError::new_err("halation component must be (gate_ev (3,2), transfer (3,3))"));
        }
        let g = gate.readonly();
        let g = g.as_slice()?;
        let t = transfer.readonly();
        let t = t.as_slice()?;
        out.push(spatial::HalationComponent {
            gate_ev: [[g[0], g[1]], [g[2], g[3]], [g[4], g[5]]],
            transfer: [[t[0], t[1], t[2]], [t[3], t[4], t[5]], [t[6], t[7], t[8]]],
        });
    }
    Ok(out)
}

/// Run area decimation on the source dtype, promoting each sample at its
/// original float64 accumulation point instead of copying the whole band.
#[allow(clippy::too_many_arguments)]
fn area_decimate_typed<T: numpy::Element + Copy + Into<f64> + Sync>(
    py: Python<'_>, rows: &Bound<'_, PyArrayDyn<T>>, y0: usize,
    h: usize, w: usize, out_h: usize, out_w: usize, acc: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let shape = rows.shape();
    let (n, c) = (shape[0], shape[2]);
    if y0 > h || n > h - y0 || shape[1] != w || out_h == 0 || out_w == 0 {
        return Err(PyValueError::new_err("invalid source row range or decimation dimensions"));
    }
    let rro = rows.readonly();
    let rdata = rro.as_array().into_dimensionality::<numpy::ndarray::Ix3>().unwrap();
    let np = py.import("numpy")?;
    let dtype = acc.getattr("dtype")?;
    if dtype.eq(np.getattr("float64")?)? {
        let mut a = acc.cast::<PyArrayDyn<f64>>()?.readwrite();
        if a.shape() != [out_h, out_w, c] {
            return Err(PyValueError::new_err("acc must be (out_h, out_w, c)"));
        }
        let data = a.as_slice_mut()?;
        py.detach(|| spatial::area_decimate_rows(rdata, n, y0, h, w, out_h, out_w, c, spatial::Acc::F64(data)));
    } else if dtype.eq(np.getattr("float32")?)? {
        let mut a = acc.cast::<PyArrayDyn<f32>>()?.readwrite();
        if a.shape() != [out_h, out_w, c] {
            return Err(PyValueError::new_err("acc must be (out_h, out_w, c)"));
        }
        let data = a.as_slice_mut()?;
        py.detach(|| spatial::area_decimate_rows(rdata, n, y0, h, w, out_h, out_w, c, spatial::Acc::F32(data)));
    } else {
        return Err(PyValueError::new_err("acc must be float64 or float32"));
    }
    Ok(())
}

/// film_optics.area_decimate_rows: float32/float64 source, accumulator in place.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn area_decimate_rows<'py>(
    py: Python<'py>, rows: &Bound<'py, PyAny>, y0: usize,
    h: usize, w: usize, out_h: usize, out_w: usize, acc: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let rows = np.getattr("asarray")?.call1((rows,))?;
    let shape: Vec<usize> = rows.getattr("shape")?.extract()?;
    let c = *shape.last().ok_or_else(|| PyValueError::new_err("rows must have a channel axis"))?;
    // Preserve the original flat-band input convention, but an already
    // shaped (including empty) view needs no reshape or contiguous copy.
    let rows = if shape.len() == 3 && shape[1] == w {
        rows
    } else if w > 0 && c > 0 {
        rows.call_method1("reshape", ((-1i64, w, c),))?
    } else {
        return Err(PyValueError::new_err("rows must reshape to (n, w, c)"));
    };
    if let Ok(a) = rows.cast::<PyArrayDyn<f32>>() {
        area_decimate_typed(py, a, y0, h, w, out_h, out_w, acc)
    } else {
        let a = as_view_array::<f64>(py, &rows, "float64")?;
        area_decimate_typed(py, &a, y0, h, w, out_h, out_w, acc)
    }
}

/// film_optics.upsample_rows -> (y1 - y0, width, c) float32.
#[pyfunction]
fn upsample_rows<'py>(
    py: Python<'py>,
    map_dec: &Bound<'py, PyAny>,
    y0: usize,
    y1: usize,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let m = as_f32_array(py, map_dec)?;
    let shape = m.shape().to_vec();
    if shape.len() != 3 {
        return Err(PyValueError::new_err("map must be (dh, dw, c)"));
    }
    let (dh, dw, c) = (shape[0], shape[1], shape[2]);
    let ro = m.readonly();
    let d = ro.as_slice()?;
    let out = py.detach(|| spatial::upsample_rows(d, dh, dw, c, y0, y1, height, width));
    Ok(PyArray1::from_vec(py, out).reshape([y1 - y0, width, c])?.into_any())
}

/// film_optics._gaussian_blur_slabbed on (h, w, c) float32 -> new array.
#[pyfunction]
fn gaussian_blur_slabbed<'py>(py: Python<'py>, img: &Bound<'py, PyAny>, sigma: f64, periodic: bool) -> PyResult<Bound<'py, PyAny>> {
    let a = as_view_array::<f32>(py, img, "float32")?;
    let shape = a.shape().to_vec();
    if shape.len() != 3 {
        return Err(PyValueError::new_err("img must be (h, w, c)"));
    }
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let ro = a.readonly();
    let view = ro.as_array().into_dimensionality::<numpy::ndarray::Ix3>().unwrap();
    let out = py.detach(|| spatial::gaussian_blur(view, sigma, periodic));
    Ok(PyArray1::from_vec(py, out).reshape([h, w, c])?.into_any())
}

/// film_optics._blur_small_sigma on one (h, w) float32 plane.
#[pyfunction]
fn blur_small_sigma<'py>(py: Python<'py>, chan: &Bound<'py, PyAny>, sigma_px: f64) -> PyResult<Bound<'py, PyAny>> {
    let a = as_view_array::<f32>(py, chan, "float32")?;
    let shape = a.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("chan must be (h, w)"));
    }
    let (h, w) = (shape[0], shape[1]);
    let ro = a.readonly();
    let view = ro.as_array().into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
    let out = py.detach(|| spatial::blur_small_sigma(view, sigma_px));
    Ok(PyArray1::from_vec(py, out).reshape([h, w])?.into_any())
}

/// film_optics.halation_layer_gate on (..., 3) float32.
#[pyfunction]
fn halation_layer_gate<'py>(py: Python<'py>, e_lin: &Bound<'py, PyAny>, e_ref: Vec<f32>, gate_ev: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let e = as_f32_array(py, e_lin)?;
    let shape = e.shape().to_vec();
    if shape.last() != Some(&3) {
        return Err(PyValueError::new_err("e_lin must be (..., 3)"));
    }
    let g = as_f32_array(py, gate_ev)?;
    if g.shape() != [3, 2] {
        return Err(PyValueError::new_err("gate_ev must be (3, 2)"));
    }
    let gro = g.readonly();
    let gs = gro.as_slice()?;
    let gate = [[gs[0], gs[1]], [gs[2], gs[3]], [gs[4], gs[5]]];
    let r = vec3_f32(e_ref, "e_ref")?;
    let ro = e.readonly();
    let d = ro.as_slice()?;
    let out = py.detach(|| {
        let mut out = vec![0.0f32; d.len()];
        for (px, o) in d.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
            let v = spatial::halation_layer_gate_px([px[0], px[1], px[2]], r, &gate);
            o.copy_from_slice(&v);
        }
        out
    });
    Ok(PyArray1::from_vec(py, out).reshape(shape)?.into_any())
}

/// film_optics.halation_pointwise_return on (..., 3) float32.
#[pyfunction]
fn halation_pointwise_return<'py>(py: Python<'py>, e_lin: &Bound<'py, PyAny>, e_ref: Vec<f32>, comps: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let e = as_f32_array(py, e_lin)?;
    let shape = e.shape().to_vec();
    if shape.last() != Some(&3) {
        return Err(PyValueError::new_err("e_lin must be (..., 3)"));
    }
    let comps = halation_comps_from_py(py, comps)?;
    let r = vec3_f32(e_ref, "e_ref")?;
    let ro = e.readonly();
    let d = ro.as_slice()?;
    let out = py.detach(|| {
        let mut out = vec![0.0f32; d.len()];
        for (px, o) in d.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
            o.copy_from_slice(&spatial::halation_pointwise_return_px([px[0], px[1], px[2]], r, &comps));
        }
        out
    });
    Ok(PyArray1::from_vec(py, out).reshape(shape)?.into_any())
}

/// film_optics.halation_component_source on (..., 3) float32 for one component.
#[pyfunction]
fn halation_component_source<'py>(py: Python<'py>, e_lin: &Bound<'py, PyAny>, e_ref: Vec<f32>, comp: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let e = as_f32_array(py, e_lin)?;
    let shape = e.shape().to_vec();
    if shape.last() != Some(&3) {
        return Err(PyValueError::new_err("e_lin must be (..., 3)"));
    }
    let list = pyo3::types::PyList::new(py, [comp])?;
    let comps = halation_comps_from_py(py, list.as_any())?;
    let r = vec3_f32(e_ref, "e_ref")?;
    let ro = e.readonly();
    let d = ro.as_slice()?;
    let out = py.detach(|| {
        let mut out = vec![0.0f32; d.len()];
        for (px, o) in d.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
            o.copy_from_slice(&spatial::halation_component_source_px([px[0], px[1], px[2]], r, &comps[0]));
        }
        out
    });
    Ok(PyArray1::from_vec(py, out).reshape(shape)?.into_any())
}

/// film_optics.capture_bloom_gate on any float32 array.
#[pyfunction]
fn capture_bloom_gate<'py>(py: Python<'py>, y: &Bound<'py, PyAny>, t0: f64, t1: f64) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f32_array(py, y)?;
    let shape = a.shape().to_vec();
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let out = py.detach(|| d.iter().map(|&v| spatial::capture_bloom_gate_px(v, t0, t1)).collect::<Vec<f32>>());
    Ok(PyArray1::from_vec(py, out).reshape(shape)?.into_any())
}

/// film_optics.capture_bloom_source_rows on (..., 3) float32.
#[pyfunction]
fn capture_bloom_source_rows<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, t0: f64, t1: f64) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f32_array(py, rgb)?;
    let shape = a.shape().to_vec();
    if shape.last() != Some(&3) {
        return Err(PyValueError::new_err("rgb must be (..., 3)"));
    }
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let out = py.detach(|| {
        let mut out = vec![0.0f32; d.len()];
        for (px, o) in d.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
            o.copy_from_slice(&spatial::capture_bloom_source_px([px[0], px[1], px[2]], t0, t1));
        }
        out
    });
    Ok(PyArray1::from_vec(py, out).reshape(shape)?.into_any())
}

/// film_optics.capture_bloom_apply_rows -> (n*width, 3) float32.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn capture_bloom_apply_rows<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    glow_map: &Bound<'py, PyAny>,
    y0: usize,
    y1: usize,
    height: usize,
    width: usize,
    t0: f64,
    t1: f64,
    core_ratio: Vec<f64>,
    save_lights: f32,
    saturation: f32,
    amount: f32,
) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f32_array(py, rgb)?;
    let n = y1 - y0;
    if a.len() != n * width * 3 {
        return Err(PyValueError::new_err("rgb must hold (y1 - y0) * width pixels"));
    }
    let m = as_f32_array(py, glow_map)?;
    let ms = m.shape().to_vec();
    if ms.len() != 3 || ms[2] != 3 {
        return Err(PyValueError::new_err("glow_map must be (dh, dw, 3)"));
    }
    if core_ratio.len() != 2 {
        return Err(PyValueError::new_err("core_ratio must have 2 elements"));
    }
    let cr = [core_ratio[0], core_ratio[1]];
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let mro = m.readonly();
    let md = mro.as_slice()?;
    let out = py.detach(|| {
        let glow = spatial::upsample_rows(md, ms[0], ms[1], 3, y0, y1, height, width);
        let mut out = vec![0.0f32; d.len()];
        for ((px, g), o) in d.chunks_exact(3).zip(glow.chunks_exact(3)).zip(out.chunks_exact_mut(3)) {
            o.copy_from_slice(&spatial::capture_bloom_apply_px(
                [px[0], px[1], px[2]], [g[0], g[1], g[2]], t0, t1, cr, save_lights, saturation, amount,
            ));
        }
        out
    });
    Ok(PyArray1::from_vec(py, out).reshape([n * width, 3])?.into_any())
}

/// film_optics.apply_scatter_mix on (h, w, 3) float32; `chans` is
/// [(s_mix, [(sigma_px, weight), ...]) x 3] as _scatter_components resolves.
#[pyfunction]
fn apply_scatter_mix<'py>(py: Python<'py>, img: &Bound<'py, PyAny>, chans: Vec<(f64, Vec<(f64, f64)>)>) -> PyResult<Bound<'py, PyAny>> {
    let a = as_view_array::<f32>(py, img, "float32")?;
    let shape = a.shape().to_vec();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("img must be (h, w, 3)"));
    }
    if chans.len() != 3 {
        return Err(PyValueError::new_err("chans must have 3 entries"));
    }
    let (h, w) = (shape[0], shape[1]);
    let sc: [spatial::ScatterChannel; 3] = [
        spatial::ScatterChannel { s_mix: chans[0].0, comps: chans[0].1.clone() },
        spatial::ScatterChannel { s_mix: chans[1].0, comps: chans[1].1.clone() },
        spatial::ScatterChannel { s_mix: chans[2].0, comps: chans[2].1.clone() },
    ];
    let ro = a.readonly();
    let view = ro.as_array().into_dimensionality::<numpy::ndarray::Ix3>().unwrap();
    let out = py.detach(|| spatial::apply_scatter_mix(view, &sc));
    Ok(PyArray1::from_vec(py, out).reshape([h, w, 3])?.into_any())
}

/// film_optics.sample_field on the cached master integral image (gh+1, gw+1, c)
/// float32 (landscape store; `rotated` samples it transposed).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_field<'py>(
    py: Python<'py>,
    ii: &Bound<'py, PyAny>,
    rotated: bool,
    height: usize,
    width: usize,
    x0: f64,
    y0: f64,
    w_mm: f64,
    h_mm: f64,
    gate_w_mm: f64,
    gate_h_mm: f64,
    phase: (i64, i64),
) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f32_array(py, ii)?;
    let shape = a.shape().to_vec();
    if shape.len() != 3 {
        return Err(PyValueError::new_err("ii must be (gh+1, gw+1, c)"));
    }
    let (store_h, store_w, c) = (shape[0] - 1, shape[1] - 1, shape[2]);
    let (gh, gw) = if rotated { (store_w, store_h) } else { (store_h, store_w) };
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let out = py.detach(|| {
        let img = spatial::IntegralImage { data: d, gh, gw, c, rotated };
        let edges = spatial::field_edges(height, width, x0, y0, w_mm, h_mm, gate_w_mm, gate_h_mm, gh, gw, phase);
        spatial::sample_field(&img, &edges, height, width)
    });
    Ok(PyArray1::from_vec(py, out).reshape([height, width, c])?.into_any())
}

/// film_optics.apply_density_grain (band_limited_gaussian_v1) -> (n, 3) float64.
#[pyfunction]
fn density_grain_v1<'py>(py: Python<'py>, amounts: &Bound<'py, PyAny>, lo: Vec<f64>, hi: Vec<f64>, field: &Bound<'py, PyAny>, sigma_mul: f64) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f64_array(py, amounts)?;
    let f = as_f32_array(py, field)?;
    if a.len() != f.len() || a.len() % 3 != 0 || lo.len() != 3 || hi.len() != 3 {
        return Err(PyValueError::new_err("amounts/field must be (n, 3), lo/hi 3-vectors"));
    }
    let aro = a.readonly();
    let ad = aro.as_slice()?;
    let fro = f.readonly();
    let fd = fro.as_slice()?;
    let n = ad.len() / 3;
    let out = py.detach(|| spatial::density_grain_v1(ad, [lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]], fd, sigma_mul));
    Ok(PyArray1::from_vec(py, out).reshape([n, 3])?.into_any())
}

/// film_optics.apply_density_grain (measured_sigma_v2) -> (n, 3) float64.
/// `tables` = [(chart_base, density_axis, sigma) x 3].
#[pyfunction]
fn density_grain_v2<'py>(py: Python<'py>, amounts: &Bound<'py, PyAny>, field: &Bound<'py, PyAny>, tables: Vec<(f32, Vec<f64>, Vec<f64>)>, amount_over_rms: f32) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f64_array(py, amounts)?;
    let f = as_f32_array(py, field)?;
    if a.len() != f.len() || a.len() % 3 != 0 || tables.len() != 3 {
        return Err(PyValueError::new_err("amounts/field must be (n, 3), tables x3"));
    }
    let tabs: [spatial::SigmaTable; 3] = [
        spatial::SigmaTable { base: tables[0].0, d: tables[0].1.clone(), sigma: tables[0].2.clone() },
        spatial::SigmaTable { base: tables[1].0, d: tables[1].1.clone(), sigma: tables[1].2.clone() },
        spatial::SigmaTable { base: tables[2].0, d: tables[2].1.clone(), sigma: tables[2].2.clone() },
    ];
    let aro = a.readonly();
    let ad = aro.as_slice()?;
    let fro = f.readonly();
    let fd = fro.as_slice()?;
    let n = ad.len() / 3;
    let out = py.detach(|| spatial::density_grain_v2(ad, fd, &tabs, amount_over_rms));
    Ok(PyArray1::from_vec(py, out).reshape([n, 3])?.into_any())
}

/// film_optics.halation_reinject_rows -> (n*width, 3) float64.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn halation_reinject_rows<'py>(
    py: Python<'py>,
    log_e: &Bound<'py, PyAny>,
    spread_map: &Bound<'py, PyAny>,
    e_ref: Vec<f32>,
    y0: usize,
    y1: usize,
    height: usize,
    width: usize,
    comps: &Bound<'py, PyAny>,
    residual: bool,
    amount: f32,
    give_lin: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let l = as_f64_array(py, log_e)?;
    let n = (y1 - y0) * width;
    if l.len() != n * 3 {
        return Err(PyValueError::new_err("log_e must hold (y1 - y0) * width pixels"));
    }
    let m = as_f32_array(py, spread_map)?;
    let ms = m.shape().to_vec();
    if ms.len() != 3 || ms[2] != 3 {
        return Err(PyValueError::new_err("spread_map must be (dh, dw, 3)"));
    }
    let comps = halation_comps_from_py(py, comps)?;
    let r = vec3_f32(e_ref, "e_ref")?;
    let give = if give_lin.is_none() {
        None
    } else {
        let g = as_f32_array(py, give_lin)?;
        if g.len() != n * 3 {
            return Err(PyValueError::new_err("give_lin must match log_e"));
        }
        Some(g)
    };
    let lro = l.readonly();
    let ld = lro.as_slice()?;
    let mro = m.readonly();
    let md = mro.as_slice()?;
    let gro = give.as_ref().map(|g| g.readonly());
    let gd = match gro.as_ref() {
        Some(g) => Some(g.as_slice()?),
        None => None,
    };
    let out = py.detach(|| {
        let up = spatial::upsample_rows(md, ms[0], ms[1], 3, y0, y1, height, width);
        spatial::halation_reinject(ld, gd, &up, r, &comps, residual, amount)
    });
    Ok(PyArray1::from_vec(py, out).reshape([n, 3])?.into_any())
}

// ---------------------------------------------------------------------------
// Stage 3 (2026-09-15): film v2 core per-pixel chain (film_v2_math.py, film_develop.py).

fn mat3_f64(v: Vec<f64>, what: &str) -> PyResult<[[f64; 3]; 3]> {
    if v.len() != 9 {
        return Err(PyValueError::new_err(format!("{what} must have 9 elements")));
    }
    Ok([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
}

fn vec3_f64(v: Vec<f64>, what: &str) -> PyResult<[f64; 3]> {
    v.as_slice().try_into().map_err(|_| PyValueError::new_err(format!("{what} must have 3 elements")))
}

fn rows3_f64<'py>(py: Python<'py>, data: Vec<f64>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_vec(py, data).reshape([n, 3])?.into_any())
}

/// film_v2_math.layer_log_exposure -> (n, 3) float64.
#[pyfunction]
fn layer_log_exposure<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, observer: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    let obs = mat3_f64(observer, "observer")?;
    if let Ok(a) = rgb.cast::<PyArrayDyn<f64>>() {
        let n = require_rgb_f64(a, "rgb")?;
        let ro = a.readonly();
        let d = ro.as_slice()?;
        let mut out = vec![0.0f64; n * 3];
        py.detach(|| film_core::par_map3(d, &mut out, |i, o| film_core::layer_log_exposure(i, &obs, o)));
        return rows3_f64(py, out, n);
    }
    let a = as_f32_array(py, rgb)?;
    let n = require_rgb(&a, "rgb")?;
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let mut out = vec![0.0f64; n * 3];
    py.detach(|| film_core::par_map3(d, &mut out, |i, o| film_core::layer_log_exposure(i, &obs, o)));
    rows3_f64(py, out, n)
}

/// film_v2_math.chroma_field_log_exposure -> (n, 3) float64.
#[pyfunction]
fn chroma_field_log_exposure<'py>(
    py: Python<'py>,
    rgb: &Bound<'py, PyAny>,
    delta_lut: &Bound<'py, PyAny>,
    domain: Vec<f64>,
    xyz_from_rec2020: Vec<f64>,
    observer: Vec<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let lut = as_f64_array(py, delta_lut)?;
    let ls = lut.shape().to_vec();
    if ls.len() != 3 || ls[0] != ls[1] || ls[0] < 2 || ls[2] != 3 {
        return Err(PyValueError::new_err("delta_lut must be (n, n, 3)"));
    }
    if domain.len() != 4 {
        return Err(PyValueError::new_err("domain must have 4 elements"));
    }
    let lro = lut.readonly();
    let ld = lro.as_slice()?;
    let field = film_core::ChromaField {
        table: ld,
        n: ls[0],
        domain: [domain[0], domain[1], domain[2], domain[3]],
        xyz_from_rec2020: mat3_f64(xyz_from_rec2020, "xyz_from_rec2020")?,
        observer: mat3_f64(observer, "observer")?,
    };
    if let Ok(a64) = rgb.cast::<PyArrayDyn<f64>>() {
        let n = require_rgb_f64(a64, "rgb")?;
        let ro = a64.readonly();
        let d = ro.as_slice()?;
        let mut out = vec![0.0f64; n * 3];
        py.detach(|| film_core::par_map3(d, &mut out, |i, o| film_core::chroma_field_log_exposure(i, &field, o)));
        return rows3_f64(py, out, n);
    }
    let a = as_f32_array(py, rgb)?;
    let n = require_rgb(&a, "rgb")?;
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let mut out = vec![0.0f64; n * 3];
    py.detach(|| film_core::par_map3(d, &mut out, |i, o| film_core::chroma_field_log_exposure(i, &field, o)));
    rows3_f64(py, out, n)
}

/// film_v2_math.characteristic_amounts -> (n, 3) float64.
#[pyfunction]
fn characteristic_amounts<'py>(py: Python<'py>, log_e: &Bound<'py, PyAny>, le_axis: Vec<f64>, table: &Bound<'py, PyAny>, ev_offset: f64) -> PyResult<Bound<'py, PyAny>> {
    let l = as_f64_array(py, log_e)?;
    let n = require_rgb_f64(&l, "log_e")?;
    let t = as_f64_array(py, table)?;
    if t.shape() != [le_axis.len(), 3] {
        return Err(PyValueError::new_err("amounts_table must be (K, 3)"));
    }
    let lro = l.readonly();
    let ld = lro.as_slice()?;
    let tro = t.readonly();
    let td = tro.as_slice()?;
    let mut out = vec![0.0f64; n * 3];
    py.detach(|| film_core::par_map3(ld, &mut out, |i, o| film_core::characteristic_amounts(i, &le_axis, td, ev_offset, o)));
    rows3_f64(py, out, n)
}

fn require_rgb_f64(arr: &Bound<'_, PyArrayDyn<f64>>, name: &str) -> PyResult<usize> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!("{name} must be (N, 3)")));
    }
    Ok(shape[0])
}

/// film_develop inter-image amplification -> (n, 3) float64 (new array).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn interimage_amplify<'py>(
    py: Python<'py>,
    amounts: &Bound<'py, PyAny>,
    log_e: &Bound<'py, PyAny>,
    le_axis: Vec<f64>,
    table: &Bound<'py, PyAny>,
    rail_lo: Vec<f64>,
    rail_hi: Vec<f64>,
    beta: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f64_array(py, amounts)?;
    let n = require_rgb_f64(&a, "amounts")?;
    let l = as_f64_array(py, log_e)?;
    if require_rgb_f64(&l, "log_e")? != n {
        return Err(PyValueError::new_err("log_e must match amounts"));
    }
    let t = as_f64_array(py, table)?;
    if t.shape() != [le_axis.len(), 3] {
        return Err(PyValueError::new_err("table must be (K, 3)"));
    }
    let lo = vec3_f64(rail_lo, "rail_lo")?;
    let hi = vec3_f64(rail_hi, "rail_hi")?;
    let mut out = a.readonly().as_slice()?.to_vec();
    let lro = l.readonly();
    let ld = lro.as_slice()?;
    let tro = t.readonly();
    let td = tro.as_slice()?;
    py.detach(|| film_core::par_map3(ld, &mut out, |i, o| film_core::interimage_amplify(o, i, &le_axis, td, lo, hi, beta)));
    rows3_f64(py, out, n)
}

/// film_develop._tetrahedral on a (n, n, n, 3) float32 LUT at (m, 3) float32 coordinates.
#[pyfunction]
fn tetrahedral<'py>(py: Python<'py>, lut: &Bound<'py, PyAny>, u: &Bound<'py, PyAny>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    let l = as_f32_array(py, lut)?;
    if l.shape() != [n, n, n, 3] {
        return Err(PyValueError::new_err("lut must be (n, n, n, 3)"));
    }
    let uu = as_f32_array(py, u)?;
    let m = require_rgb(&uu, "u")?;
    let lro = l.readonly();
    let ld = lro.as_slice()?;
    let uro = uu.readonly();
    let ud = uro.as_slice()?;
    let mut out = vec![0.0f32; m * 3];
    py.detach(|| film_core::par_map3(ud, &mut out, |i, o| film_core::tetrahedral(ld, n, i, o)));
    rows3_f32(py, out, m)
}

/// film_v2_math.film_compression_ev -> (n, 3) float64.
#[pyfunction]
fn film_compression_ev<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, impact: f64, knee_ev: f64, width_ev: f64, rho: f64) -> PyResult<Bound<'py, PyAny>> {
    let a = as_f64_array(py, rgb)?;
    let n = require_rgb_f64(&a, "rgb")?;
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let mut out = vec![0.0f64; n * 3];
    py.detach(|| film_core::par_map3(d, &mut out, |i, o| film_core::film_compression_ev(i, impact, knee_ev, width_ev, rho, o)));
    rows3_f64(py, out, n)
}

/// developed[:, c] /= np.interp(ev_y, cast_ev, cast[:, c]) with
/// ev_y = log2(max(rgb @ luma, eps) / 0.18) + offset (float32 rgb) -> new (n, 3) float32.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cast_divide<'py>(
    py: Python<'py>,
    developed: &Bound<'py, PyAny>,
    rgb: &Bound<'py, PyAny>,
    eps: f32,
    offset: f32,
    cast_ev: Vec<f64>,
    cast: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dv = as_f32_array(py, developed)?;
    let n = require_rgb(&dv, "developed")?;
    let a = as_f32_array(py, rgb)?;
    if require_rgb(&a, "rgb")? != n {
        return Err(PyValueError::new_err("rgb must match developed"));
    }
    let c = as_f64_array(py, cast)?;
    if c.shape() != [cast_ev.len(), 3] {
        return Err(PyValueError::new_err("cast must be (K, 3)"));
    }
    let mut out = dv.readonly().as_slice()?.to_vec();
    let aro = a.readonly();
    let ad = aro.as_slice()?;
    let cro = c.readonly();
    let cd = cro.as_slice()?;
    py.detach(|| {
        film_core::par_map3(ad, &mut out, |i, o| {
            let mut ev = vec![0.0f32; i.len() / 3];
            film_core::scene_ev_luma_f32(i, eps, offset, &mut ev);
            film_core::cast_divide_per_pixel(o, &ev, &cast_ev, cd);
        });
    });
    rows3_f32(py, out, n)
}

// ---------------------------------------------------------------------------
// color.apply_rgb_matrix3: float64 products, (a + b) + c, one round to float32.

fn matrix3_rows<T: Copy + Into<f64> + Sync>(d: &[T], m: &[f64; 9], out: &mut [f32]) {
    film_core::par_map3(d, out, |i, o| {
        for (px, q) in i.chunks_exact(3).zip(o.chunks_exact_mut(3)) {
            let (r, g, b): (f64, f64, f64) = (px[0].into(), px[1].into(), px[2].into());
            q[0] = ((m[0] * r + m[1] * g) + m[2] * b) as f32;
            q[1] = ((m[3] * r + m[4] * g) + m[5] * b) as f32;
            q[2] = ((m[6] * r + m[7] * g) + m[8] * b) as f32;
        }
    });
}

#[pyfunction]
fn apply_rgb_matrix3<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, matrix: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    let m: [f64; 9] = matrix.as_slice().try_into().map_err(|_| PyValueError::new_err("matrix must have 9 elements"))?;
    if let Ok(a) = rgb.cast::<PyArrayDyn<f64>>() {
        let n = require_rgb_f64(a, "rgb")?;
        let ro = a.readonly();
        let d = ro.as_slice()?;
        let mut out = vec![0.0f32; n * 3];
        py.detach(|| matrix3_rows(d, &m, &mut out));
        return rows3_f32(py, out, n);
    }
    let a = as_f32_array(py, rgb)?;
    let n = require_rgb(&a, "rgb")?;
    let ro = a.readonly();
    let d = ro.as_slice()?;
    let mut out = vec![0.0f32; n * 3];
    py.detach(|| matrix3_rows(d, &m, &mut out));
    rows3_f32(py, out, n)
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
    m.add_function(wrap_pyfunction!(sensor_rgb_clip_counts_u16, m)?)?;
    m.add_function(wrap_pyfunction!(feather_masks_f16, m)?)?;
    m.add_function(wrap_pyfunction!(crop_loss_footprint, m)?)?;
    m.add_function(wrap_pyfunction!(merge_processing_loss_f16_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gain_map_mosaic, m)?)?;
    m.add_function(wrap_pyfunction!(warp_dng, m)?)?;
    m.add_function(wrap_pyfunction!(gamut_counts, m)?)?;
    m.add_function(wrap_pyfunction!(hdr_roundtrip_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(base_roundtrip_metrics, m)?)?;
    for f in [
        wrap_pyfunction!(area_decimate_rows, m)?, wrap_pyfunction!(upsample_rows, m)?,
        wrap_pyfunction!(gaussian_blur_slabbed, m)?, wrap_pyfunction!(blur_small_sigma, m)?,
        wrap_pyfunction!(halation_layer_gate, m)?, wrap_pyfunction!(halation_pointwise_return, m)?,
        wrap_pyfunction!(halation_component_source, m)?, wrap_pyfunction!(capture_bloom_gate, m)?,
        wrap_pyfunction!(capture_bloom_source_rows, m)?, wrap_pyfunction!(capture_bloom_apply_rows, m)?,
        wrap_pyfunction!(apply_scatter_mix, m)?, wrap_pyfunction!(sample_field, m)?,
        wrap_pyfunction!(density_grain_v1, m)?, wrap_pyfunction!(density_grain_v2, m)?,
        wrap_pyfunction!(halation_reinject_rows, m)?,
    ] {
        m.add_function(f)?;
    }
    for f in [
        wrap_pyfunction!(layer_log_exposure, m)?, wrap_pyfunction!(chroma_field_log_exposure, m)?,
        wrap_pyfunction!(characteristic_amounts, m)?, wrap_pyfunction!(interimage_amplify, m)?,
        wrap_pyfunction!(tetrahedral, m)?, wrap_pyfunction!(film_compression_ev, m)?,
        wrap_pyfunction!(cast_divide, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add_function(wrap_pyfunction!(apply_rgb_matrix3, m)?)?;
    Ok(())
}
