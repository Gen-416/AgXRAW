// SPDX-License-Identifier: GPL-3.0-or-later

#include "dngscan_fast/agx_core.h"
#include "thread_budget.h"
#include "dngscan_fast/hdr_core.h"
#include "dngscan_fast/output_core.h"
#include "dngscan_fast/film_appearance_core.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

float read_float(const py::object& obj, const char* name) {
  return obj.attr(name).cast<float>();
}

bool read_bool(const py::object& obj, const char* name) {
  return obj.attr(name).cast<bool>();
}

void copy_matrix9(const py::object& seq, float out[9]) {
  const py::sequence items = py::reinterpret_borrow<py::sequence>(seq);
  if (py::len(items) != 9) {
    throw std::invalid_argument("matrix must have 9 elements");
  }
  for (int i = 0; i < 9; ++i) {
    out[i] = py::cast<float>(items[i]);
  }
}

// R2 item 6: the exact two-stage output matrices keep full float64 precision
// so the kernel reproduces NumPy's stage arithmetic bit-for-bit.
void copy_matrix9_f64(const py::object& seq, double out[9]) {
  const py::sequence items = py::reinterpret_borrow<py::sequence>(seq);
  if (py::len(items) != 9) {
    throw std::invalid_argument("matrix must have 9 elements");
  }
  for (int i = 0; i < 9; ++i) {
    out[i] = py::cast<double>(items[i]);
  }
}

dngscan_fast::CurveParams curve_from_py(const py::object& obj) {
  dngscan_fast::CurveParams c{};
  c.black_ev = read_float(obj, "black_ev");
  c.range_ev = read_float(obj, "range_ev");
  c.gamma = read_float(obj, "gamma");
  c.target_black = read_float(obj, "target_black");
  c.target_white = read_float(obj, "target_white");
  c.toe_power = read_float(obj, "toe_power");
  c.toe_transition_x = read_float(obj, "toe_transition_x");
  c.toe_transition_y = read_float(obj, "toe_transition_y");
  c.toe_scale = read_float(obj, "toe_scale");
  c.need_convex_toe = read_bool(obj, "need_convex_toe");
  c.toe_fallback_power = read_float(obj, "toe_fallback_power");
  c.toe_fallback_coefficient = read_float(obj, "toe_fallback_coefficient");
  c.slope = read_float(obj, "slope");
  c.intercept = read_float(obj, "intercept");
  c.shoulder_power = read_float(obj, "shoulder_power");
  c.shoulder_transition_x = read_float(obj, "shoulder_transition_x");
  c.shoulder_transition_y = read_float(obj, "shoulder_transition_y");
  c.shoulder_scale = read_float(obj, "shoulder_scale");
  c.need_concave_shoulder = read_bool(obj, "need_concave_shoulder");
  c.shoulder_fallback_power = read_float(obj, "shoulder_fallback_power");
  c.shoulder_fallback_coefficient = read_float(obj, "shoulder_fallback_coefficient");
  return c;
}

dngscan_fast::NativeAgxPlan plan_from_py(const py::object& obj) {
  dngscan_fast::NativeAgxPlan plan{};
  copy_matrix9(obj.attr("inset"), plan.inset);
  copy_matrix9(obj.attr("outset"), plan.outset);
  plan.curve = curve_from_py(obj.attr("curve"));
  plan.hue_restore = read_float(obj, "hue_restore");
  plan.view_brightness = read_float(obj, "view_brightness");
  plan.punch_strength = read_float(obj, "punch_strength");
  copy_matrix9(obj.attr("rec2020_to_xyz"), plan.rec2020_to_xyz);
  copy_matrix9(obj.attr("xyz_to_rec2020"), plan.xyz_to_rec2020);
  copy_matrix9(obj.attr("oklab_m1"), plan.oklab_m1);
  copy_matrix9(obj.attr("oklab_m2"), plan.oklab_m2);
  copy_matrix9(obj.attr("oklab_m1_inv"), plan.oklab_m1_inv);
  copy_matrix9(obj.attr("oklab_m2_inv"), plan.oklab_m2_inv);
  return plan;
}

dngscan_fast::NativeOutputPlan output_plan_from_py(const py::object& obj) {
  dngscan_fast::NativeOutputPlan plan{};
  copy_matrix9_f64(obj.attr("rec2020_to_xyz"), plan.rec2020_to_xyz);
  copy_matrix9_f64(obj.attr("xyz_to_output"), plan.xyz_to_output);
  copy_matrix9(obj.attr("output_to_lms"), plan.output_to_lms);
  copy_matrix9(obj.attr("lms_to_output"), plan.lms_to_output);
  copy_matrix9(obj.attr("oklab_m2"), plan.oklab_m2);
  copy_matrix9(obj.attr("oklab_m2_inv"), plan.oklab_m2_inv);
  plan.alpha = read_float(obj, "alpha");
  return plan;
}

void copy_vector3(const py::object& seq, float out[3]) {
  const py::sequence items = py::reinterpret_borrow<py::sequence>(seq);
  if (py::len(items) != 3) {
    throw std::invalid_argument("vector must have 3 elements");
  }
  for (int i = 0; i < 3; ++i) {
    out[i] = py::cast<float>(items[i]);
  }
}

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

dngscan_fast::HdrCurveTableView table_from_py(
    const py::object& obj, FloatArray& holder) {
  holder = obj.attr("values").cast<FloatArray>();
  if (holder.ndim() != 1 || holder.shape(0) < 2) {
    throw std::invalid_argument("curve table values must be a 1-D array of >= 2 samples");
  }
  dngscan_fast::HdrCurveTableView view{};
  view.ev_start = obj.attr("ev_start").cast<float>();
  view.inv_step = obj.attr("inv_step").cast<float>();
  view.values = holder.data();
  view.size = static_cast<int>(holder.shape(0));
  return view;
}

// The returned plan borrows the two table arrays; native_holder/reference_holder
// must stay alive for the duration of the kernel call.
dngscan_fast::NativeHdrPlan hdr_plan_from_py(
    const py::object& obj, FloatArray& native_holder, FloatArray& reference_holder) {
  dngscan_fast::NativeHdrPlan plan{};
  copy_matrix9(obj.attr("inset"), plan.inset);
  copy_matrix9(obj.attr("outset"), plan.outset);
  copy_matrix9(obj.attr("rec2020_to_xyz"), plan.rec2020_to_xyz);
  copy_matrix9(obj.attr("xyz_to_rec2020"), plan.xyz_to_rec2020);
  copy_matrix9(obj.attr("xyz_to_output"), plan.xyz_to_output);
  // ABI v10: the output stage reads the exact float64 copies
  copy_matrix9_f64(obj.attr("rec2020_to_xyz"), plan.rec2020_to_xyz_f64);
  copy_matrix9_f64(obj.attr("xyz_to_output"), plan.xyz_to_output_f64);
  copy_matrix9(obj.attr("oklab_m1"), plan.oklab_m1);
  copy_matrix9(obj.attr("oklab_m2"), plan.oklab_m2);
  copy_matrix9(obj.attr("oklab_m1_inv"), plan.oklab_m1_inv);
  copy_matrix9(obj.attr("oklab_m2_inv"), plan.oklab_m2_inv);
  copy_vector3(obj.attr("formation_luma"), plan.formation_luma);
  copy_vector3(obj.attr("output_luma"), plan.output_luma);
  plan.hue_restore = read_float(obj, "hue_restore");
  plan.punch_strength = read_float(obj, "punch_strength");
  plan.global_rho = read_float(obj, "global_rho");
  plan.peak = read_float(obj, "peak");
  plan.has_reference = read_bool(obj, "has_reference");
  plan.native_table = table_from_py(obj.attr("native_table"), native_holder);
  if (plan.has_reference) {
    plan.reference_table = table_from_py(obj.attr("reference_table"), reference_holder);
  } else {
    plan.reference_table = plan.native_table;
  }
  return plan;
}

void require_rgb_array(const py::array& array, const char* name) {
  if (array.ndim() != 2 || array.shape(1) != 3) {
    throw std::invalid_argument(std::string(name) + " must be (N, 3)");
  }
}

void require_same_shape(
    const py::array& input,
    const py::array& other,
    const char* name) {
  require_rgb_array(other, name);
  if (other.shape(0) != input.shape(0)) {
    throw std::invalid_argument(std::string(name) + " must match rgb shape");
  }
}

}  // namespace

PYBIND11_MODULE(_dngscan_fast, m) {
  m.def("set_thread_budget", [](unsigned budget) {
    dngscan_fast::g_thread_budget.store(budget, std::memory_order_relaxed);
  }, "Per-process native thread budget; 0 = hardware concurrency (S3).");

  m.doc() = "dngscan optional C++ AgX core";

  m.def("native_abi_version", []() { return dngscan_fast::NATIVE_ABI_VERSION; });

  m.def(
      "film_appearance_apply_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
         py::array_t<float, py::array::c_style | py::array::forcecast> scene_ev,
         py::array_t<float, py::array::c_style | py::array::forcecast> f_hue,
         py::array_t<float, py::array::c_style | py::array::forcecast> d_hue,
         py::array_t<float, py::array::c_style | py::array::forcecast> f_chroma,
         py::array_t<float, py::array::c_style | py::array::forcecast> d_chroma,
         py::array_t<float, py::array::c_style | py::array::forcecast> f_density,
         py::array_t<float, py::array::c_style | py::array::forcecast> d_density,
         py::array_t<float, py::array::c_style | py::array::forcecast> ev_knots,
         py::array_t<float, py::array::c_style | py::array::forcecast> nb_ab,
         bool has_neutral_bias, float strength, float neutral_c0,
         float chroma_knee, float chroma_power, float richness_mult,
         float density_mult,
         py::array_t<float, py::array::c_style | py::array::forcecast> m_fwd,
         py::array_t<float, py::array::c_style | py::array::forcecast> m2,
         py::array_t<float, py::array::c_style | py::array::forcecast> m2_inv,
         py::array_t<float, py::array::c_style | py::array::forcecast> m_inv) {
        if (rgb.ndim() != 2 || rgb.shape(1) != 3) {
          throw std::invalid_argument("rgb must be (N, 3) float32");
        }
        const py::ssize_t n = rgb.shape(0);
        if (scene_ev.ndim() != 1 || scene_ev.shape(0) != n) {
          throw std::invalid_argument("scene_ev must be (N,)");
        }
        const int K = static_cast<int>(ev_knots.shape(0));
        const int H = static_cast<int>(f_hue.shape(1));
        for (const auto* arr : {&f_hue, &d_hue, &f_chroma, &d_chroma,
                                &f_density, &d_density}) {
          if (arr->ndim() != 2 || arr->shape(0) != K || arr->shape(1) != H) {
            throw std::invalid_argument("field tables must be (K, H)");
          }
        }
        if (nb_ab.ndim() != 2 || nb_ab.shape(0) != K || nb_ab.shape(1) != 2) {
          throw std::invalid_argument("nb_ab must be (K, 2)");
        }
        for (const auto* mat : {&m_fwd, &m2, &m2_inv, &m_inv}) {
          if (mat->size() != 9) {
            throw std::invalid_argument("matrices must have 9 elements");
          }
        }
        py::array_t<float> out({n, static_cast<py::ssize_t>(3)});
        dngscan_fast::FilmAppearanceParams p{};
        p.f_hue = f_hue.data(); p.d_hue = d_hue.data();
        p.f_chroma = f_chroma.data(); p.d_chroma = d_chroma.data();
        p.f_density = f_density.data(); p.d_density = d_density.data();
        p.ev_knots = ev_knots.data(); p.nb_ab = nb_ab.data();
        p.k_knots = K; p.h_knots = H;
        p.has_neutral_bias = has_neutral_bias;
        p.strength = strength; p.neutral_c0 = neutral_c0;
        p.chroma_knee = chroma_knee; p.chroma_power = chroma_power;
        p.richness_mult = richness_mult; p.density_mult = density_mult;
        p.m_fwd = m_fwd.data(); p.m2 = m2.data();
        p.m2_inv = m2_inv.data(); p.m_inv = m_inv.data();
        std::int64_t neg = 0;
        {
          py::gil_scoped_release release;
          neg = dngscan_fast::film_appearance_apply(
              rgb.data(), scene_ev.data(), out.mutable_data(), n, p);
        }
        return py::make_tuple(out, neg);
      },
      "Film appearance palette kernel (E3); returns (out, pre-clamp rows).");

  m.def(
      "apply_agx_core_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb, const py::object& plan_obj) {
        if (rgb.ndim() != 2 || rgb.shape(1) != 3) {
          throw std::invalid_argument("rgb must be (N, 3) float32");
        }
        const auto plan = plan_from_py(plan_obj);
        const py::ssize_t n = rgb.shape(0);
        auto out = py::array_t<float>({n, py::ssize_t(3)});
        py::gil_scoped_release release;
        dngscan_fast::apply_agx_core_f32(
            rgb.data(),
            out.mutable_data(),
            static_cast<std::size_t>(n),
            plan);
        return out;
      },
      py::arg("rgb"),
      py::arg("plan"));

  m.def(
      "apply_hdr_formation_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
         const py::object& masks_obj,
         const py::object& plan_obj) {
        require_rgb_array(rgb, "rgb");
        FloatArray masks;
        const float* masks_ptr = nullptr;
        if (!masks_obj.is_none()) {
          masks = masks_obj.cast<FloatArray>();
          require_same_shape(rgb, masks, "clip_masks");
          masks_ptr = masks.data();
        }
        FloatArray native_holder;
        FloatArray reference_holder;
        const auto plan = hdr_plan_from_py(plan_obj, native_holder, reference_holder);
        const py::ssize_t n = rgb.shape(0);
        auto out = py::array_t<float>({n, py::ssize_t(3)});
        py::gil_scoped_release release;
        dngscan_fast::apply_hdr_formation_f32(
            rgb.data(),
            masks_ptr,
            out.mutable_data(),
            static_cast<std::size_t>(n),
            plan);
        return out;
      },
      py::arg("rgb"),
      py::arg("clip_masks"),
      py::arg("plan"));

  m.def(
      "fit_output_gamut_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
         const py::object& plan_obj) {
        require_rgb_array(rgb, "rgb");
        const auto plan = output_plan_from_py(plan_obj);
        const py::ssize_t n = rgb.shape(0);
        auto out = py::array_t<float>({n, py::ssize_t(3)});
        py::gil_scoped_release release;
        dngscan_fast::fit_output_gamut_f32(
            rgb.data(), out.mutable_data(), static_cast<std::size_t>(n), plan);
        return out;
      },
      py::arg("rgb"),
      py::arg("plan"));

  m.def(
      "finalize_rec2020_u8_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
         py::array_t<float, py::array::c_style | py::array::forcecast> noise_a,
         py::array_t<float, py::array::c_style | py::array::forcecast> noise_b,
         const py::object& plan_obj) {
        require_rgb_array(rgb, "rgb");
        require_same_shape(rgb, noise_a, "noise_a");
        require_same_shape(rgb, noise_b, "noise_b");
        const auto plan = output_plan_from_py(plan_obj);
        const py::ssize_t n = rgb.shape(0);
        auto out = py::array_t<std::uint8_t>({n, py::ssize_t(3)});
        py::gil_scoped_release release;
        dngscan_fast::finalize_rec2020_u8_f32(
            rgb.data(),
            noise_a.data(),
            noise_b.data(),
            out.mutable_data(),
            static_cast<std::size_t>(n),
            plan);
        return out;
      },
      py::arg("rgb"),
      py::arg("noise_a"),
      py::arg("noise_b"),
      py::arg("plan"));

  m.def(
      "finalize_output_u8_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
         py::array_t<float, py::array::c_style | py::array::forcecast> noise_a,
         py::array_t<float, py::array::c_style | py::array::forcecast> noise_b,
         const py::object& plan_obj) {
        require_rgb_array(rgb, "rgb");
        require_same_shape(rgb, noise_a, "noise_a");
        require_same_shape(rgb, noise_b, "noise_b");
        const auto plan = output_plan_from_py(plan_obj);
        const py::ssize_t n = rgb.shape(0);
        auto out = py::array_t<std::uint8_t>({n, py::ssize_t(3)});
        py::gil_scoped_release release;
        dngscan_fast::finalize_output_u8_f32(
            rgb.data(),
            noise_a.data(),
            noise_b.data(),
            out.mutable_data(),
            static_cast<std::size_t>(n),
            plan);
        return out;
      },
      py::arg("rgb"),
      py::arg("noise_a"),
      py::arg("noise_b"),
      py::arg("plan"));

  m.def(
      "finalize_rec2020_u8_noise_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
         py::array_t<float, py::array::c_style | py::array::forcecast> noise,
         const py::object& plan_obj) {
        require_rgb_array(rgb, "rgb");
        require_same_shape(rgb, noise, "noise");
        const auto plan = output_plan_from_py(plan_obj);
        const py::ssize_t n = rgb.shape(0);
        auto out = py::array_t<std::uint8_t>({n, py::ssize_t(3)});
        py::gil_scoped_release release;
        dngscan_fast::finalize_rec2020_u8_noise_f32(
            rgb.data(), noise.data(), out.mutable_data(),
            static_cast<std::size_t>(n), plan);
        return out;
      },
      py::arg("rgb"),
      py::arg("noise"),
      py::arg("plan"));

  m.def(
      "finalize_output_u8_noise_f32",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
         py::array_t<float, py::array::c_style | py::array::forcecast> noise,
         const py::object& plan_obj) {
        require_rgb_array(rgb, "rgb");
        require_same_shape(rgb, noise, "noise");
        const auto plan = output_plan_from_py(plan_obj);
        const py::ssize_t n = rgb.shape(0);
        auto out = py::array_t<std::uint8_t>({n, py::ssize_t(3)});
        py::gil_scoped_release release;
        dngscan_fast::finalize_output_u8_noise_f32(
            rgb.data(), noise.data(), out.mutable_data(),
            static_cast<std::size_t>(n), plan);
        return out;
      },
      py::arg("rgb"),
      py::arg("noise"),
      py::arg("plan"));

  m.def(
      "self_test",
      []() {
        dngscan_fast::NativeAgxPlan plan{};
        plan.curve.black_ev = -10.0f;
        plan.curve.range_ev = 16.5f;
        plan.curve.gamma = 2.2f;
        plan.curve.target_white = 1.0f;
        plan.curve.slope = 0.1f;
        plan.curve.toe_transition_x = 0.1f;
        plan.curve.shoulder_transition_x = 0.9f;
        plan.curve.toe_transition_y = 0.01f;
        plan.curve.shoulder_transition_y = 0.99f;
        plan.curve.toe_power = 1.5f;
        plan.curve.shoulder_power = 3.3f;
        plan.curve.toe_scale = 1.0f;
        plan.curve.shoulder_scale = 1.0f;
        plan.inset[0] = plan.inset[4] = plan.inset[8] = 1.0f;
        plan.outset[0] = plan.outset[4] = plan.outset[8] = 1.0f;
        plan.rec2020_to_xyz[0] = 0.637f;
        plan.rec2020_to_xyz[4] = 1.0f;
        plan.rec2020_to_xyz[8] = 1.0f;
        plan.xyz_to_rec2020[0] = 1.7167f;
        plan.xyz_to_rec2020[4] = 1.6165f;
        plan.xyz_to_rec2020[8] = 0.9421f;
        plan.oklab_m1[0] = plan.oklab_m1[4] = plan.oklab_m1[8] = 1.0f;
        plan.oklab_m2[0] = plan.oklab_m2[4] = plan.oklab_m2[8] = 1.0f;
        plan.oklab_m1_inv[0] = plan.oklab_m1_inv[4] = plan.oklab_m1_inv[8] = 1.0f;
        plan.oklab_m2_inv[0] = plan.oklab_m2_inv[4] = plan.oklab_m2_inv[8] = 1.0f;
        float in[3] = {0.18f, 0.18f, 0.18f};
        float out[3] = {};
        dngscan_fast::apply_agx_core_f32(in, out, 1, plan);
        dngscan_fast::NativeOutputPlan output_plan{};
        output_plan.rec2020_to_xyz[0] = 1.0;
        output_plan.rec2020_to_xyz[4] = 1.0;
        output_plan.rec2020_to_xyz[8] = 1.0;
        output_plan.xyz_to_output[0] = 1.0;
        output_plan.xyz_to_output[4] = 1.0;
        output_plan.xyz_to_output[8] = 1.0;
        output_plan.output_to_lms[0] = 1.0f;
        output_plan.output_to_lms[4] = 1.0f;
        output_plan.output_to_lms[8] = 1.0f;
        output_plan.lms_to_output[0] = 1.0f;
        output_plan.lms_to_output[4] = 1.0f;
        output_plan.lms_to_output[8] = 1.0f;
        output_plan.oklab_m2[0] = 1.0f;
        output_plan.oklab_m2[4] = 1.0f;
        output_plan.oklab_m2[8] = 1.0f;
        output_plan.oklab_m2_inv[0] = 1.0f;
        output_plan.oklab_m2_inv[4] = 1.0f;
        output_plan.oklab_m2_inv[8] = 1.0f;
        output_plan.alpha = 0.05f;
        float noise[3] = {};
        std::uint8_t encoded[3] = {};
        dngscan_fast::finalize_output_u8_f32(
            in, noise, noise, encoded, 1, output_plan);

        dngscan_fast::NativeHdrPlan hdr_plan{};
        hdr_plan.inset[0] = hdr_plan.inset[4] = hdr_plan.inset[8] = 1.0f;
        hdr_plan.outset[0] = hdr_plan.outset[4] = hdr_plan.outset[8] = 1.0f;
        hdr_plan.rec2020_to_xyz[0] = hdr_plan.rec2020_to_xyz[4] =
            hdr_plan.rec2020_to_xyz[8] = 1.0f;
        hdr_plan.xyz_to_rec2020[0] = hdr_plan.xyz_to_rec2020[4] =
            hdr_plan.xyz_to_rec2020[8] = 1.0f;
        hdr_plan.xyz_to_output[0] = hdr_plan.xyz_to_output[4] =
            hdr_plan.xyz_to_output[8] = 1.0f;
        hdr_plan.rec2020_to_xyz_f64[0] = hdr_plan.rec2020_to_xyz_f64[4] =
            hdr_plan.rec2020_to_xyz_f64[8] = 1.0;
        hdr_plan.xyz_to_output_f64[0] = hdr_plan.xyz_to_output_f64[4] =
            hdr_plan.xyz_to_output_f64[8] = 1.0;
        hdr_plan.oklab_m1[0] = hdr_plan.oklab_m1[4] = hdr_plan.oklab_m1[8] = 1.0f;
        hdr_plan.oklab_m2[0] = hdr_plan.oklab_m2[4] = hdr_plan.oklab_m2[8] = 1.0f;
        hdr_plan.oklab_m1_inv[0] = hdr_plan.oklab_m1_inv[4] =
            hdr_plan.oklab_m1_inv[8] = 1.0f;
        hdr_plan.oklab_m2_inv[0] = hdr_plan.oklab_m2_inv[4] =
            hdr_plan.oklab_m2_inv[8] = 1.0f;
        hdr_plan.formation_luma[0] = hdr_plan.formation_luma[1] =
            hdr_plan.formation_luma[2] = 1.0f / 3.0f;
        hdr_plan.output_luma[0] = hdr_plan.output_luma[1] =
            hdr_plan.output_luma[2] = 1.0f / 3.0f;
        hdr_plan.hue_restore = 0.6f;
        hdr_plan.punch_strength = 0.0f;
        hdr_plan.global_rho = 0.0f;
        hdr_plan.peak = 4.0f;
        // 0.18 sits at EV 0 on a [-8, +5] grid mapping EV linearly to [0, 1].
        const float hdr_values[2] = {0.0f, 1.0f};
        hdr_plan.native_table = {-8.0f, 1.0f / 13.0f, hdr_values, 2};
        hdr_plan.reference_table = hdr_plan.native_table;
        hdr_plan.has_reference = false;
        float hdr_out[3] = {};
        dngscan_fast::apply_hdr_formation_f32(in, nullptr, hdr_out, 1, hdr_plan);

        return out[0] >= 0.0f && out[1] >= 0.0f && out[2] >= 0.0f &&
               encoded[0] > 0 && encoded[1] > 0 && encoded[2] > 0 &&
               hdr_out[0] > 0.0f && hdr_out[0] <= 4.0f &&
               hdr_out[1] == hdr_out[0] && hdr_out[2] == hdr_out[0];
      });
}
