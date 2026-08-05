// SPDX-License-Identifier: GPL-3.0-or-later
//
// Scalar C++ port of dngscan's HDR AgX formation chunk. Behavioral reference:
//   dngscan/hdr_agx.py (_form_hdr_chunk), dngscan/hdr_curve.py (HdrCurveTable.apply),
//   dngscan/hdr_color.py (raw_gated_channel_separation, blend_native_hdr_paths,
//   fit_hdr_color_volume), dngscan/agx.py (prepare/finish formation), dngscan/punch.py.
// Each step replicates the NumPy float32 operation order; the film channel-ratio
// gain path is excluded at dispatch time instead of being reimplemented here.

#include "dngscan_fast/hdr_core.h"
#include "dngscan_fast/pixel_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace dngscan_fast {
namespace {

// hdr_curve._EPS: floor before the scene-EV log.
constexpr float TABLE_EPS = 1e-12f;
// hdr_color._EPS: luminance guard in the blend and the colour-volume fit.
constexpr float BLEND_EPS = 1e-6f;

inline float table_lookup(const HdrCurveTableView& table, float ev) {
  float u = (ev - table.ev_start) * table.inv_step;
  if (std::isnan(u)) {
    // np.clip keeps NaN and the NumPy interpolation then returns NaN as well.
    return u;
  }
  u = clampf(u, 0.0f, static_cast<float>(table.size - 1));
  int idx = static_cast<int>(u);
  idx = std::min(idx, table.size - 2);
  const float frac = u - static_cast<float>(idx);
  const float lo = table.values[idx];
  const float hi = table.values[idx + 1];
  return lo + (hi - lo) * frac;
}

inline float scene_ev(float value) {
  // np.log2(np.maximum(rgb, _EPS) / SCENE_MIDGRAY); std::max(value, eps) keeps NaN
  // in the first argument exactly like np.maximum does.
  return std::log2(std::max(value, TABLE_EPS) / PIXEL_MID_GRAY);
}

inline Rgb apply_table(const HdrCurveTableView& table, const Rgb& inset) {
  return {
      table_lookup(table, scene_ev(inset.r)),
      table_lookup(table, scene_ev(inset.g)),
      table_lookup(table, scene_ev(inset.b)),
  };
}

inline void gated_rho(const NativeHdrPlan& plan, const float* mask, float rho[3]) {
  const float base = clampf(plan.global_rho, 0.0f, 1.0f);
  if (mask == nullptr) {
    rho[0] = rho[1] = rho[2] = base;
    return;
  }
  const float m0 = clampf(mask[0], 0.0f, 1.0f);
  const float m1 = clampf(mask[1], 0.0f, 1.0f);
  const float m2 = clampf(mask[2], 0.0f, 1.0f);
  // np.partition(masks, 1)[..., 1]: the median of three, i.e. the second-largest.
  const float second =
      std::max(std::min(m0, m1), std::min(std::max(m0, m1), m2));
  const float multi_permission = 1.0f - second;
  rho[0] = base * (1.0f - 0.5f * m0) * multi_permission;
  rho[1] = base * (1.0f - 0.5f * m1) * multi_permission;
  rho[2] = base * (1.0f - 0.5f * m2) * multi_permission;
}

inline Rgb blend_native_paths(
    const Rgb& reference,
    const Rgb& native,
    const float rho[3],
    const float w[3]) {
  const float y_native = dot3(w, native);
  const float y_reference = dot3(w, reference);
  const float common_scale = y_native / std::max(y_reference, BLEND_EPS);
  Rgb common{
      reference.r * common_scale,
      reference.g * common_scale,
      reference.b * common_scale,
  };
  if (!(y_reference > BLEND_EPS)) {
    common = native;
  }
  const float r0 = clampf(rho[0], 0.0f, 1.0f);
  const float r1 = clampf(rho[1], 0.0f, 1.0f);
  const float r2 = clampf(rho[2], 0.0f, 1.0f);
  if (r0 <= 0.0f && r1 <= 0.0f && r2 <= 0.0f) {
    return common;
  }
  if (r0 >= 1.0f && r1 >= 1.0f && r2 >= 1.0f) {
    return native;
  }
  const Rgb proposal{
      (1.0f - r0) * common.r + r0 * native.r,
      (1.0f - r1) * common.g + r1 * native.g,
      (1.0f - r2) * common.b + r2 * native.b,
  };
  const float y_proposal = dot3(w, proposal);
  const float scale = y_native / std::max(y_proposal, BLEND_EPS);
  if (y_native > BLEND_EPS && y_proposal > BLEND_EPS) {
    return {proposal.r * scale, proposal.g * scale, proposal.b * scale};
  }
  return native;
}

inline Rgb fit_hdr_pixel(const Rgb& in, float limit, const float w[3]) {
  const bool needs_fit =
      in.r < 0.0f || in.r > limit ||
      in.g < 0.0f || in.g > limit ||
      in.b < 0.0f || in.b > limit;
  if (!needs_fit) {
    return in;
  }
  const float y_raw = dot3(w, in);
  const float y = clampf(y_raw, 0.0f, limit);
  const float c[3] = {in.r - y_raw, in.g - y_raw, in.b - y_raw};
  float lam = std::numeric_limits<float>::infinity();
  for (int i = 0; i < 3; ++i) {
    if (c[i] < 0.0f) {
      lam = std::min(lam, y / std::max(-c[i], BLEND_EPS));
    }
    if (c[i] > 0.0f) {
      lam = std::min(lam, (limit - y) / std::max(c[i], BLEND_EPS));
    }
  }
  lam = clampf(std::min(lam, 1.0f), 0.0f, 1.0f);
  Rgb fitted{
      y + lam * (in.r - y_raw),
      y + lam * (in.g - y_raw),
      y + lam * (in.b - y_raw),
  };
  fitted.r = clampf(fitted.r, 0.0f, limit);
  fitted.g = clampf(fitted.g, 0.0f, limit);
  fitted.b = clampf(fitted.b, 0.0f, limit);
  return fitted;
}

Rgb process_pixel(const Rgb& input, const float* mask, const NativeHdrPlan& plan) {
  Rgb in = input;
  if (std::isnan(in.r) || std::isnan(in.g) || std::isnan(in.b)) {
    // NumPy's compress_into_gamut poisons the whole pixel through its max/min
    // reductions; std::max/std::min only propagate a NaN in the first argument,
    // so replicate the poisoning explicitly.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    in = {nan, nan, nan};
  }
  const Rgb rgb = compress_into_gamut(in);
  const Rgb inset = mat3(plan.inset, rgb);

  const bool restore_hue = plan.hue_restore > 1e-6f;
  float pre_hue = 0.0f;
  if (restore_hue) {
    const Rgb inset_nonneg{
        std::max(inset.r, 0.0f),
        std::max(inset.g, 0.0f),
        std::max(inset.b, 0.0f),
    };
    pre_hue = rgb_to_hue(inset_nonneg);
  }

  const Rgb native = apply_table(plan.native_table, inset);
  Rgb formation = native;
  if (plan.has_reference) {
    const Rgb reference = apply_table(plan.reference_table, inset);
    float rho[3];
    gated_rho(plan, mask, rho);
    formation = blend_native_paths(reference, native, rho, plan.formation_luma);
  }

  if (restore_hue) {
    formation = mix_hue(formation, pre_hue, plan.hue_restore);
  }
  Rgb mapped = mat3(plan.outset, formation);

  const PunchMatrices punch{
      plan.rec2020_to_xyz,
      plan.xyz_to_rec2020,
      plan.oklab_m1,
      plan.oklab_m2,
      plan.oklab_m1_inv,
      plan.oklab_m2_inv,
  };
  mapped = apply_punch_rec2020_pixel(mapped, plan.punch_strength, punch);

  // rec2020_to_output as two chained float32 multiplies, then np.nan_to_num.
  const Rgb xyz = mat3(plan.rec2020_to_xyz, mapped);
  Rgb output_linear = mat3(plan.xyz_to_output, xyz);
  output_linear = {
      nan_to_num(output_linear.r, 0.0f, 1e6f, -1e6f),
      nan_to_num(output_linear.g, 0.0f, 1e6f, -1e6f),
      nan_to_num(output_linear.b, 0.0f, 1e6f, -1e6f),
  };
  return fit_hdr_pixel(output_linear, plan.peak, plan.output_luma);
}

}  // namespace

void apply_hdr_formation_f32(
    const float* input,
    const float* clip_masks,
    float* output,
    std::size_t pixel_count,
    const NativeHdrPlan& plan) {
  const auto process_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const Rgb in{input[i * 3 + 0], input[i * 3 + 1], input[i * 3 + 2]};
      const float* mask = clip_masks != nullptr ? clip_masks + i * 3 : nullptr;
      const Rgb out = process_pixel(in, mask, plan);
      output[i * 3 + 0] = out.r;
      output[i * 3 + 1] = out.g;
      output[i * 3 + 2] = out.b;
    }
  };

  // Same bounded fan-out as the SDR kernels: the caller may already run several
  // chunk workers, so the per-call parallelism stays capped at 8 threads.
  constexpr std::size_t kParallelThreshold = 128 * 1024;
  const unsigned available = std::max(1u, std::thread::hardware_concurrency());
  const unsigned worker_count =
      pixel_count >= kParallelThreshold ? std::min(available, 8u) : 1u;
  if (worker_count <= 1) {
    process_range(0, pixel_count);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(worker_count - 1);
  const std::size_t block = (pixel_count + worker_count - 1) / worker_count;
  for (unsigned worker = 1; worker < worker_count; ++worker) {
    const std::size_t begin = std::min(pixel_count, std::size_t(worker) * block);
    const std::size_t end = std::min(pixel_count, begin + block);
    workers.emplace_back(process_range, begin, end);
  }
  process_range(0, std::min(pixel_count, block));
  for (auto& worker : workers) {
    worker.join();
  }
}

}  // namespace dngscan_fast
