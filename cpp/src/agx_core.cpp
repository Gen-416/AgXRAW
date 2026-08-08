// SPDX-License-Identifier: GPL-3.0-or-later
//
// Scalar C++ port of dngscan's darktable-derived AgX formation + C1 curve + hue
// restoration + scene-driven punch. Behavioral reference:
//   dngscan/agx.py, dngscan/drt.py, dngscan/punch.py
// Original curve/formation derives from darktable AgX (GPL-3.0-or-later).
// Shared per-pixel primitives live in dngscan_fast/pixel_math.h.

#include "dngscan_fast/agx_core.h"
#include "dngscan_fast/pixel_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include "thread_budget.h"
#include <vector>

namespace dngscan_fast {
namespace {

float sigmoid(float x, float power) {
  const float xp = std::pow(std::max(x, 0.0f), power);
  return x / std::pow(1.0f + xp, 1.0f / power);
}

float scaled_sigmoid(
    float x,
    float scale_value,
    float slope,
    float power,
    float transition_x,
    float transition_y) {
  if (std::abs(scale_value) < EPS) {
    return transition_y;
  }
  return scale_value * sigmoid(slope * (x - transition_x) / scale_value, power) + transition_y;
}

float apply_curve_c1(float x, const CurveParams& p) {
  float out;
  if (x < p.toe_transition_x) {
    if (p.need_convex_toe) {
      out = p.target_black +
            std::max(
                0.0f,
                p.toe_fallback_coefficient * std::pow(std::max(x, 0.0f), p.toe_fallback_power));
    } else {
      out = scaled_sigmoid(
          x, p.toe_scale, p.slope, p.toe_power, p.toe_transition_x, p.toe_transition_y);
    }
  } else if (x > p.shoulder_transition_x) {
    if (p.need_concave_shoulder) {
      out = p.target_white -
            std::max(
                0.0f,
                p.shoulder_fallback_coefficient *
                    std::pow(std::max(1.0f - x, 0.0f), p.shoulder_fallback_power));
    } else {
      out = scaled_sigmoid(
          x,
          p.shoulder_scale,
          p.slope,
          p.shoulder_power,
          p.shoulder_transition_x,
          p.shoulder_transition_y);
    }
  } else {
    out = p.slope * x + p.intercept;
  }
  return clampf(out, p.target_black, p.target_white);
}

Rgb apply_c1_endpoints_rgb(const Rgb& inset, const CurveParams& curve) {
  Rgb linear;
  const float channels[3] = {inset.r, inset.g, inset.b};
  float out_channels[3];
  for (int c = 0; c < 3; ++c) {
    const float ev = std::log2(std::max(channels[c] / PIXEL_MID_GRAY, EPS));
    float x = (ev - curve.black_ev) / curve.range_ev;
    x = clampf(x, 0.0f, 1.0f);
    const float encoded = apply_curve_c1(x, curve);
    out_channels[c] = std::pow(std::max(encoded, 0.0f), curve.gamma);
  }
  linear = {out_channels[0], out_channels[1], out_channels[2]};
  return linear;
}

PunchMatrices punch_matrices(const NativeAgxPlan& plan) {
  return {
      plan.rec2020_to_xyz,
      plan.xyz_to_rec2020,
      plan.oklab_m1,
      plan.oklab_m2,
      plan.oklab_m1_inv,
      plan.oklab_m2_inv,
  };
}

Rgb process_pixel(const Rgb& input, const NativeAgxPlan& plan) {
  Rgb rgb = compress_into_gamut(input);
  Rgb inset = mat3(plan.inset, rgb);

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

  Rgb linear = apply_c1_endpoints_rgb(inset, plan.curve);
  if (std::abs(plan.view_brightness - 1.0f) > 1e-6f) {
    const float brightness = std::max(plan.view_brightness, EPS);
    const float power = brightness < 1.0f ? 1.0f / std::sqrt(brightness) : 1.0f / brightness;
    linear.r = std::pow(std::max(linear.r, 0.0f), power);
    linear.g = std::pow(std::max(linear.g, 0.0f), power);
    linear.b = std::pow(std::max(linear.b, 0.0f), power);
  }

  if (restore_hue) {
    linear = mix_hue(linear, pre_hue, plan.hue_restore);
  }

  Rgb mapped = mat3(plan.outset, linear);
  return apply_punch_rec2020_pixel(mapped, plan.punch_strength, punch_matrices(plan));
}

}  // namespace

void apply_agx_core_f32(
    const float* input,
    float* output,
    std::size_t pixel_count,
    const NativeAgxPlan& plan) {
  const auto process_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const Rgb in{input[i * 3 + 0], input[i * 3 + 1], input[i * 3 + 2]};
      const Rgb out = process_pixel(in, plan);
      output[i * 3 + 0] = out.r;
      output[i * 3 + 1] = out.g;
      output[i * 3 + 2] = out.b;
    }
  };

  // The fixed realtime preview is large enough for the pow-heavy AgX core to benefit
  // from bounded row-independent parallelism, but small buffers should avoid thread
  // startup overhead. Cap workers so preview never monopolizes the machine.
  constexpr std::size_t kParallelThreshold = 128 * 1024;
  const unsigned worker_count =
      pixel_count >= kParallelThreshold ? budgeted_workers(8u) : 1u;
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
