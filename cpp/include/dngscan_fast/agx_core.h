// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>

namespace dngscan_fast {

// v8 (R2 item 6): NativeOutputPlan's pre-merged rec2020_to_output replaced by
// the exact two-stage float64 matrices (rec2020_to_xyz + xyz_to_output).
inline constexpr int NATIVE_ABI_VERSION = 9;
inline constexpr float EPS = 1e-12f;

struct CurveParams {
  float black_ev;
  float range_ev;
  float gamma;
  float target_black;
  float target_white;

  float toe_power;
  float toe_transition_x;
  float toe_transition_y;
  float toe_scale;
  bool need_convex_toe;
  float toe_fallback_power;
  float toe_fallback_coefficient;

  float slope;
  float intercept;

  float shoulder_power;
  float shoulder_transition_x;
  float shoulder_transition_y;
  float shoulder_scale;
  bool need_concave_shoulder;
  float shoulder_fallback_power;
  float shoulder_fallback_coefficient;
};

struct NativeAgxPlan {
  float inset[9];
  float outset[9];
  CurveParams curve;
  float hue_restore;
  float view_brightness;
  float punch_strength;

  float rec2020_to_xyz[9];
  float xyz_to_rec2020[9];
  float oklab_m1[9];
  float oklab_m2[9];
  float oklab_m1_inv[9];
  float oklab_m2_inv[9];
};

struct Rgb {
  float r;
  float g;
  float b;
};

void apply_agx_core_f32(
    const float* input,
    float* output,
    std::size_t pixel_count,
    const NativeAgxPlan& plan);

}  // namespace dngscan_fast
