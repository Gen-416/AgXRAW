// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>

namespace dngscan_fast {

// v8 (R2 item 6): NativeOutputPlan's pre-merged rec2020_to_output replaced by
// the exact two-stage float64 matrices (rec2020_to_xyz + xyz_to_output).
// v10 (review batch 25, R-P2-6): NativeHdrPlan's output stage gets the same
// exact float64 two-stage matrices the SDR plan has carried since v8; the
// float32 chain was the recorded ~2.4e-6..8.5e-5 deviation of the HDR fast path.
// v11 (math review 2026-09-03): inset/outset and the punch/Oklab matrices of
// BOTH kernels become exact float64 stages (they were the dominant residual
// of the HDR fast path after v10, and the punch residual of the SDR one).
inline constexpr int NATIVE_ABI_VERSION = 11;
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
  // ABI v11 (math review 2026-09-03): every matrix stage NumPy evaluates
  // with a float64 matrix (agx._apply_matrix3 / apply_rgb_matrix3 on
  // float64 constants: inset, outset, the punch/Oklab excursion) is carried
  // as float64 and applied through mat3_exact_f64 — the same exact-stage
  // contract the output plan has carried since v8.
  double inset[9];
  double outset[9];
  CurveParams curve;
  float hue_restore;
  float view_brightness;
  float punch_strength;

  double rec2020_to_xyz[9];
  double xyz_to_rec2020[9];
  double oklab_m1[9];
  double oklab_m2[9];
  double oklab_m1_inv[9];
  double oklab_m2_inv[9];
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
