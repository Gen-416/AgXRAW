// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "dngscan_fast/agx_core.h"

#include <cstddef>
#include <cstdint>

namespace dngscan_fast {

inline constexpr int OUTPUT_GAMUT_FIT_ITERS = 16;
inline constexpr float OUTPUT_GAMUT_TOLERANCE = 1e-4f;

struct NativeOutputPlan {
  // R2 item 6 (ABI v8): the Rec.2020 -> output conversion is two exact
  // stages, each a float64-accumulated product materialized to float32 —
  // the NumPy graph's own operation order and rounding
  // (apply_rgb_matrix3(rec2020_to_xyz(rgb), XYZ_TO_RGB[space])). The old
  // pre-merged float matrix dropped one float32 rounding and was the
  // recorded 8.6e-5 deviation of the output fast path.
  double rec2020_to_xyz[9];
  double xyz_to_output[9];
  float output_to_lms[9];
  float lms_to_output[9];
  float oklab_m2[9];
  float oklab_m2_inv[9];
  float alpha;
};

void fit_output_gamut_f32(
    const float* input,
    float* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan);

void finalize_rec2020_u8_f32(
    const float* input,
    const float* noise_a,
    const float* noise_b,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan);

void finalize_output_u8_f32(
    const float* input,
    const float* noise_a,
    const float* noise_b,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan);

void finalize_rec2020_u8_noise_f32(
    const float* input,
    const float* noise,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan);

void finalize_output_u8_noise_f32(
    const float* input,
    const float* noise,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan);

}  // namespace dngscan_fast
