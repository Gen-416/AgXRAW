// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "dngscan_fast/agx_core.h"

#include <cstddef>
#include <cstdint>

namespace dngscan_fast {

inline constexpr int OUTPUT_GAMUT_FIT_ITERS = 16;
inline constexpr float OUTPUT_GAMUT_TOLERANCE = 1e-4f;

struct NativeOutputPlan {
  float rec2020_to_output[9];
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
