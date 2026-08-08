// SPDX-License-Identifier: GPL-3.0-or-later
// Fused SDR output kernel: matrix conversion, the authoritative 16-step Oklab
// gamut fit, display transfer, deterministic injected TPDF dither, and uint8.

#include "dngscan_fast/output_core.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include "thread_budget.h"
#include <vector>

namespace dngscan_fast {
namespace {

inline float clampf(float value, float low, float high) {
  return std::min(high, std::max(low, value));
}

inline float sanitize(float value) {
  if (std::isnan(value)) {
    return 0.0f;
  }
  if (std::isinf(value)) {
    return value > 0.0f ? 1e6f : -1e6f;
  }
  return value;
}

inline Rgb mat3(const float matrix[9], const Rgb& value) {
  return {
      matrix[0] * value.r + matrix[1] * value.g + matrix[2] * value.b,
      matrix[3] * value.r + matrix[4] * value.g + matrix[5] * value.b,
      matrix[6] * value.r + matrix[7] * value.g + matrix[8] * value.b,
  };
}

inline bool in_unit_gamut(const Rgb& value) {
  constexpr float low = -OUTPUT_GAMUT_TOLERANCE;
  constexpr float high = 1.0f + OUTPUT_GAMUT_TOLERANCE;
  return value.r >= low && value.r <= high &&
         value.g >= low && value.g <= high &&
         value.b >= low && value.b <= high;
}

inline Rgb oklab_to_output(
    float lab_l,
    float lab_a,
    float lab_b,
    const NativeOutputPlan& plan) {
  Rgb lms_prime = mat3(plan.oklab_m2_inv, {lab_l, lab_a, lab_b});
  const Rgb lms{
      lms_prime.r * lms_prime.r * lms_prime.r,
      lms_prime.g * lms_prime.g * lms_prime.g,
      lms_prime.b * lms_prime.b * lms_prime.b,
  };
  return mat3(plan.lms_to_output, lms);
}

inline Rgb fit_output_pixel(const Rgb& input, const NativeOutputPlan& plan) {
  const Rgb rgb{sanitize(input.r), sanitize(input.g), sanitize(input.b)};
  if (in_unit_gamut(rgb)) {
    return {
        clampf(rgb.r, 0.0f, 1.0f),
        clampf(rgb.g, 0.0f, 1.0f),
        clampf(rgb.b, 0.0f, 1.0f),
    };
  }

  Rgb lms = mat3(plan.output_to_lms, rgb);
  lms.r = std::cbrt(lms.r);
  lms.g = std::cbrt(lms.g);
  lms.b = std::cbrt(lms.b);
  const Rgb lab = mat3(plan.oklab_m2, lms);

  const float chroma = std::hypot(lab.g, lab.b);
  const float ld = lab.r - 0.5f;
  const float abs_ld = std::abs(ld);
  const float e1 = 0.5f + abs_ld + plan.alpha * chroma;
  const float sign = ld > 0.0f ? 1.0f : (ld < 0.0f ? -1.0f : 0.0f);
  const float radicand = std::max(e1 * e1 - 2.0f * abs_ld, 0.0f);
  const float l0 = 0.5f * (1.0f + sign * (e1 - std::sqrt(radicand)));

  float lo = 0.0f;
  float hi = 1.0f;
  for (int iteration = 0; iteration < OUTPUT_GAMUT_FIT_ITERS; ++iteration) {
    const float t = 0.5f * (lo + hi);
    const Rgb candidate = oklab_to_output(
        l0 * (1.0f - t) + t * lab.r,
        t * lab.g,
        t * lab.b,
        plan);
    if (in_unit_gamut(candidate)) {
      lo = t;
    } else {
      hi = t;
    }
  }

  const Rgb fitted = oklab_to_output(
      l0 * (1.0f - lo) + lo * lab.r,
      lo * lab.g,
      lo * lab.b,
      plan);
  return {
      clampf(fitted.r, 0.0f, 1.0f),
      clampf(fitted.g, 0.0f, 1.0f),
      clampf(fitted.b, 0.0f, 1.0f),
  };
}

inline float display_encode(float linear) {
  const float value = clampf(linear, 0.0f, 1.0f);
  if (value <= 0.0031308f) {
    return value * 12.92f;
  }
  return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

inline std::uint8_t quantize(float encoded, float noise_a, float noise_b) {
  const float value = std::floor(encoded * 255.0f + 0.5f + noise_a - noise_b);
  return static_cast<std::uint8_t>(clampf(value, 0.0f, 255.0f));
}

inline std::uint8_t quantize_noise(float encoded, float noise) {
  const float value = std::floor(encoded * 255.0f + 0.5f + noise);
  return static_cast<std::uint8_t>(clampf(value, 0.0f, 255.0f));
}

template <typename Function>
void parallel_for(std::size_t pixel_count, const Function& function) {
  constexpr std::size_t kParallelThreshold = 128 * 1024;
  const unsigned worker_count =
      pixel_count >= kParallelThreshold ? budgeted_workers(8u) : 1u;
  if (worker_count <= 1) {
    function(0, pixel_count);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(worker_count - 1);
  const std::size_t block = (pixel_count + worker_count - 1) / worker_count;
  for (unsigned worker = 1; worker < worker_count; ++worker) {
    const std::size_t begin = std::min(pixel_count, std::size_t(worker) * block);
    const std::size_t end = std::min(pixel_count, begin + block);
    workers.emplace_back(function, begin, end);
  }
  function(0, std::min(pixel_count, block));
  for (auto& worker : workers) {
    worker.join();
  }
}

template <bool InputIsRec2020, bool NoiseIsCombined>
void finalize_u8(
    const float* input,
    const float* noise_a,
    const float* noise_b,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan) {
  const auto process_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      Rgb rgb{input[i * 3], input[i * 3 + 1], input[i * 3 + 2]};
      if constexpr (InputIsRec2020) {
        rgb = mat3(plan.rec2020_to_output, rgb);
      }
      const Rgb fitted = fit_output_pixel(rgb, plan);
      if constexpr (NoiseIsCombined) {
        output[i * 3] = quantize_noise(display_encode(fitted.r), noise_a[i * 3]);
        output[i * 3 + 1] = quantize_noise(display_encode(fitted.g), noise_a[i * 3 + 1]);
        output[i * 3 + 2] = quantize_noise(display_encode(fitted.b), noise_a[i * 3 + 2]);
      } else {
        output[i * 3] = quantize(
            display_encode(fitted.r), noise_a[i * 3], noise_b[i * 3]);
        output[i * 3 + 1] = quantize(
            display_encode(fitted.g), noise_a[i * 3 + 1], noise_b[i * 3 + 1]);
        output[i * 3 + 2] = quantize(
            display_encode(fitted.b), noise_a[i * 3 + 2], noise_b[i * 3 + 2]);
      }
    }
  };
  parallel_for(pixel_count, process_range);
}

}  // namespace

void fit_output_gamut_f32(
    const float* input,
    float* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan) {
  const auto process_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const Rgb fitted = fit_output_pixel(
          {input[i * 3], input[i * 3 + 1], input[i * 3 + 2]}, plan);
      output[i * 3] = fitted.r;
      output[i * 3 + 1] = fitted.g;
      output[i * 3 + 2] = fitted.b;
    }
  };
  parallel_for(pixel_count, process_range);
}

void finalize_rec2020_u8_f32(
    const float* input,
    const float* noise_a,
    const float* noise_b,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan) {
  finalize_u8<true, false>(input, noise_a, noise_b, output, pixel_count, plan);
}

void finalize_output_u8_f32(
    const float* input,
    const float* noise_a,
    const float* noise_b,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan) {
  finalize_u8<false, false>(input, noise_a, noise_b, output, pixel_count, plan);
}

void finalize_rec2020_u8_noise_f32(
    const float* input,
    const float* noise,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan) {
  finalize_u8<true, true>(input, noise, nullptr, output, pixel_count, plan);
}

void finalize_output_u8_noise_f32(
    const float* input,
    const float* noise,
    std::uint8_t* output,
    std::size_t pixel_count,
    const NativeOutputPlan& plan) {
  finalize_u8<false, true>(input, noise, nullptr, output, pixel_count, plan);
}

}  // namespace dngscan_fast
