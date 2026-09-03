// SPDX-License-Identifier: GPL-3.0-or-later
//
// Per-pixel primitives shared by the SDR AgX kernel and the HDR formation kernel.
// Behavioral reference: dngscan/agx.py, dngscan/punch.py. Each helper replicates the
// float32 operation order of its NumPy counterpart; parity is enforced by
// tests/test_fast_backend.py and tests/test_hdr_native.py.
#pragma once

#include "dngscan_fast/agx_core.h"
#include "dngscan_fast/exact_matrix.h"

#include <algorithm>
#include <cmath>

namespace dngscan_fast {

constexpr float PIXEL_MID_GRAY = 0.18f;

inline float clampf(float v, float lo, float hi) {
  return std::min(hi, std::max(lo, v));
}

inline float max3(float a, float b, float c) {
  return std::max(a, std::max(b, c));
}

inline float min3(float a, float b, float c) {
  return std::min(a, std::min(b, c));
}

inline float dot3(const float w[3], const Rgb& v) {
  return w[0] * v.r + w[1] * v.g + w[2] * v.b;
}

inline Rgb mat3(const float m[9], const Rgb& v) {
  return {
      m[0] * v.r + m[1] * v.g + m[2] * v.b,
      m[3] * v.r + m[4] * v.g + m[5] * v.b,
      m[6] * v.r + m[7] * v.g + m[8] * v.b,
  };
}

inline float smoothstep(float edge0, float edge1, float x) {
  const float denom = edge1 - edge0;
  if (std::abs(denom) < 1e-9f) {
    return 0.0f;
  }
  const float t = clampf((x - edge0) / denom, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

inline float hue_in_arc(float hue_deg, float lo, float hi) {
  float h = std::fmod(hue_deg, 360.0f);
  if (h < 0.0f) {
    h += 360.0f;
  }
  bool inside;
  float edge;
  if (lo <= hi) {
    inside = (h >= lo) && (h <= hi);
    edge = std::min(h - lo, hi - h);
  } else {
    inside = (h >= lo) || (h <= hi);
    float d_lo = (h >= lo) ? (h - lo) : (360.0f - lo + h);
    float d_hi = (h <= hi) ? (hi - h) : (360.0f - h + hi);
    edge = std::min(d_lo, d_hi);
  }
  return inside ? smoothstep(0.0f, 6.0f, edge) : 0.0f;
}

inline float nan_to_num(float v, float nan_val, float posinf_val, float neginf_val) {
  if (std::isnan(v)) {
    return nan_val;
  }
  if (std::isinf(v)) {
    return v > 0.0f ? posinf_val : neginf_val;
  }
  return v;
}

// AgX opponent-luminance constants from the pinned darktable implementation.
constexpr float AGX_OPPONENT_Y[3] = {
    0.2658180370250449f,
    0.59846986045365f,
    0.1357121025213052f,
};

inline Rgb compress_into_gamut(const Rgb& rgb) {
  const float input_y = dot3(AGX_OPPONENT_Y, rgb);
  const float max_rgb = max3(rgb.r, rgb.g, rgb.b);
  const Rgb opponent{max_rgb - rgb.r, max_rgb - rgb.g, max_rgb - rgb.b};
  const float opponent_y = dot3(AGX_OPPONENT_Y, opponent);
  const float max_opponent = max3(opponent.r, opponent.g, opponent.b);
  const float y_compensate_negative = max_opponent - opponent_y + input_y;

  const float offset = std::max(-min3(rgb.r, rgb.g, rgb.b), 0.0f);
  const Rgb rgb_offset{rgb.r + offset, rgb.g + offset, rgb.b + offset};
  const float max_offset = max3(rgb_offset.r, rgb_offset.g, rgb_offset.b);
  const Rgb opponent_offset{
      max_offset - rgb_offset.r,
      max_offset - rgb_offset.g,
      max_offset - rgb_offset.b,
  };
  const float max_inverse = max3(opponent_offset.r, opponent_offset.g, opponent_offset.b);
  const float y_inverse = dot3(AGX_OPPONENT_Y, opponent_offset);
  float y_new = dot3(AGX_OPPONENT_Y, rgb_offset);
  y_new = max_inverse - y_inverse + y_new;

  float ratio = 1.0f;
  if (y_new > y_compensate_negative && y_new > EPS) {
    ratio = y_compensate_negative / y_new;
  }
  return {rgb_offset.r * ratio, rgb_offset.g * ratio, rgb_offset.b * ratio};
}

inline float rgb_to_hue(const Rgb& rgb) {
  const float maxc = max3(rgb.r, rgb.g, rgb.b);
  const float minc = min3(rgb.r, rgb.g, rgb.b);
  const float delta = maxc - minc;
  // NumPy semantics (agx._rgb_to_hsv): hue stays 0 unless delta > EPS, which also
  // keeps a NaN delta (e.g. inf - inf) on the neutral branch instead of poisoning it.
  if (!(delta > EPS)) {
    return 0.0f;
  }
  float h;
  if (maxc == rgb.r) {
    h = std::fmod((rgb.g - rgb.b) / delta, 6.0f);
  } else if (maxc == rgb.g) {
    h = (rgb.b - rgb.r) / delta + 2.0f;
  } else {
    h = (rgb.r - rgb.g) / delta + 4.0f;
  }
  h = std::fmod(h / 6.0f, 1.0f);
  if (h < 0.0f) {
    h += 1.0f;
  }
  return h;
}

inline Rgb hsv_to_rgb(float h, float s, float v) {
  h = std::fmod(h, 1.0f);
  if (h < 0.0f) {
    h += 1.0f;
  }
  s = std::max(s, 0.0f);
  const float hh = h * 6.0f;
  const int i = static_cast<int>(std::floor(hh)) % 6;
  const float f = hh - std::floor(hh);
  const float p = v * (1.0f - s);
  const float q = v * (1.0f - s * f);
  const float t = v * (1.0f - s * (1.0f - f));
  switch (i) {
    case 0:
      return {v, t, p};
    case 1:
      return {q, v, p};
    case 2:
      return {p, v, t};
    case 3:
      return {p, q, v};
    case 4:
      return {t, p, v};
    default:
      return {v, p, q};
  }
}

inline Rgb mix_hue(const Rgb& rgb_linear, float pre_hue, float restore) {
  const float post_hue = rgb_to_hue(rgb_linear);
  float delta = post_hue - pre_hue;
  delta -= std::nearbyintf(delta);
  const float restored = std::fmod(pre_hue + (1.0f - restore) * delta, 1.0f);
  const float maxc = max3(rgb_linear.r, rgb_linear.g, rgb_linear.b);
  const float minc = min3(rgb_linear.r, rgb_linear.g, rgb_linear.b);
  float sat = 0.0f;
  if (maxc > EPS) {
    sat = (maxc - minc) / maxc;
  }
  return hsv_to_rgb(restored < 0.0f ? restored + 1.0f : restored, sat, maxc);
}

// Matrices needed by the scene-driven punch operator (dngscan/punch.py).
struct PunchMatrices {
  // ABI v11: exact float64 stages (NumPy's apply_rgb_matrix3 on float64).
  const double* rec2020_to_xyz;
  const double* xyz_to_rec2020;
  const double* oklab_m1;
  const double* oklab_m2;
  const double* oklab_m1_inv;
  const double* oklab_m2_inv;
};

constexpr float PUNCH_CHROMA_MAX = 1.5f;
constexpr float PUNCH_SKIN_DAMP = 0.55f;
constexpr float SKIN_HUE_LO = 20.0f;
constexpr float SKIN_HUE_HI = 60.0f;

inline Rgb apply_punch_rec2020_pixel(
    const Rgb& rgb_in, float strength, const PunchMatrices& m) {
  if (strength <= 1e-3f) {
    return rgb_in;
  }
  const float s = std::min(1.0f, strength);
  Rgb rgb{
      nan_to_num(rgb_in.r, 0.0f, 1e6f, 0.0f),
      nan_to_num(rgb_in.g, 0.0f, 1e6f, 0.0f),
      nan_to_num(rgb_in.b, 0.0f, 1e6f, 0.0f),
  };

  Rgb xyz = mat3_exact_f64(m.rec2020_to_xyz, rgb);
  Rgb lms = mat3_exact_f64(m.oklab_m1, xyz);
  lms.r = std::cbrt(std::max(lms.r, 0.0f));
  lms.g = std::cbrt(std::max(lms.g, 0.0f));
  lms.b = std::cbrt(std::max(lms.b, 0.0f));
  Rgb lab = mat3_exact_f64(m.oklab_m2, lms);

  const float chroma = std::hypot(lab.g, lab.b);
  float hue = std::fmod(std::atan2(lab.b, lab.g) * (180.0f / static_cast<float>(M_PI)), 360.0f);
  if (hue < 0.0f) {
    hue += 360.0f;
  }

  float weight = smoothstep(0.005f, 0.03f, chroma);
  weight *= smoothstep(0.08f, 0.22f, lab.r);
  weight *= (1.0f - smoothstep(0.72f, 0.92f, lab.r));
  weight *= (1.0f - 0.35f * smoothstep(0.20f, 0.42f, chroma));
  weight *= (1.0f - (1.0f - PUNCH_SKIN_DAMP) * hue_in_arc(hue, SKIN_HUE_LO, SKIN_HUE_HI));
  const float gain = 1.0f + (PUNCH_CHROMA_MAX - 1.0f) * s * weight;

  Rgb lab_out{lab.r, lab.g * gain, lab.b * gain};
  Rgb lms_ = mat3_exact_f64(m.oklab_m2_inv, lab_out);
  lms_.r = lms_.r * lms_.r * lms_.r;
  lms_.g = lms_.g * lms_.g * lms_.g;
  lms_.b = lms_.b * lms_.b * lms_.b;
  Rgb xyz_out = mat3_exact_f64(m.oklab_m1_inv, lms_);
  Rgb out = mat3_exact_f64(m.xyz_to_rec2020, xyz_out);
  return {
      nan_to_num(out.r, 0.0f, 1e6f, -1e6f),
      nan_to_num(out.g, 0.0f, 1e6f, -1e6f),
      nan_to_num(out.b, 0.0f, 1e6f, -1e6f),
  };
}

}  // namespace dngscan_fast
