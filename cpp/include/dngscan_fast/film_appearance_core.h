// SPDX-License-Identifier: GPL-3.0-or-later
// Native palette kernel for the film appearance layer (plan §16 P6 / E3).
// The NumPy implementation in dngscan/film_appearance.py is the correctness
// oracle; this port must match it elementwise (parity gate in
// tests/test_film_appearance_p10.py). All tables, matrices and scalars come
// from Python — the declaration source stays in one place.
#pragma once

#include <cstdint>

namespace dngscan_fast {

struct FilmAppearanceParams {
  // [K, H] recipe fields with their PCHIP EV-derivative tables (row-major).
  const float* f_hue;
  const float* d_hue;
  const float* f_chroma;
  const float* d_chroma;
  const float* f_density;
  const float* d_density;
  const float* ev_knots;   // [K], strictly increasing
  const float* nb_ab;      // [K, 2] neutral-bias table (already x nb strength)
  int k_knots;
  int h_knots;
  bool has_neutral_bias;
  float strength;
  float neutral_c0;
  float chroma_knee;
  float chroma_power;      // 2.0 and 1.0 fast paths, else powf
  float richness_mult;     // 1 + richness_delta (1.0 in reference)
  float density_mult;      // 1 + color_density_delta
  // Fused colour matrices, row-major 3x3 (from _fused_oklab_matrices):
  const float* m_fwd;      // mapped Rec.2020 -> LMS (pre-cbrt)
  const float* m2;         // cbrt(LMS) -> Oklab
  const float* m2_inv;     // Oklab -> cbrt(LMS)
  const float* m_inv;      // LMS -> mapped Rec.2020
};

// rgb: [n, 3] float32 (mapped Rec.2020), scene_ev: [n] float32, out: [n, 3].
// Returns the number of rows that had any negative component before the
// final clamp (the Python side feeds the clamp_stats counters).
std::int64_t film_appearance_apply(const float* rgb, const float* scene_ev,
                                   float* out, std::int64_t n,
                                   const FilmAppearanceParams& params);

}  // namespace dngscan_fast
