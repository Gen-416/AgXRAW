// SPDX-License-Identifier: GPL-3.0-or-later
// Native film appearance palette kernel — elementwise port of the NumPy
// oracle in dngscan/film_appearance.py (§6 math: Oklab decompose, S-gated
// periodic Catmull-Rom hue x monotone-Hermite EV field sampling, richness
// shoulder, neutral-density k on L and C, opponent reconstruction).

#include "dngscan_fast/film_appearance_core.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "thread_budget.h"

namespace dngscan_fast {
namespace {

inline void mat3_apply(const float* m, float x, float y, float z, float* o) {
  o[0] = m[0] * x + m[1] * y + m[2] * z;
  o[1] = m[3] * x + m[4] * y + m[5] * z;
  o[2] = m[6] * x + m[7] * y + m[8] * z;
}

std::int64_t run_range(const float* rgb, const float* scene_ev, float* out,
                       std::int64_t begin, std::int64_t end,
                       const FilmAppearanceParams& p) {
  const int H = p.h_knots;
  const int K = p.k_knots;
  const float step = 360.0f / static_cast<float>(H);
  const float c0sq = p.neutral_c0 * p.neutral_c0;
  const float ev_lo = p.ev_knots[0];
  const float ev_hi = p.ev_knots[K - 1];
  std::int64_t neg_rows = 0;

  for (std::int64_t i = begin; i < end; ++i) {
    const float r = rgb[i * 3 + 0];
    const float g = rgb[i * 3 + 1];
    const float b_in = rgb[i * 3 + 2];

    float lms[3];
    mat3_apply(p.m_fwd, r, g, b_in, lms);
    const float cl = std::cbrt(lms[0]);
    const float cm = std::cbrt(lms[1]);
    const float cs = std::cbrt(lms[2]);
    float lab[3];
    mat3_apply(p.m2, cl, cm, cs, lab);
    const float L = lab[0];
    const float a = lab[1];
    const float b = lab[2];
    const float C = std::hypot(a, b);
    float hdeg = std::atan2(b, a) * (180.0f / 3.14159265358979323846f);
    hdeg = std::fmod(hdeg, 360.0f);
    if (hdeg < 0.0f) hdeg += 360.0f;

    const float S = C / std::max(L, 1e-6f);
    const float s2 = S * S;
    const float w_c = s2 / (s2 + c0sq);
    const float cr = S / p.chroma_knee;
    float shoulder;
    if (p.chroma_power == 2.0f) {
      shoulder = cr * cr;
    } else if (p.chroma_power == 1.0f) {
      shoulder = cr;
    } else {
      shoulder = std::pow(cr, p.chroma_power);
    }
    const float r_sh = 1.0f / (1.0f + shoulder);

    // Periodic Catmull-Rom on the hue axis.
    const float hf = hdeg / step;
    const float base = std::floor(hf);
    const float t = hf - base;
    int j1 = static_cast<int>(base) % H;
    if (j1 < 0) j1 += H;
    const int j0 = (j1 - 1 + H) % H;
    const int j2 = (j1 + 1) % H;
    const int j3 = (j1 + 2) % H;
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float w0 = -0.5f * t3 + t2 - 0.5f * t;
    const float w1 = 1.5f * t3 - 2.5f * t2 + 1.0f;
    const float w2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    const float w3 = 0.5f * t3 - 0.5f * t2;

    // Monotone Hermite bracket on the EV axis.
    const float e_raw = scene_ev[i];
    const float ec = std::min(ev_hi, std::max(ev_lo, e_raw));
    int seg = K - 2;
    for (int s = 0; s < K - 1; ++s) {
      if (ec < p.ev_knots[s + 1]) { seg = s; break; }
    }
    const float dx = p.ev_knots[seg + 1] - p.ev_knots[seg];
    const float u = (ec - p.ev_knots[seg]) / dx;
    const float u2 = u * u;
    const float u3 = u2 * u;
    const float h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
    const float h10 = (u3 - 2.0f * u2 + u) * dx;
    const float h01 = -2.0f * u3 + 3.0f * u2;
    const float h11 = (u3 - u2) * dx;

    const int r0 = seg * H;
    const int r1 = (seg + 1) * H;
    auto sample = [&](const float* f, const float* d) {
      const float row_f0 = f[r0 + j0] * w0 + f[r0 + j1] * w1
                         + f[r0 + j2] * w2 + f[r0 + j3] * w3;
      const float row_d0 = d[r0 + j0] * w0 + d[r0 + j1] * w1
                         + d[r0 + j2] * w2 + d[r0 + j3] * w3;
      const float row_f1 = f[r1 + j0] * w0 + f[r1 + j1] * w1
                         + f[r1 + j2] * w2 + f[r1 + j3] * w3;
      const float row_d1 = d[r1 + j0] * w0 + d[r1 + j1] * w1
                         + d[r1 + j2] * w2 + d[r1 + j3] * w3;
      return h00 * row_f0 + h10 * row_d0 + h01 * row_f1 + h11 * row_d1;
    };

    const float dh = sample(p.f_hue, p.d_hue);
    const float gc = sample(p.f_chroma, p.d_chroma) * p.richness_mult;
    const float dd = sample(p.f_density, p.d_density) * p.density_mult;

    const float sw = p.strength * w_c;
    const float h_new = (hdeg + dh * sw) * (3.14159265358979323846f / 180.0f);
    const float k_dens = std::exp2(dd * sw * (-1.0f / 3.0f));
    const float c_new = C * std::exp2(gc * r_sh * sw) * k_dens;
    const float l_new = L * k_dens;

    float a_new = c_new * std::cos(h_new);
    float b_new = c_new * std::sin(h_new);
    if (p.has_neutral_bias) {
      // nb_ab is pre-multiplied by the neutral-bias strength; the kernel
      // adds strength * interp(ec) exactly like the oracle's np.interp.
      const float ta = p.nb_ab[seg * 2 + 0];
      const float tb = p.nb_ab[seg * 2 + 1];
      const float na = p.nb_ab[(seg + 1) * 2 + 0];
      const float nb = p.nb_ab[(seg + 1) * 2 + 1];
      a_new += p.strength * (ta + (na - ta) * u);
      b_new += p.strength * (tb + (nb - tb) * u);
    }

    float clms[3];
    mat3_apply(p.m2_inv, l_new, a_new, b_new, clms);
    const float xl = clms[0] * clms[0] * clms[0];
    const float xm = clms[1] * clms[1] * clms[1];
    const float xs = clms[2] * clms[2] * clms[2];
    float rgb_out[3];
    mat3_apply(p.m_inv, xl, xm, xs, rgb_out);
    const bool neg = rgb_out[0] < 0.0f || rgb_out[1] < 0.0f || rgb_out[2] < 0.0f;
    neg_rows += neg ? 1 : 0;
    out[i * 3 + 0] = std::max(rgb_out[0], 0.0f);
    out[i * 3 + 1] = std::max(rgb_out[1], 0.0f);
    out[i * 3 + 2] = std::max(rgb_out[2], 0.0f);
  }
  return neg_rows;
}

}  // namespace

std::int64_t film_appearance_apply(const float* rgb, const float* scene_ev,
                                   float* out, std::int64_t n,
                                   const FilmAppearanceParams& params) {
  const unsigned workers =
      budgeted_workers(static_cast<unsigned>(std::max<std::int64_t>(1, n / 65536)));
  if (workers <= 1) {
    return run_range(rgb, scene_ev, out, 0, n, params);
  }
  std::vector<std::thread> pool;
  std::vector<std::int64_t> counts(workers, 0);
  const std::int64_t chunk = (n + workers - 1) / workers;
  for (unsigned w = 0; w < workers; ++w) {
    const std::int64_t begin = static_cast<std::int64_t>(w) * chunk;
    const std::int64_t end = std::min<std::int64_t>(n, begin + chunk);
    if (begin >= end) break;
    pool.emplace_back([&, w, begin, end]() {
      counts[w] = run_range(rgb, scene_ev, out, begin, end, params);
    });
  }
  for (auto& th : pool) th.join();
  std::int64_t total = 0;
  for (const auto c : counts) total += c;
  return total;
}

}  // namespace dngscan_fast
