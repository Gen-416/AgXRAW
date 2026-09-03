// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "dngscan_fast/agx_core.h"

namespace dngscan_fast {

// R2 item 6 (ABI v8, SDR) / review batch 25 (ABI v10, HDR): one exact NumPy
// matrix stage — float64 matrix entries times the float32 value promoted to
// double, accumulated left-to-right in double, and materialized to float32
// exactly once (apply_rgb_matrix3's `out[:, r] =` assignment). Left-
// associative double addition matches the NumPy expression tree;
// -ffp-contract=off keeps FMA from skipping the product roundings.
inline Rgb mat3_exact_f64(const double matrix[9], const Rgb& value) {
  const double r = static_cast<double>(value.r);
  const double g = static_cast<double>(value.g);
  const double b = static_cast<double>(value.b);
  return {
      static_cast<float>(matrix[0] * r + matrix[1] * g + matrix[2] * b),
      static_cast<float>(matrix[3] * r + matrix[4] * g + matrix[5] * b),
      static_cast<float>(matrix[6] * r + matrix[7] * g + matrix[8] * b),
  };
}

}  // namespace dngscan_fast
