// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "dngscan_fast/agx_core.h"

#include <cstddef>

namespace dngscan_fast {

// One compiled scene-EV -> display-linear curve on a uniform grid
// (dngscan/hdr_curve.py HdrCurveTable). Values are borrowed from the caller and
// must stay alive for the duration of the kernel call.
struct HdrCurveTableView {
  float ev_start;
  float inv_step;
  const float* values;
  int size;
};

// Full HDR formation chain plan (dngscan/hdr_agx.py _form_hdr_chunk):
//   compress_into_gamut -> inset -> curve table (native + optional reference-white
//   chroma candidate blended by RAW-gated rho) -> hue restore -> outset -> punch ->
//   Rec.2020 -> XYZ -> output RGB -> nan_to_num -> HDR colour-volume fit.
// Two film features are deliberately not represented and keep the NumPy path
// via dispatch exclusion (dngscan/_fast.py supports_hdr_formation): the full-mode
// takeover LUT (film_mode="full" with an active curve preset) and the enlarger
// colour-head LMS gain field (non-zero Y/M filtration).
struct NativeHdrPlan {
  float inset[9];
  float outset[9];

  // rec2020_to_output is applied as two chained float32 multiplies, matching the
  // NumPy rec2020_to_output(rec2020_to_xyz -> XYZ_TO_RGB[space]) operation order.
  float rec2020_to_xyz[9];
  float xyz_to_rec2020[9];
  float xyz_to_output[9];

  float oklab_m1[9];
  float oklab_m2[9];
  float oklab_m1_inv[9];
  float oklab_m2_inv[9];

  // Pre-outset formation luminance row (hdr_color.formation_luma_weights) for the
  // native/reference chroma blend, and the output space's normalized luminance row
  // (hdr_color.output_luma_weights) for the colour-volume fit.
  float formation_luma[3];
  float output_luma[3];

  float hue_restore;
  float punch_strength;
  // channel_separation * snr_gate; per-pixel rho is gated by the CFA clip masks.
  float global_rho;
  // Scene-authorized content peak, 2^H_rendered (_pack_peak); no margin is
  // applied (R2 item 10 removed it).
  float peak;

  HdrCurveTableView native_table;
  HdrCurveTableView reference_table;
  bool has_reference;
};

// clip_masks may be nullptr (no CFA evidence: rho stays the clamped global value).
void apply_hdr_formation_f32(
    const float* input,
    const float* clip_masks,
    float* output,
    std::size_t pixel_count,
    const NativeHdrPlan& plan);

}  // namespace dngscan_fast
