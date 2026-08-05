# Review findings - HDR AgX pipeline

> **ARCHIVED 2026-07-29.** This document describes the pipeline *before* the v2
> log-stop shoulder landed and before the delivery gates were recalibrated on the
> real-frame corpus: §2's "the white clamp is not a zero-derivative endpoint" is the
> exact property v2's pinned-tangent Hermite now provides, and §1's whole-frame
> pixel gates (median 1.5% / p95 8% / p99 12%) were replaced by per-profile,
> per-container block + pixel-chroma tolerances in `dngscan/delivery.py`. Kept for
> history only; the current contract is
> [`../HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md`](../HDR_AGX_V2_IMPLEMENTATION_PLAN.zh-CN.md).

Reviewed after the native extended-white curve migration on 2026-07-29. No item below
invalidates the mathematical tone path; these are the remaining calibration and delivery
boundaries.

## 1. Delivery tolerances are engineering gates

The ISO gain-map file is a lossy JPEG base plus a lossy auxiliary rendition. Current
whole-frame round-trip gates (median 1.5%, p95 8%, p99 12%) were calibrated against the
macOS Core Image writer at quality 100. They are regression limits, not a lossless claim
or constants from ISO 21496-1. A successful write still requires the per-file readback.

Cross-platform interpretation by Android, Chrome, Quick Look versions other than the test
host, and social-platform transcodes remains a real device test. Local Core Image success
does not prove that interoperability.

## 2. Tone endpoint boundary

The native HDR compiler now refuses the generic AgX `power < 1` accelerating shoulder and
reduces target white when necessary. Internal toe/latitude/shoulder joins are C1. The
finite white-EV input clamp is not a zero-derivative endpoint, however; reliable-tail
margins keep trustworthy pixels away from that boundary. A future true zero-slope endpoint
would require a different sigmoid boundary solve, not another post-curve gain layer.

## 3. Colour calibration boundaries

`HdrColorGeometry.snr_gate` remains fixed at 1.0. Production rendering must not change
according to whether diagnostics were requested, and a production-path per-channel tail
SNR metric does not exist yet.

The extended-P3 projector preserves linear Y and an RGB opponent direction. It is not a
perceptual JMh hue compressor. `rho_base`, RAW9's aggregate-confidence cap, and HDR-specific
outset geometry still need a broader EDR corpus.

## 4. Verified in this migration

- SDR golden and byte-freeze fixtures are unchanged.
- Native HDR curves remain monotone, preserve EV0 at 0.18, reach their solved endpoints,
  and never enter the accelerating fallback.
- `rho` changes chromaticity without changing the native curve's formation luminance.
- Extended-P3 projection remains bounded and preserves Y where the target volume permits.
- Core Image live tests pass outside the filesystem sandbox.
- Apple ISO gain-map writer and readback tests pass with the new HDR pixels.
