# Third-party notices

## darktable AgX (GPL-3.0-or-later)

The `agx` tone-mapping mode in `dngscan.core` ports portions of the AgX view-transform
implementation from darktable:

- https://github.com/darktable-org/darktable/blob/cf5e698c1a5afac52de785c3bf63fcbcb71707d3/src/iop/agx.c
- https://github.com/darktable-org/darktable/blob/cf5e698c1a5afac52de785c3bf63fcbcb71707d3/data/kernels/agx.cl

darktable is licensed under GPL-3.0-or-later. Because this project incorporates that
code, the combined work is distributed under **GPL-3.0-or-later** as well.
Reference copies of `agx.c` and `agx.cl` are included under `dngscan_assets/` with
their original GPL notices intact. The exact upstream commit is recorded in
`dngscan_assets/README.md` so changes in darktable `master` cannot silently redefine
dngscan's rendering baseline.

The AgX inset/outset primaries derive from Troy Sobotka's AgX family of view
transforms. Optional Blender-reference geometries follow the published construction
used by Eary Chow's AgX LUT generator:

- https://github.com/EaryChow/AgX_LUT_Gen

No third-party display or camera LUT is distributed with dngscan.

## Embedded lens metadata and RAW processing references

The Sony/Fujifilm coefficient conventions in `dngscan/embedded_lens.py` are
adapted from darktable's metadata-driven lens correction (GPL-3.0-or-later),
with tag layouts cross-checked against Exiv2:

- https://github.com/darktable-org/darktable/blob/master/src/common/exif.cc
- https://github.com/darktable-org/darktable/blob/master/src/iop/lens.cc
- https://github.com/Exiv2/exiv2/blob/main/src/fujimn_int.cpp

The implementation reads profiles authored in each photograph; no third-party
lens-profile database is bundled. This code is distributed under the project's
GPL-3.0-or-later license.

Camera-matrix reconstruction follows LibRaw's `cam_xyz_coeff` normalization and
matrix constants. DNG black-delta normalization and opcode stage conventions
were checked against the DNG specification and Adobe's DNG SDK reference:

- https://github.com/LibRaw/LibRaw/blob/master/src/utils/utils_dcraw.cpp
- https://helpx.adobe.com/camera-raw/digital-negative.html
- https://android.googlesource.com/platform/external/dng_sdk/+/refs/heads/android14-prebuilt-test/source/dng_linearization_info.cpp

These references document the numerical conventions implemented by this
project; the DNG SDK is not bundled or linked.

## RAW to ACES spectral data (Apache-2.0)

Selected camera sensitivities and training reflectances under
`dngscan_assets/spectral/` come from the Academy Software Foundation's
`rawtoaces-data` repository:

- https://github.com/AcademySoftwareFoundation/rawtoaces-data

That source repository is licensed under Apache-2.0. Derived CSV files retain
source and measurement notes in `dngscan_assets/spectral/README.md`.

## CBLD camera black levels (NOT redistributed)

The advisory black-level report line can use CBLD (Camera Black Level
Database, https://y-g-jiang.github.io/CBLD.html, by 知乎@姜尧耕). CBLD
credits its author but carries no explicit redistribution license, and
attribution is not authorization — so its data is **not** included in this
repository or in released wheels. Users who want the advisory line fetch
the data themselves for personal use:

    python tools/import_cbld.py     # writes ~/.config/dngscan/cbld.json

Without that local import the feature is silently absent. The upstream
author's own caveat applies: 仅供参考，最好以自己机器的当次拍摄为准。

## Evidence-shell test corpus (tests/data/evidence_shells)

Container structures stripped from CC0 camera samples hosted by
raw.pixls.us (bulk pixel data removed; each shell's manifest records the
source URL and SHA-256 of the original). Sources: DJI FC6310 DJI_0220.DNG,
DJI Osmo Action DJI_0254.DNG, PENTAX K-r IMGP4425.DNG — all published
under CC0 by their contributors. The shell format
(tools/make_evidence_shell.py) is an independent implementation of an
idea credited to y-g-jiang's "dngshell" test-corpus format.

## PhotonsToPhotos-derived bulk sensor priors (permission pending)

`dngscan/data/priors/p2p_bulk.json` holds derived sensor priors (unity gain,
read noise in electrons, PDR) for 135 legacy 14-bit cameras. The chain has
three layers: (1) chart data measured and published by Bill Claff at
photonstophotos.net — © William J. Claff, **all rights reserved**, no
published data-reuse policy; (2) a machine-generated compilation by
y-g-jiang (`pdr_camera_data_14bit.js`, with hletrd pixel-pitch data);
(3) our conversion (`tools/import_p2p_pdr.py`, derivations in its header).

Status recorded honestly rather than laundered: the project owner decided on
2026-08-24 to import now — dngscan is an open-source, non-commercial tool
that recomputes and analyzes rather than republishing the database — while
permission is being sought from the author. (The layer-2 compilation by
y-g-jiang is covered by his 2026-08-25 credit-based grant; the pending
question is solely Bill Claff's layer-1 chart data.) The entire footprint is that one
JSON file plus its importer; deleting the file cleanly removes the tier
(same reversibility discipline as CBLD above). The curated entries in
`dngscan/sensor_priors.json` cite the same site per-entry.

## JPTC first-party sensor measurements (y-g-jiang)

`dngscan/data/priors/jptc/*.json` are photon-transfer fits computed by
`tools/import_jptc.py` from JPTC/2 CSV measurements published by y-g-jiang
(https://y-g-jiang.github.io/, first-party bench data, e.g.
sony-a7m5-iso100-electronic.csv). The collector records only raw per-frame
statistics; all derived quantities (gain, read noise, FWC, PRNU) are our
fits with apertures and uncertainties declared in each entry.
**Permission: granted by the author on 2026-08-25** — no formal license,
use permitted with credit. Credit: measurement data by y-g-jiang (姜尧耕),
https://y-g-jiang.github.io/. Entries remain single-file removable.

## dngshell first-party shells (y-g-jiang)

Six evidence shells (sony_ilce7m5, leica_q2, fujifilm_gfx100,
hasselblad_x1d, canon_5d3, panasonic_s1rm2) were converted with
`tools/import_dngshell.py` from the DNGSHL1 corpus at
https://y-g-jiang.github.io/shells/. Their source files are the corpus
author's own captures (hence no third-party sourceUrl).
**Permission: granted by the author on 2026-08-25** — no formal license,
use permitted with credit (same grant as the JPTC measurements above).
Credit: source captures and DNGSHL1 corpus by y-g-jiang (姜尧耕),
https://y-g-jiang.github.io/. Each shell contains only container/metadata
bytes (bulk pixel and preview regions removed upstream AND re-stripped by
our walker); the manifest's `upstream` block records the true original's
SHA-256. Deleting the six .evshell files removes the tier.

## Lens/filter transmittance library (y-g-jiang)

`dngscan/data/lens_transmittance.json` bundles 118 spectral transmittance
measurements (84 lenses, 34 filters; 380-755 nm @ 1 nm) converted by
`tools/import_lens_transmittance.py` from the author's first-party bench
data at https://y-g-jiang.github.io/lens-transmittance-data/.
**Permission: granted by the author on 2026-08-25** — no formal license,
use permitted with credit (same grant as the JPTC measurements).
Credit: measurements by y-g-jiang (姜尧耕), https://y-g-jiang.github.io/.
Single-file removable.

## JPTC collect sets (y-g-jiang)

`dngscan/data/priors/jptc_collect/*.json` are derived from the author's
first-party collect sets at https://y-g-jiang.github.io/data/collect/
(formats JPTC-DARK/1, JPTC-ISOGAIN/1, JPTC-SPECTRUM/1, JPTC/2; 13 sets,
9 camera bodies, tester 姜尧耕). All derivations (gain curves anchored on
PTC fits, temporal read noise with the declared sigma-clip undone, banding
decomposition, noise-whiteness ratios) are ours and documented in
`tools/import_jptc_collect.py`. Same credit-based grant of 2026-08-25 as
the other first-party data; the directory is removable as a unit.

`dngscan/data/priors/collect_registry.json` additionally preserves the
author's set registry, dependency relationships and revision notes from
`data/collect/sets.json` at commit
`504e119bcfb9ff2b58fc9e47c204ddb77eee5879`. It is generated by
`tools/compile_calibration_registry.py` and carries the same credit-based
permission. Registry metadata does not by itself qualify a measurement for
pixel correction; partial and research-only relationships remain explicit.

## Film simulation data (dngscan/data/film) — spektrafilm, CC BY-SA 4.0

The observe film simulation (`dngscan/film/`, see docs/FILM.zh-CN.md) ships one
data asset, `dngscan/data/film/observe_v1.json`. Its per-stock curve, reversal
metered-exposure offset, hue rendition, grey-scale crossover and Stage A
observer/characteristic tables are computed from Andrea Volpato's spektrafilm
film and print profiles:

    Derived from spektrafilm by Andrea Volpato
    https://github.com/andreavolpato/spektrafilm
    Licensed CC BY-SA 4.0
    Modified by the AgXRAW / AgXFilm project (https://github.com/Gen-416).

spektrafilm's profile data is licensed under **CC BY-SA 4.0**, separately from
its GPL code; the derived quantities are distributed under the same CC BY-SA 4.0
terms. The unmodified license text ships inside the package next to the engine
(`dngscan/film/SPEKTRAFILM_LICENSE.txt`), with the list of modifications in
`dngscan/film/CHANGELOG.txt`, and the attribution is also embedded in the asset's
`license` block. The profiles used were pinned at upstream commit
`3bb2c2d2801ff68b92019cf1dbcbb133d60832bc` (profile version 0.3.2). spektrafilm
processed its profiles from Kodak and Fujifilm datasheets and scientific
publications; original measurements remain the property of their holders.
"spektrafilm" is named here for attribution only.

Not derived from spektrafilm:

- Grain sigma(D): Kodak VISION3 250D (5207) diffuse rms granularity curves,
  digitized from Eastman Kodak publication H-1-5207 (48 um aperture). Data
  remain the property of Eastman Kodak.
- Per-stock grain scales: granularity figures published in Kodak (Print
  Grain Index; diffuse rms) and Fujifilm (rms granularity) technical data
  sheets, combined as documented in AgXFilm `docs/grain_scale.json`.
- Reversal metered mid grey follows the ISO 2240 speed-point definition
  (method only; no standard text is reproduced).

The research archive, the frozen pre-split integration and the derivation
tools live in the AgXFilm repository (https://github.com/Gen-416/AgXFilm).
The pre-split film features (full print chain, colour head, development,
halation, media scatter, appearance recipes) are not part of AgXRAW.

## Spectral calibration bootstrap data and lens-filter constants

`dngscan_assets/spectral/` (used by `tools/calibrate_skin_matrix.py` and the
retained `alev_material_d55` scene-transform preset; provenance per file in
`dngscan_assets/spectral/README.md`):

- `arri_alexa_alev3_ssf_digitized.csv`: ALEV3/ALEXA spectral sensitivities
  visually transcribed at 10 nm anchors from Figure 1 of Leonhardt & Brendel,
  "Critical spectra in the color reproduction process of digital motion picture
  cameras", IS&T CIC23 (2015). Approximate digitized values, not a copy of the
  figure; the measurements belong to their authors.
- `sony_imx410_qe_zwo_asi2400mc_digitized.csv`: rough IMX410 QE proxy after the
  published ZWO ASI2400MC QE graph (https://www.zwoastro.com/product/asi2400mc-pro/).
- `sigma_fp_hot_mirror_model_420_660.csv`, `demo_*_reflectance.csv`,
  `unit_transmission.csv`: analytic models authored for this project.

Lens conversion filters (`dngscan/lens_filter.py`) are derived from Kodak's
published Wratten mired shifts and filter factors (two numbers per filter); no
spectral curve is bundled. Measured lens/filter transmittance and embedded lens
corrections are listed in their own sections above.
