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

## Status A / Status M densitometer responsivities

`dngscan_assets/spectral/densitometer/` carries the ISO 5-3 Status A and
Status M spectral responsivities as digitized in Giorgianni, Madden & Kriss,
*Digital Color Management* (Wiley 2009), p. 335, redistributed from the
agx-emulsion project's v0.2.0-legacy tree
(`agx_emulsion/data/densitometer/`, GPL-3.0-or-later — same license as this
project). Offline verification only (`tools/crosscheck_2383.py`); not used in
rendering and not packaged into wheels.

## spektrafilm film profiles (CC BY-SA 4.0)

Film stock profiles under `dngscan_assets/spectral/spektrafilm/` (spectral
sensitivities and characteristic curves for the twenty simulated stocks and their
paired print media — the full roster is listed in
`dngscan_assets/spectral/spektrafilm/README.md`) come verbatim from Andrea
Volpato's spektrafilm project:

- https://github.com/andreavolpato/agx-emulsion

spektrafilm's code is GPL-3.0-or-later; its profile data is licensed separately
under **CC BY-SA 4.0**. The full license text ships in two places: alongside the
vendored profiles (`dngscan_assets/spectral/spektrafilm/SPEKTRAFILM_LICENSE.txt`)
and inside the installed package (`dngscan/SPEKTRAFILM_LICENSE.txt`), so wheels
carrying the derived preset JSONs carry the license too. The vendored copy is
pinned to upstream commit `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc` with a
per-file SHA-256 manifest (`MANIFEST.sha256`). dngscan's film curve presets and
prefeed targets derived from these profiles are treated as direct derivatives
under the same CC BY-SA 4.0 terms, with provenance recorded in
`dngscan_assets/spectral/spektrafilm/README.md` and in each preset's `source` field.
The upstream data was processed from manufacturer datasheets and scientific papers;
original measurements remain the property of their respective manufacturers.

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
