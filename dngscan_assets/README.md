# dngscan Assets

Open-source references and calibration inputs used by dngscan.

- `darktable_agx.c` / `darktable_agx.cl` — reference copies from darktable commit
  [`cf5e698c1a5afac52de785c3bf63fcbcb71707d3`](https://github.com/darktable-org/darktable/commit/cf5e698c1a5afac52de785c3bf63fcbcb71707d3)
  (2026-05-14), redistributed under GPL-3.0-or-later. Their Git blob IDs are
  `380389e24b62a1ec5ab2d7713ffd3b797d2270d5` and
  `19c3b01dfdef26e4f86e1a22408989753644176e`. The AgX curve construction, scene-default
  primary geometry, hue-restore semantics, and C1 endpoint DRT are pinned to this source
  instead of following darktable `master` implicitly.
Spectral calibration inputs live under `spectral/` — see `spectral/README.md`.

No vendor LUT is distributed. Local LUT downloads and generated look fields are ignored
by Git and must not be committed without an explicit redistribution licence.
