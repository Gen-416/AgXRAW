# dngscan spectral calibration CSVs

These CSV files are replaceable calibration inputs for `tools/calibrate_skin_matrix.py`.
They intentionally keep measured or digitized spectral data outside the algorithm.

Current files are bootstrap-quality approximations, not authoritative measurements:

- `arri_alexa_alev3_ssf_digitized.csv`: ALEV3/ALEXA SSF digitized at 10nm anchors from Figure 1 of Leonhardt & Brendel, "Critical spectra in the color reproduction process of digital motion picture cameras", CIC23 2015 (measurements of five ALEXA cameras, 380-780nm/5nm double monochromator, ColorChecker-validated). Visual transcription accuracy ~±2-3% of peak per channel; per-channel white normalization in the calibrator cancels per-channel scale. Source PDF: https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/23/1/art00029
- `sony_imx410_qe_zwo_asi2400mc_digitized.csv`: rough IMX410 RGB QE proxy. Replace with points digitized from the ZWO ASI2400MC QE graph/specification: https://www.zwoastro.com/product/asi2400mc-pro/
- `sigma_fp_hot_mirror_model_420_660.csv`: sigmoid hot-mirror transmission model, 420nm blue-side and 660nm red-side cutoff assumption. Replace with measured transmission if available. Kolari teardown confirms the fp conversion/filter-stack context: https://kolarivision.com/the-sigma-fp-disassembly-and-teardown/
- `demo_skin_reflectance.csv`: analytic demo skin reflectance manifold. Replace with a licensed/public skin reflectance library such as Hyper-Skin if you have permission to use the released data: https://github.com/hyperspectral-skin/Hyper-Skin-2023
- `demo_cyan_reflectance.csv`: analytic cyan/blue-green material manifold. Replace or augment with measured surface spectra. The MLS dataset is one open source of real measured object/illumination spectra under CC BY-SA 4.0: https://github.com/visillect/mls-dataset

Accepted formats include wide `wavelength_nm,sample_1,sample_2,...`, tidy `sample,wavelength_nm,reflectance`, and single `wavelength_nm,value` CSVs.

## Material mode (`--preset-mode material`)

Additional optional per-material reflectance CSVs (same accepted formats); analytic
demo families are used when a file is absent:

- `foliage_reflectance.csv`: vegetation with red edge. Open sources: USGS spectral library, ECOSTRESS.
- `magenta_reflectance.csv`: magenta/purple dyes and fabrics.
- `neutral_reflectance.csv`: near-neutral ramps.

Outputs: preset `alev_material_d55` (merged into scene_transform_presets.json) and
`calibration_report.json` here (per-material before/after Oklab divergence per
illuminant, confidence, and the D55 cross-material leakage table). Windows can be
re-fit from real photos per material with `tools/fit_skin_window.py --preset
alev_material_d55 --region <name>`.

## Measured camera data (fetched 2026-07)

- `sony_a7m3_ssf_weta_measured.csv`: Sony A7 III (IMX410 colour, same sensor as Sigma fp)
  full-camera SSF measured by Weta Digital, 380-780nm/5nm, from AMPAS rawtoaces-data
  (https://github.com/AcademySoftwareFoundation/rawtoaces-data). Includes the A7III
  filter stack, so pair it with `unit_transmission.csv` — do NOT multiply the hot-mirror
  model on top. Caveat: the fp's stock hot mirror differs slightly (Kolari teardown).
- `unit_transmission.csv`: flat transmission for integrated full-camera SSFs.
- `rawtoaces_training_reflectance.csv`: the 190 AMPAS IDT training reflectances
  (rawtoaces-data), used to fit the per-illuminant camera->Rec2020 profiles.

Recalibrate with:
    python tools/calibrate_skin_matrix.py --preset-mode material \
      --imx410-qe-csv dngscan_assets/spectral/sony_a7m3_ssf_weta_measured.csv \
      --ir-transmission-csv dngscan_assets/spectral/unit_transmission.csv \
      --profile-csv dngscan_assets/spectral/rawtoaces_training_reflectance.csv
