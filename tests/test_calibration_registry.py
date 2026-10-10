# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dngscan import calibration_registry as registry, priors, calibration
from tests.test_user_calibration import collect_profile
from dngscan.gui import calibration_service
from dngscan.noise_model import _prior_spectrum, model_from_prior
from dataclasses import replace
from tests.test_noise_model import frame


class CalibrationRegistryTests(unittest.TestCase):
    def test_fixed_catalog_preserves_retirement_dependencies_and_specific_scales(self):
        value=registry._read(registry.PACKAGED);rows={v["id"]:v for v in value["sets"]}
        self.assertEqual(len(rows),13)
        self.assertEqual(value["source"]["commit"],registry.SOURCE_COMMIT)
        self.assertEqual(rows["s1r2-mech-20260812"]["capability_status"]["read_noise"],"retired")
        self.assertEqual(rows["s1r2-elec-12bit-20260820"]["derivation"]["ptc_gain_scale"],4)
        self.assertEqual(rows["s5m2-mech-12bit-20260820"]["derivation"]["ptc_gain_scale"],1)
        self.assertEqual(rows["a7rm6-elec-srhq-20260820"]["derivation"]["ptc_gain_scale"],.5)
        self.assertEqual(rows["a7rm4a-mech-20260820"]["derivation"]["gain_method"],"nominal-ISO")
        self.assertEqual(rows["m11m-mech-20260820"]["channel_semantics"],"monochrome-position-slots")
        self.assertFalse(any("s1r2-mech-20260812" in v.get("source","") for v in priors._jptc_entries()))

    def test_cycles_missing_nodes_bad_units_and_tamper_refuse(self):
        raw={"sets":[{"id":"a","gainFromSet":"b"},{"id":"b","ptcFromSet":"a"}]}
        with self.assertRaisesRegex(ValueError,"cycle"):registry.compile_catalog(raw,source_commit="a"*40,raw_sha256="b"*64)
        raw["sets"].pop()
        with self.assertRaisesRegex(ValueError,"does not exist"):registry.compile_catalog(raw,source_commit="a"*40,raw_sha256="b"*64)
        with self.assertRaisesRegex(ValueError,"positive"):
            registry.compile_catalog({"sets":[{"id":"a","ptcGainScale":0}]},source_commit="a"*40,raw_sha256="b"*64)
        value=copy.deepcopy(registry._read(registry.PACKAGED));value["source"]["commit"]="a"*40
        with self.assertRaisesRegex(ValueError,"identity"):registry.validate_release(value)

    def test_offline_install_and_rollback_change_identity(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,DNGSCAN_CALIBRATION_LIBRARY_DIR=folder):
            initial=registry.library_fingerprint()
            value=registry.compile_catalog({"sets":[{"id":"test","camera":{"brand":"Test","model":"body"}}]},source_commit="a"*40,raw_sha256="b"*64)
            path=Path(folder)/"release-input.json";path.write_text(json.dumps(value))
            installed=registry.install_library(path)
            self.assertEqual(installed["version"],value["version"])
            self.assertNotEqual(initial,registry.library_fingerprint())
            registry.select_version("packaged");self.assertEqual(initial,registry.library_fingerprint())

    def test_capability_composition_requires_exact_physical_contract_and_keeps_spectrum(self):
        base={"make_equals":"TEST","model_equals":["BODY"],"shutter":"mechanical","readout_match_status":"matched",
            "readout_contract":{"version":1,"storage_lossless":True,"libraw_raw_geometry":[4000,3000]},
            "reference_dn_range":3995.,"acquisition_contract":{"linearisation_curve":"identity"},
            "gain_log2iso_log2epd":[[math.log2(100),1.]],"source":"independent PTC"}
        dark={**base,"gain_log2iso_log2epd":[],"read_noise_dn_log2iso":[[math.log2(100),3.]],
            "noise_whiteness_h_log2iso":[[math.log2(100),.1]],"source":"individual dark"}
        joined=registry.compose_capabilities(base,[dark],100)
        self.assertEqual(joined["capability_sources"],{"gain":"independent PTC","read_noise":"individual dark"})
        self.assertAlmostEqual(priors.read_noise_e(joined,100),6.)
        self.assertEqual(_prior_spectrum(joined,100)[0],"measured-spectral-imbalance")
        for key,value in (("reference_dn_range",4000.),("shutter","electronic"),("acquisition_contract",{})):
            self.assertNotIn("capability_sources",registry.compose_capabilities(base,[{**dark,key:value}],100))
        rejected={**base,"read_noise_unresolved_isos":[100]}
        self.assertNotIn("capability_sources",registry.compose_capabilities(rejected,[dark],100))

    def test_corrupt_library_disables_only_public_source_and_gui_can_roll_back(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,DNGSCAN_CALIBRATION_LIBRARY_DIR=folder):
            (Path(folder)/"current.json").write_text('{"version":"broken"}')
            with self.assertWarnsRegex(RuntimeWarning,"disabled"):
                self.assertEqual(priors._jptc_entries(),[])
            self.assertIn("error",calibration_service.public_library_status({})["library"])
            with patch.object(calibration_service,"_invalidate_preview"):
                restored=calibration_service.select_public_library({"version":"packaged"})
            self.assertNotIn("error",restored["library"])
            self.assertTrue(priors._jptc_entries())

    def test_gui_installs_selected_offline_release_and_invalidates_preview(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,DNGSCAN_CALIBRATION_LIBRARY_DIR=folder):
            value=registry._read(registry.PACKAGED)
            with patch.object(calibration_service,"_invalidate_preview") as invalidate:
                result=calibration_service.install_public_library({"files":[{"name":"library.json","text":json.dumps(value)}]})
            self.assertEqual(result["library"]["selection"],value["version"])
            invalidate.assert_called_once()
            weak=copy.deepcopy(value)
            weak["profiles"]=[{"set_id":weak["sets"][-1]["id"],"qualification":"mode-validated-statistical-model",
                "payload":collect_profile(),"source_evidence":{}}]
            weak["profiles"][0]["sha256"]=registry.digest(weak["profiles"][0]["payload"])
            weak["version"]=registry.digest({k:v for k,v in weak.items() if k!="version"})[:24]
            with self.assertRaisesRegex(ValueError,"qualified readout"):
                registry.validate_release(weak)

    def test_installed_partial_dark_reaches_production_selection_and_spectral_constraint(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,DNGSCAN_CALIBRATION_DIR=str(Path(folder)/"user")):
            base=collect_profile();base["read_noise_log2iso_log2e"]=[]
            base["readout_contract"]={"version":1,"libraw_raw_geometry":[4000,3000],"storage_lossless":True}
            base["acquisition_contract"]={"linearisation_curve":"identity"}
            base["reference_dn_range"]=15359.
            dark=copy.deepcopy(base)
            for key in ("ptc_anchor","fwc_e","gain_jump_isos"):dark.pop(key,None)
            dark.update(id="partial dark",gain_log2iso_log2epd=[],read_noise_dn_log2iso=[[math.log2(100),1.5]],noise_whiteness_h_log2iso=[[math.log2(100),.1]])
            for index,payload in enumerate((base,dark)):
                path=Path(folder)/f"profile{index}.json";path.write_text(json.dumps(payload));calibration.import_calibration(path)
            capture={"version":1,"shutter":"electronic","libraw_raw_geometry":[4000,3000],"storage_lossless":True}
            selected=priors.find_priors("SIGMA","fp",shutter="electronic",iso=100,readout=capture)
            self.assertIn("capability_sources",selected)
            self.assertAlmostEqual(priors.read_noise_e(selected,100),6.)
            self.assertEqual(_prior_spectrum(selected,100)[0],"measured-spectral-imbalance")
            # The independent spectrum still constrains a complete variance model.
            complete={**selected,"source":"complete PTC","independent_constraint_priors":[]}
            self.assertEqual(_prior_spectrum(registry.compose_capabilities(complete,[next(c for c in selected["independent_constraint_priors"] if c.get("noise_whiteness_h_log2iso"))],100),100)[0],"measured-spectral-imbalance")

    def test_partial_release_cannot_resurrect_or_implicitly_load_packaged_collect_assets(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, DNGSCAN_CALIBRATION_LIBRARY_DIR=folder):
            packaged = registry._read(registry.PACKAGED)
            for sets in ([], [dict(row, calibrationOnly=False) for row in packaged["sets"]]):
                with self.subTest(empty=not sets):
                    value = registry.compile_catalog({"sets": sets}, source_commit="a"*40, raw_sha256="b"*64)
                    path = Path(folder)/"release.json"; path.write_text(json.dumps(value))
                    registry.install_library(path)
                    collect = [row for row in priors._jptc_entries() if row.get("source", "").startswith("JPTC collect set")]
                    self.assertFalse(any("s1r2-mech-20260812" in row["source"] for row in collect))
                    if not sets:
                        self.assertEqual(collect, [])
            registry.select_version("packaged")
            self.assertTrue(any(row.get("source", "").startswith("JPTC collect set") for row in priors._jptc_entries()))

    def test_malformed_catalogue_records_disable_public_library_without_crashing(self):
        original = registry._read(registry.PACKAGED)
        mutations = [("sets", [7]), ("sets", [{"id": "bad", "camera": []}]),
            ("sets", [{"id": "bad", "gainFromSet": []}]), ("profiles", [7]), ("source", [])]
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, DNGSCAN_CALIBRATION_LIBRARY_DIR=folder):
            for field, replacement in mutations:
                with self.subTest(field=field, replacement=replacement):
                    value = copy.deepcopy(original); value[field] = replacement
                    value["version"] = registry.digest({k:v for k,v in value.items() if k != "version"})[:24]
                    with self.assertRaises(ValueError): registry.validate_release(value)
                    path = Path(folder)/(value["version"]+".json"); path.write_text(json.dumps(value))
                    (Path(folder)/"current.json").write_text(json.dumps({"version":value["version"]}))
                    self.assertIn("error", registry.library_status())
                    with self.assertWarnsRegex(RuntimeWarning, "disabled"):
                        self.assertEqual(priors._jptc_entries(), [])
            (Path(folder)/"current.json").write_text("[]")
            self.assertIn("error", registry.library_status())
            with self.assertWarnsRegex(RuntimeWarning, "disabled"):
                self.assertEqual(priors._jptc_entries(), [])

    def test_catalogue_without_measurements_does_not_claim_measured_gain(self):
        value = registry.compile_catalog({"sets":[{"id":"no-measurements"}]}, source_commit="a"*40, raw_sha256="b"*64)
        self.assertEqual(value["sets"][0]["capability_status"]["gain"], "unmeasured")
        self.assertEqual(value["profiles"], [])
        self.assertIn("catalogue-only", value["status"])

    def test_composed_dark_keeps_stored_total_variance_through_production_noise_model(self):
        base = collect_profile(); base["read_noise_log2iso_log2e"] = []
        base["readout_contract"] = {"version":1,"libraw_raw_geometry":[4000,3000],"storage_lossless":True}
        base["acquisition_contract"] = {"linearisation_curve":"identity", "adc_step":"4", "sigma_clip_correction":"applied",
            "stored_dark_variance_measurement":"paired-frame-pre-sheppard", "stored_dark_variance_domain":"linearized-raw-dn"}
        base["reference_dn_range"] = 15359.
        dark = copy.deepcopy(base)
        for key in ("ptc_anchor","fwc_e","gain_jump_isos"): dark.pop(key,None)
        total = 1.5**2 + 4**2/12
        dark.update(gain_log2iso_log2epd=[], read_noise_dn_log2iso=[[math.log2(100),1.5]],
            stored_dark_variance_dn2_log2iso=[[math.log2(100),total]],
            noise_whiteness_v_log2iso=[[math.log2(100),.1]])
        base, dark = calibration._validated_prior(base), calibration._validated_prior(dark)
        base.update(readout_match_status="matched",source="gain only")
        dark.update(readout_match_status="matched",source="paired dark")
        joined = registry.compose_capabilities(base,[dark],100)
        self.assertEqual(calibration.stored_dark_variance(joined,100),(total,"measured-stored-dark-variance"))
        self.assertEqual(joined["capability_sources"]["stored_dark_variance"], "paired dark")
        capture = {"version":1,"shutter":"electronic","libraw_raw_geometry":[4000,3000],"storage_lossless":True}
        bundle = replace(frame(),shot_iso=100,shot_shutter="electronic",capture_readout=capture,
            white_level=16383,black_levels=[1024.]*4,camera_white_levels=[16383.]*4)
        model = model_from_prior(bundle,{i:16383. for i in range(4)},joined)
        self.assertEqual(model.status,"valid")
        self.assertAlmostEqual(model.coefficients("G1")[1],total/15359.**2,places=17)
        self.assertEqual(model.correlation,"measured-spectral-imbalance")
        self.assertIsNone(priors.read_noise_e(joined,200))  # Composition creates no new valid ISO span.
        self.assertNotIn("capability_sources",registry.compose_capabilities(base,[dark],50))
        blocked = dict(dark,read_noise_unresolved_isos=[100],read_noise_dn_log2iso=[])
        self.assertNotIn("capability_sources",registry.compose_capabilities(base,[blocked],100))

    def test_composition_retains_original_stored_variance_samples_and_interpolation_status(self):
        x0,x1 = math.log2(100),math.log2(400)
        base = {"make_equals":"TEST","model_equals":["BODY"],"shutter":"mechanical","readout_match_status":"matched",
            "readout_contract":{"version":1,"storage_lossless":True,"libraw_raw_geometry":[4000,3000]},
            "reference_dn_range":3995.,"acquisition_contract":{"linearisation_curve":"identity", "sigma_clip_correction":"applied",
                "stored_dark_variance_measurement":"paired-frame-pre-sheppard","stored_dark_variance_domain":"linearized-raw-dn"},
            "source_format":"dngscan-jptc-collect-1","gain_log2iso_log2epd":[[x0,1.],[x1,0.]],"source":"gain"}
        dark = {**base,"gain_log2iso_log2epd":[],"source":"dark", "read_noise_dn_log2iso":[[x0,1.],[x1,2.]],
            "stored_dark_variance_dn2_log2iso":[[x0,2.],[x1,5.]]}
        joined = registry.compose_capabilities(base,[dark],200)
        self.assertEqual(joined["stored_dark_variance_dn2_log2iso"],dark["stored_dark_variance_dn2_log2iso"])
        variance,status = calibration.stored_dark_variance(joined,200)
        self.assertAlmostEqual(variance,3.5)
        self.assertEqual(status,"interpolated-stored-dark-variance")
        self.assertEqual(joined["stored_dark_variance_composition"]["status"],"interpolated-stored-dark-variance")
        self.assertIsNone(priors.read_noise_e(joined,100))
        self.assertNotIn("capability_sources",registry.compose_capabilities(base,[dict(dark,gain_jump_isos=[400])],200))
        contradictory = dict(dark,stored_dark_variance_dn2_log2iso=[[x0,.1],[x1,.2]])
        self.assertNotIn("capability_sources",registry.compose_capabilities(base,[contradictory],200))


if __name__=="__main__":unittest.main()
