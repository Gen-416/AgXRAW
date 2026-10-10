# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent heldout system correction, never a fabricated camera calibration."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dngscan import pixel_calibration as pc


class PersonalPixelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(809)
        self.conditions = dict(make="SIGMA", model="fp", body_serial="test-body", readout_identity="test-mode",
            decoder_identity="test-linear-unpack", linearisation_identity="identity", domain="linearized-raw-dn-before-stage2",
            iso=640., exposure_seconds=[.001,.01], cfa_pattern="RGGB", origin=[8,12], shape=[24,32])
        self.conditions["metadata_black_identity"] = pc.metadata_black_identity(np.array(100.), (24, 32))
        row, col = np.indices((24,32)); bias = 2.+.4*np.sin(row)+.7*np.cos(col)
        self.train = 100.+bias+self.rng.normal(0,.4,(32,24,32))
        self.test = 100.+bias+self.rng.normal(0,.4,(16,24,32))
        self.source = {"training":"dark", "heldout":"dark", "training_hashes":[hashlib.sha256(f"train{i}".encode()).hexdigest() for i in range(32)],
            "heldout_hashes":[hashlib.sha256(f"test{i}".encode()).hexdigest() for i in range(16)], "kind":"synthetic-validation"}
        self.reference = {"phase_dn":{f"C{r}{c}":float(bias[r::2,c::2].mean()) for r in (0,1) for c in (0,1)},
            "phase_uncertainty_dn":dict.fromkeys(pc.PHASES,.01), "method":"independent synthetic zero", "independent_zero_evidence":True}

    def product(self, **options):
        return pc.build_dark_product(self.train,self.test, conditions=self.conditions, source=self.source,
            metadata_black=np.array(100.), black_reference=self.reference, **options)

    def capture(self):
        return {**self.conditions,"exposure_seconds":.004}

    def test_observed_mean_alone_is_not_zero_and_no_automatic_application(self):
        product = pc.build_dark_product(self.train,self.test,conditions=self.conditions,source=self.source,metadata_black=np.array(100.))
        self.assertEqual(product["manifest"]["qualification"],"observed-only")
        self.assertIsNone(product["manifest"]["diagnostics"]["qualified_black_reference_dn"])
        with self.assertRaisesRegex(ValueError,"experimental=True"): pc.apply_signed_product(self.test[0]-100,product,self.capture())
        with self.assertRaisesRegex(ValueError,"unqualified"): pc.apply_signed_product(self.test[0]-100,product,self.capture(),experimental=True,input_state="metadata-black-subtracted-signed-dn",metadata_black=np.array(100.))

    def test_independent_holdout_improves_fixed_error_preserves_random_variance_and_negative_tail(self):
        product = self.product(); diagnostics = product["manifest"]["diagnostics"]
        self.assertEqual(product["manifest"]["qualification"],"heldout-supported-experiment")
        self.assertLess(diagnostics["heldout_fixed_rmse_after_dn"], diagnostics["heldout_fixed_rmse_before_dn"]*.1)
        raw=self.test-100; saved=raw.copy()
        corrected=np.stack([pc.apply_signed_product(frame,product,self.capture(),experimental=True,input_state="metadata-black-subtracted-signed-dn",metadata_black=np.array(100.))[0] for frame in raw])
        np.testing.assert_array_equal(raw,saved)
        np.testing.assert_allclose(corrected.var(axis=0),raw.var(axis=0),rtol=1e-12,atol=1e-14)
        self.assertGreater(float(np.mean(corrected<0)),.3)
        self.assertGreater(diagnostics["calibration_mean_variance_dn2"],0)
        self.assertFalse(product["manifest"]["automatic_application"])

    def test_identity_body_origin_domain_repeat_and_clipping_refuse(self):
        product = self.product()
        for key,value in (("body_serial","other-body"),("origin",[0,0]),("decoder_identity","other"),("iso",100), ("exposure_seconds",1.)):
            with self.subTest(key=key):
                self.assertFalse(pc.match_product(product,{**self.capture(),key:value})[0])
        with self.assertRaisesRegex(ValueError,"already-applied"):
            pc.apply_signed_product(self.test[0]-100,product,self.capture(),experimental=True,input_state="metadata-black-subtracted-signed-dn",metadata_black=np.array(100.),applied_corrections=[product["manifest"]["identity"]])
        source=copy.deepcopy(self.source);source["heldout_hashes"][0]=source["training_hashes"][0]
        with self.assertRaisesRegex(ValueError,"independent"):
            pc.build_dark_product(self.train,self.test,conditions=self.conditions,source=source,metadata_black=np.array(100.))
        clipped=self.train.copy();clipped[0,0,0]=0
        result=pc.build_dark_product(clipped,self.test,conditions=self.conditions,source=self.source,metadata_black=np.array(100.),black_reference=self.reference)
        self.assertEqual(result["manifest"]["qualification"],"observed-only")

    def test_persist_hashes_and_tamper_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"dark.npz";product=self.product();pc.save_product(path,product)
            restored=pc.load_product(path);self.assertEqual(product["manifest"],restored["manifest"])
            restored["arrays"]["residual_dn"][0,0]+=1
            with self.assertRaisesRegex(ValueError,"checksum"):pc.validate_product(restored)

    def test_flat_separate_light_domain_duplicate_shading_and_variance_scaling(self):
        dark=self.product();row,col=np.indices((24,32));field=1+.2*col/31
        train=1000*field+dark["arrays"]["residual_dn"]+self.rng.normal(0,.2,(32,24,32))
        test=1000*field+dark["arrays"]["residual_dn"]+self.rng.normal(0,.2,(16,24,32))
        source={**self.source,"training":"flat","heldout":"flat",
            "training_hashes":[hashlib.sha256(f"flat{i}".encode()).hexdigest() for i in range(32)],
            "heldout_hashes":[hashlib.sha256(f"flat-test{i}".encode()).hexdigest() for i in range(16)]}
        conditions={**self.conditions,"lens_identity":"test-lens","optical_stack":"bare", "aperture":4.,"focal_length_mm":35.,"focus_distance":"infinity","illumination_identity":"uniform-test-light"}
        flat=pc.build_flat_product(train,test,conditions=conditions,source=source,dark_product=dark,illumination_verified=True,input_state=pc.INPUT_STATE,metadata_black=np.array(100.))
        capture={**conditions,"exposure_seconds":.004}
        self.assertEqual(flat["manifest"]["qualification"],"heldout-supported-experiment")
        self.assertFalse(pc.match_product(flat,capture,applied_corrections=["DNG-GainMap"])[0])
        self.assertEqual(pc.match_product(flat,capture)[1],"matched-dark-correction-not-recorded")
        self.assertTrue(pc.match_product(flat,capture,applied_corrections=[dark["manifest"]["identity"]])[0])
        x=self.rng.normal(0,1,(2000,24,32));y=x*flat["arrays"]["spatial_gain"]
        np.testing.assert_allclose(y.var(axis=0),x.var(axis=0)*flat["arrays"]["spatial_gain"]**2,rtol=1e-6)


    def flat_inputs(self):
        dark = self.product()
        row, col = np.indices((24,32))
        field = 1 + .2*col/31
        train = 1000*field + dark["arrays"]["residual_dn"] + self.rng.normal(0,.2,(32,24,32))
        test = 1000*field + dark["arrays"]["residual_dn"] + self.rng.normal(0,.2,(16,24,32))
        source = {**self.source, "training":"flat", "heldout":"flat",
            "training_hashes":[hashlib.sha256(f"flat{i}".encode()).hexdigest() for i in range(32)],
            "heldout_hashes":[hashlib.sha256(f"held-flat{i}".encode()).hexdigest() for i in range(16)]}
        conditions = {**self.conditions, "lens_identity":"test-lens", "optical_stack":"bare", "aperture":4.,
            "focal_length_mm":35., "focus_distance":"infinity", "illumination_identity":"verified-flat-light"}
        options = dict(conditions=conditions, source=source, dark_product=dark, illumination_verified=True,
                       input_state=pc.INPUT_STATE, metadata_black=np.array(100.))
        return train, test, options

    def test_complete_black_field_identity_prevents_changed_baseline_application(self):
        product = self.product()
        spatial = np.full((24,32), 100.)
        self.assertEqual(pc.metadata_black_identity(np.array(100.), (24,32)),
                         pc.metadata_black_identity(spatial, (24,32)))
        spatial[8,9] += .125
        changed = pc.metadata_black_identity(spatial, (24,32))
        self.assertNotEqual(changed, self.conditions["metadata_black_identity"])
        capture = {**self.capture(), "metadata_black_identity":changed}
        self.assertEqual(pc.match_product(product, capture)[1], "metadata_black_identity-mismatch-or-unknown")
        with self.assertRaisesRegex(ValueError, "metadata_black_identity"):
            pc.apply_signed_product(self.test[0]-spatial,product,capture,experimental=True,
                input_state=pc.INPUT_STATE,metadata_black=spatial)
        # Supplying the old identity while declaring a different removed field is rejected too.
        with self.assertRaisesRegex(ValueError, "differs from the supplied field"):
            pc.apply_signed_product(self.test[0]-101,product,self.capture(),experimental=True,
                input_state=pc.INPUT_STATE,metadata_black=np.array(101.))
        without_identity = {k:v for k,v in self.capture().items() if k != "metadata_black_identity"}
        self.assertFalse(pc.match_product(product, without_identity)[0])
        with self.assertRaisesRegex(ValueError, "declared metadata black identity"):
            pc.build_dark_product(self.train,self.test,conditions=self.conditions,source=self.source,
                metadata_black=np.array(101.),black_reference=self.reference)

    def test_flat_rejects_tampered_dark_and_missing_input_domain(self):
        train, test, options = self.flat_inputs()
        altered = copy.deepcopy(options["dark_product"])
        altered["arrays"]["residual_dn"] += .2
        with self.assertRaisesRegex(ValueError,"checksum"):
            pc.build_flat_product(train,test,**{**options,"dark_product":altered})
        for state in (None,"unsubtracted-raw-dn","already-dark-corrected"):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError,"input state"):
                pc.build_flat_product(train,test,**{**options,"input_state":state})
        with self.assertRaisesRegex(ValueError,"complete metadata black"):
            pc.build_flat_product(train,test,**{**options,"metadata_black":None})

    def test_conflicting_absolute_and_residual_reference_modes_are_rejected(self):
        for mode, nested in (("absolute","residual"),("residual","absolute")):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError,"mode conflicts"):
                pc.build_dark_product(self.train,self.test,conditions=self.conditions,source=self.source,
                    metadata_black=np.array(100.),black_reference={**self.reference,"mode":nested},mode=mode)
        matching = pc.build_dark_product(
            self.train,self.test,conditions=self.conditions,source=self.source,metadata_black=np.array(100.),
            black_reference={**self.reference,"mode":"residual"},mode="residual")
        self.assertEqual(matching["manifest"]["diagnostics"]["qualified_black_reference_dn"]["mode"],"residual")
        for bound in (float("nan"),float("inf")):
            with self.assertRaisesRegex(ValueError,"lower coding bound"):
                pc.build_dark_product(self.train,self.test,conditions=self.conditions,source=self.source,
                    metadata_black=np.array(100.),black_reference=self.reference,raw_floor_dn=bound)

    def test_flat_keeps_shared_dark_uncertainty_and_declares_incomplete_budget(self):
        train, test, options = self.flat_inputs()
        flat = pc.build_flat_product(train,test,**options)
        budget = flat["manifest"]["diagnostics"]["uncertainty_budget"]
        self.assertEqual(budget["status"],"incomplete")
        self.assertFalse(budget["automatic_qualification"])
        dark_variance = options["dark_product"]["arrays"]["mean_uncertainty_dn"].astype(float)**2
        expected = train.var(axis=0,ddof=1)/len(train) + dark_variance
        np.testing.assert_allclose(flat["arrays"]["reference_mean_uncertainty_dn"]**2,expected,rtol=2e-7)
        self.assertTrue(np.all(flat["arrays"]["reference_mean_uncertainty_dn"]**2 >= dark_variance*.999999))
        capture = {**options["conditions"],"exposure_seconds":.004}
        corrected, record = pc.apply_signed_product(test[0]-options["dark_product"]["arrays"]["residual_dn"],flat,capture,
            experimental=True,input_state=pc.INPUT_STATE,metadata_black=np.array(100.),
            applied_corrections=[options["dark_product"]["manifest"]["identity"]])
        self.assertEqual(record["uncertainty_budget_status"],"incomplete")
        self.assertFalse(record["automatic_application"])
        self.assertTrue(np.isfinite(corrected).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"flat.npz"
            pc.save_product(path,flat)
            self.assertEqual(pc.load_product(path)["manifest"]["identity"],flat["manifest"]["identity"])


    def test_cli_flat_uses_same_explicit_domain_and_black_field_contract(self):
        train, test, options = self.flat_inputs()
        tool = Path(__file__).parents[1]/"tools/build_personal_calibration.py"
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            dark_path, input_path = folder/"dark.npz", folder/"flat-input.npz"
            contract_path, out_path = folder/"contract.json", folder/"flat-product.npz"
            pc.save_product(dark_path,options["dark_product"])
            np.savez(input_path,training=train,heldout=test,metadata_black=np.array(100.))
            contract = {key:value for key,value in options.items() if key not in ("dark_product","metadata_black")}
            contract.pop("input_state")
            contract_path.write_text(json.dumps(contract))
            command = [sys.executable,str(tool),"flat",str(input_path),"--contract",str(contract_path),
                       "--dark-product",str(dark_path),"--out",str(out_path)]
            failed = subprocess.run(command,text=True,capture_output=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn("input state",failed.stderr)
            self.assertFalse(out_path.exists())
            contract["input_state"] = pc.INPUT_STATE
            contract_path.write_text(json.dumps(contract))
            passed = subprocess.run(command,text=True,capture_output=True)
            self.assertEqual(passed.returncode,0,passed.stderr)
            loaded = pc.load_product(out_path)
            self.assertEqual(loaded["manifest"]["conditions"]["metadata_black_identity"],self.conditions["metadata_black_identity"])
            self.assertEqual(loaded["manifest"]["diagnostics"]["uncertainty_budget"]["status"],"incomplete")


if __name__=="__main__":unittest.main()
