# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dngscan import calibration, readout
from dngscan.cli import parse_args
from dngscan.analysis import analyze
from dngscan.raw_io import load_raw
from dngscan.gui.preview_cache import _cache_identity, _bundle_metadata
from tests.test_spectral_fallback_pipeline import write_noise_dng
from tests.test_readout_contract import edit_ifd
from tests.test_user_calibration import collect_profile


class CaptureDeclarationTests(unittest.TestCase):
    def test_cli_current_batch_drive_and_readout_remain_separate(self):
        args = parse_args(["example.dng", "--capture-shutter", "mechanical",
                           "--capture-readout-id", "readout-A", "--capture-kind", "still"])
        self.assertEqual(args.capture_declaration, {"version": 1, "shutter": "mechanical",
            "readout_id": "readout-A", "capture_kind": "still"})

    def test_fills_only_unknown_mode_without_mutating_evidence_and_conflicts_refuse(self):
        evidence={"shutter":None,"storage_lossless":False,"codec_id":"sony-arw2"}
        declaration={"version":1,"shutter":"mechanical","readout_id":"same-measurement-mode"}
        actual=readout.with_capture_declaration(evidence,declaration)
        self.assertIsNone(evidence["shutter"])
        self.assertFalse(actual["storage_lossless"])
        self.assertEqual(actual["shutter_source"],"user-current-batch-declaration")
        drive = readout.with_capture_declaration({"capture_kind": "still"},
            {"version": 1, "capture_kind": "STILL"})
        self.assertEqual(drive["capture_kind"], "still")
        self.assertFalse(drive["user_declaration"]["fields"]["capture_kind"]["filled_unknown"])
        with self.assertRaisesRegex(ValueError,"冲突"):
            readout.with_capture_declaration({"shutter":"electronic"},declaration)
        for invalid in ({"version":1,"storage_lossless":True},{"version":1,"sensor_bits":14}, {"shutter":"mechanical"}):
            with self.subTest(value=invalid),self.assertRaises(ValueError):readout.normalize_capture_declaration(invalid)

    def test_real_decoder_matches_declared_mode_analysis_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,DNGSCAN_CALIBRATION_DIR=str(Path(folder)/"store")):
            root=Path(folder); path=root/"unknown-mode.dng";write_noise_dng(path,file_profile=False)
            edit_ifd(path,{272:(2,5,b"fp L\0")})
            profile=collect_profile();profile["camera"]="SIGMA fp L";profile["model_candidates"]=["fp L"]
            profile["shutter"]="机械快门"
            entry=root/"profile.json";entry.write_text(json.dumps(profile));calibration.import_calibration(entry)
            unknown=load_raw(path,scene_half_size=True)
            declaration={"version":1,"shutter":"mechanical"}
            declared=load_raw(path,scene_half_size=True,capture_declaration=declaration)
            np.testing.assert_array_equal(unknown.scene_rec2020_render,declared.scene_rec2020_render)
            self.assertIsNone(unknown.shot_shutter);self.assertEqual(declared.shot_shutter,"mechanical")
            analysis,_,_=analyze(declared,4)
            self.assertEqual(analysis.noise_model.status,"valid")
            self.assertNotEqual(_cache_identity(path,"clip","camera"),_cache_identity(path,"clip","camera",capture_declaration=declaration))
            metadata=json.loads(json.dumps(_bundle_metadata(declared)))
            self.assertEqual(metadata["capture_readout"]["user_declaration"]["scope"],"current-batch")


if __name__=="__main__":unittest.main()
