# SPDX-License-Identifier: GPL-3.0-or-later
"""RAW acceptance tools must reject conflicting outputs before any decoding."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dngscan.protected_output import SourceOutputConflictError
from tools import validate_coreimage_raw as coreimage  # Also sets the tools import path.
from tools import validate_native_raw as native
from tools import validate_raw_failures as failures
from tools import validate_calibrated_pipeline as calibrated
from tools import calibrate_raw9_anchors as anchors


class RawToolOutputProtectionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='agxraw-tool-output-test-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.source=self.root/'source.dng'
        self.original=b'disposable synthetic RAW placeholder'
        self.source.write_bytes(self.original)
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'frames':[{'path':str(self.source)}]}))

    def aliases(self):
        symlink=self.root/'symlink.json'
        symlink.symlink_to(self.source)
        hardlink=self.root/'hardlink.json'
        os.link(self.source,hardlink)
        return [self.source,symlink,hardlink]

    def test_all_raw_tool_reports_reject_aliases_before_processing(self):
        entries=(
            (native,'run_file',lambda out:[str(self.source),'--out',str(out)]),
            (failures,'run_cases',lambda out:[str(self.source),'--out',str(out)]),
            (calibrated,'run_file',lambda out:[str(self.source),'--pictures',str(self.root/'pictures'),'--out',str(out)]),
            (coreimage,'run_case',lambda out:['--files',str(self.source),'--out',str(out)]),
            (anchors,'measure_candidate',lambda out:[str(self.manifest),'--report',str(out)]),
        )
        for destination in self.aliases():
            for module,function,arguments in entries:
                with self.subTest(tool=module.__name__,alias=destination.name), \
                     patch.object(module,function) as process:
                    with self.assertRaises(SourceOutputConflictError):
                        module.main(arguments(destination))
                    process.assert_not_called()
                self.assertEqual(self.source.read_bytes(),self.original)

    def test_coreimage_private_worker_also_protects_its_source(self):
        request=json.dumps({'path':str(self.source),'case':{}})
        with patch.object(coreimage,'run_case') as process:
            with self.assertRaises(SourceOutputConflictError):
                coreimage.main(['--worker-config',request,'--worker-result',str(self.source)])
            process.assert_not_called()
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_anchor_asset_installation_protects_declared_raws_and_candidate(self):
        for destination in self.aliases()+[self.manifest]:
            with self.subTest(destination=destination.name), patch.object(anchors,'install_candidate') as install:
                with self.assertRaises(SourceOutputConflictError):
                    anchors.main(['--install-candidate',str(self.manifest),'--asset',str(destination)])
                install.assert_not_called()
        with self.assertRaises(SourceOutputConflictError):
            anchors.install_candidate({'frames':[{'path':str(self.source)}]},self.source)
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_comparison_sheet_is_preflighted_before_any_raw_processing(self):
        pictures=self.root/'pictures'
        pictures.mkdir()
        sheet=pictures/'source-three-way.png'
        os.link(self.source,sheet)
        with patch.object(calibrated,'run_file') as process:
            with self.assertRaises(SourceOutputConflictError):
                calibrated.main([str(self.source),'--pictures',str(pictures),'--out',str(self.root/'safe.json')])
            process.assert_not_called()
        with patch.object(calibrated,'load_raw') as decode:
            with self.assertRaises(SourceOutputConflictError):
                calibrated.run_file(self.source,half=True,pictures=pictures,scratch=self.root)
            decode.assert_not_called()
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_report_commit_rechecks_alias_after_private_write(self):
        output=self.root/'report.json'
        write_text=Path.write_text
        def write(path,*args,**kwargs):
            result=write_text(path,*args,**kwargs)
            os.link(self.source,output)
            return result
        with patch.object(Path,'write_text',write):
            with self.assertRaises(SourceOutputConflictError):
                coreimage._atomic_json(output,{'status':'tested'},protected_sources=[self.source])
        self.assertEqual(self.source.read_bytes(),self.original)
        self.assertEqual(output.read_bytes(),self.original)
        self.assertFalse(list(self.root.glob('.agxraw-delivery-*')))

    def test_damage_copy_helper_cannot_target_original(self):
        for destination in self.aliases():
            with self.subTest(destination=destination.name), self.assertRaises(SourceOutputConflictError):
                failures.make_variant(self.source,destination,'invalid-header')
        self.assertEqual(self.source.read_bytes(),self.original)


if __name__=='__main__':unittest.main()
