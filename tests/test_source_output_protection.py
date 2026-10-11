# SPDX-License-Identifier: GPL-3.0-or-later
"""Only disposable synthetic DNGs may exercise destructive export counterexamples."""
import hashlib
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.analysis import analyze
from dngscan.delivery import resolve_delivery_profile
from dngscan.delivery_transaction import DeliveryTransaction
from dngscan import export
from dngscan.protected_output import ProtectedSources, SourceOutputConflictError
from dngscan.raw_io import load_raw
from dngscan.report import write_csv
from tests.test_spectral_fallback_pipeline import write_noise_dng


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceOutputProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='agxraw-source-output-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.dng'
        write_noise_dng(self.source)
        self.original = digest(self.source)
        self.env = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root/'no-calibration'))
        self.env.start()
        self.addCleanup(self.env.stop)

    def aliases(self):
        folder = self.root/'sub'
        folder.mkdir()
        symlink = self.root/'symlink.jpg'
        symlink.symlink_to(self.source)
        hardlink = self.root/'hardlink.jpg'
        os.link(self.source, hardlink)
        directory = self.root/'directory-link'
        directory.symlink_to(self.root, target_is_directory=True)
        return [self.source, folder/'..'/'source.dng', symlink, hardlink, directory/'source.dng']

    def test_path_and_existing_file_aliases_are_all_rejected(self):
        guard = ProtectedSources(self.source)
        for destination in self.aliases():
            with self.subTest(destination=destination.name):
                with self.assertRaises(SourceOutputConflictError):
                    guard.check(destination)
        self.assertEqual(digest(self.source), self.original)

    def test_transaction_checks_before_staging_and_again_on_commit(self):
        with self.assertRaises(SourceOutputConflictError):
            with DeliveryTransaction(self.source, protected_sources=[self.source]):
                self.fail('preflight must happen before staging')
        for kind in ('symlink','hardlink'):
            destination = self.root/(kind+'.jpg')
            with self.subTest(kind=kind), self.assertRaises(SourceOutputConflictError):
                with DeliveryTransaction(destination, protected_sources=[self.source]) as transaction:
                    transaction.path.write_bytes(b'private candidate')
                    if kind == 'symlink':destination.symlink_to(self.source)
                    else:os.link(self.source, destination)
                    transaction.commit()
            self.assertEqual(digest(self.source), self.original)
            self.assertEqual(digest(destination), self.original)
        self.assertFalse(list(self.root.glob('.agxraw-delivery-*')))

    def test_original_inode_remains_protected_after_input_path_replacement(self):
        guard = ProtectedSources(self.source)
        original_alias = self.root/'old-source.dng'
        self.source.rename(original_alias)
        self.source.write_bytes(b'new source version')
        destination = self.root/'old-hardlink.jpg'
        os.link(original_alias, destination)
        with self.assertRaises(SourceOutputConflictError):guard.check(destination)
        self.assertEqual(digest(original_alias), self.original)
        self.assertEqual(self.source.read_bytes(), b'new source version')

    def test_relative_and_absolute_alias_match(self):
        relative = Path(os.path.relpath(self.source, Path.cwd()))
        with self.assertRaises(SourceOutputConflictError):ProtectedSources(relative).check(self.source)
        with self.assertRaises(SourceOutputConflictError):ProtectedSources(self.source).check(relative)

    def test_input_symlink_protects_the_original_target(self):
        source_link = self.root/'input-link.dng'
        source_link.symlink_to(self.source)
        with self.assertRaises(SourceOutputConflictError):
            export.export_jpeg(source_link,self.source,100,None,None)
        self.assertEqual(digest(self.source),self.original)

    def test_finished_pair_preserves_one_shot_source_iterables(self):
        from dngscan import gainmap
        pair=SimpleNamespace(sdr_rgb=None,hdr_rgba_f16=None,display_headroom_ev=1.)
        out=self.root/'pair.jpg'
        def writer(*args,**kwargs):
            with self.assertRaises(SourceOutputConflictError):
                ProtectedSources(kwargs['protected_sources']).check(self.source)
            return {'retained_protection':True}
        with patch.object(gainmap,'write_apple_gainmap_file',side_effect=writer):
            result=gainmap.encode_finished_pair(pair,out,None,
                protected_sources=(p for p in [self.source]))
        self.assertTrue(result['retained_protection'])

    def test_protected_gainmap_transaction_reports_published_path(self):
        from dngscan import gainmap
        output=self.root/'finished.heic'
        profile=resolve_delivery_profile('auto',container='heic')
        def write(prepared,path,**kwargs):
            path.write_bytes(b'verified private candidate')
            return {'output_path':str(path),'verified':True}
        with patch.object(gainmap,'_PreparedGainmapMaster'), \
             patch.object(gainmap,'_write_gainmap_prepared',side_effect=write):
            result=gainmap.write_apple_gainmap_file(None,None,output,1.,
                delivery=profile,protected_sources=[self.source])
        self.assertEqual(result['output_path'],str(output))
        self.assertEqual(output.read_bytes(),b'verified private candidate')
        self.assertEqual(digest(self.source),self.original)

    def test_real_fixed_and_auto_jpeg_api_reject_aliases_and_allow_safe_output(self):
        bundle = load_raw(self.source, scene_half_size=True)
        analysis,_,_ = analyze(bundle, 4, diagnostics=False)
        for profile in (None, resolve_delivery_profile('auto')):
            for destination in self.aliases() if profile is None else [self.source]:
                with self.subTest(profile=profile.name if profile else 'fixed',destination=destination.name):
                    with self.assertRaises(SourceOutputConflictError):
                        export.export_jpeg(self.source,destination,100,bundle,analysis,delivery=profile)
                    self.assertEqual(digest(self.source),self.original)
            out = self.root/('safe-auto.jpg' if profile else 'safe-fixed.jpg')
            export.export_jpeg(self.source,out,100,bundle,analysis,delivery=profile)
            self.assertTrue(out.read_bytes().startswith(b'\xff\xd8'))
            self.assertEqual(digest(self.source),self.original)

    def test_bundle_source_cannot_be_bypassed_with_different_metadata_path(self):
        bundle=SimpleNamespace(path=self.source,evidence=SimpleNamespace(path=self.source))
        with patch.object(export,'render_output_u8') as render:
            with self.assertRaises(SourceOutputConflictError):
                export.export_jpeg(self.root/'metadata-only.dng',self.source,100,bundle,None)
            render.assert_not_called()

    def test_requested_and_final_container_paths_checked_before_render(self):
        for fmt in ('sdr','sdr-heic','ultrahdr','ultrahdr-heic'):
            with self.subTest(format=fmt), patch.object(export,'render_output_u8') as render:
                with self.assertRaises(SourceOutputConflictError):
                    export.export_jpeg(self.source,self.source,100,None,None,output_format=fmt)
                render.assert_not_called()
        for fmt,source_suffix,requested_suffix in (
                ('sdr-heic','.heic','.jpg'),('ultrahdr-heic','.heic','.jpg'),
                ('ultrahdr','.jpg','.heic')):
            source=self.root/('suffix-source'+source_suffix)
            source.write_bytes(b'disposable RAW placeholder')
            requested=source.with_suffix(requested_suffix)
            with self.subTest(final_format=fmt), self.assertRaises(SourceOutputConflictError):
                export.export_jpeg(source,requested,100,None,None,output_format=fmt)
            self.assertEqual(source.read_bytes(),b'disposable RAW placeholder')

    def test_direct_source_aware_entrypoints_preflight_without_optional_backends(self):
        from dngscan.heif_delivery import save_sdr_heif
        from dngscan import heif_encoder
        from dngscan.gainmap import write_apple_gainmap_file, encode_finished_pair, encode_finished_pair_jpeg
        from dngscan.auto_encode import select_encoding, select_heif_encoding
        from dngscan.plot import plot_dashboard
        rgb=np.zeros((2,2,3),np.uint8)
        calls=[
            lambda:export.export_srgb_jpeg(self.source,self.source,100,None,None),
            lambda:export.export_ultrahdr_jpeg(self.source,self.source,100,None,None),
            lambda:save_sdr_heif(rgb,self.source,None,source_raw=self.source),
            lambda:export.save_jpeg_array(rgb,self.source,100,protected_sources=[self.source]),
            lambda:heif_encoder.encode(rgb,self.source,95,protected_sources=[self.source]),
            lambda:heif_encoder.encode_apple(rgb,self.source,95,protected_sources=[self.source]),
            lambda:write_apple_gainmap_file(None,None,self.source,1.,delivery=None,protected_sources=[self.source]),
            lambda:encode_finished_pair(None,self.source,None,protected_sources=[self.source]),
            lambda:encode_finished_pair_jpeg(None,self.source,None,protected_sources=[self.source]),
            lambda:select_encoding(self.source,None,protected_sources=[self.source]),
            lambda:select_heif_encoding(self.source,None,protected_sources=[self.source]),
            lambda:plot_dashboard(SimpleNamespace(path=self.source),None,None,None,self.source),
            lambda:export.carry_capture_metadata(self.source,self.source),
            lambda:export.carry_capture_metadata_hdr(self.source,self.source,'jpeg'),
        ]
        for i,call in enumerate(calls):
            with self.subTest(entry=i), self.assertRaises(SourceOutputConflictError):call()
            self.assertEqual(digest(self.source),self.original)

    def test_fixed_export_rechecks_destination_after_render(self):
        bundle=load_raw(self.source,scene_half_size=True)
        analysis,_,_=analyze(bundle,4,diagnostics=False)
        out=self.root/'changed-after-render.jpg'
        original_render=export.render_output_u8
        def render(*args,**kwargs):
            pixels=original_render(*args,**kwargs)
            os.link(self.source,out)
            return pixels
        with patch.object(export,'render_output_u8',side_effect=render):
            with self.assertRaisesRegex(RuntimeError,'禁止覆盖'):
                export.export_jpeg(self.source,out,100,bundle,analysis)
        self.assertEqual(digest(self.source),self.original)
        self.assertEqual(digest(out),self.original)

    def test_csv_declared_source_and_legacy_csv_row_source_are_protected(self):
        for destination in self.aliases():
            with self.subTest(destination=destination.name), self.assertRaises(SourceOutputConflictError):
                write_csv(destination,{'file':str(self.source),'metric':1})
        with self.assertRaises(SourceOutputConflictError):write_csv(self.source,{'metric':1},protected_sources=[self.source])
        safe=self.root/'report.csv'
        write_csv(safe,{'file':str(self.source),'metric':1})
        self.assertIn('metric',safe.read_text())
        self.assertEqual(digest(self.source),self.original)

    def test_automatic_candidate_search_never_swallows_source_conflicts(self):
        from dngscan.auto_encode import select_encoding, select_heif_encoding
        out=self.root/'candidate.jpg'
        out.write_bytes(b'previous delivery')
        for stage in ('reference','later','420'):
            calls=[]
            def encode(quality,path):
                calls.append(quality)
                if stage=='reference' or (stage=='later' and quality==98):
                    raise SourceOutputConflictError('source conflict')
                path.write_bytes(bytes([quality])*1000)
                return {'delivery_quality':quality,'delivery_chroma_requested':'422'}
            def encode_420(quality,path):
                raise SourceOutputConflictError('source conflict')
            with self.subTest(jpeg_stage=stage), self.assertRaises(SourceOutputConflictError):
                select_encoding(out,encode,qualities=(99,98),
                                encode_420=encode_420 if stage=='420' else None,
                                protected_sources=[self.source])
            self.assertEqual(out.read_bytes(),b'previous delivery')
            if stage=='reference':self.assertEqual(calls,[99])
        for stage in ('reference','later'):
            def encode_heif(quality,chroma,auxiliary,path):
                if stage=='reference' or quality!=95:
                    raise SourceOutputConflictError('source conflict')
                path.write_bytes(b'private reference')
                return {'delivery_quality':quality,'delivery_chroma_requested':chroma}
            with self.subTest(heif_stage=stage), self.assertRaises(SourceOutputConflictError):
                select_heif_encoding(out,encode_heif,protected_sources=[self.source])
            self.assertEqual(out.read_bytes(),b'previous delivery')
        self.assertEqual(digest(self.source),self.original)
        self.assertFalse(list(self.root.glob('.agxraw-*')))

    def test_gui_rejects_main_and_dashboard_aliases_before_writers(self):
        from dngscan.gui import service
        from tests.test_preview_cache import _analysis, _bundle
        for dashboard_conflict in (False,True):
            destination=self.root/('source_simple_pstable_scan.png' if dashboard_conflict
                                   else 'source_simple_pstable.jpg')
            os.link(self.source,destination)
            bundle=_bundle()
            bundle.path=self.source
            with ExitStack() as stack:
                stack.enter_context(patch.object(service.dg,'require_dependencies'))
                stack.enter_context(patch.object(service.PREVIEW_STORE,'peek',return_value=None))
                stack.enter_context(patch.object(service,'_load_export_scene',return_value=bundle))
                stack.enter_context(patch.object(service,'_cached_full_analysis',return_value=None))
                stack.enter_context(patch.object(service.dg,'analyze',return_value=(_analysis(),None,None)))
                stack.enter_context(patch.object(service.dg,'build_render_plan',return_value=object()))
                stack.enter_context(patch.object(service,'export_suffix_parts',return_value='simple'))
                stack.enter_context(patch.object(service,'export_plan_fingerprint',return_value='stable'))
                writer=stack.enter_context(patch.object(service.dg,'export_jpeg'))
                plot=stack.enter_context(patch.object(service.dg,'plot_dashboard'))
                with self.subTest(dashboard=dashboard_conflict), self.assertRaises(SourceOutputConflictError):
                    service.run_export({'input':str(self.source),'outdir':str(self.root),
                                        'evAuto':False,'png':True})
                writer.assert_not_called()
                plot.assert_not_called()
            self.assertEqual(digest(self.source),self.original)
            destination.unlink()

    def test_cli_preflights_all_outputs_before_any_write(self):
        png=self.root/'existing.png'
        png.write_bytes(b'previous dashboard')
        for opts in (
                ['--jpeg',str(self.source),'--jpeg-quality','100','--chroma','444'],
                ['--jpeg',str(self.source),'--delivery-profile','auto'],
                ['--out',str(png),'--csv',str(self.source)],
                ['--jpeg',str(self.source),'--output-format','sdr-heic']):
            with self.subTest(options=opts):
                p=subprocess.run([sys.executable,'-m','dngscan',str(self.source),*opts,'--ev','0'],
                    capture_output=True,text=True,timeout=30)
                self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
                self.assertIn('禁止覆盖',p.stderr)
                self.assertEqual(digest(self.source),self.original)
                self.assertEqual(png.read_bytes(),b'previous dashboard')
        for kind in ('symlink','hardlink'):
            target=self.root/('cli-'+kind+'.png')
            if kind=='symlink':target.symlink_to(self.source)
            else:os.link(self.source,target)
            p=subprocess.run([sys.executable,'-m','dngscan',str(self.source),'--out',str(target)],
                capture_output=True,text=True,timeout=30)
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(digest(self.source),self.original)


if __name__=='__main__':unittest.main()
