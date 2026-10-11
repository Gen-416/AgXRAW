# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture facts are file-authored, immutable, scoped and JSON stable."""
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from dngscan import metadata, readout
from dngscan.capture_facts import read, from_dict, public_facts, fingerprint, LensIdentity
from dngscan.capture_identity import public_capture
from dngscan.raw_io import load_raw, release_analysis_buffers
from dngscan.analysis import analyze
from dngscan.source_metadata import source_metadata_session, SourceChangedError, SourceVerificationError
from dngscan.gui import preview_cache as pc, service
from dngscan.report import processing_evidence_summary, csv_row
from tests.test_capture_identity import tiff_bytes, ascii_field
from tests.test_readout_contract import edit_ifd
from tests.test_spectral_fallback_pipeline import write_noise_dng


def rational(tag, numerator, denominator, endian="<", kind=5):
    return tag, kind, 1, struct.pack(endian+("LL" if kind==5 else "ll"),numerator,denominator)


def exif_file(path, *, root=(), exif=(), endian="<"):
    """Two IFDs with TIFF-relative offsets, including rational payloads."""
    root=list(root)+[(34665,4,1,bytes(4))]
    first=bytearray(tiff_bytes(root,endian=endian))
    second=bytearray(tiff_bytes(exif,endian=endian))
    shift=len(first)
    count=struct.unpack_from(endian+'H',second,8)[0]
    for index in range(count):
        position=10+12*index
        kind,size=struct.unpack_from(endian+'HL',second,position+2)
        if size*metadata._TYPE_SIZES[kind]>4:
            pointer=struct.unpack_from(endian+'L',second,position+8)[0]
            struct.pack_into(endian+'L',second,position+8,pointer+shift)
    for index in range(len(root)):
        position=10+12*index
        if struct.unpack_from(endian+'H',first,position)[0]==34665:
            struct.pack_into(endian+'L',first,position+8,shift+8)
    path.write_bytes(first+second)


class CaptureFactsTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='capture-facts-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        env=patch.dict(os.environ,DNGSCAN_CALIBRATION_DIR=str(self.root/'profiles'))
        env.start();self.addCleanup(env.stop)

    def actual_dng(self, *, exposure=(1,80), serial='BODY-PRIVATE', lens_serial='LENS-PRIVATE'):
        path=self.root/'sensor.dng'
        write_noise_dng(path)
        fields=[rational(33434,*exposure),rational(33437,28,10),rational(37386,70,1),
                ascii_field(42035,'SIGMA'),ascii_field(42036,'28-70mm F2.8 DG DN | Contemporary 021'),
                ascii_field(42037,lens_serial),ascii_field(50735,serial)]
        edit_ifd(path,{tag:(kind,count,value) for tag,kind,count,value in fields})
        return path

    def test_exact_rationals_and_source_provenance_both_byte_orders(self):
        for endian in ('<','>'):
            path=self.root/'capture.nef'
            exif_file(path,endian=endian,exif=[rational(33434,10,300,endian),
                rational(33437,18,10,endian),rational(37386,35,1,endian),
                ascii_field(42035,'Nikon'),ascii_field(42036,'Lens 35'),ascii_field(42037,'LENS-A')])
            facts=metadata.read_capture_facts(path)
            self.assertAlmostEqual(facts.value('exposure_seconds'),1/30)
            self.assertEqual(facts.get('exposure_seconds').observations[0].rational,(10,300))
            self.assertEqual(facts.value('aperture_fnumber'),1.8)
            self.assertEqual(facts.value('focal_length_mm'),35.)
            source=facts.get('exposure_seconds').observations[0].to_dict()['source']
            self.assertEqual((source['ifd'],source['tag'],source['adapter_version']),('IFD0.ExifIFD',33434,1))
            self.assertEqual(facts.get('lens_hardware_id').status,'missing')
            self.assertIsInstance(facts.value('lens_identity'),LensIdentity)
            self.assertEqual(facts.value('lens_identity').kind,'metadata-name')

    def test_missing_optional_fields_do_not_disable_actual_raw_decode(self):
        path=self.root/'missing.dng';write_noise_dng(path)
        bundle=load_raw(path,scene_half_size=True)
        self.assertEqual(bundle.capture_facts.get('exposure_seconds').status,'missing')
        self.assertIsNone(bundle.capture_readout['exposure_seconds'])
        self.assertEqual(bundle.capture_facts.get('temperature_c').reason,'not-measured-by-this-adapter')
        self.assertIsNone(bundle.capture_readout['focus_distance_m'])
        self.assertEqual(bundle.shot_shutter,'electronic')
        self.assertGreater(bundle.scene_rec2020_render.size,0)

    def test_zero_denominators_wrong_type_and_invalid_pointer_are_preserved(self):
        path=self.root/'invalid.dng'
        for numerator in (0,1):
            exif_file(path,exif=[rational(33434,numerator,0)])
            fact=read(path).get('exposure_seconds')
            self.assertEqual(fact.status,'invalid')
            self.assertIsNone(fact.value)
            self.assertEqual(fact.observations[0].rational,(numerator,0))
        exif_file(path,exif=[(33434,4,1,struct.pack('<L',1))])
        self.assertEqual(read(path).get('exposure_seconds').status,'invalid')
        path.write_bytes(tiff_bytes([(34665,4,1,struct.pack('<L',0xfffffff0))]))
        self.assertEqual(read(path).get('aperture_fnumber').status,'invalid')
        self.assertIsNone(read(path).value('aperture_fnumber'))

    def test_conflicting_ifds_and_duplicate_tag_values_do_not_choose_a_winner(self):
        path=self.root/'conflict.dng'
        exif_file(path,root=[rational(33434,1,80)],exif=[rational(33434,1,160)])
        facts=read(path)
        self.assertEqual(facts.get('exposure_seconds').status,'conflict')
        self.assertIsNone(facts.value('exposure_seconds'))
        self.assertEqual(len(facts.get('exposure_seconds').observations),2)
        exif_file(path,exif=[rational(33434,1,30),rational(33434,10,300)])
        self.assertEqual(read(path).get('exposure_seconds').status,'verified')
        exif_file(path,exif=[ascii_field(42036,'Lens A'),ascii_field(42036,'Lens B')])
        self.assertEqual(read(path).get('lens_identity').status,'conflict')

    def test_converted_dng_does_not_gain_native_firmware_or_lens_hardware_id(self):
        path=self.root/'converted.arw'
        exif_file(path,root=[ascii_field(271,'SONY'),ascii_field(272,'ILCE-7M5'),
            ascii_field(305,'ILCE-7M5 v2.00'),(50706,1,4,bytes([1,4,0,0]))],
            exif=[rational(33434,1,80),ascii_field(42036,'28-70mm F2.8')])
        facts=read(path)
        self.assertEqual(facts.container,'dng')
        self.assertEqual(facts.value('exposure_seconds'),1/80)
        self.assertIsNone(facts.value('firmware'))
        self.assertIsNone(facts.value('lens_hardware_id'))
        self.assertEqual(facts.value('lens_identity').namespace,'exif:LensModel')

    def test_facts_are_immutable_and_one_source_session_reuses_the_snapshot(self):
        path=self.actual_dng()
        with source_metadata_session(path):
            facts=read(path)
            self.assertIs(read(path),facts)
            bundle=load_raw(path,scene_half_size=True)
            self.assertIs(bundle.capture_facts,facts)
            self.assertIs(bundle.evidence.capture_facts,facts)
            with self.assertRaises(FrozenInstanceError):facts.get('exposure_seconds').value=1.
            with self.assertRaises(FrozenInstanceError):facts.container='mutated'
            serialized=facts.to_dict();serialized['fields']['exposure_seconds']['value']=9.
            self.assertEqual(facts.value('exposure_seconds'),1/80)
        self.assertIs(release_analysis_buffers(bundle).capture_facts,facts)

    def test_source_change_during_facts_read_aborts_the_outer_load(self):
        path=self.actual_dng()
        from dngscan import capture_facts
        observe=capture_facts._observe
        def changed(*args,**kwargs):
            result=observe(*args,**kwargs)
            with path.open('ab') as target:target.write(b'changed')
            return result
        with patch.object(capture_facts,'_observe',side_effect=changed):
            with self.assertRaises(SourceChangedError):load_raw(path,scene_half_size=True)

    def test_source_disappearance_during_facts_read_is_unverifiable_not_changed(self):
        path=self.actual_dng()
        from dngscan import capture_facts
        observe=capture_facts._observe
        def removed(*args,**kwargs):
            result=observe(*args,**kwargs)
            path.unlink(missing_ok=True)
            return result
        with patch.object(capture_facts,'_observe',side_effect=removed):
            with self.assertRaises(SourceVerificationError):read(path)

    def test_matcher_exposure_bounds_and_lens_namespace_are_exact(self):
        bundle=load_raw(self.actual_dng(),scene_half_size=True)
        values=bundle.capture_readout
        constraints=readout.measurement_fields({'readout_contract':{'version':1,
            'exposure_seconds':[.01,.02],'aperture_fnumber':2.8,'focal_length_mm':70.,
            'lens_identity':values['lens_identity'],'lens_serial':'LENS-PRIVATE'}})
        self.assertEqual(readout.match(constraints,values)[0],'matched')
        for bounds in ([.1,1.],[.001,.002]):
            mismatch=readout.measurement_fields({'readout_contract':{'version':1,'exposure_seconds':bounds}})
            self.assertEqual(readout.match(mismatch,values)[0],'mismatch')
        identity=dict(values['lens_identity'])
        for change in ({'model':identity['model']+' II'}, {'make':'OTHER'}, {'namespace':'vendor:lens-id'}):
            expected={**identity,**change}
            self.assertEqual(readout.match({'readout_contract':{'lens_identity':expected}},values)[0],'mismatch')
        self.assertEqual(readout.match({'readout_contract':{'lens_identity':{
            'namespace':'Sony:LensType','kind':'hardware-id','id':'123'}}},values)[0],'mismatch')
        unknown={**values,'exposure_seconds':None}
        self.assertEqual(readout.match(constraints,unknown)[0],'unverified')
        for invalid in (True,0,float('nan'),[1.,.5],[1.],['1',2]):
            with self.assertRaises(ValueError):readout.measurement_fields({
                'readout_contract':{'version':1,'exposure_seconds':invalid}})

    def test_fact_status_reaches_matcher_without_profile_backfill(self):
        path=self.root/'invalid-exposure.dng';write_noise_dng(path)
        edit_ifd(path,{33434:(5,1,struct.pack('<LL',1,0))})
        bundle=load_raw(path,scene_half_size=True)
        self.assertEqual(readout.match({'readout_contract':{'exposure_seconds':[.01,.02]}},bundle.capture_readout),
                         ('unverified','file-exposure-seconds-invalid'))
        self.assertIsNone(bundle.capture_facts.value('exposure_seconds'))
        with self.assertRaises(ValueError):readout.with_capture_declaration(bundle.capture_readout,
            {'version':1,'exposure_seconds':.01})

    def test_json_and_actual_proxy_disk_cache_preserve_facts(self):
        path=self.actual_dng();bundle=load_raw(path,scene_half_size=True)
        facts=bundle.capture_facts
        payload=json.loads(json.dumps(facts.to_dict(),allow_nan=False))
        self.assertEqual(from_dict(payload),facts)
        self.assertEqual(fingerprint(from_dict(payload)),fingerprint(facts))
        analysis,_,_=analyze(bundle,4)
        entry=pc.build_proxy_entry(bundle,analysis)
        self.assertEqual(entry.bundle.capture_facts,facts)
        _,entry.cache_digest=pc._cache_identity(path,'clip','camera','libraw','auto','auto','aligned',4)
        destination=self.root/'cache'/f'{entry.cache_digest}.npz'
        pc._write_disk_entry(destination,entry)
        restored=pc._read_disk_entry(destination,path,False)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.bundle.capture_facts,facts)
        self.assertEqual(restored.source_metadata,entry.source_metadata)
        self.assertEqual(json.loads(json.dumps(pc._bundle_metadata(bundle))),pc._bundle_metadata(bundle))
        envelope=service._preview_analysis_envelope(entry)
        decoded=load_raw(path,scene_half_size=True)
        with patch.object(pc,'_cache_dir',return_value=destination.parent):
            kwargs=dict(decoded_bundle=decoded,envelope=envelope)
            self.assertIsNotNone(service._cached_full_analysis(path,'clip','camera','libraw','auto','auto',**kwargs))
            changed=replace(facts,fields=tuple((name,replace(field,value=.1)) if name=='exposure_seconds'
                                             else (name,field) for name,field in facts.fields))
            decoded.capture_facts=changed
            self.assertIsNone(service._cached_full_analysis(path,'clip','camera','libraw','auto','auto',**kwargs))

    def test_public_diagnostics_redact_both_serials_and_observations(self):
        path=self.actual_dng();bundle=load_raw(path,scene_half_size=True)
        analysis,_,_=analyze(bundle,4)
        for public in (public_facts(bundle.capture_facts), public_capture(bundle.capture_readout),
                       processing_evidence_summary(bundle,analysis),csv_row(bundle,analysis,self.root/'report.csv')):
            text=json.dumps(public,allow_nan=False)
            self.assertNotIn('BODY-PRIVATE',text)
            self.assertNotIn('LENS-PRIVATE',text)
        self.assertEqual(bundle.capture_facts.value('body_serial'),'BODY-PRIVATE')
        self.assertEqual(bundle.capture_facts.value('lens_serial'),'LENS-PRIVATE')

    def test_cache_restore_rejects_mutable_or_incomplete_fact_members(self):
        facts=read(self.actual_dng()).to_dict()
        for mutate in (
            lambda data:data.update(fields=[]),
            lambda data:data['fields']['exposure_seconds'].update(reason={}),
            lambda data:data['fields']['lens_identity'].update(value=[]),
            lambda data:data['fields']['lens_model'].update(value=3),
            lambda data:data['fields']['exposure_seconds']['observations'][0].update(reason=[]),
            lambda data:data.update(source_identity={'stat':[]}),
        ):
            payload=json.loads(json.dumps(facts));mutate(payload)
            with self.assertRaises(ValueError):from_dict(payload)


if __name__=='__main__':unittest.main()
