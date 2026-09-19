# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural integrity when replacing HDR primary codestreams."""
import io,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image,JpegImagePlugin
from dngscan import jpeg_gainmap as jpg,heif_gainmap as heif


def jpeg_template(rgb):
    out=io.BytesIO();Image.fromarray(rgb).save(out,format='JPEG',quality=95)
    base=out.getvalue();aux=b'\xff\xd8opaque-gainmap\xff\xd9'
    # TIFF-relative MPEntry offsets, two images. The primary starts at SOI.
    tiff=b'II'+struct.pack('<HL',42,8)+struct.pack('<H',1)+struct.pack('<HHLL',0xb002,7,32,26)+bytes(4)
    marker_length=4+4+len(tiff)+32
    end=len(base)+marker_length
    mp=tiff+struct.pack('<LLLL',0,end,0,0)+struct.pack('<LLLL',0,len(aux),end-10,0)
    app=b'\xff\xe2'+struct.pack('>H',len(mp)+6)+b'MPF\0'+mp
    return base[:2]+app+base[2:]+aux,aux


def heif_fixture(width=16,height=12,*,hdr=True,clap=None,mirror=None):
    box=heif._box;items={1:(b'hvc1',b'original-image')}
    if hdr:items.update({2:(b'hvc1',b'gainmap-payload'),3:(b'tmap',b'tone-map-metadata')})
    props=[box(b'ispe',bytes(4)+struct.pack('>LL',width,height)),box(b'hvcC',b'codec-config')]
    if clap is not None:props.append(box(b'clap',clap))
    if mirror is not None:props.append(box(b'imir',bytes([mirror])))
    inf=box(b'iinf',bytes(4)+struct.pack('>H',len(items))+b''.join(box(b'infe',b'\x02\0\0\0'+struct.pack('>HH',i,0)+typ+b'\0') for i,(typ,_) in items.items()))
    assoc=box(b'ipma',bytes(4)+struct.pack('>L',len(items))+b''.join(struct.pack('>HB',i,len(props))+bytes(range(1,len(props)+1)) for i in items))
    prop=box(b'iprp',box(b'ipco',b''.join(props))+assoc)
    refs=box(b'iref',bytes(4)+box(b'dimg',struct.pack('>4H',3,2,1,2))) if hdr else b''
    idat=b''.join(v for _,v in items.values());offset=0;records=[]
    for i,(_,payload) in items.items():
        records.append(struct.pack('>HHHHLL',i,1,0,1,offset,len(payload)));offset+=len(payload)
    loc=box(b'iloc',b'\x01\0\0\0\x44\0'+struct.pack('>H',len(items))+b''.join(records))
    return box(b'ftyp',b'heic'+bytes(4)+b'mif1heic')+box(b'meta',bytes(4)+box(b'pitm',bytes(4)+struct.pack('>H',1))+inf+refs+prop+loc+box(b'idat',idat))


class JpegRepackTests(unittest.TestCase):
    def test_sampling_is_independent_and_auxiliary_offsets_relocate(self):
        rgb=np.random.default_rng(89).integers(0,256,(32,48,3),dtype=np.uint8)
        original,aux=jpeg_template(rgb)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gain.jpg'
            for q in (95,97,98,100):
                for chroma,index in (('420',2),('422',1),('444',0)):
                    p.write_bytes(original);jpg.replace_primary(p,rgb,q,chroma)
                    data=p.read_bytes();self.assertTrue(data.endswith(aux))
                    with Image.open(p) as im:
                        im.load();self.assertEqual(im.size,(48,32));self.assertEqual(JpegImagePlugin.get_sampling(im),index)
                    segment=next((a,b,payload) for marker,a,b,payload in jpg._segments(data) if marker==0xe2)
                    _,_,payload=segment;tiff=payload+4
                    size,off=struct.unpack_from('<LL',data,tiff+26+16+4)
                    self.assertEqual(data[tiff+off:tiff+off+size],aux)
                    primary_size=struct.unpack_from('<L',data,tiff+26+4)[0]
                    self.assertEqual(data[primary_size-2:primary_size],b'\xff\xd9')

    def test_malformed_jpeg_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.jpg';p.write_bytes(b'\xff\xd8\xff')
            with self.assertRaises(ValueError):jpg.replace_primary(p,np.zeros((8,8,3),np.uint8),97,'422')
            self.assertEqual(p.read_bytes(),b'\xff\xd8\xff')


class HeifRepackTests(unittest.TestCase):
    def test_sdr_icc_embedding_keeps_coded_payloads_and_does_not_add_hdr_brand(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'sdr.heic'
            path.write_bytes(heif_fixture(hdr=False))
            before = heif._parse(path.read_bytes())
            heif.embed_primary_icc(path, b'ICC-profile')
            after = heif._parse(path.read_bytes())
            self.assertEqual(after[7], before[7])
            self.assertEqual(after[4], before[4])
            self.assertEqual(heif.primary_icc(path), b'ICC-profile')
            self.assertEqual(after[0][0], before[0][0])

    def test_auxiliary_replacement_preserves_primary_and_iso_parameters(self):
        with tempfile.TemporaryDirectory() as td:
            path, donor = Path(td)/'pair.heic', Path(td)/'auxiliary.heic'
            path.write_bytes(heif_fixture())
            donor.write_bytes(heif_fixture(hdr=False).replace(b'original-image', b'new-gain-codes'))
            before = heif._parse(path.read_bytes())
            heif.replace_image_item(path, donor, 2)
            after = heif._parse(path.read_bytes())
            self.assertEqual(after[2], before[2])
            self.assertEqual(after[7][1], before[7][1])
            self.assertEqual(after[7][3], before[7][3])
            self.assertEqual(after[7][2], b'new-gain-codes')
            self.assertEqual(after[4], before[4])

    def test_unknown_auxiliary_encoding_is_not_reinterpreted_as_rgb_gain(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'pair.heic'
            path.write_bytes(heif_fixture())
            with self.assertRaisesRegex(ValueError, 'sample depth'):
                heif.iso_gainmap_item(path)

    def test_item_payloads_and_references_survive(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gain.heic';d=Path(td)/'donor.heic'
            p.write_bytes(heif_fixture());d.write_bytes(heif_fixture(hdr=False).replace(b'original-image',b'newcoded-image'))
            before=heif._parse(p.read_bytes());heif.replace_primary(p,d);after=heif._parse(p.read_bytes())
            self.assertEqual(after[7][1],b'newcoded-image')
            self.assertEqual(after[7][2],before[7][2]);self.assertEqual(after[7][3],before[7][3]);self.assertEqual(after[4],before[4])

    def test_clean_aperture_must_really_match_and_stay_inside(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gain.heic';d=Path(td)/'donor.heic'
            for display in (16,14,40):
                original=heif_fixture();p.write_bytes(original)
                clap=struct.pack('>4LiLiL',display,1,12,1,0,1,0,1)
                d.write_bytes(heif_fixture(width=32,height=16,hdr=False,clap=clap))
                if display==16:heif.replace_primary(p,d)
                else:
                    with self.assertRaises(ValueError):heif.replace_primary(p,d)
                    self.assertEqual(p.read_bytes(),original)

    def test_truncated_or_external_payload_is_rejected(self):
        data=heif_fixture()
        with self.assertRaises(ValueError):heif._parse(data[:-2])
        # A malformed box must never be accepted as an empty payload.
        with self.assertRaises(ValueError):list(heif._boxes(struct.pack('>L4s',7,b'meta')))

    def test_both_mirror_axes_require_an_explicit_transform(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gain.heic';d=Path(td)/'donor.heic'
            d.write_bytes(heif_fixture(hdr=False))
            for axis in (0,1):
                original=heif_fixture(mirror=axis);p.write_bytes(original)
                with self.assertRaisesRegex(ValueError,'explicit pixel transform'):
                    heif.replace_primary(p,d)
                self.assertEqual(p.read_bytes(),original)


if __name__=='__main__':unittest.main()
