# SPDX-License-Identifier: GPL-3.0-or-later
"""File-authored Sony/Fujifilm lens profiles.

Tag layout and coefficient conventions cross-checked against Exiv2 and
Darktable's metadata lens implementation (GPL-3.0-or-later). Profiles are
capture-specific; no nearest-camera/lens guesses or automatic CA estimation.
"""
from __future__ import annotations
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from ._deps import np
from . import metadata as md
from .dng_opcodes import Warp


class _OffsetFile:
    def __init__(self,file,base): self.file,self.base=file,base
    def seek(self,pos):return self.file.seek(self.base+pos)
    def read(self,n):return self.file.read(n)


@dataclass(frozen=True)
class RadialVignette:
    knots: tuple[float,...]
    attenuation: tuple[float,...]


@dataclass(frozen=True)
class LensProfile:
    warp: Warp
    vignette: RadialVignette
    source: str
    shutter: str | None = None


def _values(f,offset,endian,wanted):
    result={}
    for tag,typ,count,data in md._read_ifd_entries(f,offset,endian):
        if tag not in wanted:continue
        if count*md._TYPE_SIZES.get(typ,4)>128000:raise ValueError('lens metadata too large')
        result[tag]=(list(struct.unpack(endian+'L',data)) if typ==13 and count==1
                     else md._entry_values(f,typ,count,data,endian))
    return result


def read(path: Path) -> LensProfile | None:
    with open(path,'rb') as f:
        head=f.read(116)
        if head.startswith(md._RAF_MAGIC):
            joff,jlen=struct.unpack_from('>LL',head,84);f.seek(joff)
            tiff=md._exif_tiff_from_jpeg(f.read(min(jlen,128000)))
            crop,shutter=1.0,None
            if tiff:
                t=io.BytesIO(tiff);e='<' if tiff[:2]==b'II' else '>'
                root=_values(t,struct.unpack_from(e+'L',tiff,4)[0],e,{34665})
                if root.get(34665):
                    exif=_values(t,int(root[34665][0]),e,{37500})
                    blob=(exif.get(37500) or [b''])[0]
                    if isinstance(blob,bytes) and blob.startswith(b'FUJIFILM'):
                        maker=_values(io.BytesIO(blob),struct.unpack_from('<L',blob,8)[0],'<',{0x104d,0x1050})
                        crop=1.25 if (maker.get(0x104d) or [0])[0] in (2,4) else 1.0
                        shutter={0:'mechanical',1:'electronic',2:'electronic',3:'efcs'}.get((maker.get(0x1050) or [None])[0])
            offset=struct.unpack_from('>L',head,100)[0];t=_OffsetFile(f,offset);t.seek(0);h=t.read(8)
            if h[:4]!=b'II*\0':return None
            root=_values(t,struct.unpack_from('<L',h,4)[0],'<',{0xf000})
            if not root.get(0xf000):return None
            tags=_values(t,int(root[0xf000][0]),'<',{0xf00b,0xf00f,0xf010})
            if not all(k in tags for k in (0xf00b,0xf00f,0xf010)):return None
            d,c,v=(np.asarray(tags[k],dtype=np.float64) for k in (0xf00b,0xf00f,0xf010))
            if (len(d),len(c),len(v))==(19,29,19):
                knots=d[1:10];dist=d[10:19];red=c[10:19];blue=c[19:28];vig=v[10:19]/100
                match=np.array_equal(knots,c[1:10]) and np.array_equal(knots,v[1:10])
            elif (len(d),len(c),len(v))==(23,31,23):
                knots=d[1:12];dist=d[12:23];red=np.r_[0,c[11:21]];blue=np.r_[0,c[21:31]];vig=v[12:23]/100
                match=np.array_equal(knots,np.r_[0,c[1:11]]) and np.array_equal(knots,v[1:12])
            else:raise ValueError('unsupported Fujifilm lens coefficient layout')
            if not match:raise ValueError('inconsistent Fujifilm lens knots')
            knots=knots*crop
            if knots[0]>0:
                knots=np.r_[0,knots];dist=np.r_[0,dist];red=np.r_[0,red];blue=np.r_[0,blue];vig=np.r_[1,vig]
            # Fuji stores the field against SOURCE radius. Invert its
            # monotonic mapping before destination-to-source resampling.
            rin=np.linspace(0,max(1.,knots[-1]),1025)
            magnification=1+np.interp(rin,knots,dist)/100
            target=rin/magnification
            rgb=np.array([magnification*(1+np.interp(rin,knots,red)),magnification,
                          magnification*(1+np.interp(rin,knots,blue))])
            return _profile(target,rgb,knots,vig,'Fujifilm embedded lens',shutter)
        if head[:2] not in (b'II',b'MM'):return None
        e='<' if head[:2]==b'II' else '>'
        if struct.unpack_from(e+'H',head,2)[0]!=42:return None
        root=_values(f,struct.unpack_from(e+'L',head,4)[0],e,{330,50706,0x7032,0x7035,0x7037})
        if 50706 in root:return None  # DNG opcodes own its correction; never double apply.
        candidates=[root]+[_values(f,int(off),e,{0x7032,0x7035,0x7037}) for off in root.get(330,[])[:64]]
        tags=next((t for t in candidates if all(k in t for k in (0x7032,0x7035,0x7037))),None)
        if tags is None:return None
        d,c,v=(np.asarray(tags[k],dtype=np.float64) for k in (0x7037,0x7035,0x7032))
        n=int(d[0])
        if not 2<=n<=16 or len(d)!=n+1 or len(c)!=2*n+1 or c[0]!=2*n or len(v)!=n+1 or v[0]!=n:
            raise ValueError('invalid Sony lens coefficient count')
        knots=(np.arange(n)+.5)/(n-1)
        magnification=1+d[1:]*2**-14
        rgb=np.array([magnification*(1+c[1:n+1]*2**-21),magnification,
                      magnification*(1+c[n+1:]*2**-21)])
        vig=2**(.5-2**(v[1:]*2**-13-1))
        return _profile(knots,rgb,knots,vig,'Sony embedded lens',None)


def _profile(knots,rgb,vknots,vig,source,shutter):
    if (not all(np.isfinite(x).all() for x in (knots,rgb,vknots,vig))
            or np.any(np.diff(knots)<=0) or np.any(np.diff(vknots)<=0)
            or np.any(rgb<=0) or np.any(vig<=0)):
        raise ValueError('invalid embedded lens radial field')
    # Fill the sensor rectangle after correcting pincushion/TCA; keep a
    # deterministic crop and record the same geometry for evidence masks.
    scale=max(1.,float(np.max([np.interp(np.linspace(.5,1,256),knots,row) for row in rgb])))
    warp=Warp(((1.,0.,0.,0.,0.,0.),),.5,.5,knots=tuple(knots),
              scales=tuple(tuple(row) for row in rgb),scale=scale)
    return LensProfile(warp,RadialVignette(tuple(vknots),tuple(vig)),source,shutter)


def apply_vignette(image,op,loss=None,limits=None):
    h,w=image.shape[:2];cx,cy=w/2,h/2;radius=np.hypot(cx,cy)
    x=(np.arange(w)-cx)/radius
    for y in range(0,h,128):
        rr=np.hypot(x[None,:],(np.arange(y,min(y+128,h))[:,None]-cy)/radius)
        gain=1/np.interp(rr,op.knots,op.attenuation)
        band=image[y:y+128].astype(np.float32)*gain[...,None]
        if limits is not None:
            if loss is not None:np.maximum(loss[y:y+128],band>=limits,out=loss[y:y+128])
            np.clip(band,0,limits,out=band)
        image[y:y+128]=band.astype(image.dtype)
