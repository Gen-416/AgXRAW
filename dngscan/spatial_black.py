# SPDX-License-Identifier: GPL-3.0-or-later
"""DNG spatial black calibration, without mutating original sensor evidence."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import struct
from ._deps import np
from . import metadata as md


def sensor_tags(path: Path, wanted: set[int]) -> dict:
    """Main DNG RAW IFD only; never mix preview calibration with sensor tags."""
    with open(path,'rb') as f:
        head=f.read(8)
        if len(head)!=8 or head[:2] not in (b'II',b'MM'):return {}
        endian='<' if head[:2]==b'II' else '>'
        if struct.unpack(endian+'H',head[2:4])[0]!=42:return {}
        root=md._read_ifd_entries(f,struct.unpack(endian+'L',head[4:])[0],endian)
        if not any(t==md.TAG_DNG_VERSION for t,*_ in root):return {}
        def values(entries):
            result={}
            for tag,typ,count,data in entries:
                if tag in wanted|{254,256,257,262,330}:
                    if count*md._TYPE_SIZES.get(typ,1)>16_000_000:
                        raise ValueError('DNG calibration metadata too large')
                    result[tag]=md._entry_values(f,typ,count,data,endian)
            return result
        first=values(root);ifds=[first]
        for off in first.get(330,[])[:64]:
            ifds.append(values(md._read_ifd_entries(f,int(off),endian)))
        candidates=[v for v in ifds if v.get(262,[0])[0] in (32803,34892)
                    and not int(v.get(254,[0])[0])&1]
        return max(candidates,key=lambda v:v.get(256,[0])[0]*v.get(257,[0])[0]) if candidates else {}


@dataclass(frozen=True)
class SpatialBlack:
    horizontal: object
    vertical: object
    pattern: object
    origin: tuple[int,int]
    max_black: tuple[float,...]

    def band(self,y0,y1,width,plane=0):
        y=np.arange(y0,y1)+self.origin[0]
        x=np.arange(width)+self.origin[1]
        a=np.asarray(self.pattern)
        return (a[y[:,None]%a.shape[0],x[None,:]%a.shape[1],min(plane,a.shape[2]-1)]
                +self.vertical[y,None]+self.horizontal[None,x])


def read(path,raw) -> SpatialBlack | None:
    tags=sensor_tags(path,{277,50713,50714,50715,50716,50829})
    if not (50715 in tags or 50716 in tags):return None
    h,w=raw.raw_image_visible.shape[:2]
    area=tags.get(50829,[0,0,tags[257][0],tags[256][0]])
    ah,aw=int(area[2]-area[0]),int(area[3]-area[1])
    origin=(int(raw.sizes.top_margin-area[0]),int(raw.sizes.left_margin-area[1]))
    if min(origin)<0 or origin[0]+h>ah or origin[1]+w>aw:
        raise ValueError('DNG black-delta active area does not cover visible pixels')
    hor=np.asarray(tags.get(50715,np.zeros(aw)),dtype=np.float32)
    ver=np.asarray(tags.get(50716,np.zeros(ah)),dtype=np.float32)
    repeat=tags.get(50713,[1,1]);planes=int(tags.get(277,[1])[0])
    if len(repeat)!=2 or min(repeat)<1 or max(repeat)>64 or planes not in (1,3,4):
        raise ValueError('invalid DNG black pattern dimensions')
    black=np.asarray(tags.get(50714,[0.]),dtype=np.float32)
    if black.size != repeat[0]*repeat[1]*planes or hor.size!=aw or ver.size!=ah:
        raise ValueError('invalid DNG spatial black calibration shape')
    black=black.reshape(int(repeat[0]),int(repeat[1]),planes)
    if not all(np.isfinite(a).all() for a in (hor,ver,black)):
        raise ValueError('nonfinite DNG spatial black calibration')
    max_h=np.asarray([hor[c::int(repeat[1])].max() for c in range(int(repeat[1]))])
    max_v=np.asarray([ver[r::int(repeat[0])].max() for r in range(int(repeat[0]))])
    maxima=(black+max_v[:,None,None]+max_h[None,:,None]).max(axis=(0,1))
    if np.any(maxima>=float(raw.white_level)):
        raise ValueError('DNG black level reaches white level')
    return SpatialBlack(hor,ver,black,origin,tuple(map(float,maxima)))


def apply_to_working(raw,model,black_levels,white,loss=None):
    """Normalize against maximum black as specified by the Adobe DNG SDK.

    Feed equivalent uniform-black codes to LibRaw; its later subtraction and
    scaling reproduce the DNG range mapping. The independent evidence copy is
    untouched, including physical saturation at the original stored WhiteLevel.
    """
    image=raw.raw_image_visible
    colors=np.asarray(raw.raw_colors_visible) if image.ndim==2 else None
    levels=np.asarray(black_levels,dtype=np.float32)
    for y in range(0,image.shape[0],128):
        y1=min(y+128,image.shape[0]);band=image[y:y1]
        n=1 if image.ndim==2 else min(3,image.shape[2])
        for c in range(n):
            src=band if n==1 else band[...,c]
            black=levels[colors[y:y1]] if n==1 else levels[c]
            actual=model.band(y,y1,image.shape[1],c)
            maximum=model.max_black[min(c,len(model.max_black)-1)]
            v=(src.astype(np.float32)-actual)*(float(white)-black)/(float(white)-maximum)+black
            if loss is not None:
                target=loss[y:y1] if n==1 else loss[y:y1,...,c]
                target |= ((src<float(white)) & (v>=white)).astype(np.uint8)
            src[:]=np.rint(np.clip(v,0,min(float(white),np.iinfo(image.dtype).max))).astype(image.dtype)


def corrected_plane(bundle,yoff,xoff,ph,pw):
    """Remove spatial black structure from single-colour noise measurements."""
    plane=bundle.raw_image[yoff::ph,xoff::pw]
    model=getattr(getattr(bundle,'evidence',None),'spatial_black',None)
    if model is None:return plane
    out=plane.astype(np.float32)
    # Subtract the varying component, keeping the channel's original pedestal.
    cid=int(bundle.raw_colors[yoff,xoff]);base=float(bundle.black_levels[cid])
    for start in range(0,out.shape[0],64):
        stop=min(start+64,out.shape[0]);y0=yoff+start*ph;y1=min(bundle.raw_image.shape[0],yoff+stop*ph)
        out[start:stop]-=model.band(y0,y1,bundle.raw_image.shape[1])[::ph,xoff::pw]-base
    return out


def clip_mask(raw,colors,desc,black_levels,white_levels,model):
    """Per-position headroom in sensor coordinates, bounded to 128-row bands."""
    from .raw_io import _smoothstep
    h,w=raw.shape[:2]
    out=np.zeros((max(1,h//2),max(1,w//2),3),dtype=np.float32)
    for y in range(0,h//2*2,128):
        end=min(y+128,h//2*2)
        soft=np.zeros((end-y,w//2*2,3),dtype=np.float32)
        if raw.ndim==3:
            for c in range(3):
                black=model.band(y,end,w,c)[:,:w//2*2]
                white=white_levels[min(c,len(white_levels)-1)]
                soft[...,c]=_smoothstep(.95,.99,(raw[y:end,:w//2*2,c]-black)/np.maximum(white-black,1.))
        else:
            black=model.band(y,end,w)[:,:w//2*2]
            for cid in np.unique(colors):
                label=desc[int(cid):int(cid)+1]
                if label not in ('R','G','B'):continue
                white=white_levels[min(int(cid),len(white_levels)-1)]
                proximity=_smoothstep(.95,.99,(raw[y:end,:w//2*2]-black)/np.maximum(white-black,1.))
                plane=np.where(colors[y:end,:w//2*2]==cid,proximity,0.)
                np.maximum(soft[..., 'RGB'.index(label)],plane,out=soft[..., 'RGB'.index(label)])
        out[y//2:end//2]=soft.reshape((end-y)//2,2,w//2,2,3).max(axis=(1,3))
    return out
