# SPDX-License-Identifier: GPL-3.0-or-later
"""DNG point transforms in their declared stage and logical numeric domain."""
from __future__ import annotations
from dataclasses import dataclass
import struct
from ._deps import np


@dataclass(frozen=True)
class PointOp:
    kind: int
    area: tuple[int,...]
    values: tuple[float,...]
    stage: int


@dataclass(frozen=True)
class BadPixels:
    phase: int
    constant: int | None = None
    points: tuple = ()
    rectangles: tuple = ()


def parse(kind,payload,stage):
    if kind in (4,5):
        if kind==4:
            if len(payload)!=8:raise ValueError('invalid FixBadPixelsConstant')
            constant,phase=struct.unpack('>LL',payload)
            if constant>65535 or phase>3:raise ValueError('invalid bad-pixel constant/phase')
            return BadPixels(phase,constant)
        if len(payload)<12:raise ValueError('truncated FixBadPixelsList')
        phase,npnt,nrect=struct.unpack_from('>3L',payload)
        if phase>3 or len(payload)!=12+npnt*8+nrect*16:raise ValueError('invalid FixBadPixelsList')
        points=tuple(struct.unpack_from('>2l',payload,12+i*8) for i in range(npnt))
        rects=tuple(struct.unpack_from('>4l',payload,12+npnt*8+i*16) for i in range(nrect))
        if any(min(p)<0 for p in points) or any(min(r)<0 or r[0]>=r[2] or r[1]>=r[3] for r in rects):
            raise ValueError('invalid bad-pixel coordinates')
        return BadPixels(phase,None,points,rects)
    if len(payload)<36:raise ValueError('truncated DNG point transform')
    area=struct.unpack_from('>4l4L',payload);t,l,b,r,plane,planes,rp,cp=area
    count=struct.unpack_from('>L',payload,32)[0]
    if planes<1 or rp<1 or cp<1 or min(t,l)<0 or ((b<=t or r<=l) and (rp!=1 or cp!=1)):
        raise ValueError('invalid DNG point transform area')
    if kind==7:
        if not 1<=count<=65536 or len(payload)!=36+count*2:raise ValueError('invalid MapTable')
        values=struct.unpack_from(f'>{count}H',payload,36)
    elif kind==8:
        if count>8 or len(payload)!=36+(count+1)*8:raise ValueError('invalid MapPolynomial')
        values=struct.unpack_from(f'>{count+1}d',payload,36)
    else:
        expected=(max(b-t,0)+rp-1)//rp if kind in (10,12) else (max(r-l,0)+cp-1)//cp
        if count!=expected or len(payload)!=36+count*4:raise ValueError('invalid row/column transform count')
        values=struct.unpack_from(f'>{count}f',payload,36)
    if not np.isfinite(values).all():raise ValueError('nonfinite DNG point transform')
    return PointOp(kind,area,tuple(values),stage)


def apply(image,op,*,black=0.,white=65535.,colors=None,loss=None):
    """Apply per-band to integer camera codes; stage 2/3 values are normalized."""
    t,l,b,r,first,planes,rp,cp=op.area
    h,w=image.shape[:2]
    if b<=t or r<=l:t,l,b,r=0,0,h,w
    if t>=h or l>=w:return
    b,r=min(b,h),min(r,w)
    channels=1 if image.ndim==2 else image.shape[2]
    values=np.asarray(op.values,dtype=np.float64)
    for c in range(first,min(first+planes,channels)):
        for start in range(t,b,128*rp):
            stop=min(b,start+128*rp)
            sel=(slice(start,stop,rp),slice(l,r,cp))
            src=image[sel] if channels==1 else image[sel+(c,)]
            if not src.size:continue
            cid=colors[sel] if colors is not None else c
            bl=np.asarray(black)[cid] if np.ndim(black) else black
            wl=np.asarray(white)[cid] if np.ndim(white) else white
            span=np.maximum(wl-bl,1.)
            x=src.astype(np.float64)
            if op.stage!=1:x=(x-bl)/span
            if op.kind==7:
                code=np.rint(x if op.stage==1 else x*65535).astype(np.int64)
                out=values[np.clip(code,0,len(values)-1)]
                if op.stage!=1:out/=65535
            elif op.kind==8:
                out=np.polynomial.polynomial.polyval(x,values)
            else:
                # Row/column table indexes are relative to the declared area,
                # including pitch; clipping the overlap never shifts the table.
                v=(values[(np.arange(start,stop,rp)-t)//rp,None] if op.kind in (10,12)
                   else values[None,(np.arange(l,r,cp)-l)//cp])
                out=x+v if op.kind in (10,11) else x*v
            ceiling=65535. if op.stage==1 else 1.
            if loss is not None:
                dest=loss[sel] if channels==1 else loss[sel+(c,)]
                np.maximum(dest,(x<ceiling)&(out>=ceiling),out=dest)
            out=np.clip(out,0,ceiling)
            if op.stage!=1:out=out*span+bl
            src[:]=np.rint(out).astype(image.dtype)


def repair_bad_pixels(raw,op,loss=None):
    """Same-colour interpolation; original evidence is never overwritten."""
    image=raw.raw_image
    if image.ndim!=2:raise ValueError('bad-pixel Bayer opcode needs a mosaic')
    colors=np.asarray(raw.raw_colors)
    desc=raw.color_desc.decode() if isinstance(raw.color_desc,bytes) else raw.color_desc
    yy2,xx2=np.indices((min(12,image.shape[0]),min(12,image.shape[1])))
    green=np.asarray([v=='G' for v in desc])[colors[:yy2.shape[0],:yy2.shape[1]]]
    expected=(yy2+xx2+op.phase+(op.phase>>1))%2==0
    if not np.array_equal(green,expected):
        raise ValueError('bad-pixel opcode Bayer phase does not match the sensor')
    bad=image==op.constant if op.constant is not None else np.zeros(image.shape,bool)
    for y,x in op.points:
        if y>=image.shape[0] or x>=image.shape[1]:raise ValueError('bad pixel outside RAW')
        bad[y,x]=True
    for t,l,b,r in op.rectangles:
        if b>image.shape[0] or r>image.shape[1]:raise ValueError('bad rectangle outside RAW')
        bad[t:b,l:r]=True
    yy,xx=np.nonzero(bad)
    if yy.size>1_000_000:raise ValueError('bad-pixel region exceeds supported interpolation coverage')
    # Use unchanged good neighbours at increasing same-CFA spacing, excluding
    # every listed defect. Contiguous defects cannot contaminate each other.
    for start in range(0,len(yy),4096):
        y,x=yy[start:start+4096],xx[start:start+4096]
        samples=[]
        for radius in (2,4,6,8):
            for dy,dx in ((-radius,0),(radius,0),(0,-radius),(0,radius),(-radius,-radius),(-radius,radius),(radius,-radius),(radius,radius)):
                ny,nx=y+dy,x+dx
                inside=(ny>=0)&(nx>=0)&(ny<image.shape[0])&(nx<image.shape[1])
                ny,nx=np.clip(ny,0,image.shape[0]-1),np.clip(nx,0,image.shape[1]-1)
                valid=inside & ~bad[ny,nx] & (colors[ny,nx]==colors[y,x])
                samples.append(np.where(valid,image[ny,nx],np.nan))
            if np.all(np.isfinite(np.asarray(samples)).sum(axis=0)>=4):break
        sample=np.asarray(samples)
        if np.any(np.all(~np.isfinite(sample),axis=0)):
            raise ValueError('bad-pixel region has no usable same-colour neighbours')
        image[y,x]=np.rint(np.nanmedian(sample,axis=0)).astype(image.dtype)
    if loss is not None:
        top,left=raw.sizes.top_margin,raw.sizes.left_margin
        loss[:]=np.maximum(loss,bad[top:top+loss.shape[0],left:left+loss.shape[1]])
