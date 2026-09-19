# SPDX-License-Identifier: GPL-3.0-or-later
"""Repackage an ISO gain-map HEIF with a separately encoded HEVC primary.

Restricted to still-image meta containers written by ImageIO/libheif. Unknown
item addressing, external data, movies, or unexpected item layouts fail closed.
All surviving payloads are copied byte-for-byte, including ISO tmap metadata and
auxiliary images. Item extents and property associations are rebuilt.
"""
from __future__ import annotations
import struct
from pathlib import Path


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack('>L4s',len(payload)+8,kind)+payload


def _boxes(data: bytes):
    pos=0
    while pos < len(data):
        if pos+8 > len(data): raise ValueError('truncated HEIF box')
        size,kind=struct.unpack_from('>L4s',data,pos); head=8
        if size==1:
            if pos+16>len(data): raise ValueError('truncated large HEIF box')
            size=struct.unpack_from('>Q',data,pos+8)[0];head=16
        if size==0: size=len(data)-pos
        if size<head or pos+size>len(data): raise ValueError('invalid HEIF box extent')
        yield kind,data[pos+head:pos+size]
        pos+=size


def _one(boxes,kind):
    values=[v for k,v in boxes if k==kind]
    if len(values)!=1: raise ValueError(f'expected one HEIF {kind!r}')
    return values[0]


class _Reader:
    def __init__(self,data): self.data,self.pos=data,0
    def take(self,n):
        if n<0 or self.pos+n>len(self.data): raise ValueError('truncated HEIF item table')
        out=self.data[self.pos:self.pos+n];self.pos+=n;return out
    def uint(self,n): return int.from_bytes(self.take(n),'big')
    def end(self):
        if self.pos!=len(self.data): raise ValueError('trailing HEIF table bytes')


def _parse(data):
    top=list(_boxes(data))
    if any(k not in (b'ftyp',b'meta',b'mdat',b'free',b'skip') for k,_ in top):
        raise ValueError('unsupported HEIF top-level layout')
    meta=_one(top,b'meta')
    if meta[:4]!=bytes(4): raise ValueError('unsupported HEIF meta version')
    children=list(_boxes(meta[4:]))
    pitm=_one(children,b'pitm')
    if pitm[:4]!=bytes(4) or len(pitm)!=6: raise ValueError('unsupported primary item ID')
    primary=int.from_bytes(pitm[4:],'big')
    iinf=_one(children,b'iinf')
    if iinf[:4]!=bytes(4): raise ValueError('unsupported iinf version')
    infos={}
    for kind,info in _boxes(iinf[6:]):
        if kind!=b'infe' or info[0]!=2 or len(info)<13: raise ValueError('unsupported infe layout')
        item=int.from_bytes(info[4:6],'big')
        if item in infos or info[6:8]!=b'\0\0': raise ValueError('duplicate/protected HEIF item')
        infos[item]=info
    if len(infos)!=int.from_bytes(iinf[4:6],'big'): raise ValueError('invalid HEIF item count')
    refs=[]
    for kind,payload in children:
        if kind!=b'iref':continue
        if payload[:4]!=bytes(4):raise ValueError('unsupported HEIF iref version')
        for typ,ref in _boxes(payload[4:]):
            r=_Reader(ref);src=r.uint(2);n=r.uint(2);targets=[r.uint(2) for _ in range(n)];r.end()
            refs.append((typ,src,targets))
    iprp=list(_boxes(_one(children,b'iprp')))
    props=list(_boxes(_one(iprp,b'ipco')))
    assocs={}
    for kind,payload in iprp:
        if kind!=b'ipma':continue
        r=_Reader(payload);v=r.uint(1);flags=r.uint(3)
        if v!=0 or flags & ~1:raise ValueError('unsupported HEIF ipma version')
        count=r.uint(4)
        for _ in range(count):
            item=r.uint(2);n=r.uint(1);pairs=[]
            for _ in range(n):
                val=r.uint(2 if flags&1 else 1);flag=0x8000 if flags&1 else 0x80
                essential=bool(val&flag);idx=val&(flag-1)
                if idx>len(props):raise ValueError('invalid HEIF property index')
                if idx:pairs.append((essential,idx))
            assocs.setdefault(item,[]).extend(pairs)
        r.end()
    idats=[p for k,p in children if k==b'idat']
    r=_Reader(_one(children,b'iloc'));version=r.uint(1);r.uint(3)
    if version not in (0,1):raise ValueError('unsupported HEIF iloc version')
    sizes=r.uint(1);more=r.uint(1);osize,lsize=sizes>>4,sizes&15;bsize,isize=more>>4,more&15
    if any(s not in (0,4,8) for s in (osize,lsize,bsize,isize)):
        raise ValueError('unsupported HEIF item extent width')
    payloads={};count=r.uint(2)
    for _ in range(count):
        item=r.uint(2);method=r.uint(2) if version else 0;ref=r.uint(2);base=r.uint(bsize);n=r.uint(2)
        if ref or method not in (0,1):raise ValueError('external/derived HEIF item addressing')
        source=data if method==0 else _one(children,b'idat')
        chunks=[]
        for _ in range(n):
            if version and isize:r.uint(isize)
            off,ln=base+r.uint(osize),r.uint(lsize)
            if not ln or off+ln>len(source):raise ValueError('invalid HEIF payload extent')
            chunks.append(source[off:off+ln])
        if item in payloads:raise ValueError('duplicate HEIF iloc')
        payloads[item]=b''.join(chunks)
    r.end()
    if set(payloads)!=set(infos) or primary not in infos:
        raise ValueError('inconsistent HEIF item tables')
    return top,children,primary,infos,refs,props,assocs,payloads


def replace_primary(path: Path, donor: Path) -> None:
    """Install a single-item HEVC donor, preserving all original HDR auxiliaries."""
    top,children,primary,infos,refs,props,assocs,payloads=_parse(path.read_bytes())
    dtop,_,dp,dinfos,drefs,dprops,dassoc,dpayloads=_parse(donor.read_bytes())
    def dimensions(properties, associations, item):
        values=[properties[i-1][1] for _,i in associations.get(item,[]) if properties[i-1][0]==b'ispe']
        if len(values)!=1 or len(values[0])!=12:raise ValueError('invalid primary dimensions')
        w,h=struct.unpack_from('>LL',values[0],4)
        apertures=[properties[i-1][1] for _,i in associations.get(item,[]) if properties[i-1][0]==b'clap']
        if apertures:
            if len(apertures)!=1 or len(apertures[0])!=32:raise ValueError('invalid HEIF clean aperture')
            wn,wd,hn,hd,xn,xd,yn,yd=struct.unpack('>4LiLiL',apertures[0])
            if not all((wd,hd,xd,yd)) or wn%wd or hn%hd:
                raise ValueError('unsupported fractional HEIF clean aperture')
            cw,ch=wn//wd,hn//hd
            if min(cw,ch)<=0 or cw+2*abs(xn/xd)>w or ch+2*abs(yn/yd)>h:
                raise ValueError('HEIF clean aperture outside coded image')
            return cw,ch
        return w,h
    if dimensions(props,assocs,primary)!=dimensions(dprops,dassoc,dp):
        raise ValueError('HEIF primary dimensions differ')
    if len(dinfos)!=1 or drefs or dinfos[dp][8:12]!=b'hvc1':
        raise ValueError('HEIF donor must contain one HEVC image without auxiliaries')
    if not any(info[8:12]==b'tmap' for info in infos.values()):
        raise ValueError('HEIF source lacks ISO tone-map item')
    removed=set()
    def descendants(item,seen):
        if item in seen:raise ValueError('cyclic HEIF image grid')
        for typ,src,targets in refs:
            if typ==b'dimg' and src==item:
                for child in targets:
                    removed.add(child);descendants(child,seen|{item})
    descendants(primary,set())
    # Grid descendants must not also supply the gain map or another rendition.
    for typ,src,targets in refs:
        if src!=primary and src not in removed and any(t in removed for t in targets):
            raise ValueError('shared primary grid tiles cannot be replaced')
    infos={k:v for k,v in infos.items() if k not in removed}
    payloads={k:v for k,v in payloads.items() if k not in removed}
    assocs={k:v for k,v in assocs.items() if k not in removed}
    refs=[r for r in refs if r[1] not in removed and not (r[0]==b'dimg' and r[1]==primary)]
    infos[primary]=dinfos[dp][:4]+struct.pack('>H',primary)+dinfos[dp][6:]
    payloads[primary]=dpayloads[dp]
    # Codec/geometry/colour belong to the newly encoded item. Preserve only
    # unrelated properties, e.g. HDR viewing environment metadata.
    replaced={b'hvcC',b'ispe',b'pixi',b'colr',b'clap',b'pasp',b'irot',b'imir'}
    prior=[(e,i) for e,i in assocs.get(primary,[]) if props[i-1][0] not in replaced]
    for e,i in assocs.get(primary,[]):
        if props[i-1][0] == b'imir' or (props[i-1][0] == b'irot' and any(props[i-1][1])):
            raise ValueError('rotated HEIF primary requires explicit pixel transform')
    offset=len(props);props+=dprops
    assocs[primary]=prior+[(e,i+offset) for e,i in dassoc.get(dp,[])]
    if len(props)>=0x8000:raise ValueError('too many HEIF properties')
    iinf=_box(b'iinf',bytes(4)+struct.pack('>H',len(infos))+b''.join(_box(b'infe',v) for v in infos.values()))
    iref=_box(b'iref',bytes(4)+b''.join(_box(t,struct.pack('>HH',src,len(targets))+b''.join(struct.pack('>H',i) for i in targets)) for t,src,targets in refs))
    assocdata=[]
    for item,values in assocs.items():
        if len(values)>255:raise ValueError('too many HEIF item properties')
        assocdata.append(struct.pack('>HB',item,len(values))+b''.join(struct.pack('>H',i|(0x8000 if e else 0)) for e,i in values))
    iprp=_box(b'iprp',_box(b'ipco',b''.join(_box(k,v) for k,v in props))+_box(b'ipma',b'\0\0\0\1'+struct.pack('>L',len(assocs))+b''.join(assocdata)))
    # Codec brands describe the new primary, tmap remains explicitly advertised.
    ftyp=_one(dtop,b'ftyp')
    brands=[ftyp[i:i+4] for i in range(8,len(ftyp),4)]
    if b'tmap' not in brands:brands.append(b'tmap')
    prefix=_box(b'ftyp',ftyp[:8]+b''.join(brands))
    def make_meta(start):
        records=[]
        for item,payload in payloads.items():
            records.append(struct.pack('>HHHHQQ',item,0,0,1,start,len(payload)))
            start+=len(payload)
        iloc=_box(b'iloc',b'\1\0\0\0\x88\0'+struct.pack('>H',len(records))+b''.join(records))
        out=[]
        replacements={b'iinf':iinf,b'iref':iref,b'iprp':iprp,b'iloc':iloc}
        for kind,value in children:
            if kind==b'idat':continue
            out.append(replacements.get(kind,_box(kind,value)))
        return _box(b'meta',bytes(4)+b''.join(out))
    meta=make_meta(0);meta=make_meta(len(prefix)+len(meta)+8)
    result=prefix+meta+_box(b'mdat',b''.join(payloads.values()))
    checked=_parse(result)
    if checked[7]!=payloads:raise ValueError('HEIF relocation failed verification')
    path.write_bytes(result)
