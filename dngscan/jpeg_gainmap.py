# SPDX-License-Identifier: GPL-3.0-or-later
"""Replace the primary JPEG codestream while preserving ISO gain-map packaging.

The gain map is derived from the finished, uncompressed rendition pair. ImageIO
owns that calculation/metadata; libjpeg owns independent primary quality and
sampling. MPF offsets are relocated, never copied with stale byte addresses.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path


def _segments(data: bytes):
    if data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    pos = 2
    while pos < len(data):
        start = pos
        if data[pos] != 255:
            raise ValueError('invalid JPEG marker')
        while pos < len(data) and data[pos] == 255:
            pos += 1
        if pos + 3 > len(data):
            raise ValueError('truncated JPEG marker')
        marker = data[pos]
        pos += 1
        if marker == 0xd9:
            return
        if marker in (0xd8, 0x01) or 0xd0 <= marker <= 0xd7:
            raise ValueError('unexpected standalone JPEG marker')
        size = int.from_bytes(data[pos:pos+2], 'big')
        end = pos + size
        if size < 2 or end > len(data):
            raise ValueError('truncated JPEG segment')
        yield marker, start, end, pos + 2
        if marker == 0xda:
            return
        pos = end


def replace_primary(path: Path, rgb, quality: int, chroma: str) -> None:
    from PIL import Image
    source = path.read_bytes()
    segments = list(_segments(source))
    if not segments or segments[-1][0] != 0xda:
        raise ValueError('JPEG has no scan')
    if not any(m == 0xc0 for m,*_ in segments):
        raise ValueError('gain-map primary replacement requires baseline JPEG')
    eoi = source.find(b'\xff\xd9', segments[-1][2])
    if eoi < 0:
        raise ValueError('JPEG scan has no end')
    old_end = eoi + 2
    metadata = [(m, source[a:b], a, p-a) for m,a,b,p in segments if 0xe0 <= m <= 0xef]
    if sum(blob[off:off+4] == b'MPF\0' for _,blob,_,off in metadata) != 1:
        raise ValueError('ISO gain-map JPEG must have exactly one MPF index')
    encoded = io.BytesIO()
    Image.fromarray(rgb).save(encoded, format='JPEG', quality=int(quality),
                             subsampling={'444':0,'422':1,'420':2}[chroma], optimize=True)
    coded = encoded.getvalue()
    parts = [coded[a:b] for m,a,b,_ in _segments(coded) if not 0xe0 <= m <= 0xef]
    scan_end = list(_segments(coded))[-1][2]
    body = b''.join(parts) + coded[scan_end:]
    new_end = 2 + sum(len(blob) for _,blob,_,_ in metadata) + len(body)
    relocated = []
    new_pos = 2
    for _,blob,old_pos,payload in metadata:
        if blob[payload:payload+4] == b'MPF\0':
            blob = bytearray(blob)
            tiff = payload + 4
            order = bytes(blob[tiff:tiff+2])
            if order not in (b'II', b'MM'):
                raise ValueError('invalid MPF byte order')
            endian = '<' if order == b'II' else '>'
            if struct.unpack_from(endian+'H', blob, tiff+2)[0] != 42:
                raise ValueError('invalid MPF TIFF header')
            ifd = tiff + struct.unpack_from(endian+'L', blob, tiff+4)[0]
            n = struct.unpack_from(endian+'H',blob,ifd)[0]
            found = False
            for i in range(n):
                tag, typ, count, offset = struct.unpack_from(endian+'HHLL',blob,ifd+2+12*i)
                if tag != 0xb002:
                    continue
                if typ != 7 or count % 16 or count < 32 or tiff+offset+count > len(blob):
                    raise ValueError('invalid MPEntry array')
                found = True
                for j in range(count//16):
                    at = tiff+offset+16*j
                    old_size, old_offset = struct.unpack_from(endian+'LL',blob,at+4)
                    if j == 0:
                        if old_offset != 0 or old_size != old_end:
                            raise ValueError('MPF primary boundaries do not match JPEG')
                        struct.pack_into(endian+'L',blob,at+4,new_end)
                    else:
                        absolute = old_pos+tiff+old_offset
                        if absolute < old_end or absolute+old_size > len(source):
                            raise ValueError('MPF auxiliary extent is outside JPEG')
                        offset_new = absolute + new_end-old_end - (new_pos+tiff)
                        struct.pack_into(endian+'L',blob,at+8,offset_new)
            if not found:
                raise ValueError('missing MPEntry array')
        relocated.append(bytes(blob))
        new_pos += len(blob)
    path.write_bytes(b'\xff\xd8'+b''.join(relocated)+body+source[old_end:])
