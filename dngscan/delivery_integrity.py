# SPDX-License-Identifier: GPL-3.0-or-later
"""Compressed rendition identity across best-effort capture-metadata rewrites.

Offsets and item IDs may move. Image samples, decoder properties, colour profiles
and HDR rendition relationships must remain unchanged after a verified encode.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def encoded_content_signature(path: Path, container: str):
    data = path.read_bytes()
    if container == "jpeg":
        from .jpeg_gainmap import _segments

        segments = list(_segments(data))
        if not segments or segments[-1][0] != 0xda or not any(s[0] == 0xc0 for s in segments):
            raise ValueError("metadata carry requires a baseline JPEG")
        scan = segments[-1][2]
        end = data.find(b"\xff\xd9", scan)
        if end < 0:
            raise ValueError("truncated JPEG scan")
        digest = sha256()
        for marker, start, stop, payload in segments:
            # EXIF/XMP and comments may change. MPF contains relocated offsets.
            # Keep ICC, ISO gain-map metadata, Adobe colour transform, coding
            # tables, frame and scan headers. Auxiliary JPEG bytes stay exact.
            if marker in (0xe1, 0xfe) or (marker == 0xe2 and data[payload:payload+4] == b"MPF\0"):
                continue
            digest.update(data[start:stop])
        digest.update(data[scan:])
        return digest.digest()
    if container != "heic":
        raise ValueError("unsupported delivery container")
    from .heif_gainmap import _parse

    _, _, primary, infos, refs, props, assocs, payloads = _parse(data)
    nodes = {}
    for item, info in infos.items():
        kind = info[8:12]
        if kind in (b"Exif", b"mime", b"uri "):
            continue
        # ImageIO may remove an explicit zero rotation. This is identity;
        # imir=0 is still a reflection and must never be normalized away.
        descriptive, transforms = [], []
        for essential, idx in assocs.get(item, []):
            prop, value = props[idx-1]
            if (prop, value) == (b"irot", b"\0"):
                continue
            # Descriptive properties have no application order. ImageIO also
            # marks the unchanged ICC colr as essential; every supported
            # decoder here already consumes that property in either case.
            entry = (prop, essential or prop == b"colr", sha256(value).digest())
            (descriptive if prop in (b"hvcC", b"colr", b"ispe", b"pixi", b"auxC")
             else transforms).append(entry)
        properties = tuple(sorted(descriptive)), tuple(transforms)
        nodes[item] = (kind, sha256(payloads[item]).digest(), properties)
    if primary not in nodes:
        raise ValueError("HEIF primary is not an image")
    links = []
    for kind, source, targets in refs:
        if source not in nodes:
            continue  # Capture metadata's cdsc references may be added.
        if any(target not in nodes for target in targets):
            raise ValueError("image reference points outside rendition items")
        links.append((kind, nodes[source], tuple(nodes[target] for target in targets)))
    return nodes[primary], tuple(sorted(nodes.values())), tuple(sorted(links))
