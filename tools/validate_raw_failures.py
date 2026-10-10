#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check explicit RAW failure on damaged temporary copies, never original files.

This checks chosen structural corruption, not arbitrary pixel tampering. A RAW
without authenticated integrity data cannot distinguish every altered, still
syntactically valid pixel stream from an intentional exposure.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dngscan import metadata as md


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def main_cfa_ifd(path):
    """Bounded metadata-only lookup for a controlled test mutation."""
    with path.open("rb") as source:
        head = source.read(8)
        if len(head) != 8 or head[:2] not in (b"II",b"MM"):
            return None
        endian = "<" if head[:2] == b"II" else ">"
        if struct.unpack(endian+"H",head[2:4])[0] != 42:
            return None
        first, = struct.unpack(endian+"L",head[4:])
        visited, candidates = set(), []
        def visit(offset):
            if not offset or offset in visited or len(visited) >= 64:
                return
            visited.add(offset)
            entries = md._read_ifd_entries(source,offset,endian)
            identity = {tag:md._entry_values(source,typ,count,data,endian)
                        for tag,typ,count,data in entries
                        if tag in (254,262,330) and count <= 64}
            if ((identity.get(254) or [0])[0] == 0
                    and (identity.get(262) or [0])[0] == 32803):
                candidates.append((offset,entries,endian))
            for sub in identity.get(330,[]):
                visit(int(sub))
        offset = first
        while offset and offset not in visited and len(visited)<64:
            visit(offset)
            source.seek(offset)
            count_bytes = source.read(2)
            if len(count_bytes) != 2:
                break
            count, = struct.unpack(endian+"H",count_bytes)
            source.seek(offset+2+12*count)
            following = source.read(4)
            offset = struct.unpack(endian+"L",following)[0] if len(following)==4 else 0
        return candidates[0] if len(candidates)==1 else None


def make_variant(source, target, mutation):
    """Create one copy and return its exact structural mutation description."""
    if mutation in ("prefix-4096", "truncated-half"):
        limit = min(4096, source.stat().st_size//2) if mutation == "prefix-4096" else source.stat().st_size//2
        with source.open("rb") as original, target.open("wb") as broken:
            remaining = limit
            while remaining:
                block = original.read(min(1048576,remaining))
                if not block:
                    break
                broken.write(block)
                remaining -= len(block)
        return {"kind":mutation,"retained_bytes":limit}
    if mutation == "invalid-header":
        shutil.copyfile(source,target)
        with target.open("r+b") as broken:
            broken.write(b"FAIL")
        return {"kind":mutation,"offset":0,"replacement_hex":"4641494c"}
    if mutation == "unsupported-main-cfa-compression":
        found = main_cfa_ifd(source)
        if found is None:
            return None
        offset,entries,endian = found
        for index,(tag,typ,count,data) in enumerate(entries):
            if tag == 259 and typ in (3,4) and count == 1:
                previous = md._entry_values(io.BytesIO(),typ,count,data,endian)[0]
                shutil.copyfile(source,target)
                with target.open("r+b") as broken:
                    broken.seek(offset+2+12*index+8)
                    broken.write(struct.pack(endian+{3:"H",4:"L"}[typ],65534))
                return {"kind":mutation,"ifd_offset":offset,"original_compression":previous,
                        "replacement_compression":65534}
        return None
    raise ValueError("unknown test mutation")


def probe(path):
    import numpy as np
    from dngscan.raw_io import load_raw
    try:
        bundle = load_raw(path,scene_half_size=True)
    except Exception as exc:
        return {"decode_supported":False,"render_supported":False,"measurement_qualified":False,
                "analysis_attempted":False,"published_output":False,
                "status":"explicit-decoder-failure","exception_type":type(exc).__name__,"error":str(exc)}
    # Unexpected acceptance fails this check. Do not continue to physical
    # analysis or export simply because a buffer was returned.
    raw = bundle.raw_image
    return {"decode_supported":True,"render_supported":False,"measurement_qualified":False,
            "analysis_attempted":False,"published_output":False,"status":"decoded",
            "raw_shape":list(raw.shape),"raw_nonzero_fraction":float(np.count_nonzero(raw)/raw.size)}


def isolated_probe(path, temporary):
    """A damaged decoder input must not take down the parent acceptance run."""
    clean = lambda value: str(value).replace(str(temporary),"<temporary>")
    try:
        child = subprocess.run([sys.executable,str(Path(__file__).resolve()),"--probe",str(path)],
                               capture_output=True,text=True,timeout=60)
    except subprocess.TimeoutExpired:
        return {"process_returncode":None,"stderr":"decoder probe exceeded 60 seconds",
                "result":{"status":"probe-timeout"}}
    try:
        result = json.loads(child.stdout.splitlines()[-1]) if child.returncode == 0 else {}
    except (ValueError,IndexError):
        result = {}
    return {"process_returncode":child.returncode,"stderr":clean(child.stderr[-2048:]),
            "result":{key:clean(value) if key=="error" else value for key,value in result.items()}}


def run_cases(path, temporary):
    before = digest(path)
    row = {"file":path.name,"source_sha256":before,"source_bytes":path.stat().st_size,"cases":[]}
    control = isolated_probe(path,temporary)
    row["control"] = {**control,"passed":control["process_returncode"]==0
                      and control["result"].get("decode_supported") is True}
    for mutation in ("invalid-header","prefix-4096","truncated-half","unsupported-main-cfa-compression"):
        target = temporary/(mutation+path.suffix)
        description = make_variant(path,target,mutation)
        if description is None:
            row["cases"].append({"mutation":mutation,"status":"not-applicable","passed":None,
                                 "reason":"no-unique-classic-TIFF-main-CFA-IFD"})
            continue
        outcome = isolated_probe(target,temporary)
        row["cases"].append({"mutation":description,"copy_sha256":digest(target),"copy_bytes":target.stat().st_size,
                **outcome,"passed":outcome["process_returncode"]==0
                and outcome["result"].get("status")=="explicit-decoder-failure"})
        target.unlink()
    row["original_unchanged"] = before == digest(path)
    return row


def main():
    if len(sys.argv)==3 and sys.argv[1]=="--probe":
        print(json.dumps(probe(Path(sys.argv[2])),allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files",type=Path,nargs="+")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    import rawpy
    report = {"schema":"agxraw-raw-failure-acceptance-1","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "working_changes":True,"runtime":{"python":platform.python_version(),"rawpy":rawpy.__version__,
                         "libraw_version":list(rawpy.libraw_version),"libraw_commit":"e419de08001de28ae6988ecb22df47e52b9c5eaa"},
            "parameters":{"decoder":"libraw","half_size":True,"highlight":"clip","analysis_on_failure":False},
            "scope":"chosen structural corruption only; no guarantee to detect arbitrary syntactically valid pixel mutation", "samples":[]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agxraw-failure-acceptance-") as temporary:
        for path in args.files:
            print(f"Checking damaged copies of {path.name}",flush=True)
            report["samples"].append(run_cases(path,Path(temporary)))
            args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    return int(any(not row["control"]["passed"] or not row["original_unchanged"]
                   or any(case["passed"] is False for case in row["cases"])
                   for row in report["samples"]))


if __name__=="__main__":
    raise SystemExit(main())
