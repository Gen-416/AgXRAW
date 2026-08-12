#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import the CBLD camera black-level database into a pinned asset.

CBLD (Camera Black Level Database, https://y-g-jiang.github.io/CBLD.html,
by 知乎@姜尧耕) publishes MEASURED per-camera, per-ISO, per-readout-mode
four-channel black levels with sub-DN precision and high-ISO clipping
flags — exactly the per-camera calibration corpus the policy register
names for the black-level/clip family. The site ships the data embedded
in its JS bundle; this tool extracts it, converts it to a stable JSON
asset with provenance, and pins the channel-order contract (CBLD order is
R, G1, B, G2).

The data is ADVISORY in dngscan (the author's own header says 仅供参考,
最好以自己机器的当次拍摄为准): metadata black levels stay authoritative
(A9 doctrine) and CBLD rides the report as a measured reference.

    python tools/import_cbld.py            # fetch live bundle and refresh
    python tools/import_cbld.py bundle.js  # convert a local copy
"""
from __future__ import annotations

import datetime
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dngscan" / "data" / "cbld.json"
PAGE = "https://y-g-jiang.github.io/CBLD.html"


def _bundle_url() -> str:
    html = urllib.request.urlopen(PAGE, timeout=30).read().decode()
    m = re.search(r'src="([^"]*assets/index-[^"]+\.js)"', html)
    if not m:
        raise RuntimeError("CBLD bundle script not found on the page")
    src = m.group(1)
    return src if src.startswith("http") else f"https://y-g-jiang.github.io/{src.lstrip('/')}"


def extract(bundle: str) -> list[dict]:
    start = bundle.find("=[{id:")
    # find the array assignment that contains shootingModes
    for m in re.finditer(r'\w+=\[\{"?id"?:', bundle):
        i = bundle.find("[", m.start())
        depth = 0
        for j in range(i, len(bundle)):
            if bundle[j] == "[":
                depth += 1
            elif bundle[j] == "]":
                depth -= 1
                if depth == 0:
                    seg = bundle[i:j + 1]
                    break
        else:
            continue
        if "shootingModes" not in seg:
            continue
        seg = re.sub(r"ee\(([-\d.eE]+),([-\d.eE]+),([-\d.eE]+)\)",
                     r'{"avg":\1,"range":[\2,\3]}', seg)
        seg = re.sub(r"Pn\(([-\d.eE]+)\)", r'{"avg":\1,"range":[\1,\1]}', seg)
        seg = seg.replace(":!0", ":true").replace(":!1", ":false")
        seg = re.sub(r"([{,])([A-Za-z_][A-Za-z0-9_]*):", r'\1"\2":', seg)
        return json.loads(seg)
    raise RuntimeError("no camera array with shootingModes found in bundle")


# LibRaw (make_contains, model_contains) aliases per CBLD id. Extend as the
# upstream database grows; unknown new ids import fine but stay unmatched
# until an alias is added here.
ALIASES = {
    "nikon-D810": ("NIKON", ["D810"]),
    "nikon-z6ii": ("NIKON", ["Z 6_2", "Z6 II", "Z 6 II"]),
    "olympus-om1": ("OM DIGITAL", ["OM-1"]),
    "olympus-em1m3": ("OLYMPUS", ["E-M1 MARK III", "E-M1MarkIII".upper()]),
    "panasonix-s1rm2": ("PANASONIC", ["DC-S1RM2", "S1RM2"]),
    "panasonic-g9": ("PANASONIC", ["DC-G9", "G9"]),
}


def main() -> int:
    if len(sys.argv) > 1:
        bundle = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
        source = sys.argv[1]
    else:
        url = _bundle_url()
        bundle = urllib.request.urlopen(url, timeout=60).read().decode(
            "utf-8", errors="replace"
        )
        source = url
    cams = extract(bundle)
    for cam in cams:
        alias = ALIASES.get(cam["id"])
        cam["libraw_make_contains"] = alias[0] if alias else None
        cam["libraw_model_contains"] = alias[1] if alias else []
        cam["measured"] = "实测" in cam.get("name", "")
    payload = {
        "schema": 1,
        "source": source,
        "page": PAGE,
        "author": "知乎@姜尧耕 (y-g-jiang.github.io)",
        "advisory": "仅供参考，最好以自己机器的当次拍摄为准（上游原话）",
        "channel_order": ["R", "G1", "B", "G2"],
        "fetched": datetime.date.today().isoformat(),
        "cameras": cams,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(cams)} cameras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
