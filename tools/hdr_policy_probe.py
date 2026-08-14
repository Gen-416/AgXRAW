# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR policy-constant probe over a local corpus (R2 item 7 groundwork).

The HDR latitude constants (RHO_BASE, the clip/P3 confidence slopes, the
white margins/minimums, the shoulder starts — see dngscan/policy.py) are
declared "awaiting corpus calibration". This tool prints, per frame, every
quantity those constants gate, so the owner can re-pin them against a real
EDR corpus with evidence in hand instead of adjectives.

It changes NOTHING. Any value change goes through the consuming module,
a POLICY_VERSION bump and a new fingerprint (dngscan/policy.py).

Usage:
    python tools/hdr_policy_probe.py ~/Pictures/AgXRAW样张/*.DNG
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dngscan.analysis import analyze
from dngscan.grade import RENDER_MODE
from dngscan.hdr_agx_plan import (
    compile_channel_separation,
    compile_hdr_agx_plan,
    compile_tail_snr_gate,
)
from dngscan.raw_io import load_raw
from dngscan.tone import build_render_plan


def probe(path: Path) -> None:
    bundle = load_raw(path, scene_half_size=True)
    analysis, _, _ = analyze(bundle, margin=4)
    plan = build_render_plan(bundle, analysis, RENDER_MODE, "p3")
    hdr = compile_hdr_agx_plan(
        plan, analysis=analysis, scene_decoder=str(bundle.scene_decoder)
    )
    scene = getattr(plan, "scene", None)
    k_all = getattr(analysis, "cell_k_of_all_pct", None) or {}
    multi_pct = sum(float(k_all.get(k, 0.0) or 0.0) for k in (2, 3, 4))
    gamut = getattr(analysis, "gamut_out_pct", None) or {}
    print(f"== {path.name}")
    print(
        f"  sparse_emitter={bool(getattr(scene, 'sparse_emitter_tail', False))}"
        f"  reliable_tail={hdr.tone.reliable_tail_ev:+.2f}EV"
        f"  white={hdr.tone.white_ev:.2f}EV"
        f"  margin={hdr.tone.white_margin_ev:.2f}EV"
        f"  shoulder_start={hdr.tone.shoulder_start_ev:.2f}EV"
    )
    print(
        f"  multi_clip={multi_pct:.3f}%"
        f"  p3_out={float(gamut.get('Display P3', gamut.get('P3', 0.0)) or 0.0):.3f}%"
        f"  tail_snr_gate={compile_tail_snr_gate(analysis):.3f}"
        f"  rho={compile_channel_separation(analysis, str(bundle.scene_decoder)):.3f}"
        f"  (x gate -> {hdr.color.channel_separation * hdr.color.snr_gate:.3f})"
    )
    print(
        f"  requested={hdr.tone.requested_headroom_ev:.2f}EV"
        f"  rendered={hdr.tone.rendered_headroom_ev:.2f}EV"
        f"  alpha={hdr.tone.shoulder_alpha:.3f}"
        f"  segments={len(hdr.tone.shoulder_segments)}"
    )


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        try:
            probe(p)
        except Exception as exc:  # a broken frame should not end the survey
            print(f"== {p.name}\n  探测失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
