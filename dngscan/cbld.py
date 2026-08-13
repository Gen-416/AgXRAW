# SPDX-License-Identifier: GPL-3.0-or-later
"""CBLD advisory black-level priors (Camera Black Level Database).

Upstream: https://y-g-jiang.github.io/CBLD.html by 知乎@姜尧耕 — measured
per-camera, per-ISO, per-readout-mode four-channel black levels with
sub-DN precision and high-ISO clipping flags.

REDISTRIBUTION (review R1 item 8): the upstream page credits its author
but carries no explicit redistribution license, and attribution is not
authorization — so the database is NOT shipped in this repository or in
the wheel. A user who wants the advisory line runs
``python tools/import_cbld.py`` themselves, which fetches the data from
the upstream site for their own local use (default
``~/.config/dngscan/cbld.json``; ``DNGSCAN_CBLD`` overrides the path).
Without a local import the module is silent — no line, no warning.

Doctrine: metadata black levels stay AUTHORITATIVE (A9 — the same rule
as WhiteLevel); CBLD is an advisory measured reference surfaced in the
report so a fractional-DN mismatch (the classic deep-shadow colour-cast
cause) is visible instead of silent. The sanctioned override path
remains the user's own dark frame.

Channel order contract: CBLD publishes R, G1, B, G2.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

CHANNEL_ORDER = ("R", "G1", "B", "G2")


def data_path() -> Path:
    """Where the user-imported database lives (R1 item 8: user-import
    only, never a packaged asset). ``DNGSCAN_CBLD`` wins; the default is
    the per-user config location tools/import_cbld.py writes to."""
    env = os.environ.get("DNGSCAN_CBLD")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "dngscan" / "cbld.json"


def _load_db() -> dict[str, Any]:
    try:
        payload = json.loads(data_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cameras": []}
    if payload.get("channel_order") != list(CHANNEL_ORDER):
        # fail closed on a contract change rather than mislabel channels
        return {"cameras": []}
    return payload


@lru_cache(maxsize=1)
def _db_cached(_path: str) -> dict[str, Any]:
    return _load_db()


def _db() -> dict[str, Any]:
    # keyed by the resolved path so tests (and env changes) reload
    return _db_cached(str(data_path()))


def find_black_levels(
    make: str | None, model: str | None, iso: float | None,
) -> dict[str, Any] | None:
    """Best advisory match for (make, model, iso), or None.

    Measured entries (实测值) win over recommended ones; the first
    shooting mode (the plain single-shot readout) is used — burst modes
    are surfaced through the entry for a caller that wants them; the
    nearest tabulated ISO is chosen and reported as `iso_matched`.
    """
    if not make or not model:
        return None
    make_u, model_u = make.upper(), model.upper()
    best = None
    for cam in _db().get("cameras", []):
        mk = cam.get("libraw_make_contains")
        if not mk or mk not in make_u:
            continue
        if not any(alias.upper() in model_u
                   for alias in cam.get("libraw_model_contains", [])):
            continue
        if best is None or (cam.get("measured") and not best.get("measured")):
            best = cam
    if best is None:
        return None
    modes = best.get("shootingModes") or []
    if not modes:
        return None
    rows = modes[0].get("data") or []
    if not rows:
        return None
    if iso is None:
        row = rows[0]
    else:
        row = min(rows, key=lambda r: abs(float(r.get("iso", 0)) - float(iso)))
    return {
        "camera": best.get("name", best.get("id", "?")),
        "measured": bool(best.get("measured")),
        "mode": modes[0].get("modeName", "?"),
        "iso_matched": row.get("iso"),
        "values": tuple(float(row[k]["avg"]) for k in ("r", "g1", "b", "g2")),
        "clipping": bool(row.get("clipping", False)),
        "advisory": _db().get("advisory", ""),
    }


def report_line(
    make: str | None, model: str | None, iso: float | None,
    metadata_levels: tuple[float, ...] | list[float] | None,
    color_desc: str | None = None,
) -> str | None:
    """One advisory report line, or None when CBLD has no match.

    The channel-wise mismatch comparison runs only when the decoder's
    color_desc is RGBG, i.e. LibRaw's channel order is R, G1, B, G2 and
    lines up with CBLD's published order — the upstream tutorial itself
    warns other cameras may report RG1G2B-style orders, and comparing
    across a mislabeled order would invent a mismatch. The reference
    values are still shown either way.
    """
    hit = find_black_levels(make, model, iso)
    if hit is None:
        return None
    vals = "/".join(f"{v:g}" for v in hit["values"])
    # Matching honesty (review R1 item 8): the row was chosen by substring
    # model match, the FIRST published shooting mode, and the NEAREST
    # tabulated ISO — a heuristic lookup, and only 实测 when the entry
    # itself is measured (推荐值 otherwise). The line says all of that
    # instead of presenting the number as this frame's measurement.
    kind = "实测" if hit["measured"] else "推荐值"
    line = (
        f"CBLD 黑电平参考({kind};启发式匹配 {hit['camera']},"
        f"模式={hit['mode']}(首个),ISO {hit['iso_matched']}(最近档)): "
        f"R/G1/B/G2 = {vals}"
    )
    if hit["clipping"]:
        line += "（该 ISO 黑电平有削底,均值偏高）"
    order_comparable = (color_desc or "").upper() == "RGBG"
    if metadata_levels and order_comparable:
        meta = list(metadata_levels)[:4]
        if meta and any(m > 0 for m in meta):
            worst = max(
                abs(float(m) - v) for m, v in zip(meta, hit["values"])
            ) if len(meta) >= 4 else None
            if worst is not None and worst >= 0.5:
                line += f"；与元数据最大差 {worst:.2f} DN(暗部偏色候选成因)"
    elif metadata_levels and not order_comparable:
        line += "；通道顺序非 RGBG,不与元数据逐通道比较"
    line += "；数据:知乎@姜尧耕,仅供参考"
    return line
