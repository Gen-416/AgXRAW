# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference-print appearance layer: plan object and asset contract
(FILM_APPEARANCE_RECIPE_PLAN §9/§11, phase P1).

P1 ships the INFRASTRUCTURE and nothing visible: the immutable plan, the
fail-closed recipe loader, and the strict fast paths. The palette kernel
itself is P2; until it lands, only IDENTITY recipes (every field zero) are
accepted, and a non-identity asset refuses to load rather than silently
doing nothing — an asset that promises a look and delivers a no-op would be
the quiet version of that lie.

Contracts that hold from day one:

- `technical` mode never touches an asset, never allocates, and returns the
  developed frame object unchanged — the P1 exit gate is that every frozen
  byte in the repo stays identical.
- The compiled `FilmAppearancePlan` is the ONLY thing the runtime consults
  (the A3 lesson, inherited here rather than re-learned): recipe arrays ride
  the plan as immutable payload, resolved and hash-verified at compile time.
- Unknown schema, wrong stock/medium pairing, bad hash, non-finite or
  mis-shaped fields: ValueError at load, never a default.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

APPEARANCE_SCHEMA = 1
APPEARANCE_DIR = Path(__file__).with_name("data") / "film_appearance"
MANIFEST_PATH = APPEARANCE_DIR / "MANIFEST.json"

APPEARANCE_MODES = ("technical", "reference")

# §6.6 axes the P2 kernel will interpolate over. Declared here so the loader
# can validate shapes before any kernel exists to consume them.
EV_KNOTS = (-6.0, -3.0, 0.0, 3.0, 6.0)
HUE_KNOT_COUNT = 24
STRENGTH_MAX = 1.5


@dataclass(frozen=True)
class FilmAppearancePlan:
    """Immutable compiled appearance state (plan §11).

    `recipe` holds the loaded, validated field arrays for reference mode —
    resolved at compile so the runtime never touches the filesystem or any
    mutable registry. Arrays are wrapped read-only.
    """

    mode: str = "technical"
    recipe_id: str = ""
    strength: float = 1.0
    provenance: str = ""
    asset_sha256: str = ""
    recipe: dict | None = field(default=None, compare=False)


_RECIPE_CACHE: dict[tuple, dict] = {}


def _manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise ValueError(
            f"外观层资产清单缺失:{MANIFEST_PATH}(fail closed,不使用未钉扎资产)"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def recipe_path(recipe_id: str) -> Path:
    return APPEARANCE_DIR / f"{recipe_id}.npz"


def load_recipe(recipe_id: str, *, stock_id: str, medium_id: str) -> dict:
    """Load and validate one recipe asset, fail-closed on every axis.

    The (stock, medium) pairing is part of the asset's IDENTITY: a recipe
    authored against Portra-on-Endura silently applied to another stock
    would be exactly the unaccountable LUT the plan exists to avoid.
    """
    rid = str(recipe_id)
    path = recipe_path(rid)
    if not path.is_file():
        raise ValueError(f"外观 recipe '{rid}' 不存在:{path}")
    stat = path.stat()
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    # The pairing is part of the cache key: identity validation must run
    # for every DISTINCT claim, or a correct earlier load would let a wrong
    # stock ride the cache past the checks (caught by P1's own gate test).
    cache_key = (rid, stat.st_mtime_ns, sha, str(stock_id), str(medium_id))
    got = _RECIPE_CACHE.get(cache_key)
    if got is not None:
        return got

    pinned = _manifest().get("files", {}).get(path.name)
    if pinned != sha:
        raise ValueError(
            f"外观 recipe '{rid}' 哈希不匹配清单(资产被改动或清单未再生)"
        )

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(np.asarray(z["meta"])))
        if int(meta.get("schema", -1)) != APPEARANCE_SCHEMA:
            raise ValueError(
                f"外观 recipe schema {meta.get('schema')} != {APPEARANCE_SCHEMA}"
            )
        for key, expect in (
            ("recipe_id", rid), ("stock_id", str(stock_id)),
            ("medium_id", str(medium_id)),
        ):
            if str(meta.get(key)) != expect:
                raise ValueError(
                    f"外观 recipe 身份不符:{key}={meta.get(key)!r} != {expect!r}"
                )
        if str(meta.get("process_space")) != "display-linear-rec2020/oklab+scene-ev":
            raise ValueError(
                f"外观 recipe 处理空间未知:{meta.get('process_space')!r}"
            )
        provenance = str(meta.get("provenance", ""))
        if provenance not in ("editorial-authored", "empirical-own-target"):
            raise ValueError(f"外观 recipe provenance 未知:{provenance!r}")

        ev = np.asarray(z["ev_knots"], dtype=np.float64)
        hue = np.asarray(z["hue_knots_deg"], dtype=np.float64)
        if ev.shape != (len(EV_KNOTS),) or not np.allclose(ev, EV_KNOTS):
            raise ValueError("外观 recipe EV 轴与声明网格不符")
        if hue.shape != (HUE_KNOT_COUNT,) or not np.allclose(
            hue, np.arange(HUE_KNOT_COUNT) * (360.0 / HUE_KNOT_COUNT)
        ):
            raise ValueError("外观 recipe hue 轴与声明网格不符")

        fields: dict = {}
        for name, shape in (
            ("hue_delta_deg", (len(EV_KNOTS), HUE_KNOT_COUNT)),
            ("log_chroma_gain", (len(EV_KNOTS), HUE_KNOT_COUNT)),
            ("density_ev", (len(EV_KNOTS), HUE_KNOT_COUNT)),
            ("neutral_bias_ab", (len(EV_KNOTS), 2)),
        ):
            arr = np.asarray(z[name], dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"外观 recipe 字段 {name} 形状 {arr.shape} != {shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"外观 recipe 字段 {name} 含非有限值")
            arr.setflags(write=False)
            fields[name] = arr

        # P1 hard gate: the palette kernel is P2. Until it exists, a recipe
        # carrying non-zero fields would load, apply nothing, and lie about
        # it — refuse instead. P2 removes this gate together with its test.
        if any(np.abs(fields[k]).max() > 0.0 for k in fields):
            raise ValueError(
                "外观 recipe 含非恒等字段,但 P2 palette 内核尚未落地——"
                "P1 只接受 identity recipe(fail closed)"
            )

        out = {
            "meta": meta,
            "provenance": provenance,
            "sha256": sha,
            **fields,
        }
    _RECIPE_CACHE.clear()
    _RECIPE_CACHE[cache_key] = out
    return out


def compile_appearance_plan(
    mode: str, strength: float, *, stock_id: str, medium_id: str
) -> FilmAppearancePlan:
    """Resolve the user's appearance selection into the immutable plan."""
    mode = str(mode or "technical")
    if mode not in APPEARANCE_MODES:
        raise ValueError(f"film_appearance={mode!r} 未知(可选 technical/reference)")
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= STRENGTH_MAX:
        raise ValueError(
            f"film_appearance_strength={strength!r} 域为 [0, {STRENGTH_MAX}]"
        )
    if mode == "technical":
        return FilmAppearancePlan(mode="technical")
    rid = f"{stock_id}__{medium_family(medium_id)}_reference_v1"
    recipe = load_recipe(rid, stock_id=stock_id, medium_id=medium_id)
    return FilmAppearancePlan(
        mode="reference",
        recipe_id=rid,
        strength=strength,
        provenance=recipe["provenance"],
        asset_sha256=recipe["sha256"],
        recipe=recipe,
    )


def medium_family(medium_id: str) -> str:
    """Collapse a concrete medium asset id to its recipe family name.

    Recipes are authored per stock x medium FAMILY (endura, 2383, direct):
    the translated/native variants of one paper share a palette intent, and
    a per-variant recipe would multiply authoring work without a measured
    reason. The concrete id still participates in load-time identity via
    meta.medium_id.
    """
    m = str(medium_id)
    if "endura" in m:
        return "endura"
    if "2383" in m or "2393" in m:
        return "print2383"
    if m.startswith("direct"):
        return "direct"
    return m or "unknown"


def apply_film_appearance(developed: np.ndarray, plan: FilmAppearancePlan) -> np.ndarray:
    """Apply the appearance layer to mapped Rec.2020 (post-neutralization,
    pre-delivery). P1: `technical` and strength 0 are STRICT identities (the
    same object back); reference mode carries only identity recipes, so the
    result is numerically identical — the call exists so the P2 kernel drops
    into an already-wired, already-cached slot.
    """
    if plan is None or plan.mode == "technical" or plan.strength == 0.0:
        return developed
    if plan.recipe is None:
        raise ValueError("reference 模式的 plan 缺少已解析 recipe(编译期错误)")
    # P2 kernel slot. With the P1 identity gate above, every field is zero
    # and the transform is exactly the identity; return unchanged.
    return developed
