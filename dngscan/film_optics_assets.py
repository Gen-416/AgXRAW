# SPDX-License-Identifier: GPL-3.0-or-later
"""Analog-optics assets and the compiled runtime plan
(FILM_OPTICS_V2 §7, phase P1).

There is no longer one global `OpticsProfile`. Spatial imaging happens in
three different materials with three different owners, and collapsing them
into a single struct is what let a print-medium scatter constant get read as
a film property and a modelled halo radius get quoted as if it were measured.
So the data splits the way the physics does:

    stock optics    negative/reversal grain, in-emulsion scatter, backing
                    halation, anti-halation class
    print optics    formation scatter, positive-medium grain, viewing
                    scatter, and (for now) the legacy print scatter that the
                    GUI still calls "Bloom"
    capture bloom   an EDITORIAL lens-and-emulsion glow, declared as such

Every asset carries its own provenance — `measured`, `derived`, `modelled` or
`editorial` — because the honest answer differs field by field, and a report
that says "modelled" for a whole profile is either too pessimistic about the
measured parts or too generous about the invented ones.

Loading is fail-closed: unknown schema, unknown model name, a missing file, a
hash mismatch or a non-finite number all raise. A spatial operator that
silently falls back to a default is a spatial operator whose output nobody can
attribute.

Naming note (§12.2): the operator the GUI calls "Bloom" is, in this version,
the positive medium's own conservative scatter. It is `legacy_print_scatter`
here. The editorial capture bloom that the name will eventually mean does not
exist yet; P3 introduces it, and the two must never share a field.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
ASSET_DIR = Path(__file__).resolve().parent / "data" / "film_optics"
MANIFEST_PATH = ASSET_DIR / "MANIFEST.json"

PROVENANCE = ("measured", "derived", "modelled", "editorial")


class OpticsAssetError(ValueError):
    """Any refusal to load or compile an optics asset."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise OpticsAssetError(msg)


def _finite(value: Any, name: str) -> float:
    v = float(value)
    _require(np.isfinite(v), f"{name} is not finite")
    return v


def _finite_triple(value: Any, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64)
    _require(arr.shape == (3,), f"{name} must have three components")
    _require(bool(np.isfinite(arr).all()), f"{name} is not finite")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _provenance(value: Any, name: str) -> str:
    s = str(value)
    _require(s in PROVENANCE, f"{name} provenance {s!r} not in {PROVENANCE}")
    return s


# --------------------------------------------------------------------------
# component assets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GrainAsset:
    """Granularity of one emulsion.

    `model` names the maths, and the runtime dispatches on it rather than on
    which fields happen to be present. P4 adds a second model; the two must be
    distinguishable by a reader of the asset alone.
    """

    provenance: str
    medium: str                  # negative | reversal | print_film | paper
    model: str                   # band_limited_gaussian_v1
    pitch_um: float
    size_um: float
    sigma0: float
    layer_corr: float

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "GrainAsset":
        model = str(raw.get("model", ""))
        _require(model == "band_limited_gaussian_v1",
                 f"{where}: unknown grain model {model!r}")
        pitch = _finite(raw["pitch_um"], f"{where}.pitch_um")
        size = _finite(raw["size_um"], f"{where}.size_um")
        _require(pitch > 0.0 and size > 0.0, f"{where}: grain scales must be positive")
        sigma0 = _finite(raw["sigma0"], f"{where}.sigma0")
        _require(sigma0 >= 0.0, f"{where}: sigma0 must be non-negative")
        corr = _finite(raw["layer_corr"], f"{where}.layer_corr")
        _require(0.0 <= corr <= 1.0, f"{where}: layer_corr outside [0,1]")
        return cls(
            provenance=_provenance(raw["provenance"], where),
            medium=str(raw["medium"]),
            model=model,
            pitch_um=pitch, size_um=size, sigma0=sigma0, layer_corr=corr,
        )


@dataclass(frozen=True)
class HalationComponent:
    """One radial component of the backing reflection (§5.3).

    Three exist in practice and they are not the same effect at three
    strengths: `local` is the main halo hugging a high-contrast edge, `global`
    is a low-amplitude wide red glare, and `aura` is the very large return a
    stock without remjet or with a weak anti-halation layer produces. Each
    carries its own radius, its own per-layer trigger and its own non-negative
    transfer matrix, which is what lets a white source come back warm at the
    inner ring and red at the outer one.
    """

    name: str
    radius_mm: float
    gate_ev: Any                 # [3, 2] per-layer smootherstep (t0, t1) in EV
    transfer: Any                # [3, 3] non-negative, rows = destination layer

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "HalationComponent":
        radius = _finite(raw["radius_mm"], f"{where}.radius_mm")
        _require(radius > 0.0, f"{where}: radius must be positive")
        gate = np.asarray(raw["gate_ev"], dtype=np.float64)
        _require(gate.shape == (3, 2), f"{where}.gate_ev must be [3, 2]")
        _require(bool(np.isfinite(gate).all()), f"{where}.gate_ev is not finite")
        # A zero-width gate is a hard threshold, which §10.1 gate 4 forbids:
        # the source mask has to stay C1 or the halo grows a visible contour.
        _require(bool(np.all(gate[:, 1] - gate[:, 0] >= 0.5)),
                 f"{where}.gate_ev needs at least 0.5 EV of smootherstep width")
        transfer = np.asarray(raw["transfer"], dtype=np.float64)
        _require(transfer.shape == (3, 3), f"{where}.transfer must be [3, 3]")
        _require(bool(np.isfinite(transfer).all()),
                 f"{where}.transfer is not finite")
        _require(bool((transfer >= 0.0).all()),
                 f"{where}.transfer must be non-negative")
        return cls(
            name=str(raw["name"]), radius_mm=radius,
            gate_ev=gate, transfer=transfer,
        )


@dataclass(frozen=True)
class HalationAsset:
    """Backing-reflection halation.

    `dc_mode` is the field R1 exists for. `additive` reinjects the spread
    itself, which double-counts the uniform-field response already baked into
    the characteristic curve (measured at +0.95% frame-wide red in P0);
    `residual` reinjects only the spatial part. It stays a declared mode
    rather than an implicit property of the code so an asset says which
    physics it was authored against.
    """

    provenance: str
    model: str                   # layer_components_v1
    dc_mode: str                 # additive | residual
    anti_halation_class: str
    components: tuple[HalationComponent, ...]

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "HalationAsset":
        model = str(raw.get("model", ""))
        _require(model == "layer_components_v1",
                 f"{where}: unknown halation model {model!r}")
        dc_mode = str(raw.get("dc_mode", ""))
        _require(dc_mode in ("additive", "residual"),
                 f"{where}: unknown dc_mode {dc_mode!r}")
        comps = raw.get("components") or ()
        _require(len(comps) >= 1, f"{where}: needs at least one component")
        parsed = tuple(
            HalationComponent.from_json(c, f"{where}.components[{i}]")
            for i, c in enumerate(comps)
        )
        names = [c.name for c in parsed]
        _require(len(set(names)) == len(names),
                 f"{where}: duplicate component names {names}")
        return cls(
            provenance=_provenance(raw["provenance"], where),
            model=model, dc_mode=dc_mode,
            anti_halation_class=str(raw["anti_halation_class"]),
            components=parsed,
        )

    def total_return(self) -> np.ndarray:
        """Summed transfer over components — the per-layer energy a fully
        gated source hands back. The number a report should quote when it
        claims a profile is 'strong' or 'weak'."""
        return np.sum([c.transfer for c in self.components], axis=0)


@dataclass(frozen=True)
class PrintScatterAsset:
    """The positive medium's own conservative scatter — the GUI's "Bloom".

    Named for what it is. §12.2: the user-facing control keeps its label for
    now, but nothing internal may call this bloom, or the editorial glow P3
    introduces will end up sharing its amount by accident.
    """

    provenance: str
    model: str                   # conservative_pyramid_v1
    levels: int
    threshold: float
    strength: float

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "PrintScatterAsset":
        model = str(raw.get("model", ""))
        _require(model == "conservative_pyramid_v1",
                 f"{where}: unknown print-scatter model {model!r}")
        levels = int(raw["levels"])
        _require(levels >= 1, f"{where}: levels must be >= 1")
        threshold = _finite(raw["threshold"], f"{where}.threshold")
        _require(threshold >= 0.0, f"{where}: threshold must be non-negative")
        strength = _finite(raw["strength"], f"{where}.strength")
        _require(strength >= 0.0, f"{where}: strength must be non-negative")
        return cls(
            provenance=_provenance(raw["provenance"], where),
            model=model, levels=levels, threshold=threshold, strength=strength,
        )


# --------------------------------------------------------------------------
# family assets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StockOpticsAsset:
    asset_id: str
    provenance: str
    gate_reference_mm: tuple[float, float]
    anti_halation: str
    grain: GrainAsset | None
    halation: HalationAsset | None
    emulsion_scatter: Any = None          # P2
    source_notes: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "StockOpticsAsset":
        _require(str(raw.get("kind")) == "stock_optics",
                 f"{where}: not a stock_optics asset")
        gate = np.asarray(raw["gate_reference_mm"], dtype=np.float64)
        _require(gate.shape == (2,) and bool((gate > 0).all()),
                 f"{where}: gate_reference_mm must be two positive lengths")
        grain = raw.get("grain")
        halation = raw.get("halation")
        _require(raw.get("emulsion_scatter") is None,
                 f"{where}: emulsion_scatter is a P2 field and must be null here")
        return cls(
            asset_id=str(raw["asset_id"]),
            provenance=_provenance(raw["provenance"], where),
            gate_reference_mm=(float(gate[0]), float(gate[1])),
            anti_halation=str(raw["anti_halation"]),
            grain=GrainAsset.from_json(grain, f"{where}.grain") if grain else None,
            halation=(
                HalationAsset.from_json(halation, f"{where}.halation")
                if halation else None
            ),
            source_notes=tuple(str(s) for s in raw.get("source_notes", ())),
        )


@dataclass(frozen=True)
class PrintOpticsAsset:
    asset_id: str
    provenance: str
    print_scatter: PrintScatterAsset | None
    formation_scatter: Any = None         # P3
    positive_grain: GrainAsset | None = None   # P4
    viewing_scatter: Any = None           # deliberately never enabled in v2
    source_notes: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "PrintOpticsAsset":
        _require(str(raw.get("kind")) == "print_optics",
                 f"{where}: not a print_optics asset")
        _require(raw.get("formation_scatter") is None,
                 f"{where}: formation_scatter is a P3 field and must be null here")
        _require(raw.get("positive_grain") is None,
                 f"{where}: positive_grain is a P4 field and must be null here")
        _require(raw.get("viewing_scatter") is None,
                 f"{where}: viewing_scatter has no measured PSF and stays null")
        scatter = raw.get("legacy_print_scatter")
        return cls(
            asset_id=str(raw["asset_id"]),
            provenance=_provenance(raw["provenance"], where),
            print_scatter=(
                PrintScatterAsset.from_json(
                    scatter, f"{where}.legacy_print_scatter"
                ) if scatter else None
            ),
            source_notes=tuple(str(s) for s in raw.get("source_notes", ())),
        )


@dataclass(frozen=True)
class CaptureBloomAsset:
    """Editorial lens-and-emulsion glow. Inert until P3.

    Present as an explicitly inactive asset rather than as a missing one, so
    the compiled plan's shape does not change when P3 lands and so a reader
    can see that the slot exists and is deliberately empty.
    """

    asset_id: str
    provenance: str
    active: bool

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "CaptureBloomAsset":
        _require(str(raw.get("kind")) == "capture_bloom",
                 f"{where}: not a capture_bloom asset")
        active = bool(raw["active"])
        _require(not active, f"{where}: capture bloom has no P1 implementation")
        return cls(
            asset_id=str(raw["asset_id"]),
            provenance=_provenance(raw["provenance"], where),
            active=active,
        )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

_CACHE: dict[str, Any] = {}


def _manifest() -> dict:
    got = _CACHE.get("__manifest__")
    if got is None:
        _require(MANIFEST_PATH.is_file(), f"missing optics manifest {MANIFEST_PATH}")
        got = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        _require(int(got.get("schema", -1)) == SCHEMA,
                 f"optics manifest schema {got.get('schema')} != {SCHEMA}")
        _CACHE["__manifest__"] = got
    return got


def _load_json(name: str) -> dict:
    """Read one asset, verifying its manifest hash before parsing.

    The hash is checked against the BYTES, before json.loads: a corrupted or
    substituted asset must be refused as a file, not diagnosed field by field
    after half of it has already been trusted.
    """
    path = ASSET_DIR / f"{name}.json"
    _require(path.is_file(), f"missing optics asset {path}")
    payload = path.read_bytes()
    pinned = _manifest().get("files", {}).get(f"{name}.json")
    _require(pinned is not None, f"{name}.json is not pinned in the manifest")
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == pinned,
             f"{name}.json hash {digest[:12]} != pinned {str(pinned)[:12]}")
    raw = json.loads(payload.decode("utf-8"))
    _require(int(raw.get("schema", -1)) == SCHEMA,
             f"{name}: schema {raw.get('schema')} != {SCHEMA}")
    return raw


def load_stock_optics(asset_id: str) -> StockOpticsAsset:
    key = f"stock:{asset_id}"
    if key not in _CACHE:
        _CACHE[key] = StockOpticsAsset.from_json(
            _load_json(f"stock__{asset_id}"), f"stock__{asset_id}"
        )
    return _CACHE[key]


def load_print_optics(asset_id: str) -> PrintOpticsAsset:
    key = f"print:{asset_id}"
    if key not in _CACHE:
        _CACHE[key] = PrintOpticsAsset.from_json(
            _load_json(f"print__{asset_id}"), f"print__{asset_id}"
        )
    return _CACHE[key]


def load_capture_bloom(asset_id: str) -> CaptureBloomAsset:
    key = f"bloom:{asset_id}"
    if key not in _CACHE:
        _CACHE[key] = CaptureBloomAsset.from_json(
            _load_json(f"bloom__{asset_id}"), f"bloom__{asset_id}"
        )
    return _CACHE[key]


def asset_digest(name: str) -> str:
    return str(_manifest()["files"][f"{name}.json"])


# --------------------------------------------------------------------------
# the compiled plan
# --------------------------------------------------------------------------

# One asset family per class for now. When per-stock optics land, this map
# gains entries and the compiler picks by stock id; it does NOT gain a
# fallback, because "this stock has no measured optics so it borrows another
# stock's" is exactly the claim the provenance rules forbid.
DEFAULT_STOCK_OPTICS = "modelled_default"
DEFAULT_PRINT_OPTICS = "modelled_default"
DEFAULT_CAPTURE_BLOOM = "none"


@dataclass(frozen=True)
class FilmOpticsPlan:
    """Immutable per-render optics declaration (§7.2).

    Runtime code eats the whole plan and never re-reads a GUI string or picks
    a default of its own. Pixel geometry is deliberately NOT here: the plan is
    a function of (stock, medium, amounts, seed) only, so it can be hashed for
    the render fingerprint and cached across every preview, crop and export of
    the same settings. The per-render pixel mapping lives in the spatial
    context, which knows the image size.
    """

    stock: StockOpticsAsset
    print_medium: PrintOpticsAsset
    capture_bloom: CaptureBloomAsset
    grain_amount: float
    halation_amount: float
    print_scatter_amount: float
    seed: int
    asset_ids: tuple[str, ...] = ()
    asset_hashes: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    @property
    def engaged(self) -> bool:
        return (
            self.grain_amount > 0.0
            or self.halation_amount > 0.0
            or self.print_scatter_amount > 0.0
        )

    def report(self) -> dict:
        """What the render report must print: which asset, and how honest.

        §12.2 refuses "模拟光学 standard" as a report line. A reader has to be
        able to tell a modelled halo radius from a measured one without
        opening the source.
        """
        return {
            "stock_optics": self.stock.asset_id,
            "print_optics": self.print_medium.asset_id,
            "capture_bloom": self.capture_bloom.asset_id,
            "amounts": {
                "grain": self.grain_amount,
                "halation": self.halation_amount,
                "print_scatter": self.print_scatter_amount,
            },
            "seed": self.seed,
            "provenance": {
                "grain": self.stock.grain.provenance if self.stock.grain else None,
                "halation": (
                    self.stock.halation.provenance if self.stock.halation else None
                ),
                "print_scatter": (
                    self.print_medium.print_scatter.provenance
                    if self.print_medium.print_scatter else None
                ),
            },
            "halation_dc_mode": (
                self.stock.halation.dc_mode if self.stock.halation else None
            ),
            "asset_sha256": dict(zip(self.asset_ids, self.asset_hashes)),
        }


def _amount(plan: Any, name: str) -> float:
    v = float(getattr(plan, name, 0.0) or 0.0)
    _require(np.isfinite(v) and v >= 0.0, f"{name} must be finite and non-negative")
    return v


def compile_film_optics_plan(tone_plan: Any) -> FilmOpticsPlan | None:
    """Compile the tone plan's optics declarations into a FilmOpticsPlan.

    Returns None when nothing is engaged, which is what keeps the amount-0
    strict-identity fast path free: no asset is read, no context is built.
    """
    grain = _amount(tone_plan, "film_grain")
    halation = _amount(tone_plan, "film_halation")
    scatter = _amount(tone_plan, "film_bloom")
    if grain <= 0.0 and halation <= 0.0 and scatter <= 0.0:
        return None

    stock = load_stock_optics(DEFAULT_STOCK_OPTICS)
    medium = load_print_optics(DEFAULT_PRINT_OPTICS)
    bloom = load_capture_bloom(DEFAULT_CAPTURE_BLOOM)

    # An engaged amount with no asset behind it is a configuration error, not
    # a silent no-op: the user asked for an effect this material does not
    # declare, and answering with an unchanged image teaches them the slider
    # is broken.
    _require(grain <= 0.0 or stock.grain is not None,
             f"stock optics '{stock.asset_id}' declares no grain")
    _require(halation <= 0.0 or stock.halation is not None,
             f"stock optics '{stock.asset_id}' declares no halation")
    _require(scatter <= 0.0 or medium.print_scatter is not None,
             f"print optics '{medium.asset_id}' declares no print scatter")

    names = (
        f"stock__{stock.asset_id}",
        f"print__{medium.asset_id}",
        f"bloom__{bloom.asset_id}",
    )
    return FilmOpticsPlan(
        stock=stock,
        print_medium=medium,
        capture_bloom=bloom,
        grain_amount=grain,
        halation_amount=halation,
        print_scatter_amount=scatter,
        seed=int(getattr(tone_plan, "film_optics_seed", 0) or 0),
        asset_ids=names,
        asset_hashes=tuple(asset_digest(n) for n in names),
        provenance=(stock.provenance, medium.provenance, bloom.provenance),
    )
