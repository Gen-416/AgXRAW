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
                    scatter
    capture bloom   an EDITORIAL lens-and-emulsion glow, declared as such

Every asset carries its own provenance — `measured`, `derived`, `modelled` or
`editorial` — because the honest answer differs field by field, and a report
that says "modelled" for a whole profile is either too pessimistic about the
measured parts or too generous about the invented ones.

Loading is fail-closed: unknown schema, unknown model name, a missing file, a
hash mismatch or a non-finite number all raise. A spatial operator that
silently falls back to a default is a spatial operator whose output nobody can
attribute.

Naming note (§12.2): "Bloom" is the editorial capture bloom (P3). The old
post-B2 conservative print scatter that once wore the name was retained
through P3-P4 as `legacy_print_scatter` for acceptance comparison and was
DELETED in P5e — the V2 operators are the default and the comparison had
served its purpose.
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
    model: str                   # band_limited_gaussian_v1 | measured_sigma_v2
    pitch_um: float
    size_um: float
    sigma0: float                # v1 only; 0.0 under measured_sigma_v2
    layer_corr: float
    # measured_sigma_v2 (grain V2 P4): per-channel diffuse rms sigma(D)
    # measured at `aperture_um` in the CHART's densitometry coordinate.
    # channel order (R, G, B); per channel: (chart_base, chart_max) and a
    # monotone-in-D ((D, sigma), ...) table. Empty tuples under v1.
    aperture_um: float = 0.0
    chart_density: tuple = ()    # ((base, max),) * 3
    sigma_density: tuple = ()    # (((D, sigma), ...),) * 3
    # multi-band spectrum (P4 particle oracle): ((size_um, weight), ...).
    # Empty means one band at size_um (the pre-oracle behaviour). A size
    # at or below half the pitch renders as per-cell white noise.
    bands: tuple = ()

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "GrainAsset":
        model = str(raw.get("model", ""))
        _require(model in ("band_limited_gaussian_v1", "measured_sigma_v2"),
                 f"{where}: unknown grain model {model!r}")
        pitch = _finite(raw["pitch_um"], f"{where}.pitch_um")
        size = _finite(raw["size_um"], f"{where}.size_um")
        _require(pitch > 0.0 and size > 0.0, f"{where}: grain scales must be positive")
        corr = _finite(raw["layer_corr"], f"{where}.layer_corr")
        _require(0.0 <= corr <= 1.0, f"{where}: layer_corr outside [0,1]")
        if model == "band_limited_gaussian_v1":
            sigma0 = _finite(raw["sigma0"], f"{where}.sigma0")
            _require(sigma0 >= 0.0, f"{where}: sigma0 must be non-negative")
            _require("channels" not in raw and "aperture_um" not in raw,
                     f"{where}: v1 must not carry measured-model fields")
            return cls(
                provenance=_provenance(raw["provenance"], where),
                medium=str(raw["medium"]),
                model=model,
                pitch_um=pitch, size_um=size, sigma0=sigma0, layer_corr=corr,
            )
        # measured_sigma_v2: the maths has no free sigma0 — a leftover one
        # would silently double-scale, so its presence is a hard error.
        _require("sigma0" not in raw,
                 f"{where}: measured_sigma_v2 carries no sigma0")
        aperture = _finite(raw["aperture_um"], f"{where}.aperture_um")
        _require(aperture > 0.0, f"{where}: aperture_um must be positive")
        chans = raw["channels"]
        _require(set(chans) == {"R", "G", "B"},
                 f"{where}: channels must be exactly R, G, B")
        bases: list[tuple] = []
        tables: list[tuple] = []
        for name in ("R", "G", "B"):
            ch = chans[name]
            base = _finite(ch["chart_density"][0], f"{where}.{name}.chart_density[0]")
            dmax = _finite(ch["chart_density"][1], f"{where}.{name}.chart_density[1]")
            _require(dmax > base >= 0.0,
                     f"{where}.{name}: chart_density range must be increasing")
            rows = ch["sigma_density"]
            _require(len(rows) >= 4, f"{where}.{name}: sigma_density too short")
            prev_d = None
            tab = []
            for i, row in enumerate(rows):
                d = _finite(row[0], f"{where}.{name}.sigma_density[{i}][0]")
                s = _finite(row[1], f"{where}.{name}.sigma_density[{i}][1]")
                _require(0.0 < s < 0.2,
                         f"{where}.{name}: sigma {s} outside (0, 0.2)")
                _require(prev_d is None or d > prev_d,
                         f"{where}.{name}: sigma_density must be strictly "
                         "increasing in D (fold duplicates at import)")
                prev_d = d
                tab.append((d, s))
            bases.append((base, dmax))
            tables.append(tuple(tab))
        bands: list[tuple] = []
        if "bands" in raw:
            wsum = 0.0
            prev_size = 0.0
            for i, row in enumerate(raw["bands"]):
                bsize = _finite(row[0], f"{where}.bands[{i}][0]")
                bwgt = _finite(row[1], f"{where}.bands[{i}][1]")
                _require(bsize > prev_size,
                         f"{where}.bands: sizes must be positive ascending")
                _require(bwgt > 0.0, f"{where}.bands: weights must be positive")
                prev_size = bsize
                wsum += bwgt
                bands.append((bsize, bwgt))
            _require(abs(wsum - 1.0) < 1e-6,
                     f"{where}.bands: weights must sum to 1 (got {wsum})")
        return cls(
            provenance=_provenance(raw["provenance"], where),
            medium=str(raw["medium"]),
            model=model,
            pitch_um=pitch, size_um=size, sigma0=0.0, layer_corr=corr,
            aperture_um=aperture,
            chart_density=tuple(bases),
            sigma_density=tuple(tables),
            bands=tuple(bands),
        )


@dataclass(frozen=True)
class ScatterKernelAsset:
    """A per-channel energy-conserving scatter mix (P5, plan §5.1/§6.2).

    E' = (1-s)E + s[(1-w)·G_sigma*E + w·G_tail*E], kernels normalized,
    scales in film-plane micrometres — a two-Gaussian mix, the exact form
    the operator executes (R10 item 3: renamed from "core_tail"; no
    exponential PSF was ever applied at runtime). Parameters come from the
    fitted datasheet MTF curves (dngscan/data/mtf/,
    tools/import_kodak_mtf.py). Identifiability contract: an active tail
    (w > 0) must have tail_sigma_um >= 2*sigma_um, else the components are
    degenerate; `w` and `tail_sigma_um` are 0 for a single-Gaussian fit."""

    provenance: str
    model: str                    # bi_gaussian_v1 | gaussian_v1
    s: tuple                      # (R, G, B) mix fractions
    w: tuple                      # (R, G, B) tail weights
    sigma_um: tuple               # (R, G, B) Gaussian core scales
    tail_sigma_um: tuple          # (R, G, B) Gaussian tail scales
    source: str = ""

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "ScatterKernelAsset":
        model = str(raw.get("model", ""))
        _require(model in ("bi_gaussian_v1", "gaussian_v1"),
                 f"{where}: unknown scatter model {model!r}")
        chans = raw["channels"]
        _require(set(chans) == {"R", "G", "B"},
                 f"{where}: channels must be exactly R, G, B")
        s, w, sig, lam = [], [], [], []
        for name in ("R", "G", "B"):
            ch = chans[name]
            sv = _finite(ch["s"], f"{where}.{name}.s")
            _require(0.0 < sv <= 1.0, f"{where}.{name}: s outside (0, 1]")
            sgv = _finite(ch["sigma_um"], f"{where}.{name}.sigma_um")
            _require(0.0 < sgv < 30.0, f"{where}.{name}: sigma_um implausible")
            wv = _finite(ch.get("w", 0.0), f"{where}.{name}.w")
            _require(0.0 <= wv <= 1.0, f"{where}.{name}: w outside [0, 1]")
            tsv = _finite(ch.get("tail_sigma_um", 0.0), f"{where}.{name}.tail_sigma_um")
            _require(0.0 <= tsv < 60.0, f"{where}.{name}: tail_sigma_um implausible")
            _require((wv > 0.0) == (tsv > 0.0),
                     f"{where}.{name}: tail weight and scale must be zero together")
            # 0.1% tolerance: the fit may sit exactly on the 2x bound and
            # 3-decimal rounding of both fields must not flip the contract
            _require(wv == 0.0 or tsv >= 2.0 * sgv * 0.999,
                     f"{where}.{name}: degenerate tail (tail_sigma < 2*sigma)")
            _require(model != "gaussian_v1" or (wv == 0.0 and tsv == 0.0),
                     f"{where}.{name}: gaussian_v1 carries no tail fields")
            s.append(sv); w.append(wv); sig.append(sgv); lam.append(tsv)
        return cls(
            provenance=_provenance(raw["provenance"], where),
            model=model,
            s=tuple(s), w=tuple(w),
            sigma_um=tuple(sig), tail_sigma_um=tuple(lam),
            source=str(raw.get("source", "")),
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
class StockOpticsAsset:
    asset_id: str
    provenance: str
    gate_reference_mm: tuple[float, float]
    anti_halation: str
    grain: GrainAsset | None
    halation: HalationAsset | None
    emulsion_scatter: "ScatterKernelAsset | None" = None   # P5 (§5.1)
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
        # P5 activated the reserved slot: the stock's own emulsion scatter,
        # fitted from the datasheet MTF (plan §5.1). Absence means the
        # medium declares none — not an error.
        scat = raw.get("emulsion_scatter")
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
            emulsion_scatter=(
                ScatterKernelAsset.from_json(scat, f"{where}.emulsion_scatter")
                if scat else None
            ),
            source_notes=tuple(str(s) for s in raw.get("source_notes", ())),
        )


@dataclass(frozen=True)
class PrintOpticsAsset:
    asset_id: str
    provenance: str
    formation_scatter: "ScatterKernelAsset | None" = None  # P5 (§6.2)
    positive_grain: GrainAsset | None = None   # P4
    viewing_scatter: Any = None           # deliberately never enabled in v2
    source_notes: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "PrintOpticsAsset":
        _require(str(raw.get("kind")) == "print_optics",
                 f"{where}: not a print_optics asset")
        form = raw.get("formation_scatter")
        _require(raw.get("viewing_scatter") is None,
                 f"{where}: viewing_scatter has no measured PSF and stays null")
        _require(raw.get("legacy_print_scatter") is None,
                 f"{where}: legacy_print_scatter was deleted in P5e — "
                 "the V2 formation scatter is the medium's scatter")
        # P4 activated the reserved slot: the positive medium's own grain
        # (2383 print film measured tables), applied on the PRINT dye
        # amounts before B2 viewing. A paper/print asset without one simply
        # contributes no grain — unlike the stock side, absence is not an
        # error because grain is a stock-anchored user amount.
        pos = raw.get("positive_grain")
        return cls(
            asset_id=str(raw["asset_id"]),
            provenance=_provenance(raw["provenance"], where),
            positive_grain=(
                GrainAsset.from_json(pos, f"{where}.positive_grain")
                if pos else None
            ),
            formation_scatter=(
                ScatterKernelAsset.from_json(form, f"{where}.formation_scatter")
                if form else None
            ),
            source_notes=tuple(str(s) for s in raw.get("source_notes", ())),
        )


@dataclass(frozen=True)
class BloomScale:
    """One rung of the source-detection ladder (§6.1, R1).

    `detect_um` is the low-pass the LUMINANCE is seen through before the gate,
    i.e. the smallest source this rung can still find; `diffuse_um` is how far
    that rung's light then travels. Keeping them separate is the whole point:
    a filament and a blown window should not receive the same halo, and one
    detector radius cannot tell them apart however its threshold is set.
    """

    detect_um: float
    diffuse_um: float
    gate_ev: tuple[float, float]
    weight: float

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "BloomScale":
        detect = _finite(raw["detect_um"], f"{where}.detect_um")
        diffuse = _finite(raw["diffuse_um"], f"{where}.diffuse_um")
        _require(detect >= 0.0, f"{where}: detect_um must be non-negative")
        _require(diffuse > 0.0, f"{where}: diffuse_um must be positive")
        gate = np.asarray(raw["gate_ev"], dtype=np.float64)
        _require(gate.shape == (2,), f"{where}.gate_ev must be (t0, t1)")
        _require(bool(np.isfinite(gate).all()), f"{where}.gate_ev is not finite")
        _require(gate[1] - gate[0] >= 0.5,
                 f"{where}.gate_ev needs at least 0.5 EV of smootherstep width")
        weight = _finite(raw["weight"], f"{where}.weight")
        _require(weight >= 0.0, f"{where}: weight must be non-negative")
        return cls(detect, diffuse, (float(gate[0]), float(gate[1])), weight)


@dataclass(frozen=True)
class CaptureBloomAsset:
    """Editorial lens-and-emulsion glow, applied to the SCENE (§6.1).

    Declared editorial and meant to be: it is the combined impression of lens
    flare and in-emulsion spreading, authored for look. What it is NOT is a
    measured property of any print medium — which is exactly what the old
    post-B2 operator was mistaken for.

    `active=False` keeps the slot legible when a configuration deliberately
    has no glow, so the compiled plan's shape never changes.
    """

    asset_id: str
    provenance: str
    active: bool
    scales: tuple[BloomScale, ...] = ()
    save_lights: float = 0.0
    core_ratio: tuple[float, float] = (0.2, 0.8)
    saturation: float = 1.0

    @classmethod
    def from_json(cls, raw: dict, where: str) -> "CaptureBloomAsset":
        _require(str(raw.get("kind")) == "capture_bloom",
                 f"{where}: not a capture_bloom asset")
        active = bool(raw["active"])
        if not active:
            return cls(
                asset_id=str(raw["asset_id"]),
                provenance=_provenance(raw["provenance"], where),
                active=False,
            )
        scales = raw.get("scales") or ()
        _require(len(scales) >= 2,
                 f"{where}: a scale space needs at least two rungs")
        parsed = tuple(
            BloomScale.from_json(sc, f"{where}.scales[{i}]")
            for i, sc in enumerate(scales)
        )
        _require(parsed[0].detect_um == 0.0,
                 f"{where}: the finest rung must detect at full resolution "
                 "(detect_um 0), or point sources are averaged away")
        _require(all(b.detect_um > a.detect_um for a, b in zip(parsed, parsed[1:])),
                 f"{where}: detection scales must increase")
        _require(all(b.diffuse_um > a.diffuse_um for a, b in zip(parsed, parsed[1:])),
                 f"{where}: diffusion radii must increase with the source size")
        save = _finite(raw.get("save_lights", 0.0), f"{where}.save_lights")
        _require(0.0 <= save <= 1.0, f"{where}: save_lights outside [0,1]")
        ratio = np.asarray(raw.get("core_ratio", [0.2, 0.8]), dtype=np.float64)
        _require(ratio.shape == (2,) and ratio[1] - ratio[0] >= 0.1,
                 f"{where}.core_ratio must be a widening pair")
        sat = _finite(raw.get("saturation", 1.0), f"{where}.saturation")
        _require(sat >= 0.0, f"{where}: saturation must be non-negative")
        return cls(
            asset_id=str(raw["asset_id"]),
            provenance=_provenance(raw["provenance"], where),
            active=True, scales=parsed, save_lights=save,
            core_ratio=(float(ratio[0]), float(ratio[1])), saturation=sat,
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
DEFAULT_CAPTURE_BLOOM = "modelled_default"


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
    capture_bloom_amount: float
    seed: int
    asset_ids: tuple[str, ...] = ()
    asset_hashes: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    @property
    def engaged(self) -> bool:
        return (
            self.grain_amount > 0.0
            or self.halation_amount > 0.0
            or self.capture_bloom_amount > 0.0
        )

    @property
    def has_media_scatter(self) -> bool:
        """Whether the declared media carry any scatter stage (R3 item 3)."""
        return (
            self.stock.emulsion_scatter is not None
            or self.print_medium.formation_scatter is not None
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
                "capture_bloom": self.capture_bloom_amount,
            },
            "seed": self.seed,
            "provenance": {
                "grain": self.stock.grain.provenance if self.stock.grain else None,
                "halation": (
                    self.stock.halation.provenance if self.stock.halation else None
                ),
                "capture_bloom": self.capture_bloom.provenance,
                "emulsion_scatter": (
                    self.stock.emulsion_scatter.provenance
                    if self.stock.emulsion_scatter else None
                ),
                "formation_scatter": (
                    self.print_medium.formation_scatter.provenance
                    if self.print_medium.formation_scatter else None
                ),
                "positive_grain": (
                    self.print_medium.positive_grain.provenance
                    if self.print_medium.positive_grain else None
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
    # §12.2: `film_bloom` now drives the EDITORIAL CAPTURE BLOOM. The old
    # post-B2 conservative operator it used to drive is `legacy_print_scatter`
    # and is no longer reachable from a user amount — a medium property must
    # not ride a look slider, and the two must never share a field or the
    # rename buys nothing.
    bloom = _amount(tone_plan, "film_bloom")
    # R3 item 3: the media scatter (§5.1 emulsion / §6.2 formation) is a
    # property of the DECLARED media, so under the default
    # film_media_scatter="declared" a full-film plan engages it even with
    # every look slider at zero — the contract that motivated the flag in
    # review R1 item 4. Only "off" (or no full-film chain at all, where no
    # media exists to scatter) compiles to the strict-identity fast path.
    media_scatter_on = (
        str(getattr(tone_plan, "film_media_scatter", "declared") or "declared")
        != "off"
        and str(getattr(tone_plan, "film_mode", "observe")) == "full"
        and str(getattr(tone_plan, "curve_preset", "none")) != "none"
    )
    if grain <= 0.0 and halation <= 0.0 and bloom <= 0.0 and not media_scatter_on:
        return None

    # R1 item 3: ONE generic profile serves every curve_preset and print
    # medium today — deliberately, and honestly labelled: its 5207/2383-
    # sourced blocks carry `derived` provenance, never the rendered
    # material's own measurement. Per-material binding (keying these loads
    # on curve_preset / film_print_medium, failing closed where no data
    # exists) is the recorded follow-up.
    stock = load_stock_optics(DEFAULT_STOCK_OPTICS)
    medium = load_print_optics(DEFAULT_PRINT_OPTICS)
    bloom_asset = load_capture_bloom(DEFAULT_CAPTURE_BLOOM)

    # An engaged amount with no asset behind it is a configuration error, not
    # a silent no-op: the user asked for an effect this material does not
    # declare, and answering with an unchanged image teaches them the slider
    # is broken.
    _require(grain <= 0.0 or stock.grain is not None,
             f"stock optics '{stock.asset_id}' declares no grain")
    _require(halation <= 0.0 or stock.halation is not None,
             f"stock optics '{stock.asset_id}' declares no halation")
    _require(bloom <= 0.0 or bloom_asset.active,
             f"capture bloom '{bloom_asset.asset_id}' is inactive")

    names = (
        f"stock__{stock.asset_id}",
        f"print__{medium.asset_id}",
        f"bloom__{bloom_asset.asset_id}",
    )
    return FilmOpticsPlan(
        stock=stock,
        print_medium=medium,
        capture_bloom=bloom_asset,
        grain_amount=grain,
        halation_amount=halation,
        capture_bloom_amount=bloom,
        seed=int(getattr(tone_plan, "film_optics_seed", 0) or 0),
        asset_ids=names,
        asset_hashes=tuple(asset_digest(n) for n in names),
        provenance=(stock.provenance, medium.provenance,
                    bloom_asset.provenance),
    )
