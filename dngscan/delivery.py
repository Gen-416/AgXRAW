# SPDX-License-Identifier: GPL-3.0-or-later
"""Delivery profiles: encode settings that never feed back into formation.

Formation produces finished SDR/HDR masters at full precision. This module only
describes how those masters are packaged. Archive keeps the historical Ultrahdr
contract (quality 100, 4:4:4, tight round-trip gates). Share lowers JPEG quality so
Core Image may emit 4:2:0 and uses wider engineering gates calibrated for that loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DELIVERY_PROFILE_CHOICES = ("archive", "share")
DEFAULT_DELIVERY_PROFILE = "archive"
DELIVERY_CONTAINER_CHOICES = ("jpeg", "heic")

# Share defaults for Ultrahdr / SDR when the user picks the profile without overriding
# quality/chroma. Archive keeps the historical q100 / 444 Ultrahdr defaults.
SHARE_JPEG_QUALITY = 90
SHARE_CHROMA = "420"
ARCHIVE_JPEG_QUALITY = 100
ARCHIVE_CHROMA = "444"


@dataclass(frozen=True)
class DeliveryTolerances:
    """Engineering round-trip gates for one encode profile.

    These are not ISO 21496-1 constants. They separate low-frequency colour/tone drift
    from encoder-dependent high-frequency loss at a stated quality/chroma operating point.
    """

    base_mean_code_error: float
    base_channel_bias_code_error: float
    base_block_p99_code_error: float
    hdr_block_median_relative_error: float
    hdr_block_p95_relative_error: float
    hdr_block_p99_relative_error: float
    hdr_block_chroma_error: float
    # Pixel-scale chroma p99 gate. 8x8 block means are computed on the same grid 4:2:0
    # averages chroma over, so they are nearly blind to subsampling damage; this is the
    # one statistic that still sees it. Dropping it while introducing 4:2:0 delivery
    # would leave chroma quality entirely ungated.
    hdr_pixel_chroma_error: float
    require_chroma_444: bool
    # Optional Apple gain-map auxiliary downsample. None leaves Core Image default.
    gainmap_subsample_factor: int | None = None


# All tolerance sets below were calibrated 2026-07-29 against the macOS Core Image
# writer on the three-frame regression corpus (_SDI0150 daylight, _SDI0199 stage,
# _SDI0133 bar), worst case per metric with ~1.3x margin. The synthetic-ramp numbers
# they replace under-reported real-content loss by an order of magnitude on the stage
# frame; a gate that no representative frame can pass is not a contract, it is a
# post-render crash. These remain engineering regression limits, not quality claims.

# Quality-100 / 4:4:4 Core Image ISO gain-map delivery (JPEG and HEVC both comfortably
# inside these on the corpus).
ARCHIVE_TOLERANCES = DeliveryTolerances(
    base_mean_code_error=4.0,
    base_channel_bias_code_error=1.0,
    base_block_p99_code_error=2.0,
    hdr_block_median_relative_error=0.015,
    hdr_block_p95_relative_error=0.05,
    hdr_block_p99_relative_error=0.06,
    hdr_block_chroma_error=0.015,
    # NOT the historical flat 0.02: high-ISO chroma noise legitimately reaches 0.073 at
    # q100/4:4:4 on the stage frame -- the very case that got the old pixel gate removed.
    # Corpus worst with margin; silent 4:2:0 measured 0.086+ even on clean frames, so
    # this still separates subsampling damage from noise, layered with the 4:4:4 check.
    hdr_pixel_chroma_error=0.10,
    require_chroma_444=True,
    gainmap_subsample_factor=None,
)

# q90 / 4:2:0 JPEG delivery. Still rejects broad tone/chroma transforms.
SHARE_TOLERANCES = DeliveryTolerances(
    base_mean_code_error=10.0,
    base_channel_bias_code_error=2.0,
    base_block_p99_code_error=5.0,
    hdr_block_median_relative_error=0.05,
    hdr_block_p95_relative_error=0.11,
    hdr_block_p99_relative_error=0.14,
    hdr_block_chroma_error=0.04,
    # Legitimate 4:2:0-plus-noise reaches 0.276 on the stage frame; backstop only.
    hdr_pixel_chroma_error=0.35,
    require_chroma_444=False,
    gainmap_subsample_factor=2,
)

# Share HEVC loses visibly more than share JPEG at the same nominal settings (block p95
# roughly 1.8x on the corpus, and larger files). A separate calibration keeps that an
# honest, stated operating point instead of judging HEVC against JPEG numbers.
SHARE_HEIC_TOLERANCES = DeliveryTolerances(
    base_mean_code_error=10.0,
    base_channel_bias_code_error=2.0,
    base_block_p99_code_error=10.0,
    hdr_block_median_relative_error=0.06,
    hdr_block_p95_relative_error=0.19,
    hdr_block_p99_relative_error=0.26,
    hdr_block_chroma_error=0.08,
    hdr_pixel_chroma_error=0.45,
    require_chroma_444=False,
    gainmap_subsample_factor=2,
)


def tolerances_for(name: str, container: str = "jpeg") -> DeliveryTolerances:
    """The calibrated gate set for one (profile, container) operating point."""
    if name == "archive":
        return ARCHIVE_TOLERANCES
    return SHARE_HEIC_TOLERANCES if container == "heic" else SHARE_TOLERANCES


@dataclass(frozen=True)
class DeliveryProfile:
    """How a finished rendition pair is encoded. Does not alter AgX/HDR math."""

    name: str
    quality: int
    chroma: str
    container: str = "jpeg"
    # None derives the gates from the profile name, so a hand-built profile cannot
    # silently judge share-quality encodes against archive tolerances (or vice versa).
    tolerances: DeliveryTolerances | None = None

    def __post_init__(self) -> None:
        if self.tolerances is None:
            object.__setattr__(
                self, "tolerances", tolerances_for(self.name, self.container)
            )

    @property
    def is_archive(self) -> bool:
        return self.name == "archive"


def resolve_delivery_profile(
    name: str,
    *,
    quality: int | None = None,
    chroma: str | None = None,
    container: str = "jpeg",
) -> DeliveryProfile:
    """Build a delivery profile from a named preset plus optional overrides.

    Explicit quality/chroma win over preset defaults for *share*. Archive keeps the
    historical Ultrahdr contract and refuses softer encode knobs -- use share instead.
    """
    key = str(name or DEFAULT_DELIVERY_PROFILE).strip().lower()
    if key not in DELIVERY_PROFILE_CHOICES:
        raise ValueError(
            f"未知 delivery profile：{name}（可选：{'/'.join(DELIVERY_PROFILE_CHOICES)}）"
        )
    if key == "archive":
        if quality is not None and int(quality) != ARCHIVE_JPEG_QUALITY:
            raise ValueError(
                "delivery profile=archive 固定 JPEG quality 100；"
                "投递用更小文件请加 --delivery-profile share"
            )
        if chroma is not None and str(chroma) != ARCHIVE_CHROMA:
            raise ValueError(
                "delivery profile=archive 固定 chroma 4:4:4；"
                "投递用 4:2:0 请加 --delivery-profile share"
            )
        q = ARCHIVE_JPEG_QUALITY
        c = ARCHIVE_CHROMA
    else:
        q = SHARE_JPEG_QUALITY if quality is None else int(quality)
        c = SHARE_CHROMA if chroma is None else str(chroma)
    if not 1 <= int(q) <= 100:
        raise ValueError("JPEG quality 必须在 1-100 之间")
    if str(c) not in ("444", "422", "420"):
        raise ValueError(f"未知 chroma：{c}")
    cont = str(container or "jpeg").strip().lower()
    if cont not in DELIVERY_CONTAINER_CHOICES:
        raise ValueError(
            f"未知 delivery container：{container}（可选：{'/'.join(DELIVERY_CONTAINER_CHOICES)}）"
        )
    # Tolerances derive from (profile, container) in __post_init__, so HEVC delivery is
    # judged against its own calibration rather than JPEG's.
    return DeliveryProfile(
        name=key,
        quality=int(q),
        chroma=str(c),
        container=cont,
    )


def reprofile_for_container(
    profile: DeliveryProfile, container: str
) -> DeliveryProfile:
    """Rebuild a profile for another container without dragging stale gates along.

    Gates are calibrated per (profile, container): moving a share profile from JPEG to
    HEIC must pick up the HEVC calibration, not keep judging HEVC by JPEG numbers.
    Deliberately custom tolerance objects are preserved as-is.
    """
    if profile.container == container:
        return profile
    standard = profile.tolerances == tolerances_for(profile.name, profile.container)
    return DeliveryProfile(
        name=profile.name,
        quality=profile.quality,
        chroma=profile.chroma,
        container=container,
        tolerances=None if standard else profile.tolerances,
    )


def is_hdr_output_format(output_format: str) -> bool:
    return str(output_format) in ("ultrahdr", "ultrahdr-heic")


def container_for_output_format(output_format: str) -> str:
    return "heic" if str(output_format) == "ultrahdr-heic" else "jpeg"


def profile_from_encode_settings(
    quality: int, chroma: str, container: str = "jpeg"
) -> DeliveryProfile:
    """Infer archive vs share from explicit encode knobs (CLI without --delivery-profile).

    Exactly q100 + 4:4:4 keeps the strict archive contract the historical Ultrahdr
    defaults were calibrated on. Every other explicit combination uses share gates:
    archive's resolver pins its two knobs, and the profile system has no reason to
    refuse an encode setting that was legal before profiles existed.
    """
    q = int(quality)
    c = str(chroma)
    if q == ARCHIVE_JPEG_QUALITY and c == ARCHIVE_CHROMA:
        return resolve_delivery_profile("archive", quality=q, chroma=c, container=container)
    return resolve_delivery_profile("share", quality=q, chroma=c, container=container)


@dataclass(frozen=True)
class FinishedPair:
    """Formation masters ready for any encoder. Pixels are already display-referred."""

    sdr_rgb_u8: Any
    hdr_rgba_f16: Any
    display_headroom_ev: float
    output_gamut: str = "p3"
