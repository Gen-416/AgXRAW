# SPDX-License-Identifier: GPL-3.0-or-later
"""Small product contract for coarse colour cleanup, independent of rendering.

This policy never authorizes an unavailable noise model. Rendering must still
check the actual decoder, geometry, calibration and invalid-source masks.
Fine chroma and correlated adaptive-decoder propagation remain unavailable.
Existing recipes that only contain a numeric strength keep custom semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

POLICY_VERSION = "coarse-chroma-policy-1"
CONSERVATIVE_AMOUNT = .10


@dataclass(frozen=True)
class ChromaPolicy:
    mode: str
    amount: float
    coarse_chroma: str
    fine_chroma: str = "experimental-unavailable"
    algorithm: str = "profiled-independent-coarse-garrote"
    version: str = POLICY_VERSION
    evidence: str = "synthetic-independent-noise-tradeoff; real-camera-validation-pending"

    def to_dict(self):
        return asdict(self)

    @property
    def cache_identity(self):
        return (self.version, self.mode, self.amount, self.algorithm)


def resolve_chroma_policy(mode=None, *, custom_amount=0.):
    """Resolve off/conservative/custom without silently changing old recipes.

    Omitting ``mode`` means an existing numeric recipe, hence custom. An
    explicit conservative choice is opt-in and uses only the established
    coarse operator. "conditional" is permission to attempt existing support
    checks, not a statement that correction was applied to a photograph.
    """
    selected = "custom" if mode is None else str(mode).strip().lower()
    if selected not in ("off", "conservative", "custom"):
        raise ValueError("chroma cleanup must be off, conservative or custom")
    if selected == "off":
        amount = 0.
    elif selected == "conservative":
        amount = CONSERVATIVE_AMOUNT
    else:
        try:
            amount = float(custom_amount)
        except (ValueError, TypeError):
            raise ValueError("custom chroma strength must be finite in 0..1") from None
        if not math.isfinite(amount) or not 0 <= amount <= 1:
            raise ValueError("custom chroma strength must be finite in 0..1")
    return ChromaPolicy(selected, amount, "disabled" if amount == 0 else "conditional")
