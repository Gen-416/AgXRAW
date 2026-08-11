# SPDX-License-Identifier: GPL-3.0-or-later
"""The empirical-policy register (A8 advisory): the register and the live
constants must agree, so drift on either side fails loudly and every
value change passes through a POLICY_VERSION bump."""
from __future__ import annotations

import unittest

from dngscan import constants as c
from dngscan import policy


class RegisterConsistencyTests(unittest.TestCase):
    def test_every_entry_matches_its_live_constant(self) -> None:
        from dngscan import hdr_agx_plan as h

        live = {
            "RHO_BASE": h.RHO_BASE,
            "MULTICHANNEL_CLIP_ZERO_CONFIDENCE_PCT": h.MULTICHANNEL_CLIP_ZERO_CONFIDENCE_PCT,
            "P3_PRESSURE_ZERO_CONFIDENCE_PCT": h.P3_PRESSURE_ZERO_CONFIDENCE_PCT,
            "UNALIGNED_DECODER_RHO_CAP": h.UNALIGNED_DECODER_RHO_CAP,
            "NORMAL_WHITE_MARGIN_EV": h.NORMAL_WHITE_MARGIN_EV,
            "SPARSE_EMITTER_WHITE_MARGIN_EV": h.SPARSE_EMITTER_WHITE_MARGIN_EV,
            "NORMAL_MINIMUM_WHITE_EV": h.NORMAL_MINIMUM_WHITE_EV,
            "SPARSE_EMITTER_MINIMUM_WHITE_EV": h.SPARSE_EMITTER_MINIMUM_WHITE_EV,
            "MAXIMUM_WHITE_EV": h.MAXIMUM_WHITE_EV,
            "NORMAL_SHOULDER_START_EV": h.NORMAL_SHOULDER_START_EV,
            "SPARSE_EMITTER_SHOULDER_START_EV": h.SPARSE_EMITTER_SHOULDER_START_EV,
            "MIDGRAY_HEADROOM_STOPS": c.MIDGRAY_HEADROOM_STOPS,
            "CEILING_MIN_PILE_PIXELS": c.CEILING_MIN_PILE_PIXELS,
            "CEILING_MIN_PILE_FRACTION": c.CEILING_MIN_PILE_FRACTION,
            "CEILING_PLAUSIBLE_FRACTION": c.CEILING_PLAUSIBLE_FRACTION,
            "CEILING_NEAR_WINDOW_SCALE": c.CEILING_NEAR_WINDOW_SCALE,
            "SNR_TILE": c.SNR_TILE,
            "SNR_LOW_PERCENTILE": c.SNR_LOW_PERCENTILE,
            "SNR_BRIGHT_UNRELIABLE_STOP": c.SNR_BRIGHT_UNRELIABLE_STOP,
            "DEFAULT_HDR_HEADROOM_EV": c.DEFAULT_HDR_HEADROOM_EV,
            "MAX_HDR_PEAK_NITS": c.MAX_HDR_PEAK_NITS,
        }
        for e in policy.ENTRIES:
            if e.name == "CLIP_MARGIN_DN":
                continue   # pinned against the CLI default below
            with self.subTest(name=e.name):
                self.assertIn(e.name, live, "register names a dead constant")
                self.assertEqual(float(e.value), float(live[e.name]))
        for name in live:
            with self.subTest(name=name):
                self.assertIsNotNone(policy.entry(name),
                                     "live policy constant missing from register")

    def test_the_version_fingerprint_pins_the_value_set(self) -> None:
        """A9 item 6: editing a value and the register together without a
        POLICY_VERSION bump fails here — the stored fingerprint for the
        current version no longer matches the recomputed one."""
        self.assertIn(policy.POLICY_VERSION, policy.POLICY_FINGERPRINTS)
        self.assertEqual(
            policy._fingerprint(policy.ENTRIES),
            policy.POLICY_FINGERPRINTS[policy.POLICY_VERSION],
        )

    def test_the_fingerprint_separates_every_field(self) -> None:
        """A11 item 3: the naive joined-string hash collided when content
        moved across field boundaries (rationale="a~b"/constrained_by="c"
        vs "a"/"b~c"). Canonical JSON must (a) not collide on exactly that
        pair and (b) change when ANY single field mutates."""
        import dataclasses

        base = policy.PolicyEntry(
            name="X", value=1.0, unit="u",
            rationale="a~b", constrained_by="c", history=("h",),
        )
        moved = dataclasses.replace(base, rationale="a", constrained_by="b~c")
        self.assertNotEqual(policy._fingerprint((base,)),
                            policy._fingerprint((moved,)))
        for field_name, new_val in (
            ("value", 2.0), ("unit", "v"), ("rationale", "r2"),
            ("constrained_by", "c2"), ("history", ("h", "h2")),
        ):
            mutated = dataclasses.replace(base, **{field_name: new_val})
            with self.subTest(field=field_name):
                self.assertNotEqual(policy._fingerprint((base,)),
                                    policy._fingerprint((mutated,)))

    def test_clip_margin_matches_the_cli_default(self) -> None:
        import argparse

        from dngscan.cli import parse_args

        args = parse_args(["x.dng"])
        self.assertEqual(float(policy.entry("CLIP_MARGIN_DN").value),
                         float(args.margin))

    def test_entries_carry_provenance(self) -> None:
        for e in policy.ENTRIES:
            with self.subTest(name=e.name):
                self.assertTrue(e.rationale.strip())
                self.assertTrue(e.constrained_by.strip())
                self.assertTrue(e.unit.strip())

    def test_the_report_names_the_register(self) -> None:
        line = policy.policy_line()
        self.assertIn(f"v{policy.POLICY_VERSION}", line)
        self.assertIn("policy.py", line)


if __name__ == "__main__":
    unittest.main()
