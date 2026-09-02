# SPDX-License-Identifier: GPL-3.0-or-later
"""External review batch 22 (2026-09-02 handoff): pins for the verified findings.

1. R-P0-1: PR #160 taught the CLI to pass ``chroma_nr`` to
   ``resolve_export_ev`` while the auto-EV chain (resolve_export_ev ->
   compute_auto_ev -> max_safe_ev -> reference build_render_plan) knew
   nothing of it: ``--ev auto`` raised TypeError on every invocation. The
   chain now carries the dial end to end, and the CLI's keyword set is
   bound to the callee's signature so the next dial cannot drift.
2. R-P2-1: the legacy public ``scene_render_to_agx_u8`` entry did not
   forward the Analysis (batch 21 item 2 closed the main paths only), so a
   gated plan rendered through it built guidance without sensor-SNR
   evidence.
"""
from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _call_kwargs(source: str, callee: str) -> list[set[str]]:
    """Keyword names of every ``callee(...)`` call in ``source``."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == callee:
                out.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return out


class AutoEvChainCarriesChromaNr(unittest.TestCase):
    def test_cli_keywords_bind_to_resolve_export_ev_signature(self) -> None:
        from dngscan import auto_ev

        src = (ROOT / "dngscan" / "cli.py").read_text(encoding="utf-8")
        calls = _call_kwargs(src, "resolve_export_ev")
        self.assertTrue(calls, "cli.py no longer calls resolve_export_ev")
        params = set(inspect.signature(auto_ev.resolve_export_ev).parameters)
        for kwargs in calls:
            self.assertLessEqual(kwargs, params, kwargs - params)
        self.assertIn("chroma_nr", calls[0])

    def test_chain_signatures_and_forwarding(self) -> None:
        from dngscan import auto_ev

        for fn in (auto_ev.resolve_export_ev, auto_ev.compute_auto_ev, auto_ev.max_safe_ev):
            self.assertIn("chroma_nr", inspect.signature(fn).parameters, fn.__name__)
        src = (ROOT / "dngscan" / "auto_ev.py").read_text(encoding="utf-8")
        # both reference plans, the compute->max_safe hop and the resolve->compute hop
        for callee in ("build_render_plan", "max_safe_ev", "compute_auto_ev"):
            for kwargs in _call_kwargs(src, callee):
                if callee == "build_render_plan" and "film_curve" not in kwargs:
                    # the sample-probe fallback plan compiles without any film
                    # declaration by design; only the reference plans that
                    # mirror the export's full declaration must carry the dial
                    continue
                self.assertIn("chroma_nr", kwargs, f"{callee} call drops chroma_nr")

    def test_resolve_export_ev_hands_the_dial_to_the_reference_plan(self) -> None:
        from dngscan import auto_ev

        seen: list[float] = []

        def fake_plan(*args, **kwargs):
            seen.append(float(kwargs.get("chroma_nr", -1.0)))
            raise RuntimeError("stop after plan compile")

        bundle = mock.MagicMock()
        bundle.lens_filter = "none"
        with mock.patch.object(auto_ev, "build_render_plan", fake_plan), \
                mock.patch.object(auto_ev, "replace", lambda b, **kw: b), \
                mock.patch.object(auto_ev, "compute_exposure_gain", return_value=1.0), \
                mock.patch.object(auto_ev, "exposure_mode_for_tone_core", return_value="x"):
            with self.assertRaises(RuntimeError):
                auto_ev.resolve_export_ev("auto", bundle, mock.MagicMock(), "p3", chroma_nr=0.3)
        self.assertEqual(seen, [0.3])

    def test_manual_ev_still_bypasses_the_chain(self) -> None:
        from dngscan import auto_ev

        self.assertEqual(
            auto_ev.resolve_export_ev(0.5, mock.MagicMock(), mock.MagicMock(), "p3", chroma_nr=0.7),
            (0.5, None),
        )


class LegacyU8EntryForwardsAnalysis(unittest.TestCase):
    def test_scene_render_to_agx_u8_passes_analysis(self) -> None:
        from dngscan import render

        self.assertIn("analysis", inspect.signature(render.scene_render_to_agx_u8).parameters)
        seen = {}

        def fake_linear(*args, **kwargs):
            seen["analysis"] = kwargs.get("analysis")
            raise RuntimeError("stop")

        sentinel = object()
        with mock.patch.object(render, "scene_render_to_display_linear", fake_linear), \
                mock.patch.object(render, "plan_with_look_overrides", lambda p, *a: p):
            with self.assertRaises(RuntimeError):
                render.scene_render_to_agx_u8(mock.MagicMock(), mock.MagicMock(), analysis=sentinel)
        self.assertIs(seen["analysis"], sentinel)


if __name__ == "__main__":
    unittest.main()
