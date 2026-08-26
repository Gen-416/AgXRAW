# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate: the compiled film-optics assets' chart-derived blocks must match
the chart digitizations they cite.

Found 2026-08-25 by the owner: three rounds of digitization precision work
updated dngscan/data/{grain,mtf}/ while the rendering assets silently kept
the old tables (the asset SAID "digitized from granularity_5207.json" while
no longer matching it — 声明失实). tools/sync_film_optics_from_charts.py is
the standing compiler; this test makes staleness impossible to reintroduce.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


class ChartSyncTests(unittest.TestCase):
    def test_compiled_assets_match_chart_sources(self):
        import json
        from sync_film_optics_from_charts import (MAPPING, OPTICS, GRAIN, MTF,
                                                  compiled_blocks)

        for asset_name, gkey, gfile, skey, mfile, sfields in MAPPING:
            with self.subTest(asset=asset_name):
                asset = json.loads((OPTICS / asset_name).read_text())
                grain_ch, scatter_ch, scatter_model = compiled_blocks(
                    json.loads((GRAIN / gfile).read_text()),
                    json.loads((MTF / mfile).read_text()), sfields)
                self.assertEqual(asset[gkey]["channels"], grain_ch,
                                 f"{asset_name}.{gkey} stale vs {gfile} — "
                                 "run tools/sync_film_optics_from_charts.py")
                self.assertEqual(asset[skey]["channels"], scatter_ch,
                                 f"{asset_name}.{skey} stale vs {mfile}")
                self.assertEqual(asset[skey]["model"], scatter_model)

    def test_tail_identifiability_contract(self):
        """w == 0 must come with tail_sigma_um == 0 (inert component), and
        an active tail must be clearly wider than the core (>= 2x) — a
        degenerate two-Gaussian fit is a single Gaussian wearing noise as
        its weight (R10 item 3: the G channel had tail == core to 0.01um)."""
        import json
        from sync_film_optics_from_charts import MTF

        for mfile in ("mtf_5207.json", "mtf_2383.json"):
            fits = json.loads((MTF / mfile).read_text())["fit"]
            for ch, fit in fits.items():
                w = fit.get("w")
                if w is None:
                    continue
                if w == 0.0:
                    self.assertEqual(fit.get("tail_sigma_um"), 0.0,
                                     f"{mfile}.{ch}: inert tail not zeroed")
                else:
                    # 0.1% tolerance mirrors the loader (boundary rounding)
                    self.assertGreaterEqual(
                        fit["tail_sigma_um"], 2.0 * fit["sigma_um"] * 0.999,
                        f"{mfile}.{ch}: degenerate tail")


if __name__ == "__main__":
    unittest.main()
