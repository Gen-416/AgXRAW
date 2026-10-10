# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-shutter ISO ratios and per-phase black subtraction have separate contracts."""
import csv
import io
import math
from pathlib import Path
import random
import tempfile
import unittest

from dngscan import calibration


def write_csv(path, columns, rows, **header):
    text = io.StringIO()
    for key, value in header.items():
        text.write(f"#{key}: {value}\n")
    writer = csv.writer(text)
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(text.getvalue())


class GainGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gain = self.root / 'gain-levels.csv'
        self.dark = {iso: {'bl': 1024.} for iso in (100, 200, 400, 800, 1600)}

    def ladder(self, rows, protocol='paired-shutter'):
        write_csv(self.gain, ['ISO', 'ShutterSec', 'ShutterGroup', 'ColorIndex', 'ClipFrac', 'Mean'],
                  rows, Ladder=protocol)
        return calibration.read_isogain(self.gain, self.dark, return_diagnostics=True)

    def test_nominal_shutter_error_cancels_on_two_links(self):
        rows = [[100,.01,'A',1,0,2249], [200,.01,'A',1,0,3474],
                [200,.005,'B',1,0,2299], [400,.005,'B',1,0,3574]]
        relative, facts = self.ladder(rows)
        for iso, expected in ((100,1.), (200,.5), (400,.25)):
            self.assertAlmostEqual(relative[iso], expected, places=14)
        self.assertEqual(facts['policy'], 'paired-edges-only')
        self.assertAlmostEqual(facts['components'][0]['residual_rms_ev'], 0.)
        for row in rows:
            row[1] *= 3.7 if row[2] == 'A' else .23
        changed, _ = self.ladder(rows)
        self.assertEqual(changed, relative)

    def test_duplicates_average_inside_the_group_and_order_is_irrelevant(self):
        rows = [[100,.01,'A',1,0,2229], [100,.01,'A',1,0,2269],
                [200,.01,'A',1,0,3474], [200,.005,'B',1,0,2299],
                [400,.005,'B',1,0,3574]]
        relative, facts = self.ladder(rows)
        random.Random(147).shuffle(rows)
        repeated, changed = self.ladder(rows)
        self.assertEqual(relative, repeated)
        self.assertEqual(facts, changed)
        self.assertAlmostEqual(relative[200], .5)
        edge = next(e for e in facts['components'][0]['edges'] if e['group'] == 'A')
        self.assertEqual(edge['repeat_counts'], [2,1])
        self.assertAlmostEqual(edge['weight'], 4/3)

    def test_disconnected_and_isolated_isos_do_not_form_a_complete_curve(self):
        rows = [[100,1,'A',1,0,1524], [200,1,'A',1,0,2024],
                [400,.5,'B',1,0,1524], [800,.5,'B',1,0,2024],
                [1600,.1,'C',1,0,1524]]
        relative, facts = self.ladder(rows)
        self.assertEqual(sorted(relative), [100,200])
        self.assertEqual(facts['disconnected_isos'], [400,800,1600])
        self.assertEqual([c['isos'] for c in facts['components']], [[100,200],[400,800],[1600]])
        other = calibration.read_isogain(self.gain, self.dark, anchor_iso=800)
        self.assertEqual(sorted(other), [400,800])
        self.assertAlmostEqual(other[800], .5)

    def test_mixed_keeps_auto_shutter_measurements_as_separate_evidence(self):
        relative, facts = self.ladder([[100,1,'A',1,0,1524], [200,1,'A',1,0,2024],
                                      [400,.7,'B',1,0,4024]], protocol='mixed')
        self.assertEqual(sorted(relative), [100,200])
        self.assertEqual(facts['policy'], 'paired-edges-only')
        self.assertEqual(sorted(facts['auto_shutter_components'][0]['relative_gain']), [100,200,400])

    def test_clipped_or_near_black_bridge_is_not_reintroduced_using_nominal_time(self):
        relative, facts = self.ladder([[100,1,'A',1,0,1524], [200,1,'A',1,.5,2024],
                                      [200,.5,'B',1,0,1124], [400,.5,'B',1,0,2024]])
        self.assertFalse(relative)
        self.assertEqual({r['reason'] for r in facts['rejected_rows']}, {'clipped','signal-below-200-DN'})

    def test_closure_drift_is_visible_in_edge_residuals(self):
        _, facts = self.ladder([[100,1,'A',1,0,1524], [200,1,'A',1,0,2024],
                               [200,.5,'B',1,0,1524], [400,.5,'B',1,0,2024],
                               [100,.25,'C',1,0,1524], [400,.25,'C',1,0,3224]])
        self.assertGreater(facts['components'][0]['residual_rms_ev'], .01)
        self.assertTrue(any(abs(e['residual_ev']) > .01 for e in facts['components'][0]['edges']))

    def test_auto_shutter_still_declares_and_uses_nominal_time(self):
        rows = [[100,1,'A',1,0,1524], [200,.5,'B',1,0,1524]]
        relative, facts = self.ladder(rows, protocol='auto-shutter')
        self.assertAlmostEqual(relative[200], .5)
        self.assertEqual(facts['policy'], 'nominal-time-dependent')
        rows[1][1] = .4
        relative, _ = self.ladder(rows, protocol='auto-shutter')
        self.assertAlmostEqual(relative[200], .4)

    def test_contradictory_named_shutter_group_rejects(self):
        with self.assertRaisesRegex(ValueError, 'different shutter'):
            self.ladder([[100,1,'A',1,0,1524],[200,.5,'A',1,0,2024]])

    def measured_dark(self):
        path = self.root / 'dark-scalars.csv'
        rows = [[iso,channel,cid,black,black,2.] for iso in (100,200)
                for channel,cid,black in [('C00',0,980),('C01',1,1000),('C10',3,1040),('C11',2,1010)]]
        write_csv(path, ['ISO','Channel','ColorIndex','BlackA','BlackB','StdDiffClipped'], rows,
                  AdcStep=0, ClipVarianceFactor=1)
        _, self.dark = calibration.read_dark(path)

    def test_unequal_green_black_levels_are_subtracted_before_ratios(self):
        self.measured_dark()
        self.assertEqual(self.dark[100]['bl'], 1020.)
        self.assertEqual(self.dark[100]['phases']['C01']['bl'], 1000.)
        self.assertEqual(self.dark[100]['phases']['C10']['bl'], 1040.)
        self.assertEqual(set(self.dark[100]['phases']), {'C00','C01','C10','C11'})
        write_csv(self.gain, ['ISO','Channel','ColorIndex','ClipFrac','Mean','ShutterSec'],
                  [[100,'C01',1,0,1400,1],[100,'C10',3,0,1440,1],
                   [200,'C01',1,0,1800,1],[200,'C10',3,0,1840,1]])
        self.assertAlmostEqual(calibration.read_isogain(self.gain, self.dark)[200], .5, places=15)

    def test_unequal_green_response_and_missing_green_do_not_shift_gain(self):
        self.measured_dark()
        columns = ['ISO','Channel','ColorIndex','ClipFrac','Mean','ShutterSec']
        first = [100,'C01',1,0,1400,1]
        extra = [100,'C10',3,0,2240,1]
        second = [200,'C01',1,0,1800,1]
        write_csv(self.gain, columns, [first,extra,second])
        self.assertAlmostEqual(calibration.read_isogain(self.gain, self.dark)[200], .5, places=15)
        write_csv(self.gain, columns, [first,extra,second,[200,'C10',3,0,3440,1]])
        self.assertAlmostEqual(calibration.read_isogain(self.gain, self.dark)[200], .5, places=15)


if __name__ == '__main__':
    unittest.main()
