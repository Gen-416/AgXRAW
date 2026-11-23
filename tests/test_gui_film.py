# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI film controls: options from the asset, pinned controls while a film is
active, stash/restore of the user's choices, SDR-only gating, payload keys."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest

from dngscan import film
from dngscan.gui.page import render_page

NODE = shutil.which("node")


def _html() -> str:
    return render_page("/tmp", "test-token").decode()


class GuiFilmMarkupTests(unittest.TestCase):
    def test_every_stock_is_offered_once_and_grouped(self):
        html = _html()
        select = html.split('<select id="film"', 1)[1].split("</select>", 1)[0]
        values = re.findall(r'<option value="([^"]+)"', select)
        self.assertEqual(values[0], "none")
        self.assertEqual(sorted(values[1:]), sorted(k for k in film.film_choices() if k != "none"))
        self.assertEqual(re.findall(r'<optgroup label="([^"]+)"', select), ["彩色负片", "电影负片", "反转片"])

    def test_settings_and_payload_carry_film(self):
        html = _html()
        ids = html.split("const SETTINGS_IDS=[", 1)[1].split("];", 1)[0]
        self.assertIn('"film"', ids)
        self.assertIn('"filmGrain"', ids)
        payload = html.split("function payload(", 1)[1].split("return p;", 1)[0]
        self.assertIn('film:$("#film").value', payload)
        self.assertIn('filmGrain:+$("#filmGrain").value', payload)


HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const src = require('node:fs').readFileSync(0, 'utf8');
const elements = new Map();
const el = (value, extra={}) => ({value, disabled:false, textContent:'', title:'', style:{}, dataset:{}, options:[], ...extra});
const $ = sel => { const id = sel.slice(1); if (!elements.has(id)) elements.set(id, el('')); return elements.get(id); };
for (const [id, v] of [['film','none'],['filmGrain','1'],['toneCore','gated'],['agxPrimaries','punchy'],
                       ['sceneTransform','arri_skin_d55'],['format','ultrahdr'],['punch','1']]) elements.set(id, el(v));
const calls = [];
const context = {$, console, setStatus:(m,k)=>calls.push(['status',k]),
  updateSceneTransformUi:()=>calls.push('st'), updateToneCoreUi:()=>calls.push('tc'),
  updateHdrOptionGate:()=>calls.push('hdr')};
vm.createContext(context);
vm.runInContext(src + '\n;globalThis.updateFilmUi=updateFilmUi;globalThis.filmActive=filmActive;', context);

// off: nothing pinned
context.updateFilmUi();
assert.equal($('#toneCore').value, 'gated'); assert.equal($('#toneCore').disabled, false);
assert.equal($('#filmGrainBlock').style.display, 'none');

// on: pinned + disabled, user choices stashed, HDR -> SDR
$('#film').value = 'provia100f';
context.updateFilmUi();
assert.deepEqual([$('#toneCore').value, $('#agxPrimaries').value, $('#sceneTransform').value], ['agx','base','none']);
assert.ok($('#toneCore').disabled && $('#agxPrimaries').disabled && $('#sceneTransform').disabled && $('#punch').disabled);
assert.deepEqual([$('#toneCore').dataset.filmStash, $('#agxPrimaries').dataset.filmStash, $('#sceneTransform').dataset.filmStash],
                 ['gated','punchy','arri_skin_d55']);
assert.equal($('#format').value, 'sdr');
assert.ok(calls.some(c => Array.isArray(c) && c[1] === 'warn'));
assert.equal($('#filmGrainBlock').style.display, 'block');
assert.match($('#filmFact').textContent, /反转片/);
assert.match($('#filmFact').textContent, /测光中灰 -0\.41 EV/);

// a second activation (e.g. after a decoder switch) keeps the original stash
context.updateFilmUi();
assert.equal($('#toneCore').dataset.filmStash, 'gated');

// off again: user choices come back, controls re-enabled
$('#film').value = 'none';
context.updateFilmUi();
assert.deepEqual([$('#toneCore').value, $('#agxPrimaries').value, $('#sceneTransform').value], ['gated','punchy','arri_skin_d55']);
assert.ok(!$('#toneCore').disabled && !$('#punch').disabled);
assert.equal($('#toneCore').dataset.filmStash, undefined);
assert.equal($('#filmFact').style.display, 'none');
"""


@unittest.skipUnless(NODE, "Node.js is required for GUI callback execution")
class GuiFilmRuntimeTests(unittest.TestCase):
    def test_pin_stash_restore_and_sdr_gate(self):
        html = _html()
        start = html.index("const FILM_META=")
        source = html[start:html.index("const CORE_FACTS=", start)]
        result = subprocess.run([NODE, "-e", HARNESS], input=source, text=True,
                                capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_meta_matches_the_asset(self):
        html = _html()
        meta = json.loads(html.split("const FILM_META=", 1)[1].split(";\n", 1)[0])
        stocks = film._asset()["stocks"]
        self.assertEqual(set(meta), set(stocks))
        self.assertEqual(meta["portra800"]["grain_scale"], stocks["portra800"]["grain_scale"])


if __name__ == "__main__":
    unittest.main()
