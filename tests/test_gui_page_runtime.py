# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute selection callbacks: syntax checks cannot catch missing globals."""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest

from dngscan.gui.page import PAGE


NODE = shutil.which("node")
HARNESS = r"""
const assert = require("node:assert/strict");
const vm = require("node:vm");
const payload = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const calls = [], statuses = [], listeners = new Map();
const file = {name: "sample RAW.DNG"};
const elements = new Map([
  ["#filePicker", {files: [file], value: "selected", disabled: false}],
  ["#input", {value: "/old/source.dng"}],
  ["#outdir", {value: ""}],
  ["#revealBtn", {style: {display: "block"}}],
  ["#coreimageVersion", {value: "8"}],
]);
for (const [id, element] of elements) {
  element.addEventListener = (event, callback) => {
    assert.equal(event, "change");
    listeners.set(id, callback);
  };
}
const context = vm.createContext({
  $: id => {assert.ok(elements.has(id), `unexpected DOM lookup ${id}`); return elements.get(id);},
  INIT_DIR: "/exports",
  lastSavedPath: "/old/export.jpg",
  beginPreviewSession: () => calls.push("begin"),
  saveSettings: () => calls.push("save"),
  fetchDecodeSupport: path => {assert.equal(path, "/uploads/sample.dng"); calls.push("support");},
  preparePreview: async () => {calls.push("prepare");},
  setStatus: (text, kind) => statuses.push({text, kind}),
  apiFetch: async (url, options) => {
    assert.equal(url, "/upload?name=sample%20RAW.DNG");
    assert.equal(options.method, "POST");
    assert.equal(options.headers["Content-Type"], "application/octet-stream");
    assert.equal(options.body, file);
    assert.equal(elements.get("#filePicker").disabled, true);
    assert.equal(elements.get("#input").value, "");
    assert.deepEqual(calls, ["begin"]);
    calls.push("upload");
    return {json: async () => ({ok: true, path: "/uploads/sample.dng"})};
  },
});
// Use the page's actual declarations, so a deleted global is not supplied by
// the harness and silently hidden. Seed stale entries before dispatch.
vm.runInContext(payload.declarations, context);
vm.runInContext('RAW9_PROBES.set("old", {}); RAW9_PROBE_REQUESTS.set("old", {});', context);
vm.runInContext(payload.listener, context);
(async () => {
  await listeners.get(payload.selector)();
  if (payload.selector === "#filePicker") {
    assert.equal(statuses.some(status => status.kind === "err"), false,
      JSON.stringify(statuses));
    assert.deepEqual(calls, ["begin", "upload", "save", "support", "prepare"]);
    assert.equal(vm.runInContext("RAW9_PROBES.size + RAW9_PROBE_REQUESTS.size", context), 0);
    assert.equal(elements.get("#input").value, "/uploads/sample.dng");
    assert.equal(elements.get("#outdir").value, "/exports");
    assert.equal(elements.get("#filePicker").disabled, false);
    assert.equal(elements.get("#revealBtn").style.display, "none");
    assert.equal(context.lastSavedPath, "");
    assert.equal(statuses.at(-1).kind, "ok");
  } else {
    assert.deepEqual(calls, ["save", "prepare"]);
    assert.equal(elements.get("#coreimageVersion").value, "8");
    assert.equal(vm.runInContext("RAW9_PROBES.size + RAW9_PROBE_REQUESTS.size", context), 2);
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
"""


@unittest.skipUnless(NODE, "Node.js is required for GUI callback execution")
class PageSelectionRuntimeTests(unittest.TestCase):
    def _run_listener(self, selector: str) -> None:
        start = PAGE.index(f'$("{selector}").addEventListener("change",')
        if selector == "#filePicker":
            end = PAGE.index("\nasync function listOutDir", start)
        else:
            end = PAGE.index("\n", start)
        declarations_start = PAGE.index("const RAW9_PROBES=")
        declarations_end = PAGE.index("function raw9Probe(", declarations_start)
        result = subprocess.run(
            [NODE, "-e", HARNESS],
            input=json.dumps({
                "selector": selector,
                "listener": PAGE[start:end],
                "declarations": PAGE[declarations_start:declarations_end],
            }),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_uploaded_file_reaches_support_probe_and_preview(self) -> None:
        self._run_listener("#filePicker")

    def test_explicit_apple_version_change_reprepares_preview(self) -> None:
        self._run_listener("#coreimageVersion")


if __name__ == "__main__":
    unittest.main()
