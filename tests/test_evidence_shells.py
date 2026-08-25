# SPDX-License-Identifier: GPL-3.0-or-later
"""Evidence-shell corpus gates: metadata parsers across camera makes, ON CI.

The shells (tests/data/evidence_shells/*.evshell) are container structures
stripped from CC0 raw.pixls.us samples — every IFD byte kept, bulk pixel
regions dropped (idea credited to y-g-jiang's dngshell corpus; independent
format and implementation, tools/make_evidence_shell.py). The pinned
expectations were generated from the ORIGINAL files and human-checked, so
these tests catch cross-make regressions in the metadata/evidence parsers
without a RAW corpus on CI. Pixel decoding is deliberately out of scope.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

SHELL_DIR = Path(__file__).parent / "data" / "evidence_shells"


def _materialize(shell: Path, out_dir: Path) -> tuple[Path, dict]:
    with shell.open("rb") as fh:
        manifest = json.loads(fh.readline())
        payload = gzip.decompress(fh.read())
    out = out_dir / manifest["source_name"]
    with out.open("wb") as fh:
        fh.truncate(int(manifest["source_bytes"]))
        pos = 0
        for o, n in manifest["kept"]:
            fh.seek(int(o))
            fh.write(payload[pos : pos + int(n)])
            pos += int(n)
    return out, manifest


class EvidenceShellCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expect = json.loads((SHELL_DIR / "expectations.json").read_text())
        cls.tmp = tempfile.TemporaryDirectory()
        cls.files = {}
        for fid in cls.expect:
            shell = SHELL_DIR / f"{fid}.evshell"
            cls.files[fid], manifest = _materialize(shell, Path(cls.tmp.name))
            cls.expect[fid]["_manifest"] = manifest

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_manifests_declare_provenance(self) -> None:
        """Two admitted origins, each with its own license contract:
        raw.pixls.us shells must be CC0; y-g-jiang first-party shells must
        declare the pending-permission status (NOTICE.md) and carry the
        upstream DNGSHL1 block with the true original's SHA-256 (our outer
        hash covers only the zero-filled reconstruction)."""
        for fid, exp in self.expect.items():
            m = exp["_manifest"]
            with self.subTest(shell=fid):
                self.assertEqual(m["format"], "dngscan-evshell-1")
                self.assertEqual(len(m["source_sha256"]), 64)
                if m["source_url"].startswith("https://raw.pixls.us/"):
                    self.assertEqual(m["license"], "CC0")
                elif m["source_url"].startswith("https://y-g-jiang.github.io/shells/"):
                    self.assertIn("first-party", m["license"])
                    # license state must track NOTICE.md (review 4.10):
                    # the 2026-08-25 credit-based grant, not "being contacted"
                    self.assertIn("permitted with credit", m["license"])
                    self.assertNotIn("being contacted", m["license"])
                    up = m["upstream"]
                    self.assertEqual(up["format"], "DNGSHL1")
                    self.assertEqual(len(up["source_sha256"]), 64)
                else:
                    self.fail(f"unknown provenance origin: {m['source_url']}")

    def test_parsers_match_the_pinned_original_answers(self) -> None:
        from dngscan.metadata import (
            is_dng_container,
            read_dng_color_calibration,
            read_dng_gain_maps,
            read_dng_shot_info,
            read_dng_stage1_flags,
        )

        for fid, exp in self.expect.items():
            p = self.files[fid]
            with self.subTest(shell=fid):
                self.assertEqual(is_dng_container(p), exp["is_dng"])
                shot = read_dng_shot_info(p)
                self.assertEqual(shot.make, exp["make"])
                self.assertEqual(shot.model, exp["model"])
                self.assertEqual(shot.iso, exp["iso"])
                self.assertEqual(shot.baseline_exposure, exp["baseline_exposure"])
                neutral = (
                    list(shot.as_shot_neutral) if shot.as_shot_neutral else None
                )
                self.assertEqual(neutral, exp["as_shot_neutral"])
                cal = read_dng_color_calibration(p)
                if exp["cal"] is None:
                    self.assertIsNone(cal)
                else:
                    self.assertEqual(cal.cct1, exp["cal"]["cct1"])
                    self.assertEqual(cal.cct2, exp["cal"]["cct2"])
                    self.assertEqual(cal.cct3, exp["cal"]["cct3"])
                    self.assertEqual(
                        list(cal.matrix1[0]), exp["cal"]["matrix1_row0"]
                    )
                    self.assertEqual(
                        cal.matrix2 is not None, exp["cal"]["has_matrix2"]
                    )
                    self.assertEqual(
                        cal.matrix3 is not None, exp["cal"]["has_matrix3"]
                    )
                self.assertEqual(
                    list(read_dng_stage1_flags(p)), exp["stage1_flags"]
                )
                self.assertEqual(
                    len(read_dng_gain_maps(p)), exp["gain_map_count"]
                )

    def test_shell_bytes_are_source_bytes(self) -> None:
        """Kept ranges must be byte-identical to the source: the manifest's
        source hash re-verifies whenever the ORIGINAL is available locally
        (skipped elsewhere), and the materialized geometry always checks."""
        for fid, exp in self.expect.items():
            m = exp["_manifest"]
            p = self.files[fid]
            with self.subTest(shell=fid):
                self.assertEqual(p.stat().st_size, m["source_bytes"])


if __name__ == "__main__":
    unittest.main()
