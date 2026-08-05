# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import rawpy

from dngscan.libraw_policy import (
    PINNED_LIBRAW_COMMIT,
    RAWPY_SOURCE_COMMIT,
    RAWPY_SOURCE_URL,
    RAWPY_VERSION,
    rawpy_runtime_problem,
)


class LibRawPolicyTests(unittest.TestCase):
    def test_source_pin_is_shared_with_build_script(self) -> None:
        values = {}
        pin_file = Path(__file__).parents[1] / "tools" / "libraw-pin.env"
        for line in pin_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value
        self.assertEqual(values["RAWPY_VERSION"], RAWPY_VERSION)
        self.assertEqual(values["RAWPY_SOURCE_URL"], RAWPY_SOURCE_URL)
        self.assertEqual(values["RAWPY_SOURCE_COMMIT"], RAWPY_SOURCE_COMMIT)
        self.assertEqual(values["LIBRAW_COMMIT"], PINNED_LIBRAW_COMMIT)

    def test_install_manifests_pin_the_same_rawpy_commit(self) -> None:
        root = Path(__file__).parents[1]
        requirement = f"rawpy @ git+{RAWPY_SOURCE_URL}@{RAWPY_SOURCE_COMMIT}"
        self.assertIn(requirement, (root / "requirements.txt").read_text(encoding="utf-8"))
        self.assertIn(requirement, (root / "pyproject.toml").read_text(encoding="utf-8"))

    def test_runtime_uses_upgraded_rawpy(self) -> None:
        self.assertEqual(rawpy.__version__, RAWPY_VERSION)
        self.assertEqual(rawpy.libraw_source_commit, PINNED_LIBRAW_COMMIT)
        self.assertGreaterEqual(tuple(rawpy.libraw_version), (0, 22, 0))

    def test_runtime_gate_rejects_the_stock_wheel(self) -> None:
        stock = SimpleNamespace(__version__="0.27.0")
        problem = rawpy_runtime_problem(stock)
        self.assertIsNotNone(problem)
        self.assertIn("force-reinstall", problem)

    def test_runtime_gate_accepts_the_project_build(self) -> None:
        pinned = SimpleNamespace(
            __version__=RAWPY_VERSION,
            libraw_source_commit=PINNED_LIBRAW_COMMIT,
        )
        self.assertIsNone(rawpy_runtime_problem(pinned))


if __name__ == "__main__":
    unittest.main()
