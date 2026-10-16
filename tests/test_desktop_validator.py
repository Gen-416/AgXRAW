# SPDX-License-Identifier: GPL-3.0-or-later
"""Native entry validation must never publish its report over a source RAW."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DesktopValidatorProtectionTests(unittest.TestCase):
    def test_source_and_alias_outputs_are_rejected_before_native_launch(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.dng'
            source.write_bytes(b'private RAW evidence fixture')
            hardlink = Path(folder)/'hardlink.json'; os.link(source, hardlink)
            symlink = Path(folder)/'symlink.json'; symlink.symlink_to(source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            for destination in (source, hardlink, symlink):
                with self.subTest(destination=destination.name):
                    result = subprocess.run([sys.executable, str(root/'tools/validate_desktop_entry.py'),
                        str(source), '--output', str(destination), '--binary', '/does-not-exist'],
                        capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('禁止覆盖', result.stderr)
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
