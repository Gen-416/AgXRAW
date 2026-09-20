# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep a verified delivery private until metadata and final readback complete."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


class DeliveryTransaction:
    def __init__(self, destination: Path):
        self.destination = Path(destination)
        self._directory = None
        self.path = None

    def __enter__(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(
            prefix=".agxraw-delivery-", dir=self.destination.parent)
        self.path = Path(self._directory.name) / ("verified" + self.destination.suffix)
        return self

    def commit(self):
        if self.path is None or not self.path.is_file():
            raise RuntimeError("delivery transaction has no verified candidate")
        os.replace(self.path, self.destination)

    def __exit__(self, *_exc):
        self._directory.cleanup()
