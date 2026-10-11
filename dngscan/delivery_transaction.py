# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep a verified delivery private until metadata and final readback complete."""
from __future__ import annotations

from pathlib import Path
import tempfile
from .protected_output import ProtectedSources


class DeliveryTransaction:
    def __init__(self, destination: Path, *, protected_sources=None):
        self.destination = Path(destination)
        self.protection = ProtectedSources(protected_sources)
        self._directory = None
        self.path = None

    def __enter__(self):
        self.protection.check(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(
            prefix=".agxraw-delivery-", dir=self.destination.parent)
        self.path = Path(self._directory.name) / ("verified" + self.destination.suffix)
        return self

    def commit(self):
        if self.path is None or not self.path.is_file():
            raise RuntimeError("delivery transaction has no verified candidate")
        self.protection.replace(self.path, self.destination)

    def __exit__(self, *_exc):
        self._directory.cleanup()
