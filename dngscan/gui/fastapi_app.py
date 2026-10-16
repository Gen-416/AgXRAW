# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility import for dngscan.transports.http."""
import sys
from dngscan.transports import http as _implementation
sys.modules[__name__] = _implementation
