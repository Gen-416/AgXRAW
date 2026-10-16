# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility import for dngscan.transports.server."""
import sys
from dngscan.transports import server as _implementation
sys.modules[__name__] = _implementation
