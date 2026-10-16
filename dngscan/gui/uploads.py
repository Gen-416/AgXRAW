# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility import for dngscan.transports.uploads."""
import sys
from dngscan.transports import uploads as _implementation
sys.modules[__name__] = _implementation
