# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility import; implementation lives in dngscan.application.scheduler."""
import sys
from dngscan.application import scheduler as _implementation

# Alias the module itself: legacy monkeypatches, singletons and imports must
# share precisely the same state with new application consumers.
sys.modules[__name__] = _implementation
