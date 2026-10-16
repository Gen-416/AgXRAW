# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility import; implementation lives in dngscan.application.constants."""
import sys
from dngscan.application import constants as _implementation

# Alias the module itself: legacy monkeypatches, singletons and imports must
# share precisely the same state with new application consumers.
sys.modules[__name__] = _implementation
