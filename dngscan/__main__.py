#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run dngscan as a module: python -m dngscan."""

import sys

from .core import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
