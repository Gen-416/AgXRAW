# SPDX-License-Identifier: GPL-3.0-or-later
"""Local web GUI for dngscan."""


def main() -> int:
    """Load the optional HTTP adapter only when the GUI is launched."""
    from .server import main as server_main

    return server_main()

__all__ = ["main"]
