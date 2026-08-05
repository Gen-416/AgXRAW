# SPDX-License-Identifier: GPL-3.0-or-later
"""Pinned source policy for the LibRaw evidence provider."""

RAWPY_VERSION = "0.27.0+libraw.e419de08"
RAWPY_SOURCE_URL = "https://github.com/Gen-416/rawpy.git"
RAWPY_SOURCE_COMMIT = "cc7b4748c7b3e87da319198fdfcdb46e17c9c2a6"
PINNED_LIBRAW_COMMIT = "e419de08001de28ae6988ecb22df47e52b9c5eaa"


def rawpy_runtime_problem(rawpy_module: object) -> str | None:
    """Return a diagnostic when an environment bypasses the project pin."""
    actual_version = getattr(rawpy_module, "__version__", "unknown")
    actual_libraw_commit = getattr(rawpy_module, "libraw_source_commit", "unknown")
    if (
        actual_version == RAWPY_VERSION
        and actual_libraw_commit == PINNED_LIBRAW_COMMIT
    ):
        return None
    return (
        "wrong rawpy/LibRaw build: installed rawpy "
        f"{actual_version} (LibRaw source {actual_libraw_commit}), expected rawpy "
        f"{RAWPY_VERSION} (LibRaw source {PINNED_LIBRAW_COMMIT}). "
        "Reinstall the project dependencies with "
        "`python -m pip install --force-reinstall -r requirements.txt`."
    )
