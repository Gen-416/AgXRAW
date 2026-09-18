# SPDX-License-Identifier: GPL-3.0-or-later
"""Build shim for the optional Rust kernels (setuptools-rust).

The package metadata lives in pyproject.toml; this file only declares the
native extension. DNGSCAN_BUILD_NATIVE=OFF builds the pure-Python wheel
(the NumPy reference path stays fully functional), and `optional=True`
keeps a host without a Rust toolchain on that same fallback instead of
failing the install — the same contract the CMake build declared.
"""
import os

from setuptools import setup

native = os.environ.get("DNGSCAN_BUILD_NATIVE", "ON").strip().upper() not in {"OFF", "0", "FALSE"}
rust_extensions = []
if native:
    from setuptools_rust import Binding, RustExtension

    rust_extensions.append(
        RustExtension(
            "dngscan._dngscan_fast",
            path="rust/Cargo.toml",
            binding=Binding.PyO3,
            optional=os.environ.get("DNGSCAN_REQUIRE_NATIVE", "0") != "1",
            debug=False,
        )
    )

setup(rust_extensions=rust_extensions)
