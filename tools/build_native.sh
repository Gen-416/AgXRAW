#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build the Rust kernels in place (dngscan/_dngscan_fast*.so) for development
# and CI. Needs a Rust toolchain (rustup: https://rustup.rs) and setuptools-rust.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="$(command -v python)"
  fi
fi
if ! command -v cargo >/dev/null 2>&1; then
  # Homebrew's rustup keeps the proxies out of PATH by default.
  for candidate in "$HOME/.cargo/bin" "$(brew --prefix rustup 2>/dev/null)/bin"; do
    if [[ -x "$candidate/cargo" ]]; then export PATH="$candidate:$PATH"; break; fi
  done
fi
command -v cargo >/dev/null 2>&1 || { echo "cargo not found: install Rust via rustup (https://rustup.rs)" >&2; exit 1; }
"$PYTHON" -c "import setuptools_rust" 2>/dev/null || "$PYTHON" -m pip install --quiet "setuptools-rust>=1.10"
DNGSCAN_REQUIRE_NATIVE=1 PYO3_PYTHON="$PYTHON" "$PYTHON" setup.py build_rust --inplace --release
NATIVE_MODULE="$(find dngscan -maxdepth 1 -name '_dngscan_fast*.so' -print -quit)"
if [[ -z "$NATIVE_MODULE" ]]; then
  echo "native module was not produced" >&2
  exit 1
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Re-sign the fresh bundle: macOS SIGKILLs an importer of a bundle whose
  # signature no longer matches its inode contents (cs_invalid_page).
  codesign --force --sign - "$NATIVE_MODULE"
fi
"$PYTHON" -c 'from dngscan import _dngscan_fast as n; from dngscan.fast_plan import NATIVE_ABI_VERSION; assert n.native_abi_version() == NATIVE_ABI_VERSION and n.self_test(), "native ABI/self-test failed"'
echo "Installed native module into dngscan/ ($NATIVE_MODULE)"
# This dev copy lives INSIDE the package directory; build release wheels from
# a clean checkout so it is not shipped alongside the wheel-built one.
echo "NOTE: remove dngscan/*.so before building a wheel (dev copy must not ship)."
