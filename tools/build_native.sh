#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
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
CMAKE_BIN="$("$PYTHON" -c 'import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), "data", "bin", "cmake"))')"
PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
"$CMAKE_BIN" -S . -B build/native \
  -DDNGSCAN_BUILD_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYTHON" \
  -Dpybind11_DIR="$PYBIND11_DIR"
"$CMAKE_BIN" --build build/native -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
NATIVE_MODULE="$(find build/native/cpp -maxdepth 1 -name '_dngscan_fast*.so' -print -quit)"
if [[ -z "$NATIVE_MODULE" ]]; then
  echo "native module was not produced" >&2
  exit 1
fi
TARGET_MODULE="dngscan/$(basename "$NATIVE_MODULE")"
STAGED_MODULE="${TARGET_MODULE}.new"
cp "$NATIVE_MODULE" "$STAGED_MODULE"
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Re-sign after the copy and atomically replace the old inode. Overwriting a loaded,
  # linker-signed bundle in place leaves macOS with a stale cs_mtime and SIGKILLs the
  # next importer with cs_invalid_page.
  codesign --force --sign - "$STAGED_MODULE"
fi
mv -f "$STAGED_MODULE" "$TARGET_MODULE"
echo "Installed native module into dngscan/"
# R4: this dev copy lives INSIDE the package directory. scikit-build-core
# packages the source tree, so a wheel built from this working tree would
# ship the stale dev binary alongside the CMake-installed one (dead weight,
# wrong ABI on other interpreters). Build release wheels from a clean
# checkout: git stash -u / fresh clone, or delete dngscan/*.so first.
echo "NOTE: remove dngscan/*.so before building a wheel (dev copy must not ship)."
