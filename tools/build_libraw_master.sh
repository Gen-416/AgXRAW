#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Reinstall the project's rawpy fork with its pinned LibRaw *master* snapshot.
#
# Why: rawpy wheels bundle the newest LibRaw *release* (0.22.x), whose camera
# colour tables lag ~1-2 years behind master. Sony A7 V, for example, decodes
# with an empty colour matrix on the wheel build (see docs/SENSOR_SUPPORT.zh-CN.md).
# Master's soname bump (25 -> 26) makes a dylib
# swap unsafe, so the supported path is this source rebuild: rawpy's own sdist
# build machinery compiles and bundles whatever sits in external/LibRaw.
#
# The fallback matrix table (dngscan/camera_matrices.py) stays useful either
# way: it covers bodies missing even from master (A7R VI as of the pin below).
#
# Normal installs already use this exact dependency via requirements.txt. This
# script is the explicit repair path for an existing or accidentally stale venv.
#
# Usage:  sh tools/build_libraw_master.sh
# Optional: DNGSCAN_VENV=/path/to/venv sh tools/build_libraw_master.sh
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/tools/libraw-pin.env"
# Move the pin deliberately: rerun the full test suite and re-baseline the
# SDR freeze/golden fixtures if decode output shifts.
VENV=${DNGSCAN_VENV:-"$ROOT/.venv"}
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"
test -x "$PIP"
test -x "$PYTHON"

"$PIP" install --force-reinstall --no-deps \
  "rawpy @ git+$RAWPY_SOURCE_URL@$RAWPY_SOURCE_COMMIT"

cd "$ROOT"
"$PYTHON" - <<'EOF'
import rawpy
from dngscan.libraw_policy import PINNED_LIBRAW_COMMIT, RAWPY_VERSION

assert rawpy.__version__ == RAWPY_VERSION, (rawpy.__version__, RAWPY_VERSION)
assert rawpy.libraw_source_commit == PINNED_LIBRAW_COMMIT, (
    rawpy.libraw_source_commit,
    PINNED_LIBRAW_COMMIT,
)
print(
    "rawpy:", rawpy.__version__,
    "libraw:", rawpy.libraw_version,
    "source-pin:", rawpy.libraw_source_commit,
)
EOF
echo "Done. Run the full test suite; expect SDR freeze/golden drift if decode changed."
