# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile a fixed local Collect registry. No network is used by this tool."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dngscan.calibration_registry import compile_catalog, digest, validate_release

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("sets_json", type=Path)
p.add_argument("--source-commit", required=True)
p.add_argument("--profile-record", action="append", type=Path, default=[],
               help="Reviewed release record: set_id, qualification, source_evidence, payload; no automatic cross-mode promotion")
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
data = args.sets_json.read_bytes()
result = compile_catalog(json.loads(data), source_commit=args.source_commit,
                         raw_sha256=hashlib.sha256(data).hexdigest())
for path in args.profile_record:
    record = json.loads(path.read_text())
    record["sha256"] = digest(record["payload"])
    result["profiles"].append(record)
result["version"] = digest({k: v for k, v in result.items() if k != "version"})[:24]
validate_release(result)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(result["version"])
