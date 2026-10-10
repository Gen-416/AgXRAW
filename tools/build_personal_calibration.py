# SPDX-License-Identifier: GPL-3.0-or-later
"""Build/inspect experimental body products from explicitly prepared signed-DN data.

Input NPZ has training, heldout and complete metadata_black arrays for BOTH
products. Dark frames remain pre-black RAW DN; flat buffers must have exactly
that metadata black field removed, with contract input_state explicitly equal
to metadata-black-subtracted-signed-dn. This state is never inferred. Contract
JSON has conditions, source, black_reference (dark) or illumination_verified
(flat). RAW unpack/LUT qualification must happen before this tool: arbitrary
photos, DNG suffixes and clipped integer buffers cannot establish that domain.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dngscan import pixel_calibration as pc

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("kind", choices=("inspect", "dark", "flat"))
p.add_argument("input", type=Path)
p.add_argument("--contract", type=Path)
p.add_argument("--dark-product", type=Path)
p.add_argument("--out", type=Path)
args = p.parse_args()
if args.kind == "inspect":
    result = pc.load_product(args.input)
else:
    if args.contract is None or args.out is None: p.error("build requires --contract and --out")
    options = json.loads(args.contract.read_text())
    with np.load(args.input, allow_pickle=False) as data:
        if args.kind == "dark":
            result = pc.build_dark_product(data["training"], data["heldout"], conditions=options["conditions"],
                source=options["source"], metadata_black=data["metadata_black"],
                black_reference=options.get("black_reference"), mode=options.get("mode", "residual"),
                raw_floor_dn=options.get("raw_floor_dn", 0.))
        else:
            if args.dark_product is None: p.error("flat requires --dark-product")
            result = pc.build_flat_product(data["training"], data["heldout"], conditions=options["conditions"],
                source=options["source"], dark_product=pc.load_product(args.dark_product),
                metadata_black=data["metadata_black"], input_state=options.get("input_state"),
                illumination_verified=options.get("illumination_verified", False))
    pc.save_product(args.out, result)
print(json.dumps(result["manifest"], ensure_ascii=False, indent=2, allow_nan=False))
