# SPDX-License-Identifier: GPL-3.0-or-later
"""Refit a scene-transform region window (mu/cov) from real DNG samples.

Replaces the demo spectral-model anchor with statistics measured on actual pixels
(e.g. a face). The fitted window is stored back in the DAYLIGHT calibration frame:
measured chromaticities are divided by the shot's von Kries transport ratio
(wb_adaptation_ratios), so the preset stays illuminant-portable and the runtime
adaptation keeps working across lighting. The region's 3x3 matrix (a spectral
crosstalk correction) is left untouched — only the chromaticity window moves.

Usage:
    python tools/fit_skin_window.py --dng photo.DNG \
        --bbox 0.47,0.27,0.68,0.46 --bbox 0.49,0.46,0.62,0.50 \
        --preset arri_skin_d55 --region skin [--write]

--bbox is x0,y0,x1,y1 normalized to the rendered (EXIF-rotated) frame, same
orientation you see in any exported JPEG. Multiple --dng/--bbox pairs accumulate;
each bbox applies to the most recent --dng.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dngscan.core as dg  # noqa: E402
from dngscan.scene_transform import SCENE_TRANSFORM_PRESETS_JSON, wb_adaptation_ratios  # noqa: E402

MIN_SAMPLES = 2000
VAR_FLOOR_RG = 0.0015  # keep the window from collapsing to a needle
VAR_FLOOR_BG = 0.0008


def collect_chroma(dng: Path, bboxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    bundle = dg.load_raw(dng, "clip", scene_half_size=True)
    scene = bundle.scene_rec2020_render
    h, w = scene.shape[:2]
    ratios = wb_adaptation_ratios(bundle.wb_mode, bundle.camera_wb, bundle.daylight_wb) or (1.0, 1.0)
    print(f"{dng.name}: buffer {w}x{h}  传输比 r={tuple(round(v, 3) for v in ratios)}")
    chunks = []
    for (x0, y0, x1, y1) in bboxes:
        xs, xe = sorted((int(x0 * w), int(x1 * w)))
        ys, ye = sorted((int(y0 * h), int(y1 * h)))
        patch = scene[ys:ye, xs:xe].reshape(-1, 3).astype(np.float64) / float(bundle.scene_scale)
        good = (patch[:, 1] > 1e-4) & (patch.max(axis=1) > 0.004)
        patch = patch[good]
        if patch.shape[0] == 0:
            print(f"  bbox {x0},{y0},{x1},{y1}: 无有效像素", file=sys.stderr)
            continue
        chroma = np.stack([patch[:, 0] / patch[:, 1], patch[:, 2] / patch[:, 1]], axis=1)
        # store in the daylight calibration frame: divide out this shot's transport
        chroma[:, 0] /= ratios[0]
        chroma[:, 1] /= ratios[1]
        chunks.append(chroma)
        print(f"  bbox ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f}): {chroma.shape[0]} 像素")
    if not chunks:
        raise SystemExit("没有可用样本")
    return np.concatenate(chunks)


def robust_fit(chroma: np.ndarray, iterations: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Trimmed Gaussian fit: bbox samples include hair/glasses/background bits."""
    keep = chroma
    mu = np.median(keep, axis=0)
    cov = np.cov(keep.T)
    for _ in range(iterations):
        inv = np.linalg.pinv(cov)
        d = keep - mu
        mahal = np.einsum("ij,jk,ik->i", d, inv, d)
        keep = keep[mahal < 4.0]  # ~2 sigma in 2D
        if keep.shape[0] < MIN_SAMPLES // 2:
            break
        mu = keep.mean(axis=0)
        cov = np.cov(keep.T)
    cov[0, 0] = max(cov[0, 0], VAR_FLOOR_RG)
    cov[1, 1] = max(cov[1, 1], VAR_FLOOR_BG)
    return mu, cov


MIN_COMPONENT_SAMPLES = 1500


def mixture_fit(chroma: np.ndarray, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """k-means split, then a trimmed Gaussian per cluster.

    Single-Gaussian trimming rejects secondary illumination clusters wholesale (measured:
    shade-lit skin sits ~0.55 B/G from tungsten-lit skin in the daylight frame); k-means
    keeps each lighting mode as its own component. Undersized clusters are dropped and K
    effectively shrinks. Returns (mu, cov, fraction) sorted by fraction, descending."""
    rng = np.random.default_rng(seed)
    k = max(1, int(k))
    if k == 1 or chroma.shape[0] < 2 * MIN_COMPONENT_SAMPLES:
        mu, cov = robust_fit(chroma)
        return [(mu, cov, 1.0)]
    centers = chroma[rng.choice(chroma.shape[0], size=k, replace=False)]
    labels = np.zeros(chroma.shape[0], dtype=np.int64)
    for _ in range(25):
        dist = ((chroma[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            member = chroma[labels == j]
            if member.shape[0]:
                centers[j] = member.mean(axis=0)
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    total = float(chroma.shape[0])
    for j in range(k):
        member = chroma[labels == j]
        if member.shape[0] < MIN_COMPONENT_SAMPLES:
            print(f"  分量 {j}: 仅 {member.shape[0]} 样本，丢弃（K 自动收缩）")
            continue
        mu, cov = robust_fit(member)
        out.append((mu, cov, member.shape[0] / total))
    if not out:
        mu, cov = robust_fit(chroma)
        out = [(mu, cov, 1.0)]
    out.sort(key=lambda item: -item[2])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dng", type=Path, action="append", required=True)
    parser.add_argument("--bbox", type=str, action="append", required=True,
                        help="x0,y0,x1,y1 归一化坐标；归属最近一个 --dng")
    parser.add_argument("--preset", default="arri_skin_d55")
    parser.add_argument("--region", default="skin")
    parser.add_argument("--write", action="store_true", help="写回 preset JSON（否则只打印）")
    parser.add_argument("--scale", type=float, default=1.5,
                        help="写回的区域 scale（实测 cov 不需要 demo 的 2.5 通胀，默认 1.5）")
    parser.add_argument("--components", type=int, default=1,
                        help="混合高斯分量数（>1 时按 k-means 分簇逐簇拟合；用于覆盖日光/荫下/钨丝灯"
                             "等多光照聚类。样本不足的簇会被丢弃并自动降 K）")
    args = parser.parse_args()

    # pair bboxes to dngs by CLI order
    per_dng: dict[Path, list[tuple[float, float, float, float]]] = {}
    argv = sys.argv[1:]
    current: Path | None = None
    for i, tok in enumerate(argv):
        if tok == "--dng":
            current = Path(argv[i + 1])
            per_dng.setdefault(current, [])
        elif tok == "--bbox" and current is not None:
            per_dng[current].append(tuple(float(v) for v in argv[i + 1].split(",")))  # type: ignore[arg-type]

    samples = np.concatenate([collect_chroma(d, boxes) for d, boxes in per_dng.items() if boxes])
    if samples.shape[0] < MIN_SAMPLES:
        raise SystemExit(f"样本太少（{samples.shape[0]} < {MIN_SAMPLES}），扩大 bbox 或加图")
    mu, cov = robust_fit(samples)
    print(f"\n拟合结果（日光标定系）: mu_rg_bg=({mu[0]:.5f}, {mu[1]:.5f})")
    print(f"cov = [[{cov[0,0]:.8f}, {cov[0,1]:.8f}], [{cov[1,0]:.8f}, {cov[1,1]:.8f}]]")
    print(f"σ_rg={np.sqrt(cov[0,0]):.4f}  σ_bg={np.sqrt(cov[1,1]):.4f}  样本数={samples.shape[0]}")

    if not args.write:
        print("\n(--write 未指定，未写回 JSON)")
        return 0

    raw = json.loads(SCENE_TRANSFORM_PRESETS_JSON.read_text(encoding="utf-8"))
    transforms = raw.get("transforms", raw)
    preset = transforms[args.preset]
    hit = False
    for region in preset.get("regions", []):
        if region.get("name") == args.region:
            region["mu_rg_bg"] = [float(mu[0]), float(mu[1])]
            region["cov_rg_bg"] = [[float(cov[0, 0]), float(cov[0, 1])], [float(cov[1, 0]), float(cov[1, 1])]]
            region["scale"] = float(args.scale)
            hit = True
    if not hit:
        raise SystemExit(f"preset {args.preset} 中找不到 region {args.region}")
    note = preset.get("note", "")
    preset["note"] = note + f" [{args.region} window refit from real DNG samples {date.today().isoformat()}]"
    SCENE_TRANSFORM_PRESETS_JSON.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n已写回 {SCENE_TRANSFORM_PRESETS_JSON}（region={args.region}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
