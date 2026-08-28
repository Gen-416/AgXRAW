# SPDX-License-Identifier: GPL-3.0-or-later
"""Compose a labelled comparison plate FROM SCRATCH: rows = films, columns = renders.

Unlike PlateSpec in regen_showcases.py (which pastes panels into an old plate
to keep its caption strips), this draws the strips itself (Hiragino Sans GB),
so a plate can change layout. Used by manifests' "composed_plates" entries.
Usage: compose_plate.py OUT.jpg SCENE_STEM COL1=label,COL2=label,... ROW1=label,...
Panels are cut to a common aspect from the full renders (center crop) and
resized to PANEL_W; a caption strip (black, PingFang) sits under each row."""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
PANEL_W = 800; GAP = 6; STRIP = 52; ASPECT = 3/2
def font(size):
    for p in ('/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/Hiragino Sans GB.ttc', '/System/Library/Fonts/STHeiti Medium.ttc', '/System/Library/Fonts/Supplemental/Songti.ttc'):
        try: return ImageFont.truetype(p, size, index=0)
        except Exception: continue
    return ImageFont.load_default()
def panel(path, aspect):
    im = Image.open(path).convert('RGB'); w, h = im.size
    tw, th = (w, int(w/aspect)) if w/h >= aspect else (int(h*aspect), h)
    tw, th = min(tw, w), min(th, h)
    x0, y0 = (w-tw)//2, (h-th)//2
    return im.crop((x0, y0, x0+tw, y0+th)).resize((PANEL_W, int(PANEL_W/aspect)), Image.LANCZOS)
def compose(out, spec):
    """spec: {"cols":[{"key":..,"label":..}], "rows":[{"key":..,"label":..,"files":{colkey: path}}]}"""
    cols, rows = spec['cols'], spec['rows']
    # plate aspect follows the source: portrait scenes get portrait panels (no crop)
    first = Image.open(rows[0]['files'][cols[0]['key']]); aspect = first.size[0] / first.size[1]
    aspect = ASPECT if aspect >= 1 else aspect
    ph = int(PANEL_W/aspect)
    W = len(cols)*PANEL_W + (len(cols)-1)*GAP
    H = len(rows)*(ph+STRIP) + (len(rows)-1)*GAP
    plate = Image.new('RGB', (W, H), (0, 0, 0)); d = ImageDraw.Draw(plate); f = font(26)
    for r, row in enumerate(rows):
        y = r*(ph+STRIP+GAP)
        for c, col in enumerate(cols):
            x = c*(PANEL_W+GAP)
            plate.paste(panel(row['files'][col['key']], aspect), (x, y))
            d.text((x+12, y+ph+12), f"{row['label']} · {col['label']}", font=f, fill=(235, 235, 235))
    plate.save(out, quality=92); print('wrote', out, plate.size)
if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')); compose(sys.argv[1], spec)
