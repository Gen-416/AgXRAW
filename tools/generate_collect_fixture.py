# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the pinned upstream pure writers, with synthetic statistics (no photos).

Usage: python tools/generate_collect_fixture.py /path/to/JiangtherapeeTesterView
Does not install or execute the Electron app. The fixture contains writer
output and source hashes, not vendored upstream implementation code.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

COMMIT = "57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c"
SCRIPT = r'''
import {buildFileStem} from './src/analysis/pairing.mjs';
import {writeDarkScalarCsv,writeDarkSpectrumCsv} from './src/output/darkCsv.mjs';
import {writeIsoGainCsv,writePtcCsv} from './src/output/entryCsv.mjs';
const meta={toolVersion:'0.1.0',librawVersion:'0.22.2',generated:'2026-10-10T00:00:00Z',camera:'NIKON TestCamera',firmware:'1.00',shutterType:'mechanical',compression:'lossless',longExposureNr:'off',stabiliser:'off',imageWidth:6000,imageHeight:4000,rawWidth:6000,rawHeight:4000,cfaPattern:'RGBG',adcStep:1,curveIsIdentity:true,black:100,maximum:4095,cropW:512,cropH:512,planeW:256,planeH:256,cropSize:512,planeSize:256,clipSigma:3.5,clipVarianceFactor:1,window:'hann',ladder:'paired-shutter',iso:100};
const power=Array.from({length:129},(_,i)=>i?4/256:0);
const channels=m=>[0,1,3,2].map(color=>({color,measured:{...m},spectra:{single:{h:power,v:power},diff:{h:power.map(x=>x*2),v:power.map(x=>x*2)}}}));
const dark=[100,200].map(iso=>({iso,shutter:.004,fileA:`dark${iso}a.NEF`,fileB:`dark${iso}b.NEF`,channels:channels({n:65536,blackA:100,blackB:100,stdA:2,stdB:2,stdAMasked:2,stdDiffRaw:Math.sqrt(8),stdDiffClipped:Math.sqrt(8),diffMean:0,rejected:0,withinRowVarSingle:4,withinColVarSingle:4,withinRowVarDiff:8,withinColVarDiff:8})}));
const gain=[100,200].map(iso=>({iso,shutter:.01,aperture:8,shutterGroup:'same',file:`gain${iso}.NEF`,channels:channels({n:65536,mean:100+iso*10,std:20,clipFrac:0})}));
const signal=Array.from({length:20},(_,i)=>10*Math.pow(130,i/19)).concat([3995,3995]);
const ptc=iso=>signal.map((s,i)=>({iso,shutter:(i+1)/1000,aperture:8,fileA:`ptc${i}a.NEF`,fileB:`ptc${i}b.NEF`,channels:channels({n:65536,meanA:100+s,meanB:100+s,stdA:i<20?Math.sqrt(s/(iso===100?2:1)+4):0,stdB:i<20?Math.sqrt(s/(iso===100?2:1)+4):0,stdAMasked:i<20?Math.sqrt(s/(iso===100?2:1)+4):0,stdDiffRaw:i<20?Math.sqrt(2*s/(iso===100?2:1)+8):0,stdDiffClipped:i<20?Math.sqrt(2*s/(iso===100?2:1)+8):0,diffMean:0,rejected:0,clipFrac:i<20?0:1})}));
const stem=(entry,range)=>buildFileStem(meta,meta.camera,range,entry,'20261010');
console.log(JSON.stringify([
{name:`${stem('dark','100-200')}_scalars.csv`,text:writeDarkScalarCsv(dark,meta)},
{name:`${stem('dark','100-200')}_spectrum-h.csv`,text:writeDarkSpectrumCsv(dark,meta,'h')},
{name:`${stem('dark','100-200')}_spectrum-v.csv`,text:writeDarkSpectrumCsv(dark,meta,'v')},
{name:`${stem('gain','100-200')}_levels.csv`,text:writeIsoGainCsv(gain,meta)},
...[100,200].map(iso=>({name:`${stem('ptc','100-200')}_ptc_iso${iso}.csv`,text:writePtcCsv(ptc(iso),{...meta,iso})}))]));
'''


def main():
    upstream = Path(sys.argv[1]).resolve()
    paths = ["src/analysis/pairing.mjs", "src/analysis/exifShutter.mjs",
             "src/analysis/shutterText.mjs", "src/analysis/channelOrder.mjs",
             "src/output/darkCsv.mjs", "src/output/entryCsv.mjs"]
    hashes = {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest() for name in paths}
    files = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", SCRIPT], cwd=upstream, text=True))
    folder = Path(__file__).resolve().parents[1] / "tests/fixtures/collect-writer"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "official.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            entry = zipfile.ZipInfo(item["name"], date_time=(2026, 10, 10, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, item["text"])
    metadata = {"upstream": "https://github.com/y-g-jiang/JiangtherapeeTesterView", "commit": COMMIT,
                "source_sha256": hashes, "fixture": "synthetic measured statistics, not real camera captures",
                "zip_sha256": hashlib.sha256((folder / "official.zip").read_bytes()).hexdigest()}
    (folder / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
