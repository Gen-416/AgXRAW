# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only RAW input; codec measurements against a single uncompressed master."""
import sys, json, time, gc, math, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, JpegImagePlugin
from dngscan.raw_io import load_raw, release_analysis_buffers
from dngscan.analysis import analyze
from dngscan.auto_ev import compute_auto_ev
from dngscan.scene_scale import with_intent_exposure
from dngscan.tone import build_render_plan
from dngscan.render import render_output_u8
from dngscan.export import save_jpeg_array
from dngscan.auto_encode import coding_metrics
from dngscan import gainmap
from dngscan.delivery import resolve_delivery_profile
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.hdr_agx import render_ultrahdr_agx_pair, to_gainmap_alternate
from dngscan.models import HdrDisplayTarget
ROOT = Path('.')

def encode_ci(base, path, q, container, hdr=None, request=None, factor=2):
    import Quartz as Q
    from Foundation import NSURL, NSNumber
    p3 = Q.CGColorSpaceCreateWithName(Q.kCGColorSpaceDisplayP3)
    rgba = np.empty(base.shape[:2] + (4,), np.uint8)
    rgba[:, :, :3] = base
    rgba[:, :, 3] = 255
    image, data = gainmap._ciimage_from_rgba(rgba, Q.kCIFormatRGBA8, p3)
    image = image.imageBySettingContentHeadroom_(1.0)
    options = {Q.kCGImageDestinationLossyCompressionQuality: q / 100.0}
    req = {Q.kCGImageDestinationEncodeBaseIsSDR: gainmap._nsnumber_bool(True)}
    if request:
        req[Q.kCGImageDestinationEncodeBasePixelFormatRequest] = NSNumber.numberWithUnsignedInt_(int.from_bytes(request.encode(), 'big'))
    if hdr is not None:
        linear = Q.CGColorSpaceCreateWithName(Q.kCGColorSpaceExtendedLinearDisplayP3)
        hi, hd = gainmap._ciimage_from_rgba(hdr, Q.kCIFormatRGBAh, linear)
        hi = hi.imageBySettingContentHeadroom_(float(hdr[:, :, :3].max()))
        options[Q.kCIImageRepresentationHDRImage] = hi
        options[Q.kCIImageRepresentationHDRGainMapAsRGB] = gainmap._nsnumber_bool(True)
        options[Q.kCGImageDestinationEncodeRequest] = Q.kCGImageDestinationEncodeToISOGainmap
        if factor is not None:
            req[Q.kCGImageDestinationEncodeGainMapSubsampleFactor] = NSNumber.numberWithInt_(factor)
    options[Q.kCGImageDestinationEncodeRequestOptions] = req
    ctx = Q.CIContext.contextWithOptions_({Q.kCIContextCacheIntermediates: gainmap._nsnumber_bool(False)})
    url = NSURL.fileURLWithPath_(str(path))
    if container == 'heic':
        result = ctx.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_(image, url, Q.kCIFormatRGBA8, p3, options, None)
    else:
        result = ctx.writeJPEGRepresentationOfImage_toURL_colorSpace_options_error_(image, url, p3, options, None)
    ok, error = result if isinstance(result, tuple) else (bool(result), None)
    if not ok:
        raise RuntimeError(str(error))

def detail_metrics(decoded, reference):
    sumerr = sumref = count = 0.0
    rgbss = 0.0
    h, w = reference.shape[:2]
    for y in range(0, h, 128):
        a = decoded[y:min(y + 129, h)].astype(np.float32)
        b = reference[y:min(y + 129, h)].astype(np.float32)
        rgbss += float(np.square(a[:128] - b[:128]).sum(dtype=np.float64))
        ay = a @ np.array([0.299, 0.587, 0.114], np.float32)
        by = b @ np.array([0.299, 0.587, 0.114], np.float32)
        for axis in (0, 1):
            ga = np.diff(ay if axis == 0 else ay[:128], axis=axis)
            gb = np.diff(by if axis == 0 else by[:128], axis=axis)
            mask = np.abs(gb) >= 5
            sumerr += float(np.square((ga - gb)[mask]).sum(dtype=np.float64))
            sumref += float(np.square(gb[mask]).sum(dtype=np.float64))
            count += np.count_nonzero(mask)
    return {'edge_gradient_rmse': math.sqrt(sumerr / max(count, 1)), 'edge_error_ratio': math.sqrt(sumerr / max(sumref, 1e-12)), 'rgb_psnr': 10 * math.log10(255 ** 2 / (rgbss / (h * w * 3))) if rgbss else 100.0}

def one(p, mode, encoders='system'):
    folder = ROOT / p.stem
    folder.mkdir(exist_ok=True)
    master = folder / (mode + '-base.npy')
    hdrfile = folder / 'hdr-alternate.npy'
    if master.exists():
        raise ValueError(f'Benchmark master already exists: {master}; use a fresh --out directory to avoid stale render data')
    if not master.exists():
        b = load_raw(p, 'clip')
        a, _, _ = analyze(b, 4, diagnostics=False, gamut_names=('P3',))
        ev = compute_auto_ev(b, a)
        b = with_intent_exposure(b, user_ev=ev.ev)
        plan = build_render_plan(b, a, 'agx', 'p3')
        b = release_analysis_buffers(b)
        if mode == 'hdr':
            hp = compile_hdr_agx_plan(plan, HdrDisplayTarget(peak_nits=800), analysis=a, scene_decoder='libraw')
            base, linear = render_ultrahdr_agx_pair(b, a, plan, hp)
            hdr = to_gainmap_alternate(linear, hp.tone.peak_linear)
            np.save(hdrfile, hdr)
            del linear, hdr
        else:
            base = render_output_u8(b, a, 'p3', plan)
        np.save(master, base)
        (folder / (mode + '-source.json')).write_text(json.dumps({'raw': str(p), 'raw_bytes': p.stat().st_size, 'shape': base.shape, 'ev': ev.ev, 'highlight': 'clip', 'gamut': 'p3', 'tone': 'agx/base'}))
        Image.fromarray(base).resize((int(base.shape[1] * min(1000 / base.shape[1], 1000 / base.shape[0])), int(base.shape[0] * min(1000 / base.shape[1], 1000 / base.shape[0])))).save(folder / (mode + '-overview.jpg'))
        del b, a, plan, base
        gc.collect()
    base = np.load(master, mmap_mode='r')
    hdr = np.load(hdrfile, mmap_mode='r') if mode == 'hdr' else None
    records = folder / (mode + '-records.jsonl')
    completed = {}
    if records.exists():
        for line in records.read_text().splitlines():
            r = json.loads(line)
            completed[r['codec'], r['q'], r['requested_sampling']] = True
    configs = [('pillow-jpeg', q, s) for q in range(95, 101) for s in ('420', '422', '444')] if mode == 'sdr' else [('coreimage-jpeg', q, 'native') for q in range(95, 101)]
    configs += [('coreimage-heic', q, 'native') for q in range(95, 101)]
    if encoders == 'tunable':
        configs = [('tunable-jpeg',q,s) for q in range(95,101) for s in ('420','422','444')]
        configs += [('x265-heic',q,s) for q in (80,85,90,95) for s in ('420','422','444')]
    elif encoders == 'auto':
        if mode != 'hdr':
            raise ValueError('--encoders auto currently measures the HDR pair; use CLI for SDR auto')
        for container in ('jpeg','heic'):
            path=folder/('auto-tunable.'+('jpg' if container=='jpeg' else 'heic'))
            start=time.perf_counter()
            try:
                info=gainmap.write_apple_gainmap_file(base,hdr,path,3.,
                    delivery=resolve_delivery_profile('auto',container=container))
                info.update(sample=p.stem,seconds=time.perf_counter()-start)
            except Exception as exc:
                info={'sample':p.stem,'container':container,'error':str(exc)}
            (folder/('auto-tunable-'+container+'.json')).write_text(json.dumps(info,indent=2))
            print(json.dumps(info),flush=True)
        return
    for codec, q, sampling in configs:
        if (codec, q, sampling) in completed:
            continue
        dest = folder / ('candidate' + ('.heic' if codec.endswith('heic') else '.jpg'))
        start = time.perf_counter()
        r = {'sample': p.stem, 'mode': mode, 'codec': codec, 'q': q, 'requested_sampling': sampling}
        try:
            if codec == 'pillow-jpeg':
                save_jpeg_array(base, dest, q, 'p3', {'420': 2, '422': 1, '444': 0}[sampling])
            elif codec in ('tunable-jpeg','x265-heic'):
                if mode == 'hdr':
                    # Hold auxiliary quality constant even at q100: this
                    # matrix isolates primary quality and chroma changes.
                    gainmap.write_apple_gainmap_file(base,hdr,dest,3.,
                        delivery=resolve_delivery_profile('share',quality=q,chroma=sampling,
                            container='heic' if codec.endswith('heic') else 'jpeg'))
                elif codec=='x265-heic':
                    from dngscan.heif_encoder import encode
                    encode(base,dest,q,sampling,bit_depth=10)
                else:
                    save_jpeg_array(base,dest,q,'p3',{'420':2,'422':1,'444':0}[sampling])
            else:
                encode_ci(base, dest, q, 'heic' if codec.endswith('heic') else 'jpeg', hdr=hdr, factor=None if q == 100 else 2)
            r['encode_seconds'] = time.perf_counter() - start
            r['bytes'] = dest.stat().st_size
            props = gainmap.inspect_gainmap_file(dest)
            if codec == 'pillow-jpeg':
                with Image.open(dest) as image:
                    props['chroma_subsampling'] = {0: '4:4:4', 1: '4:2:2', 2: '4:2:0'}[JpegImagePlugin.get_sampling(image)]
            r['container'] = props
            decoded = gainmap.read_primary_rgb_u8(dest)
            r.update(coding_metrics(decoded, base))
            r.update(detail_metrics(decoded, base))
            if mode == 'hdr':
                hr = gainmap._roundtrip_error(dest, hdr)
                r['hdr_metrics'] = hr
                profile = resolve_delivery_profile('archive' if q == 100 and encoders=='system' else 'share', container='heic' if codec.endswith('heic') else 'jpeg')
                r['hdr_gates_pass'] = gainmap._hdr_roundtrip_is_acceptable(hr, profile.tolerances)
            if mode == 'sdr' and (codec == 'pillow-jpeg' and q in (95, 97, 99) and (sampling in ('420', '422', '444')) or (codec.endswith('heic') and q in (95, 97, 99))):
                dest.replace(folder / f'{mode}-{codec}-{q}-{sampling}{dest.suffix}')
            else:
                dest.unlink(missing_ok=True)
            del decoded
        except Exception as e:
            r['error'] = str(e)
            dest.unlink(missing_ok=True)
        r['total_seconds'] = time.perf_counter() - start
        with records.open('a') as f:
            f.write(json.dumps(r) + '\n')
        print(json.dumps(r), flush=True)
        gc.collect()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['sdr', 'hdr'], required=True)
    parser.add_argument('--encoders',choices=['system','tunable','auto'],default='system',
                        help='Historical system baseline, independent primary matrix, or production HDR auto')
    parser.add_argument('--out', type=Path, required=True, help='Private benchmark output directory; contains RAW-derived images')
    parser.add_argument('raws', nargs='+', type=Path)
    args = parser.parse_args()
    ROOT = args.out.expanduser().resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    for raw_path in args.raws:
        one(raw_path.expanduser().resolve(), args.mode, args.encoders)
