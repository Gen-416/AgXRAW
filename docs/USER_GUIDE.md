# dngscan User Guide

This is the manual without the math. It answers three questions: which cameras this tool
can handle, what every number on screen means, and what to pick when exporting. For the
pipeline internals and technical detail, see the
[architecture notes](ARCHITECTURE.md).
[中文版使用说明在这里](USER_GUIDE.zh-CN.md).

dngscan turns RAW photos into faithful JPEG or HEIF files for saving and sharing,
with SDR and HDR output. HDR highlights light up on compatible screens and viewers.

---

## 1. Supported cameras

**The default decoder (LibRaw)** covers the vast majority of RAW formats on the market,
including but not limited to:

| Brand | Formats |
|---|---|
| Canon | CR2 / CR3 |
| Nikon | NEF / NRW |
| Sony | ARW |
| Fujifilm | RAF (including X-Trans sensors) |
| Panasonic / OM System / Leica / Pentax | RW2 / ORF / DNG / PEF and others |
| Sigma, iPhone (ProRAW), and anything that writes DNG | DNG |

No per-camera plugin or personal noise calibration is required. Local acceptance
covers Sigma fp DNG, Sony ARW, Fujifilm X-Trans RAF and iPhone 16 Pro single-frame
Bayer DNG. These iPhone samples are not Linear RGB ProRAW; see the
[support and fallback scope](SENSOR_SUPPORT.zh-CN.md) for the actual boundaries.

**Brand-new bodies work too**: cameras so recent that built-in data tables haven't
caught up (the A7 V / GR IV / X-E5 generation) are not refused — the tool degrades
gracefully and the report states plainly that calibration data is incomplete and
results may deviate, while the photo exports normally. Several new bodies
(A7 V / A7S III / A7R VI / GR IV / Nikon Zf / X100VI / X-E5) already ship with
measured sensor data, plus colour matrices where published coefficients exist —
the A7R VI's matrix is deliberately absent rather than guessed (the declared
degradation path covers it), and the GR IV's DNG carries its own calibration.

**One known exception: Nikon High Efficiency (HE/HE\*) NEFs.** If a Z9/Z8/Z6III/
Z50II-generation body shot with "High Efficiency" compression, that format cannot be
decoded by LibRaw for codec-licensing reasons (true of all open-source tools); the
error message gives specific guidance. Fixes: convert to DNG with the free Adobe DNG
Converter (full functionality after conversion), or switch the camera to "Lossless
Compression" RAW. Lossless NEFs from the same bodies work normally.

**The Apple RAW decoder (optional — set "解码器 / Decoder" to Apple RAW)** uses the RAW
engine built into macOS. Its coverage follows Apple's official camera RAW compatibility
list — mainstream Canon, Nikon, Sony, Fujifilm, Panasonic and Sigma bodies plus iPhone
are on it, but **the newest decode model (RAW 9) is only open to the more recent bodies
among them**. You never need to look this up yourself: after you pick a file the tool
probes what that exact file supports, and if RAW 9 is unavailable it tells you plainly
and lets you choose an older version or fall back to LibRaw — it never switches
algorithms silently.

The two decoders have slightly different color personalities (Apple's denoising and
highlight reconstruction feel more "camera-manufacturer"); both produce SDR and HDR
normally. When in doubt, use the default LibRaw.

This setting changes scene reconstruction only. Sensor Evidence is always acquired by
the same LibRaw provider, so switching between LibRaw and Apple RAW cannot change the
analysis inputs. A file that LibRaw cannot open therefore cannot bypass the Evidence
requirement by selecting Apple RAW.

Two more decode-level settings sit on the RAW decode card and rarely need touching;
they differ in kind. **Apple RAW 亮度基准 — CI scale** (CLI `--coreimage-scale`, shown only when
the decoder is Apple RAW) decides at what scale Core Image's scene-linear values enter
the rest of the pipeline: **与 LibRaw 对齐 · 默认** (aligned) matches the LibRaw decode
file by file, **Apple 原始数值** (unity) keeps Core Image's native units, and **旧版固定系数**
(measured) is the old fixed ratio, kept only for reproducing earlier results; changing
it re-runs preparation (the decode). **过曝判定余量 DN — clip margin** (CLI `--margin`,
integer 0–64, default 4) is how many DN below full well each channel's clipping
threshold sits. It does not change the reconstruction; it changes the analysis
criterion — the RAW clipping share, the hard-clip numbers next to the RAW 过曝标记 toggle
and the whole Detected Parameters card are recomputed against it, so changing it
re-decodes and re-analyses this RAW (a few seconds) in the current implementation. Neither needs attention day to day; exported filenames carry
`ciscale-unity` / `ciscale-measured` or `margin{n}` only when the value is not the
default. The **色度降噪 — chroma NR** slider on the same row (CLI `--chroma-nr`, 0–1,
default 0 = strict identity) uses an independent noise model to smooth chroma at
roughly 8–128 sensor pixels (octave-aligned, with boundaries accurate to about √2).
It preserves luminance at this scene-linear stage, but can still attenuate real colour
structure. Missing calibration, Apple RAW or unsupported noise-transfer operations
skip this optional stage with a reason. Where applicable, SDR and both legs of an HDR
pair use the same corrected scene. A nonzero value names the file `cnr{x}`. See the
[current algorithm and limits](CHROMA_NR.zh-CN.md).

Use **个人标定 · 可选** on the RAW decode card to import JPTC Collect directories or
calibration JSON, then enable, disable or remove local profiles. Per-file matching
checks declared shutter, ISO, DN scale, RAW geometry and storage constraints, and
distinguishes unknown conditions from mismatches. Stored bit depth is not ADC precision.
The GUI shows noise-model availability separately from optional chroma-NR capability:
Apple, non-Bayer or unsupported correction transfers can disable the slider even with
a valid file noise model. Its value is preserved, and preview/export report actual
application separately. Ordinary missing calibration is a normal fallback, while
rejected or unresolved evidence remains visible. Auto exposure, AgX and export continue;
HDR latitude depends on the current photo's reliable highlights. See the
[calibration instructions](NOISE_CALIBRATION.zh-CN.md).

For 10-bit HEIF, the output dialog and delivery report identify the floating SDR
master (including the SDR base of HDR HEIF), final encoding depth and float readback.
Apple HDR allows either 8-bit or 10-bit. Explicit Apple SDR requires manual, 8-bit,
4:2:0 settings; incompatible combinations are explained before export. The preview
and its display histogram remain 8-bit SDR and do not establish file precision.

---

## 2. Basic workflow

1. **Pick a file** — the RAW file selector at the top;
2. **Read the Detected Parameters card** — the tool has already analyzed the photo;
   look here before touching anything (next section explains each number);
3. **Adjust exposure and tone** — if needed. When you cannot tell whether a
   highlight wants less exposure or a different shoulder, switch on the
   **RAW 过曝标记 (RAW clipping)** toggle on the preview card first (end of
   section 3);
4. **Choose the output** — ordinary JPEG or HDR, for sharing or for archiving
   (section 9);
5. **Update the preview to confirm, then export.**

There is no "analysis report" in the GUI: it produces images only, and the
Delivery Report shown after an export lists just the measured facts of the HDR
container (section 9). For the full analysis report — evidence, curve
endpoints, colour-matrix health κ, Stage A residuals and so on — run the CLI
with `--report`. Without it the CLI prints only the files it wrote
(`JPEG 图像: …` / `PNG 图像: …`); diagnostic runs with `--scan` or `--csv`
include the report automatically.

Apart from those report and diagnostic outputs (`--report`, `--csv`, `--support`,
`--hdr-debug-dir` and the like) the GUI now covers every CLI dial: anything that
shapes the image on the CLI has a control on the page, and the ones hidden or greyed
are simply those that do not apply in the current state (section 10). The six-panel
dashboard can ride along via the export dialog's 附带分析图 checkbox (section 9).

---

## 3. What the Detected Parameters mean

This card shows the tool's measurements of your photo — the same analysis the final
render will use.

**First, EV**: EV means "stops", photography's exposure unit. +1 EV = twice as bright,
−1 EV = half. Every EV in this tool is anchored at **middle gray** (the 18% gray of a
correct exposure — roughly the midtone brightness of a properly exposed face): +3 EV is
three stops brighter than middle gray, −5 EV five stops darker.

| Field | Meaning | Why it matters |
|---|---|---|
| **RAW clipping** | Share of sensor pixels that blew out ("dead white") | High values mean burned highlights; the colors there are guesses, and the tool automatically trusts them less |
| **Reliable highlight tail** | The brightest content that was **genuinely measured without clipping** (clipped pixels excluded — they don't count) | This is the sole budget for HDR brightness: if a neon sign measured +6 EV, HDR can honestly light it to +6 |
| **Earned HDR headroom** | The part of the reliable tail above "paper white" (a normal white on screen) | Exactly how many stops brighter than an ordinary photo this HDR can go; 0 means there is nothing worth HDR in this shot |
| **Subject median** | Roughly which stop the scene's body sits at | Strongly negative = a dark scene (night); near 0 = standard exposure |
| **Scene type** | Normal / sparse emitters | "Sparse emitters" = mostly-dark frames with small very-bright sources (night lights, stages); the tool automatically switches to a highlight policy suited to them |
| **Compiled curve** | The dark-to-bright range and contrast this render actually adopted | Reference values, so you know what the tool decided for this image |

**A typical read of this card**: earned headroom +1.5 EV → worth exporting HDR; RAW
clipping 8% → the sky may be burned; clipped regions are automatically trusted less; subject
median −3 EV → this is a night scene, don't force the exposure up.

### The RAW clipping overlay: where it clipped, and which channel

The "RAW clipping" percentage says how many pixels blew out, not where or in
which channel. The **RAW 过曝标记** toggle at the top right of the preview card adds
that layer: switched on, it paints the pixels at or above ~97% of full well
(the region where the render's chroma retreat engages — near full well, not
necessarily clipped: the soft mask is feathered and resized, so a marked pixel
may still hold valid data) onto the preview, per CFA channel — **red / green /
blue = that channel, white = all three**. Next to the toggle it reports two numbers:
**hard clip R · G · B** (full-resolution, ≥ full well − margin, the same
criterion as the detected-parameters card — this is the authoritative "how
much is over-exposed") and the **marked share** (≥97% of full well, the area
the layer covers); "没有接近上限的像素" when there are none.

What it shows is **decode evidence, not the rendered result**: the data is the
same evidence mask the render uses for clip retreat and the HDR chroma gate —
derived from the decode, independent of white balance, fetched once per
prepared preview and composited over every live frame on the client. That is
why it **does not move with the exposure slider**: pull EV down and the image
darkens while the marks stay exactly where they were, because those sensor
pixels carry no information any more.

That is precisely its use: **telling "the RAW already burned" apart from "it is
merely rendered too bright"**. When a highlight looks harsh, switch the overlay
on first. Sparse or absent marks mean the RAW is fine and the tone curve is the
issue — go to the tone card and work the shoulder (shoulder white, highlight
transition, section 6) or ease EV down; the gradation comes back. A large white
patch (all three channels) was already flat in the RAW: lowering EV only turns
it into flat grey, and no shoulder setting can invent gradation — either accept
it as dead white or let AgX (section 8) fade it naturally toward white. Marks in
only one or two channels mean the colour there is a guess, which the tool
already trusts less; when a local colour looks wrong, check this layer before
suspecting other settings.

Under the Apple RAW (Core Image) decoder the toggle is greyed with the reason
beside it: Core Image decoding has no per-pixel CFA evidence, so the clipping
display is unavailable.

---

## 4. White balance: As Shot vs fixed Kelvin

Besides "As Shot", the white balance selector offers a set of **fixed color
temperatures**. These are not for balancing by eye — eyeballing neutrality on a screen
never beats the camera's metering (your eyes chromatically adapt while you look). They
are **declared standard references**: the values come from industry calibrations, and
the multipliers are solved precisely from the photo file's own color calibration, with
no eye in the loop.

| Option | What it is | When |
|---|---|---|
| **As Shot** | The balance the camera metered at capture | Default; everyday output |
| **6500K · D65** | The standard white point of sRGB/Rec.709 displays | Aligning with display-industry standards |
| **5500K · photographic daylight** | Fixed daylight reference | Keep the same reference across a series; tungsten lighting remains warm |
| **3400K · Type A** | Fixed 3400 K reference | Warm studio lighting |
| **3200K · Type B** | Fixed 3200 K reference | Studio tungsten/halogen |
| **9300K · Japanese broadcast white** | The traditional white point of Japanese television (cool blue) | The "old Japanese TV" cool look — for fun |

Fixed Kelvin works on both decoders. A visible color cast after choosing one is
**expected behavior**: this fixes the reference rather than neutralizing every illuminant.

## 5. The other EVs in the interface

- **Exposure EV** (the slider): brightens/darkens everything, +1 = one stop up. 0 keeps
  the brightness relationships from capture. The RAW 过曝标记 marks on the preview do
  not move with it — they show decode evidence (section 3).
- **Brightness reference** (button): the tool measures the subject and aligns it to a
  standard exposure while limiting highlight overflow. "Expose this for me, once" — the
  result is written back to the slider and can be adjusted further.
- **HDR headroom ceiling** (output card): how many stops above paper white your screen
  or use case allows at most. It is a **ceiling, not a target** — actual usage is
  decided by the earned headroom, whichever is smaller. The default +3.0 (roughly an
  800-nit screen) rarely needs changing.
- **HDR latitude dials** (collapsed section on the output card; collapsed = auto):
  three subjective quantities — **chroma freedom ρ** (how much per-channel highlight
  colour at full evidence confidence, default 0.5), **white margin** (stops above the
  reliable tail the white endpoint sits, default 0.30 normal / 0.50 sparse emitters),
  and **shoulder start** (where the HDR shoulder leaves the body, default 0.20 / 0.00).
  The defaults are the mathematical policy; measurement cannot decide these three, so
  they are dials. The evidence gates (clip / noise / gamut pressure) always multiply
  and cannot be bypassed. CLI: `--hdr-rho` / `--hdr-white-margin` /
  `--hdr-shoulder-start`.
## 6. Tone card: endpoint mode and toe/shoulder offsets

- **Endpoint mode**: where the curve's black/white endpoints come from.
  **Scene-adaptive** (default) follows this frame's luminance percentiles — right for
  most photos, but large deep-shadow areas (a backlit bridge underside, a dark alley)
  can be declared "black" by the percentiles. **Evidence** pins the endpoints to
  independent evidence instead: black uses the matched noise model's read-noise floor,
  and white trusts only the reliable RAW tail. Reconstructed highlights do not count;
  missing evidence falls back with a note. Local texture variation is never substituted
  for a measured read-noise floor. The exposure anchor does not
  move — 0 EV still maps to 18% gray — so overall brightness stays put.
- **Toe end** (EV slider): the scene EV at which the curve lands at near-black.
  Dragging left pushes that point deeper — deeper shadows stay readable and dive to
  black later, implemented by re-solving the toe shape; **the black point, white
  point and sky highlights do not move**. Dragging right closes the shadows earlier.
- **Shoulder white** (EV slider): the scene EV at which the curve reaches near-white
  (90% of the black-floor-to-white span). Dragging right delays that point — highlight
  gradations merge later and the roll-off softens; dragging left closes white earlier
  and hardens the shoulder. Implemented by re-solving the shoulder curvature; **the
  black point, white point and shoulder start do not move**.
- **Shadow transition / highlight transition** (暗部过渡 / 高光过渡, −1…+1
  sliders): the **fine trims** for the toe and shoulder. Unlike toe end and
  shoulder white they do not name an EV coordinate; they multiply the curvature
  of the toe/shoulder segment by a bounded factor (about ±37% at full travel)
  while the black point, white point and mid-grey anchor all stay put. **They
  are restrained by design**: a full-travel move changes the finished image by
  only a few code values at peak (measured roughly 3–9/255, concentrated in one
  brightness window of the deep shadows or the highlight roll-off), and the
  whole-image per-pixel p99 rarely exceeds 10/255 — barely seeing it in the
  preview is normal behaviour, not a broken slider. Full-travel shadow
  transition is roughly a 0.6–0.7 stop move of toe end; full-travel highlight
  transition is under 0.1 stop of shoulder white. For a visible move, use toe
  end / shoulder white directly. On photos whose shoulder has no room to bend
  (see the editing tutorial's "shoulder white does not move" section),
  highlight transition and shoulder white fail together, with under 1/255 of
  difference at full travel — the same geometric reason.
- **Highlight fade** (高光褪白, colour card, −1…+1 slider): adjusts only the
  chroma of **coloured pixels approaching display white** — right fades them
  toward white earlier, left keeps more colour; luminance is untouched. Neutral
  (colourless) brights change by exactly zero, so a frame with no "bright and
  coloured" content (sunset clouds, coloured lampshades) shows no change at
  all; with such content, full travel moves a single channel by at most about
  15/255. **Disabled under HDR output** (HDR's highlight colour geometry is
  handled independently and the value is forced to 0 at export); shadow
  transition still applies under HDR, while highlight transition takes no part
  in the shape of HDR highlights (the HDR shoulder is solved independently
  above mid-grey).
- The measured line at the bottom of the tone card reports the **compiled actual
  values** (toe-end EV, shoulder-white EV, endpoint provenance). Out-of-range
  requests are clamped by the curve legality guards; the line always shows what
  actually took effect.
- **Not sure whose problem a highlight is**: switch on RAW 过曝标记 on the preview
  card first (section 3). If the RAW did not clip and the render is merely too
  bright, shoulder white / highlight transition and EV all work; where the RAW
  already overflowed in all three channels, no curve setting can invent
  gradation.

## 7. The two live histograms

While previewing, two histograms with different scopes sit next to the controls
they belong to, and both refresh with every preview frame:

- **Scene EV histogram** (exposure card, under the EV slider): the x-axis is
  scene brightness in stops relative to 18% grey (−10..+4 EV), the y-axis a log
  count. It plots **reliable scene brightness** — exactly the same sample the
  curve planner uses: RAW-clipped samples and floor-clamped black samples are
  excluded, so the population you see is the population the render decisions
  saw. Annotations: **black/white** are the two endpoints of the curve actually
  in effect (when the EV slider moves the endpoints stay put and the population
  shifts as a whole, so you can watch it cross an endpoint and get crushed); the
  **0EV** dashed line is the 18% grey reference; **p99.99** is the brightest
  trustworthy signal (the basis of the HDR budget). When reliable evidence is
  insufficient (large clipped areas and the like) the p99.99 line is honestly
  omitted rather than replaced by another number.
- **Display code-value histogram** (preview card, under the preview image):
  RGB three channels plus luma, 0–255 code values, log count, taken from the
  rendered 1920 px preview frame. It answers "what does this frame actually
  output look like" — clipped whites, gaps and crushed blacks are visible at a
  glance. With an HDR output format selected it shows the SDR base image's
  histogram and notes "HDR earned headroom +X.X EV" in the corner (again omitted
  when scene evidence is insufficient).

Both histograms are display-only for now: no hover, no range selection.

The preview card carries one more per-frame layer that is not a histogram: the
RAW 过曝标记 marks (section 3). It complements the display histogram — the
histogram's clipped whites tell you the output hit 255, the overlay tells you
whether the RAW itself had already hit full well. The former can be rescued
with exposure and curve; the latter cannot.

---

## 8. The compression cores, and which to pick

RAW records a far wider brightness range than any screen can show; the "compression
core" is how the former is fitted into the latter. It decides the overall look —
especially how highlights behave in color.

| Option | One line | When |
|---|---|---|
| **AgX · default** | Film-style highlight handling: bright areas fade naturally toward white instead of staying garishly saturated | **Use this when unsure** — 95% of the time |
| **RAW-gated · fidelity** | Brightness comes from the luminance-only curve; the color is chosen per pixel between the untouched color ratios and AgX's path to white according to RAW evidence: colors the sensor genuinely measured are kept, clipped or noisy highlights are handed to AgX | When you want colors more "faithful to the sensor"; LibRaw decoding only |
| **Scene C1 · luminance only** | Compresses brightness only, never touches color ratios | A control group: switch here to see what AgX's color handling is actually doing |
| **Fixed curve · diagnostic** | A fixed curve that ignores the scene | For troubleshooting, not for daily use |

The last two are comparison/diagnostic tools, not finishing tools.

---

## 9. Output: formats and delivery profiles

**Formats**:
- **SDR JPEG** — an ordinary photo for broad sharing;
- **SDR HEIC** — a normal-dynamic-range photo using HEIF/HEVC compression; the recipient needs HEIC support;
- **HDR gain-map · JPEG** — the recommended HDR format. The file carries both a normal
  rendition and the highlight-boost information; on capable screens (iPhone, Mac,
  Android 15+) highlights genuinely light up, everywhere else it gracefully shows the
  normal version — nothing breaks;
- **HDR gain-map · HEIC** — the same two renditions in a HEIC container. Size and
  compression quality depend on the image and encoder settings; see the measurements below.

**Delivery profiles**:

- **Automatic (default)** — JPEG searches q95–q99. SDR prefers 4:2:2 and uses 4:2:0 only when its chroma error stays within budget and it saves at least another 5%; HDR JPEG defaults to 4:2:2. With x265 available, HEIF defaults to 10-bit / 4:4:4 and searches its own q80–q95 scale. The image dimensions, exposure and AgX rendering are unchanged.
- **High-quality sharing · 97 / 4:2:0** (`--delivery-profile share-hq`) — fixed JPEG quality and sampling at the original dimensions, for SDR JPEG and HDR gain-map JPEG. The final file, including metadata and any gain map, is checked against **20,000,000 bytes (20 MB)**. An oversized result is kept with a warning; the app does not silently lower quality or resize it. This profile is unavailable for HEIF; switching to HEIC in the GUI returns it to Automatic.
- **Manual** (`--delivery-profile share`) — initially q95 / 4:2:0, with independent quality and sampling controls.
- **Archive** (`--delivery-profile archive`) — explicit q100 / 4:4:4, with larger files.

Automatic mode takes additional encoding and readback time. The report shows the selected quality, sampling, actual size and savings against the reference candidate. See the [full-resolution JPEG / HEIF study](DELIVERY_QUALITY_STUDY.zh-CN.md) for the measured corpus and its limits.

Tunable HEIF needs `brew install libheif` with an x265 encoder. The GUI exposes bit depth, speed and texture strategy for HEIC; the CLI equivalents include `--heif-encoder x265 --heif-bit-depth 10 --heif-preset slow --heif-tune ssim`. `--jpeg-quality` also supplies HEIF quality, but its numeric scale belongs to HEVC. Without libheif, the automatic backend uses the Apple system path; unsupported sampling combinations are reported explicitly. The product exposes 8/10-bit output, as some 12-bit combinations failed local readback.

**Sharing**: texture, noise, resolution and the HDR gain map affect file size. The 20 MB check is a local reminder, not a promise that every photo stays below it or that Discord, WeChat or QQ will accept the file as an image, avoid recompression or retain HDR. Check the actual size in the export report and the result received at the destination.

**After exporting, read the Delivery Report** — the collapsible panel above the
preview. It states what actually happened: file size, how many stops of HDR were really
used, how small the compression error measured. Automatic encoding and HDR delivery
perform readback checks; failed candidates are not published as the final file. The report
describes the current delivery, not the RAW analysis, which the CLI prints with `--report`
(end of section 2).

**附带分析图 — "attach the dashboard"** (checkbox in the output dialog): also
writes the six-panel diagnostic PNG at export, the equivalent of the CLI's
`--scan`. It needs the optional matplotlib dependency (`pip install
'dngscan[scan]'`); when that is missing the checkbox is greyed with the reason
instead of failing after the full-resolution analysis has already run.

---

## 10. Which options grey themselves out

The GUI's rule is: **an option that needs a particular environment or asset is
greyed with the reason shown beside it when that is missing**, rather than
letting you choose it and failing at export. Currently handled this way:

- **Decoder · Apple RAW** — needs Core Image on macOS (PyObjC Quartz); without
  it the decoder is locked to LibRaw;
- **HDR gain-map · JPEG / HEIC** — the page probes the HDR backend once on load
  (`/hdr-status`, a read-back verification); if it fails the formats are greyed
  with the reason and a selected HDR format snaps back to SDR;
- The **按 RAW 数据自动 · 保留真实颜色** (RAW-gated) compression core is unavailable under
  Apple RAW — it gates the colour path on per-pixel CFA evidence, which Core
  Image does not provide;
- **附带分析图** — needs matplotlib (section 9);
- **RAW 过曝标记** — Apple RAW decoding has no per-pixel CFA evidence (section 3).

Two more cases **warn without greying**: Apple RAW's RAW 9/8/7 version is probed
per file, and an unsupported file is intercepted before submission so you can
choose (section 1); fixed-Kelvin white balance on a file without colour
calibration degrades to As Shot, flagged with ⚠ on the Detected Parameters
card.

Apple RAW brightness reference appears only while Apple RAW is selected.

---

## 11. FAQ

**HDR export fails with "the reliable highlight tail supports no HDR headroom"?**
The photo has no genuinely measured highlight content (its earned headroom is 0). This
is not a malfunction — an HDR of a photo with no bright content would look identical
anyway. Export SDR instead.

**Chose Apple RAW and got told the file only supports RAW 8?**
Your camera/file is outside RAW 9's coverage. Continue with the older version as
prompted, or switch back to LibRaw; both produce normal output.

**Will the preview differ from the export?**
The preview uses a low-resolution proxy for speed; framing and tone match. Judge
sharpness and noise from the full-size export. Numbers labeled "full-resolution truth"
in the export status are the final ones.

**Where can I see the HDR effect?**
Open the exported file in macOS Preview/Finder, iPhone Photos, an Android 15+ gallery,
or Chrome. On ordinary monitors or older systems it is simply a normal JPEG.

**Why are quality and chroma subsampling sometimes locked?**
Automatic chooses encoding parameters; Archive pins q100 / 4:4:4; High-quality sharing pins q97 / 4:2:0. Select Manual to change them yourself. Both SDR and HDR JPEG support independent sampling controls. HEIF controls also depend on the selected encoder, and unsupported system-encoder combinations fail explicitly.
