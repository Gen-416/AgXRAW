# dngscan User Guide

This is the manual without the math. It answers three questions: which cameras this tool
can handle, what every number on screen means, and what to pick when exporting. For the
pipeline internals and technical detail, see the
[architecture notes](ARCHITECTURE.md).
[中文版使用说明在这里](USER_GUIDE.zh-CN.md).

dngscan does one thing: it turns RAW photos into faithful JPEGs — ordinary JPEGs, or HDR
photos that genuinely light up on capable screens (iPhone / Mac / recent Android photo
apps all display them natively).

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

It works out of the box — no per-camera plugins. Development and calibration happened
primarily on Sigma fp DNGs and iPhone ProRAW, so those are the most thoroughly
validated; other mainstream cameras decode and convert normally, and sample files for
anything misbehaving are welcome.

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
they differ in kind. **CI 尺度 — CI scale** (CLI `--coreimage-scale`, shown only when
the decoder is Apple RAW) decides at what scale Core Image's scene-linear values enter
the rest of the pipeline: **对齐 LibRaw · 默认** (aligned) matches the LibRaw decode
file by file, **原生单位** (unity) keeps Core Image's native units, and **实测补偿**
(measured) is the old fixed ratio, kept only for reproducing earlier results; changing
it re-runs preparation (the decode). **剪切回退 DN — clip margin** (CLI `--margin`,
integer 0–64, default 4) is how many DN below full well each channel's clipping
threshold sits. It does not change the reconstruction; it changes the analysis
criterion — the RAW clipping share, the hard-clip numbers next to the RAW 满阱 toggle
and the whole Detected Parameters card are recomputed against it, so changing it
re-decodes and re-analyses this RAW (a few seconds) in the current implementation. Neither needs attention day to day; exported filenames carry
`ciscale-unity` / `ciscale-measured` or `margin{n}` only when the value is not the
default.

---

## 2. Basic workflow

1. **Pick a file** — the RAW file selector at the top;
2. **Read the Detected Parameters card** — the tool has already analyzed the photo;
   look here before touching anything (next section explains each number);
3. **Adjust exposure and tone** — if needed. When you cannot tell whether a
   highlight wants less exposure or a different shoulder, switch on the
   **RAW 满阱 (RAW clipping)** toggle on the preview card first (end of
   section 3);
4. **Choose the output** — ordinary JPEG or HDR, for sharing or for archiving
   (section 10);
5. **Update the preview to confirm, then export.**

There is no "analysis report" in the GUI: it produces images only, and the
Delivery Report shown after an export lists just the measured facts of the HDR
container (section 10). For the full analysis report — evidence, curve
endpoints, colour-matrix health κ, Stage A residuals and so on — run the CLI
with `--report`. Without it the CLI prints only the files it wrote
(`JPEG 图像: …` / `PNG 图像: …`); diagnostic runs with `--scan` or `--csv`
include the report automatically.

Apart from those report and diagnostic outputs (`--report`, `--csv`, `--support`,
`--hdr-debug-dir` and the like) the GUI now covers every CLI dial: anything that
shapes the image on the CLI has a control on the page, and the ones hidden or greyed
are simply those that do not apply in the current state (section 11). The six-panel
dashboard can ride along via the export dialog's 附带分析图 checkbox (section 10).

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
which channel. The **RAW 满阱** toggle at the top right of the preview card adds
that layer: switched on, it paints the pixels at or above ~97% of full well
(the region where the render's chroma retreat engages — near full well, not
necessarily clipped: the soft mask is feathered and resized, so a marked pixel
may still hold valid data) onto the preview, per CFA channel — **red / green /
blue = that channel, white = all three**. Next to the toggle it reports two numbers:
**hard clip R · G · B** (full-resolution, ≥ full well − margin, the same
criterion as the detected-parameters card — this is the authoritative "how
much is over-exposed") and the **marked share** (≥97% of full well, the area
the layer covers); "无 ≥97% 满阱像素" when there are none.

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
transition, section 7) or ease EV down; the gradation comes back. A large white
patch (all three channels) was already flat in the RAW: lowering EV only turns
it into flat grey, and no shoulder setting can invent gradation — either accept
it as dead white or let AgX (section 9) fade it naturally toward white. Marks in
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
| **5500K · photographic daylight** | The calibration temperature of daylight-balanced film | **The correct starting point for film simulation**: one fixed 5500K for the whole roll, letting the actual light's warmth or coolness pass through — tungsten light *should* look orange, exactly as real film behaves |
| **3400K · Type A** | Type A tungsten film (photoflood lamps) | Tungsten film simulation |
| **3200K · Type B** | Type B tungsten film (3200K studio tungsten/halogen) | Same, the more common tungsten calibration |
| **9300K · Japanese broadcast white** | The traditional white point of Japanese television (cool blue) | The "old Japanese TV" cool look — for fun |

Fixed Kelvin works on both decoders. A visible color cast after choosing one is
**expected behavior** — it is precisely how film sees the world, not a malfunction.

## 5. Film observation positions (20 stocks + 5 theatrical variants)

Up front: **this is not a one-tap filter**. It decomposes "how a roll of film saw
the world" into independent, declared layers. The payoff is that every layer can
be understood and adjusted on its own; the price is two minutes to build the
mental model. These paragraphs are those two minutes.

### The mental model: film decides what was seen; AgX decides how to develop it

Picking a stock sets five controls at once (**all visible, all adjustable —
nothing is baked**):

| Layer | What it does | When to touch it |
|---|---|---|
| **White balance** (RAW decode card) | Locks the stock's calibration temperature: 5500K daylight, 3200K tungsten cine | Orange tungsten scenes are **by design** (real film behaves this way); switch back to As Shot if you don't want it |
| **Spectral prefeed** (prefeed card) | How this stock's layers **separate** colour (its skin/foliage character), from datasheets | Usually leave it; set to none to drop the stock's colour separation entirely |
| **Separation strength** (slider) | Intensity of that separation. The combo sets a per-stock suggestion (Velvia ×1.6) | **Too mild → push up; too strong → pull down.** 1.0 = calibration strength; the suggestion is editorial taste, not measurement |
| **Development curve** (tone card) | The stock + paired medium's tone signature: black floor, latitude, highlight rolloff. Fixed per roll, no scene adaptation | Usually leave it; the tone trims still stack on top |
| **AgX primaries** (imaging card) | The density/saturation "punch": base/punchy/muted. Paired per stock (Velvia→punchy, Kodachrome→muted) | **The bigger style lever** — switch here for more bite or more restraint |

How colour finally *develops* onto your screen (path-to-white, hue behaviour)
always belongs to AgX — the most thoroughly validated part of the pipeline, and
the reason this film simulation stays stable.

### Three steps to start

1. Pick a stock and export — the combo already carries its suggested style;
2. Adjust to taste: **too mild** → raise separation strength or switch primaries
   to punchy; **too strong** → the reverse; **dislike the colour-temperature
   cast** → set WB back to As Shot (keeping only the separation and curve);
3. Want the no-film baseline? Set film to none — pure AgX.

### Common intents

- Rich landscape chrome → Velvia 100 (ships punchy + ×1.6)
- Soft portraits → Portra 400 (switch to muted for softer still)
- Restrained vintage → Kodachrome 64
- The raw high-contrast "2383 print on a monitor" look → the **theatrical** variants
- Flat wide-latitude cine scans → the Vision3 family
- Film tone only, no colour separation → prefeed none, keep the curve preset

### Boundaries worth knowing

- Slides and cinema prints target **dark projection rooms** (~1.5× contrastier);
  delivery to an everyday screen translates that away using the classic
  surround constants — the translation carries the appearance term only. The
  calibration describes THE MEDIUM (black = the paper's or slide's own Dmax);
  viewing-room flare is no longer baked into film curves;
- **Two development roles** (imaging card 显影分工, CLI `--film-mode
  observe|full`): the default **观察 · AgX 显影** (observe — AgX develops) is
  everything above; use it day to day. **接管 · 胶片显影链** (takeover — the
  film development chain) hands development to the film model as well and
  reveals a further row of controls — the next subsection covers them;
- No vignette — it changes *how the camera saw the world*; grain and halation
  exist only in takeover mode's declared 模拟光学 (analog optics) tiers, never
  in observe mode;
- **HDR keeps working**: observe mode as always; takeover-mode Ultra HDR is
  "film print + scene HDR extension" — the SDR base IS the film print
  (byte-identical), and reliable scene highlights gain smoothly above the
  print's reference white. No claim of physical film HDR is made.

### Takeover: the film development chain (controls that appear under 接管)

In observe mode the film only decides what was seen and AgX still develops the
colour; takeover hands development to the film too. Scene colour first passes
through the stock's Stage A into three emulsion exposures (per stock, a
held-out cross-validated, exposure-homogeneous chromaticity-field correction,
or the constrained 3×3 observer where the field did not earn its place — the
report names which one and its residual), through the characteristic curves
into negative dye density, then through the factorized print chain: negative
density → paper-layer exposure → print timing → paper development → viewing
colour. AgX keeps only delivery-side gamut safety, so choosing takeover locks
the compression core to AgX. The illuminant assumption is fixed at D55 —
measurement showed tungsten and high-CRI LED scenes land in the same class as
daylight once white-balanced — so **there is no illuminant tier to choose**.
This chain (film v2), the appearance layer, optics V2 and observe mode have all
landed (status table in [docs/README.md](README.md)); the GUI no longer labels
any of it experimental.

The one-line hints on the GUI controls are deliberately basic; the full meaning
lives here:

- **灰阶中性化 — grey-scale neutralization** (CLI `--film-neutralization`):
  how the grey axis returns to neutral. **跟随胶片解释 · 默认** (follow the film
  interpretation) resolves from the interpretation control — 技术中和 → digital
  neutral, 参考印相 → print-balanced (the extended scan-reference variant's
  recipe declares digital neutral and the compiler follows); leaving the CLI
  flag out hands the same resolution to the compiler. **数字中性** — digital
  neutral (`technical-neutral`): the chain's output is divided, per pixel by
  luminance exposure, by the package's bounded neutral-tint curve, so greys
  within two stops of neutral stay strictly neutral. **印相平衡** —
  print-balanced (`print-balanced`): one constant balance solved at the EV0
  anchor, mid-grey neutral by construction while both ends of the grey scale
  keep the medium's own exposure-dependent crossover. **数据手册漂移** —
  datasheet drift (`native`): the chain verbatim with no correction, mid-grey
  anchored by the print solve and shadows tinting per each stock's datasheet —
  cine negatives green-teal, Kodachrome amber, Velvia mildly cool. In the CLI,
  `bounded`/`datasheet` are deprecated aliases of
  `technical-neutral`/`native`; the old `--film-crossover off|datasheet` is
  deprecated too — `--film-crossover datasheet` equals `--film-neutralization
  native` — and giving both is a hard error.
- **胶片曝光 — film exposure** (±2 EV, CLI `--film-exposure`): the emulsion's
  exposure state relative to its nominal EI — was the roll over- or
  under-exposed — **not the output exposure**. It changes the negative itself
  (colour, contrast, toe and shoulder); the overall brightness of the print is
  decided by the print timing below.
- **印相 timing — print timing** (CLI `--film-print-timing`): **固定 · 默认**
  (fixed) keeps the print time jointly solved at EV0, the same enlarger
  settings even when film exposure changes; **随胶片曝光重定时** (retimed with
  film exposure) re-prints darkroom-style as the film exposure moves
  (interpolated from a 0.25 EV table), overall brightness nearly constant while
  colour/contrast/toe-shoulder follow the emulsion state — available for **all
  21 negatives** (theatrical variants included) as long as the stock's retimed
  print asset is present; **自定义 · 色头+印相曝光** (custom — colour head +
  print exposure) is manual printing: on top of the fixed timing it adds a
  print-exposure slider (±2 EV, CLI `--film-print-exposure`) and the enlarger
  colour head Δτ (resolved inside the paper-layer exposure model, reported as
  modelled). Custom is open to negatives only and requires the neutralization
  to be 数据手册漂移 — the point of manual printing is to keep what came out of
  the printer — and the GUI switches it over automatically (the CLI needs an
  explicit `--film-neutralization native`). Slides have no print step: their
  timing is always fixed and the other two options are greyed with the reason.
- **印相介质 — print medium** (CLI `--film-print-medium`): defaults to the
  stock's factory-paired paper; the dropdown only appears for stocks with a
  second baked medium (at the time of writing Portra 400 also has Supra Endura,
  Vision3 250D also has 2393 print stock). Changing medium re-prints the same
  negative on different paper without double tone mapping.
- **胶片解释 — film interpretation** (CLI `--film-appearance`, CLI default
  technical): **技术中和** (technical neutral) is the spectral chain itself;
  **参考印相 · 默认** (reference print) adds the stock's palette on its paired
  paper as an appearance layer, with a strength slider (0 = not applied, 1 = the
  recipe's declared value, up to 3 extrapolated); **自定义** (custom) is
  reference plus three bounded modifiers — richness (−1…1), colour density
  (−1…1, darkens without changing saturation), grey-axis bias (0…2, a zero
  field in the current recipes, reserved). Recipe coverage is still narrow:
  for stocks without one, reference/custom are greyed and the control falls
  back to technical (at the time of writing recipes exist for Portra 400,
  Ektar 100, Velvia 100 and Vision3 250D).
- **解释变体 — interpretation variant** (CLI `--film-appearance-variant`):
  **参考印相 · 默认** is the print reading; **扫描对照 extended** is the
  scan/telecine reference reading — same family direction at 0.6 amplitude,
  grey axis digitally neutral. The dropdown only appears for stocks with an
  extended asset (at the time of writing only Vision3 250D).
- **层间放大 — inter-image amplification** (CLI `--film-interimage`): how
  much development coupling amplifies colour differences. **声明 · 默认**
  (declared) uses the stock's modelled table value (declared range across
  stocks 0.32–1.05); **关 · 光谱基线** (off — spectral baseline) is the pure
  spectral base, a debugging setting; **自定义 β** exposes a [0, 1.5] slider (0
  is equivalent to off) and the report labels it an editorial dial.
- **模拟光学 — analog optics** (CLI `--film-grain/--film-halation/
  --film-bloom`): three tiers — **关闭 · 默认** (off) / **轻** light (grain
  0.25 · halation 0.20 · bloom 0.15) / **标准** standard (grain 0.50 ·
  halation 0.40 · bloom 0.30) — or **自定义** with three 0…1 sliders. What
  they are: **grain** is a band-limited density grain field in the negative's
  millimetre coordinates, its response taken from measured σ(D) (per-channel
  lookup of the 5207 chart, calibrated at a 48 µm aperture), with a fixed
  statistical master field and one random spatial arrangement per photo; the
  negative and the paper take independent phases, so the two realizations are
  uncorrelated. **Halation** is bright scene exposure back-scattered through
  the base onto the red-sensitive layer and re-injected into layer exposure,
  before the characteristic curve. **Bloom** is an additive capture glow
  before the emulsion, declared editorial — not a conservative medium scatter.
  Two small controls sit beside the tier dropdown. **光学种子 — optics seed**
  (CLI `--film-optics-seed auto|N`) decides only the grain's spatial
  arrangement, never its size, spectrum, density response or cross-layer
  covariance: leave it empty for auto, which draws one fixed seed when the RAW
  is loaded so preview and export match (the report prints the effective
  seed); type a non-negative integer for a permanently reproducible
  realization. **介质散射 — media scatter** (CLI `--film-media-scatter
  declared|off`) is the media's own scatter (emulsion scatter and
  print-formation scatter, fitted from MTF), which belongs to the declared
  medium rather than to a look amount: under **按声明 · 默认** (declared) it
  applies from the compiled profile whenever the optics chain is engaged,
  independent of the three sliders; **关闭** (off) is the operator-isolation
  setting the measurement tooling uses, not for everyday work.
- **显影配方 — developer recipe** (CLI `--film-development`): which
  development the characteristic curves are solved for. **实测默认**
  (`measured_default`) is the datasheet development with the three
  perturbations locked at 0; **自定义显影** (`editorial_custom`) reveals three
  bounded sliders, and the report labels the result editorial development:
  **显影对比 — development contrast** (−0.5…0.5, CLI `--film-dev-contrast`)
  scales the characteristic curve's logE axis about the mid-grey anchor — the
  contrast dimension of push/pull processing, mid-grey held by construction;
  **显影灰雾 — development fog** (0…0.3, CLI `--film-dev-fog`) adds uniform
  density to all three layers — real chemical fog brightens the whole print,
  and the tool applies no hidden compensation; **显影色密度 — development
  colour density** (−0.5…0.5, CLI `--film-dev-density`) scales the developed
  dye amount about the mid-grey anchor, again without moving mid-grey. Two
  coupling rules match the CLI and are enforced server-side as well: custom
  development requires the neutralization to be 数据手册漂移 (the bounded
  neutralization's cast curve is solved for the measured development) — the
  GUI switches it automatically with a status message and greys the other
  neutralization options, while the CLI needs an explicit
  `--film-neutralization native`; and custom development is mutually
  exclusive with 随胶片曝光重定时 (the retimed τ table is likewise solved for
  the measured development) — the GUI greys retimed and falls back to fixed;
  fixed and custom timing both work.
- **胶片压缩 — film compression** (0…1, default 0, CLI `--film-compression`):
  a C1 saturating compression of scene luminance EV before the emulsion — above
  a knee it eases hard digital highlights into the negative's latitude,
  declared as an editorial bridge; 0 is the strict identity. Above 0 two more
  sliders appear: **压缩 knee — compression knee** (0…6 EV above mid-grey,
  default 2, CLI `--film-compression-knee`) sets the stop at which compression
  starts; **高光色密度 — highlight colour density** (0…2, default 0, CLI
  `--film-highlight-density`) lets the compressed highlights converge toward
  luminance-preserving neutral — the look of negative highlight dye density
  approaching saturation. The latter is only meaningful with compression > 0:
  at 0 the GUI zeroes it and the CLI and server reject a non-zero value.

The developer recipe, film compression, media scatter and optics seed are
takeover-only: they appear only while 显影分工 is set to 接管, and switching
back to observe resets all of them to their defaults so no stale value rides
along in the payload (the server also rejects non-default values in observe
mode). Exported filenames add a token only when a value is not the default, so
a dialled render never overwrites the default one: `dev-c…f…d…` for custom
development, `comp{c}k{knee}` for film compression (plus `hd{x}` when highlight
colour density > 0), `scatteroff` for media scatter off, and `seed{n}` for an
explicit seed.

### The enlarger colour head (negatives only)

A colour negative has no colour of its own — the negative is an intermediate
record, and the final colour is decided by a **person** under the enlarger:
the Y/M filter settings on the colour head are that decision. Selecting any
**negative** preset (Portra / Gold / Superia / the Vision3 family and so on)
shows two sliders on the imaging card:

- **Real darkroom units**: CC filter density, 0–200 in steps of 5 (the GUI slider covers the 0–40 working band by default; tick "色头量程扩展到 200 CC" to expose the full hardware travel); 30CC = 0.30
  optical density ≈ one stop of print-exposure attenuation for that separation.
  After a change the exposure time is re-solved darkroom-style, so mid-grey
  brightness does not move;
- **Direction follows the darkroom rule**: add the filter of the colour the
  print leans toward — too yellow, add Y (the yellow filter absorbs blue,
  exposes the paper's blue-sensitive layer less, forms less yellow dye); too
  magenta, add M;
- **Practical scale**: real darkroom fine-tuning moves in 2–10CC steps; 30CC and
  above is "this print is badly off" coarse correction, and the range to 200
  only reproduces the physical travel of the hardware dial. From the film
  tutorial's samples: Y +5CC visibly lightens a warm cast and Y +10CC crosses
  neutral; the M axis is stronger at the same scale — M +10CC already pushes
  clearly toward green, so take half-size steps on the magenta–green axis;
- The response models the real printing light path, and brightness is held
  constant automatically (implementation in the
  [architecture notes](ARCHITECTURE.md));
- **Slides (Velvia / Provia / Ektachrome / Kodachrome) grey both sliders to
  zero** with the reason shown — the slide is itself the display medium, so
  physically there is no printing step;
- **In takeover mode switch the print timing to 自定义 first**: fixed/retimed
  prints are decided by the joint solve, so the colour head is greyed and zeroed
  with a hint until then; under custom timing it takes part in the manual print
  as a modelled Δτ. In observe mode any negative can use it at any time;
- Both at 0 is exactly the same as not engaging it (the preset's factory
  neutral print decision).

**Lens filters** (RAW decode card) are the companion control: Wratten conversion
glass simulated from Kodak's published parameters (85B daylight-to-tungsten, 80A the
reverse, and others), for recreating historical workflows like "tungsten film + 85B in
daylight". There is no strength slider — glass has no half-installed state.

## 6. The other EVs in the interface

- **Exposure EV** (the slider): brightens/darkens everything, +1 = one stop up. 0 keeps
  the brightness relationships from capture. The RAW 满阱 marks on the preview do
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
- **Inter-image β** (imaging card; appears when 层间放大 is set to 自定义 β in
  takeover mode): the development-coupling colour-difference amplification.
  Default "declared" = the stock's modelled table value (0.32–1.05); a custom
  value is reported as an editorial dial. CLI:
  `--film-interimage custom --film-interimage-beta`.
- **Compression knee** (imaging card; appears in takeover mode once 胶片压缩 is
  above 0): the stop above mid-grey at which the saturating compression starts,
  0…6 EV, default 2; only scene luminance above the knee is compressed. CLI:
  `--film-compression-knee`.

## 7. Tone card: endpoint mode and toe/shoulder offsets

- **Endpoint mode**: where the curve's black/white endpoints come from.
  **Scene-adaptive** (default) follows this frame's luminance percentiles — right for
  most photos, but large deep-shadow areas (a backlit bridge underside, a dark alley)
  can be declared "black" by the percentiles. **Evidence** pins the endpoints to
  sensor evidence instead: black = the measured noise-floor EV (the sensor prior's
  published read noise when available; a single-frame estimate otherwise, truthfully
  noted), and white trusts only the reliable RAW tail (reconstructed highlights do
  not count; absent evidence falls back with a note). The exposure anchor does not
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
- **Not sure whose problem a highlight is**: switch on RAW 满阱 on the preview
  card first (section 3). If the RAW did not clip and the render is merely too
  bright, shoulder white / highlight transition and EV all work; where the RAW
  already overflowed in all three channels, no curve setting can invent
  gradation.

## 8. The two live histograms

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
RAW 满阱 marks (section 3). It complements the display histogram — the
histogram's clipped whites tell you the output hit 255, the overlay tells you
whether the RAW itself had already hit full well. The former can be rescued
with exposure and curve; the latter cannot.

---

## 9. The compression cores, and which to pick

RAW records a far wider brightness range than any screen can show; the "compression
core" is how the former is fitted into the latter. It decides the overall look —
especially how highlights behave in color.

| Option | One line | When |
|---|---|---|
| **AgX · default** | Film-style highlight handling: bright areas fade naturally toward white instead of staying garishly saturated | **Use this when unsure** — 95% of the time |
| **RAW-gated · fidelity** | Same brightness as AgX, but the color processing strength is decided per pixel by RAW evidence: colors the sensor genuinely measured are preserved harder | When you want colors more "faithful to the sensor"; LibRaw decoding only |
| **Scene C1 · luminance only** | Compresses brightness only, never touches color ratios | A control group: switch here to see what AgX's color handling is actually doing |
| **Fixed curve · diagnostic** | A fixed curve that ignores the scene | For troubleshooting, not for daily use |

The last two are comparison/diagnostic tools, not finishing tools.

---

## 10. Output: formats and delivery profiles

**Formats**:
- **SDR JPEG** — an ordinary photo, viewable everywhere;
- **HDR gain-map · JPEG** — the recommended HDR format. The file carries both a normal
  rendition and the highlight-boost information; on capable screens (iPhone, Mac,
  Android 15+) highlights genuinely light up, everywhere else it gracefully shows the
  normal version — nothing breaks;
- **HDR gain-map · HEIC** — the same content in a HEIC container. **Measured on this
  machine it is neither smaller nor better** — choose it only when a downstream
  requires HEIC.

**Delivery profiles**:
- **Archive · fidelity** — maximum quality (~60 MB per frame), for high-quality
  keeping;
- **Share · streaming** — for publishing (~11–27 MB per frame): visually
  near-identical, HDR information fully preserved, and **even the largest frames stay
  inside WeChat's 25 MB original-image limit**.

**WeChat notes**: in chats you must tick **原图 (original image)** for HDR to survive
to the recipient (iPhone WeChat shows full HDR when viewing the original); **Moments
always recompresses to an ordinary photo** — that is WeChat's behavior and no tool can
bypass it. For deliveries that matter, sending as a "file" is the safest path.

**After exporting, read the Delivery Report** — the collapsible panel above the
preview. It states what actually happened: file size, how many stops of HDR were really
used, how small the compression error measured. Every exported file passed an automatic
verification; files that fail it are never kept. It lists only the measured facts of the
HDR container; it is not the analysis report, which the CLI prints with `--report`
(end of section 2).

**附带分析图 — "attach the dashboard"** (checkbox in the output dialog): also
writes the six-panel diagnostic PNG at export, the equivalent of the CLI's
`--scan`. It needs the optional matplotlib dependency (`pip install
'dngscan[scan]'`); when that is missing the checkbox is greyed with the reason
instead of failing after the full-resolution analysis has already run.

---

## 11. Which options grey themselves out

The GUI's rule is: **an option that needs a particular environment or asset is
greyed with the reason shown beside it when that is missing**, rather than
letting you choose it and failing at export. Currently handled this way:

- **Decoder · Apple RAW** — needs Core Image on macOS (PyObjC Quartz); without
  it the decoder is locked to LibRaw;
- **HDR gain-map · JPEG / HEIC** — the page probes the HDR backend once on load
  (`/hdr-status`, a read-back verification); if it fails the formats are greyed
  with the reason and a selected HDR format snaps back to SDR;
- The **RAW 门控 · 保真** (RAW-gated) compression core is unavailable under
  Apple RAW — it gates the colour path on per-pixel CFA evidence, which Core
  Image does not provide;
- **胶片解释** reference/custom and the **解释变体** scan reference — only open
  for stocks with a recipe/asset (section 5);
- **印相 timing** retimed/custom and the **colour head** — slides are always
  fixed and the colour head is greyed and zeroed; in takeover mode the colour
  head additionally needs custom timing (section 5);
- **自定义显影** (the takeover-mode developer recipe) — locks the grey-scale
  neutralization to 数据手册漂移, greys the other neutralization options with a
  status message, and greys 随胶片曝光重定时 on the print timing, falling back
  to fixed (section 5). Custom timing likewise locks the neutralization to
  数据手册漂移;
- **附带分析图** — needs matplotlib (section 10);
- **RAW 满阱** — Apple RAW decoding has no per-pixel CFA evidence (section 3).

Two more cases **warn without greying**: Apple RAW's RAW 9/8/7 version is probed
per file, and an unsupported file is intercepted before submission so you can
choose (section 1); fixed-Kelvin white balance on a file without colour
calibration degrades to As Shot, flagged with ⚠ on the Detected Parameters
card.

A few more controls are **hidden by state rather than greyed**: CI 尺度 appears only
while the decoder is Apple RAW; the developer recipe, film compression, media scatter
and optics seed appear only in takeover mode and reset on the way back to observe;
compression knee and highlight colour density appear only while film compression is
above 0.

---

## 12. FAQ

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
The archive profile pins maximum quality by definition; in HDR formats the chroma
subsampling is decided by the system encoder from the quality setting — the tool shows
the truth rather than pretending it is adjustable.
