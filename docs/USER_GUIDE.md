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

---

## 2. Basic workflow

1. **Pick a file** — the RAW file selector at the top;
2. **Read the Detected Parameters card** — the tool has already analyzed the photo;
   look here before touching anything (next section explains each number);
3. **Adjust exposure and tone** — if needed;
4. **Choose the output** — ordinary JPEG or HDR, for sharing or for archiving
   (section 9);
5. **Update the preview to confirm, then export.**

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
- **Two modes** (CLI `--film-mode`): the default **observe** is everything above —
  use it day to day. **full** is experimental: the film development model takes
  over wholesale — scene colour passes through a constrained observer inverse
  into three emulsion exposures, through each layer's characteristic curve, the
  the characteristic curves into negative dye density, then through the
  FACTORIZED print chain (negative density -> paper-layer exposure -> print
  timing -> paper development -> viewing colour). Under the honest spectral
  chain it corroborates observe directionally; AgX compression core only.
  Declarable full-mode state: `--film-exposure` (emulsion state, ±2 EV),
  `--film-print-medium` (print medium; cross-medium pairings re-print the
  same negative without double tone mapping), `--film-print-timing
  fixed|retimed|custom` (custom unlocks the modelled colour-head Δτ and
  `--film-print-exposure`), `--film-neutralization
  technical-neutral|print-balanced|native`
  (GUI: 灰阶中性化, defaulting to "follow the film interpretation" —
  technical resolves to digital neutral, reference print to print-balanced;
  `bounded`/`datasheet` and the old `--film-crossover` are deprecated
  aliases — technical-neutral is the digitally neutralized variant with
  grays within two stops of neutral held strictly neutral, print-balanced
  solves one constant balance at the EV0 anchor so mid-grey is neutral by
  construction while both ends of the grey scale keep the medium's own
  exposure-dependent crossover, native serves the chain verbatim
  with shadows tinting per each stock's data: cine negatives green-teal,
  Kodachrome amber, Velvia mildly cool, mid-grey anchored by the print
  solve), `--film-development editorial_custom` developer recipes (bounded
  contrast/fog/colour-density perturbations, honestly labelled in the
  report), `--film-compression` (C1 highlight compression plus highlight
  colour density), and the analog optics `--film-grain` / `--film-halation`
  / `--film-bloom` (GUI: 模拟光学 tiers; grain = measured sigma(D) tables
  on a fixed-statistics master field with a per-photo random spatial
  arrangement — the negative and the print paper each take an independent
  phase, so the two grain realizations are uncorrelated; bloom = the
  EDITORIAL capture glow (additive, pre-emulsion,
  declared as editorial — the old conservative medium scatter was retired);
  preview
  and full-size export share one realization, --film-optics-seed auto|N
  controls randomness/reproduction and the report prints the effective seed;
  `--film-media-scatter declared|off` declares the media's OWN scatter
  separately — the emulsion scatter (§5.1) and the print-formation scatter
  (§6.2) are properties of the declared media, not look amounts: "declared"
  (default) applies the compiled profile whenever the optics chain is
  engaged, "off" disables both explicitly (also the operator-isolation
  setting the measurement tooling uses) — no longer an anonymous side
  effect of the look sliders);
- No grain, no vignette — it changes *how the camera saw the world*;
- **HDR keeps working**: observe mode as always; full-mode Ultra HDR is
  "film print + scene HDR extension" — the SDR base IS the film print
  (byte-identical), and reliable scene highlights gain smoothly above the
  print's reference white. No claim of physical film HDR is made.

**Lens filters** (RAW decode card) are the companion control: Wratten conversion
glass simulated from Kodak's published parameters (85B daylight-to-tungsten, 80A the
reverse, and others), for recreating historical workflows like "tungsten film + 85B in
daylight". There is no strength slider — glass has no half-installed state.

## 6. The other EVs in the interface

- **Exposure EV** (the slider): brightens/darkens everything, +1 = one stop up. 0 keeps
  the brightness relationships from capture.
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
- **Inter-image β** (film card, appears under "custom β"): the development-coupling
  colour-difference amplification. Default "declared" = the stock's modelled table
  value (0.32–1.05); a custom value is reported as an editorial dial. CLI:
  `--film-interimage custom --film-interimage-beta`.

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
- The measured line at the bottom of the tone card reports the **compiled actual
  values** (toe-end EV, shoulder-white EV, endpoint provenance). Out-of-range
  requests are clamped by the curve legality guards; the line always shows what
  actually took effect.

---

## 8. The compression cores, and which to pick

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

## 9. Output: formats and delivery profiles

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
verification; files that fail it are never kept.

---

## 10. FAQ

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
