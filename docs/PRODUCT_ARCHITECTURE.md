# AgXRAW product architecture and domain model

[简体中文](PRODUCT_ARCHITECTURE.zh-CN.md) · [Imaging pipeline and technical details](ARCHITECTURE.md)

> This document answers “which responsibilities, use cases, and domain objects make up the
> product.” It uses four software architecture layers: Presentation, Application, Domain, and
> Infrastructure. The existing [architecture and technical details](ARCHITECTURE.md) answers
> “which processing stages do the pixels pass through.” Capture, Tone, Color geometry, and
> Delivery in that document are imaging-pipeline stages, not the software layers used here.

![AgXRAW four-layer product architecture and domain model](assets/product-architecture.svg)

## How to read the diagram

- The four large horizontal regions are software architecture layers. Upper layers initiate use
  cases; lower layers supply rules or technical capabilities.
- Capture Evidence, Development, and Delivery are peer bounded contexts inside one Domain layer.
- A heavy border marks an aggregate root; a regular border marks a domain object, strategy, or
  policy.
- Contexts exchange explicit domain contracts: evidence enters Development, and finished images
  enter Delivery.
- A Development Plan is not one bag of parameters. It governs separate decision rights for its
  **Tone Plan** and **Color Plan**.

The architecture protects three product boundaries: facts do not choose aesthetics, development
does not choose packaging, and encoding cannot feed back into already formed pixels.

## 01 Presentation layer

The Presentation layer accepts human actions or programmatic requests and presents results,
progress, explanations, and errors. The product currently exposes four presentation capabilities:

- local visual interface;
- command-line batch processing;
- programmatic invocation;
- report and diagnostic presentation.

Presentation may validate input shape and display state, but it does not own the rules for reliable
evidence, legal development, or valid delivery. Different entry points must produce the same domain
result for the same use case.

## 02 Application layer

The Application layer organizes one work session. It determines call order, gathers inputs, keeps
session state, and propagates failures without redefining domain invariants.

| Use case | Orchestration behaviour | Domain result |
|---|---|---|
| Import and prepare a capture | Accept RAW/DNG, select decoding capability, establish the session | A capture ready for analysis |
| Analyze capture evidence | Request sensor and scene measurements, collect diagnostics | Capture Analysis and Spatial Color Permission |
| Set viewing and development intent | Combine automatic results with explicit user choices | Development Intent |
| Compile a development plan | Submit evidence and intent for domain adjudication | A compiled read-only Development Plan |
| Preview and apply bounded adjustments | Apply constrained biases and reform the preview | An explainable, reproducible preview |
| Export SDR / HDR renditions | Form SDR and optional HDR under the selected formation contract, then hand them to Delivery | A Finished Rendition Set |
| Generate explanations and diagnostics | Collect facts, decisions, degradation, and verification | Reports and error explanations |

Application code must not promote interface defaults into domain rules or bypass the Development
Plan by assembling renderer parameters directly.

## 03 Domain layer

The Domain layer is the sole owner of the ubiquitous language and business invariants. Three peer
contexts collaborate while retaining separate facts and decision rights.

### Capture Evidence context

This context answers “what did the sensor record, and which parts remain trustworthy?” It does not
answer “what should the photograph look like?”

| Domain object | Main behaviours | Invariant |
|---|---|---|
| **Capture (aggregate root)** | Holds sensor evidence and the scene frame; preserves decoding, calibration, and geometry provenance; reuses reconstruction when changing the declared white-balance choice | Changing the scene decoder cannot rewrite evidence provenance |
| Capture Analysis | Resolves full well, clipping, noise, and dynamic range; separates the reliable body and tail from reconstructed regions | Reconstructed highlights cannot become sensor evidence retroactively |
| Spatial Color Permission | Compiles available color-path freedom from per-pixel headroom, clipping class, and SNR | No spatial evidence means no fabricated per-pixel mask |

Capture Evidence publishes **evidence** to Development, never an aesthetic conclusion.

### Development context

This context answers “given the facts and declared intent, how may an image be formed legally?” The
user declares intent; the system adjudicates the plan.

| Domain object | Main behaviours | Invariant |
|---|---|---|
| Development Intent | Declares white balance, exposure bias, film observation position, display target, and explicit viewing choices; separates automatic results from user bias | User bias cannot rewrite capture facts |
| **Development Plan (aggregate root)** | Freezes scene metrics, governs Tone and Color decision rights separately, and validates formation, film, and output-mode compatibility at aggregate scope | Consumed as a read-only contract after compilation; downstream code may not reinterpret it |
| Tone Plan | Fixes the exposure anchor and EV0 pivot; compiles black/white endpoints, dynamic range, toe, shoulder, and local contrast; derives an independent HDR shoulder from the reliable tail | Color pressure and output gamut cannot recompile tone; display capacity may only cap the HDR peak and re-solve the shoulder above K, never rewrite the exposure anchor, scene white endpoint, or body below K |
| Color Plan | Plans hue paths, chroma retreat, and primaries geometry; grants color freedom according to CFA reliability; fits SDR gamut or HDR color volume | May form color only under the established luminance authority; cannot recompile the exposure anchor or tone endpoints |
| Display Formation | Composes Tone and Color plans under the selected formation mode; executes AgX, RAW-gated, luminance-only, neutral diagnostic, or film takeover formation | Standard AgX HDR forms directly from the scene, independently of the SDR rendition; film-full keeps the completed film-print SDR as its base and extends only scene-earned highlights above print reference white |

#### Tone Plan and Color Plan

They are separately governed responsibilities inside one Development Plan—not architecture layers
and not filters that overwrite one another in sequence. This separation concerns decision rights;
it does not claim every formation algorithm is mathematically separable.

| | Tone Plan | Color Plan |
|---|---|---|
| Core question | How is scene luminance mapped to display luminance? | Along which path does color enter the target display space? |
| Decisions owned | Exposure coordinate, black/white endpoints, dynamic range, pivot, toe/shoulder, local contrast, and an independent HDR shoulder constrained by display capacity | Hue paths, chroma retreat, primaries geometry, evidence-gated color freedom, SDR/HDR gamut fitting |
| Decisions not owned | Gamut packaging and encoding parameters | Tone endpoints, exposure anchor, tone curve |
| Implementation vocabulary | Tone Compression Plan | Color Geometry Plan |

White balance belongs to Development Intent and scene-frame formation. It affects the scene sample
observed by plan compilation, but the Color Plan may not rewrite it silently. Film and HDR may each
constrain both subplans, so the aggregate Development Plan validates formation mode, film mode, and
SDR/HDR compatibility. Joint formation such as film full may act on Tone and Color together, but it
must still declare the decision rights and invariants on both sides.

The current Render Plan already carries scene, tone, and color; when film is active it also carries
validated film exposure, development, print, and analog-finish plans. The domain contract requires
read-only consumption after compilation; this states the object-behaviour boundary without claiming
that every internal data structure has completed a physical split or deep-immutability migration.

HDR therefore has two explicit formation contracts. Standard AgX HDR forms directly from the scene,
independently of the SDR rendition. Film-full HDR uses the completed film-print SDR as its body and
adds only scene-earned highlight extension above print reference white; it neither redevelops the
body nor claims to model a physical film HDR process.

### Delivery context

This context answers “how do already formed pixels reach the outside world reliably?” Formation
decides pixels; Delivery decides packaging.

| Domain object | Main behaviours | Invariant |
|---|---|---|
| **Finished Rendition Set (aggregate root)** | Holds one SDR rendition, or an SDR base plus HDR alternate; declares display headroom and output gamut | Pixel formation is complete before encoding; Delivery may read only finished renditions |
| Delivery Policy | Selects container, quality, and chroma sampling; performs atomic writing, read-back, and round-trip tolerance checks | Encoding parameters cannot feed back into image formation |

Development publishes **finished images** to Delivery, not scene data for an encoder to reinterpret.

## 04 Infrastructure layer

Infrastructure implements technical capabilities required by the domain:

- sensor-evidence provision;
- scene decoding;
- native compute acceleration;
- image and Gain Map encoding;
- calibration-asset provision;
- local-file and system-image-service access.

Infrastructure implementations may be replaced, but they do not own evidence, development, or
delivery rules. A new decoder joins by satisfying the common scene contract; a new encoder may only
consume finished images and cannot participate in development retroactively.

## Context contracts

| Contract | Producer | Consumer | Required guarantee |
|---|---|---|---|
| Capture Evidence | Capture Evidence context | Development context | Provenance, reliability, spatial applicability, and degradation state remain traceable |
| Development Plan | Development context | Display Formation | Scene metrics, Tone Plan, and Color Plan are consumed as a read-only contract and cannot exceed their decision rights |
| Finished Rendition Set | Development context | Delivery context | Contains one SDR or an SDR/HDR pair; pixel formation is complete and gamut/headroom declarations are explicit |
| Delivery Result | Delivery context | Presentation / external caller | Container, profile, Gain Map, pixel error, and failure state are verifiable |

## Extension test

Use these questions when placing a new capability:

1. Does it only change interaction or presentation? Put it in Presentation.
2. Does it only orchestrate existing rules? Put it in Application.
3. Does it introduce domain language, an invariant, or new decision rights? Put it in the relevant
   bounded context.
4. Does it only provide decoding, compute, encoding, storage, or calibration? Put it in
   Infrastructure.
5. Does it change both tone and color? First declare each side's decision rights and cross-plan
   constraints, then validate and execute them jointly at Development Plan scope.

For pixel execution order, algorithms, SDR/HDR branches, and delivery verification details, see the
[imaging pipeline and technical details](ARCHITECTURE.md).
