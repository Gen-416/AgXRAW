# AgXRAW desktop workspace

The desktop entry uses React + TypeScript + Vite and Tauri 2. It calls the
same Python application services as the existing browser GUI. It does not
implement exposure, calibration qualification, rendering or encoding rules.

This is a **development entry**, not a self-contained installer. The Python
engine still needs an environment with the project's locked dependencies and
GUI extra. Bundling that engine, signing and release updates remain separate
work. The existing browser workbench and batch CLI remain available.

From this directory:

```sh
npm ci
# Optional when the repository's ../.venv is not the engine environment:
cp .env.example .env.local
# Set AGXRAW_PYTHON in .env.local to an absolute interpreter path.
npm run desktop
```

To open a source from the launcher: `npm run desktop -- -- --open /absolute/photo.dng`.

`npm run dev` runs the same frontend in a browser with a local engine and hot
reload. `npm run dev:ui` is the UI-only server used by Tauri. Stop the dev
launcher with Ctrl+C; it closes the engine's ownership pipe.

```sh
npm test
npm run build
npm run tauri -- build --debug --no-bundle
# With npm run dev already running, optionally test a private RAW:
AGXRAW_TEST_RAW=/absolute/path/to/photo.dng npx playwright test
```

No RAWs are included. End-to-end exports and screenshots go to a temporary
directory. The production preview remains SDR JPEG; choosing HDR affects
export, not the preview's display technology.

The initial workspace covers open/prepare, automatic or manual exposure,
basic tone controls, decoder/WB selection, and verified full-size SDR/HDR
delivery. The **original full workbench** button retains access to calibration
management, histograms and other advanced controls during gradual migration.

`src/api/generated.ts` is generated from `dngscan.application.contracts`:

```sh
python ../tools/generate_application_types.py
python ../tools/generate_application_types.py --check
```

After building, `python ../tools/validate_desktop_entry.py /absolute/photo.dng
--output /tmp/native-entry.json` opens a temporary real Tauri window, waits for
its prepare/render commands, verifies the shared frame cache and source hash,
then closes its own process and checks backend shutdown. It is a developer
smoke test, not a display-colour or HDR visual qualification.

The Python module supplies fresh defaults. The frontend preserves automatic
EV as a result rather than rewriting user intent, and freezes the entire
request for export. Legacy imports alias the canonical application modules,
so caches, scheduling and monkeypatches do not create duplicate owners.

See [the migration and verification guide](../docs/DESKTOP_DEVELOPMENT.zh-CN.md).
