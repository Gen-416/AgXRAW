# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI page information-display and responsive-layout contract.

The docs state the evidence layer's limits; the GUI must state them *at
interaction time*. Three load-bearing wires, each snapped silently once:

1. /prepare failures carry the format-gap guidance (TicoRAW diagnosis, DNG
   Converter / lossless-compression outs) — the page must display j.error
   instead of swallowing it, or the user selects an HE NEF and sees nothing.
2. That guidance is multi-line; the status area needs pre-line whitespace or
   it collapses into an unreadable wall.
3. The per-file two-decoder tier report fires on selection for every decoder,
   not only when Apple RAW is chosen.

The responsive contract additionally freezes the supported phone viewport
matrix, the five mobile control groups, safe-area/touch requirements, and the
portrait/landscape composition. Browser geometry tests remain the release
gate; these assertions stop the contract wires from disappearing before that
gate runs.

These are substring assertions against the served HTML: crude, but they turn
"someone refactored preparePreview and the error path went quiet again" from
a field report into a test failure.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from dngscan.gui.page import PAGE, render_page


MOBILE_VIEWPORT_CONTRACT = (
    (375, 667, "portrait"),
    (393, 852, "portrait"),
    (412, 915, "portrait"),
    (852, 393, "landscape"),
)
MOBILE_PAGE_CONTRACT = (
    ("mobileDecodeTab", "mobileDecodeCard", "decode"),
    ("mobileExposureTab", "mobileExposureCard", "exposure"),
    ("mobileToneTab", "toneAdjustCard", "tone"),
    ("mobileImagingTab", "mobileImagingCard", "imaging"),
    ("mobileColorTab", "colorPanel", "color"),
)
MOBILE_CONTRACT_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "REALTIME_PREVIEW_PLAN.zh-CN.md"
)


class PageInformationDisplayTests(unittest.TestCase):
    def test_prepare_failure_is_surfaced_not_swallowed(self) -> None:
        self.assertIn('setStatus(j.error,"err");renderDetectedParams(null);', PAGE)

    def test_realtime_preview_is_fixed_1920_and_has_no_resolution_control(self) -> None:
        served_page = render_page("").decode("utf-8")
        self.assertIn('id="previewLiveBadge">实时 · 1920px', served_page)
        self.assertNotIn('id="previewBtn"', PAGE)
        self.assertNotIn("更新预览", PAGE)
        self.assertNotIn("previewLongEdge", PAGE)
        self.assertNotIn("previewResolution", PAGE)

    def test_realtime_preview_uses_generation_abort_and_automatic_first_frame(self) -> None:
        self.assertIn("PREVIEW_GENERATION", PAGE)
        self.assertIn("new AbortController()", PAGE)
        self.assertIn("body.generation=generation", PAGE)
        self.assertIn("j.superseded", PAGE)
        prepare = PAGE[PAGE.index("async function preparePreview()") :]
        prepare = prepare[: prepare.index("\n}")]
        self.assertIn("PREVIEW_READY=true", prepare)
        self.assertIn("await requestPreview();", prepare)

    def test_image_controls_schedule_live_preview(self) -> None:
        wiring = PAGE[PAGE.index('$("#ev").oninput=') : PAGE.index("restoreSettings();")]
        for control in ("ev", "gradeStrength", "punch", "sceneTransformStrength"):
            with self.subTest(control=control):
                anchor = f'$("#{control}")'
                start = wiring.index(anchor)
                self.assertIn("scheduleLivePreview()", wiring[start : start + 180])
        for control in (
            "midtoneBrightness",
            "midtoneContrast",
            "shadowTransition",
            "highlightTransition",
            "highlightFade",
        ):
            self.assertIn(f'"{control}"', wiring)
        loop = wiring[wiring.index('"midtoneBrightness"') :]
        self.assertIn('forEach(id=>$("#"+id).oninput=', loop)
        self.assertIn("scheduleLivePreview()", loop)

    def test_demosaic_change_rebuilds_the_cold_proxy(self) -> None:
        wiring = PAGE[PAGE.index('$("#demosaic").addEventListener') :]
        wiring = wiring[: wiring.index("\n")]
        self.assertIn("saveSettings()", wiring)
        self.assertIn("preparePreview()", wiring)

    def test_status_area_renders_multiline_guidance(self) -> None:
        start = PAGE.index("#status{")
        self.assertIn("white-space:pre-line", PAGE[start:PAGE.index("}", start)])

    def test_status_is_an_accessible_overlay_that_cannot_resize_preview(self) -> None:
        status_rule_start = PAGE.index("#status{")
        status_rule = PAGE[status_rule_start:PAGE.index("}", status_rule_start)]
        self.assertIn("position:absolute", status_rule)
        self.assertIn("pointer-events:none", status_rule)

        preview_wrap = PAGE[
            PAGE.index('<div id="previewWrap">'):
            PAGE.index('<canvas id="displayHist"')
        ]
        self.assertIn('id="status" role="status" aria-live="polite"', preview_wrap)
        self.assertIn('s.title=t||""', PAGE)

    def test_tier_report_runs_once_per_file_not_every_prepare(self) -> None:
        selection = PAGE[PAGE.index('$("#filePicker").addEventListener') :]
        selection = selection[: selection.index("async function listOutDir")]
        self.assertIn("fetchDecodeSupport(result.path);", selection)
        prepare = PAGE[PAGE.index("async function preparePreview()"):]
        prepare = prepare[:prepare.index("\n}")]
        self.assertNotIn("fetchDecodeSupport", prepare)
        self.assertIn("RAW9_PROBE_REQUESTS", PAGE)
        self.assertEqual(PAGE.count('postJob("/raw9-support"'), 1)

    def test_layout_targets_desktop_landscape(self) -> None:
        # The desktop shell is a single-viewport dashboard: neither the page nor
        # either column scrolls, and the preview consumes the remaining height.
        self.assertIn("max-width:1900px", PAGE)
        self.assertIn("html,body{width:100%;height:100%;overflow:hidden}", PAGE)
        self.assertIn("height:100dvh", PAGE)
        self.assertIn("grid-template-rows:auto minmax(0,1fr)", PAGE)
        self.assertIn(".previewCard{height:100%;min-height:0", PAGE)
        self.assertIn("object-fit:contain", PAGE)
        self.assertNotIn("max-height:calc(100vh", PAGE)

    def test_file_picker_sits_after_title_and_uses_header_space(self) -> None:
        top_bar = PAGE[
            PAGE.index('<div class="topBar">'):
            PAGE.index("</div>", PAGE.index('<div class="topBar">'))
        ]
        self.assertLess(top_bar.index("<h1>"), top_bar.index('id="filePicker"'))

        top_bar_start = PAGE.index(".topBar{")
        top_bar_rule = PAGE[top_bar_start:PAGE.index("}", top_bar_start)]
        self.assertIn("display:grid", top_bar_rule)
        self.assertIn("grid-template-columns:max-content minmax(280px,1fr)", top_bar_rule)

        start = PAGE.index(".topBar input[type=file]{")
        rule = PAGE[start:PAGE.index("}", start)]
        self.assertIn("width:100%", rule)
        self.assertNotIn("max-width", rule)

    def test_controls_are_partitioned_into_accessible_dashboard_tabs(self) -> None:
        control_panel = PAGE[
            PAGE.index('<div class="controlPanel">'):
            PAGE.index('<div class="card previewCard">')
        ]
        for tab_id, panel_id in (
            ("captureTab", "capturePanel"),
            ("toneTab", "tonePanel"),
            ("colorTab", "colorPanel"),
        ):
            with self.subTest(panel=panel_id):
                self.assertIn(f'id="{tab_id}" role="tab"', control_panel)
                self.assertIn(f'aria-controls="{panel_id}"', control_panel)
                self.assertIn(f'id="{panel_id}" role="tabpanel"', control_panel)
        self.assertIn("function setDashboardPanel(", PAGE)
        self.assertIn('tab.addEventListener("click"', PAGE)
        self.assertIn('tab.addEventListener("keydown"', PAGE)

    def test_mobile_ui_contract_names_the_release_viewports_and_invariants(self) -> None:
        contract = MOBILE_CONTRACT_DOC.read_text(encoding="utf-8")
        for width, height, _orientation in MOBILE_VIEWPORT_CONTRACT:
            with self.subTest(viewport=(width, height)):
                self.assertIn(f"`{width}×{height}`", contract)
        for invariant in (
            "document/body 尺寸等于 viewport",
            "预览和导航可见",
            "活动卡片及其控件不越界",
            "触控目标达标",
            "不建立移动端专用的颜色、预览或导出算法",
        ):
            with self.subTest(invariant=invariant):
                self.assertIn(invariant, contract)

    def test_mobile_breakpoints_cover_the_release_viewport_matrix(self) -> None:
        self.assertIn(
            "@media (max-width:767px), (max-width:900px) and (max-height:500px)",
            PAGE,
        )
        for width, height, orientation in MOBILE_VIEWPORT_CONTRACT:
            with self.subTest(viewport=(width, height)):
                matches = width <= 767 or (width <= 900 and height <= 500)
                self.assertTrue(matches)
                self.assertEqual(orientation, "portrait" if height > width else "landscape")

    def test_mobile_layout_keeps_preview_and_pages_current_control_group(self) -> None:
        self.assertIn("viewport-fit=cover", PAGE)
        self.assertIn("grid-template-rows:minmax(160px,32%) minmax(0,1fr) auto", PAGE)
        self.assertIn('<nav class="mobileNav" role="tablist"', PAGE)
        mobile_nav = PAGE[
            PAGE.index('<nav class="mobileNav"'):
            PAGE.index("</nav>", PAGE.index('<nav class="mobileNav"'))
        ]
        for tab_id, controlled_id, target in MOBILE_PAGE_CONTRACT:
            with self.subTest(tab=tab_id):
                self.assertEqual(mobile_nav.count(f'id="{tab_id}"'), 1)
                self.assertIn(f'aria-controls="{controlled_id}"', mobile_nav)
                self.assertIn(f'data-mobile-target="{target}"', mobile_nav)
        for card_id, target in (
            ("mobileDecodeCard", "decode"),
            ("mobileExposureCard", "exposure"),
            ("toneAdjustCard", "tone"),
            ("mobileImagingCard", "imaging"),
        ):
            self.assertIn(f'id="{card_id}" data-mobile-card="{target}"', PAGE)
        color_panel = PAGE[
            PAGE.index('<section class="dashboardPanel" id="colorPanel"'):
            PAGE.index('</section>', PAGE.index('<section class="dashboardPanel" id="colorPanel"'))
        ]
        self.assertEqual(color_panel.count('data-mobile-card="color"'), 3)
        self.assertIn('data-mobile-target="color"', PAGE)
        self.assertIn("function setMobileCard(", PAGE)
        self.assertIn('tab.setAttribute("aria-selected",active?"true":"false")', PAGE)
        self.assertIn('tab.addEventListener("keydown"', PAGE)
        for key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"):
            self.assertIn(f'"{key}"', PAGE)

    def test_mobile_layout_honours_safe_areas_and_touch_targets(self) -> None:
        mobile = PAGE[
            PAGE.index("@media (max-width:767px),"):
            PAGE.index("@media (max-width:900px) and (max-height:500px)")
        ]
        for edge in ("top", "right", "bottom", "left"):
            self.assertIn(f"env(safe-area-inset-{edge},0px)", mobile)
        for touch_rule in (
            ".topBar input[type=file]{min-width:0;min-height:44px",
            "input[type=text],input[type=number],select{min-height:44px",
            "input[type=range]{height:44px",
            ".modes button,.modes button#evReferenceBtn{min-width:0;min-height:44px",
            "button.go,button.ghost{min-height:44px",
            ".mobileNav button{min-width:0;min-height:48px",
        ):
            with self.subTest(rule=touch_rule):
                self.assertIn(touch_rule, mobile)
        self.assertIn(".dialogPanel::-webkit-scrollbar{display:none}", mobile)

    def test_mobile_landscape_uses_preview_controls_and_vertical_navigation(self) -> None:
        media = PAGE[PAGE.index("@media (max-width:900px) and (max-height:500px)") :]
        media = media[:media.index("</style>")]
        self.assertIn(
            "grid-template-columns:minmax(250px,42%) minmax(0,1fr) 54px",
            media,
        )
        self.assertIn("grid-template-rows:repeat(5,minmax(0,1fr))", media)

    def test_measured_facts_sit_next_to_their_controls(self) -> None:
        # Data-to-function adjacency: each measured fact renders inside the
        # block whose control consumes it, not in a separate overview card the
        # user must scroll back to while dragging a slider.
        for fact_id, anchor in (
            ("hdrSceneFact", 'id="hdrHeadroom"'),   # scene headroom by the HDR slider
            ("evFact", 'id="evReferenceBtn"'),      # body median by the EV controls
            ("wbFact", 'id="lensFilter"'),          # WB degradation closes the WB row
            ("clipFact", 'id="highlight"'),         # clip share by highlight recovery
            ("toneFact", 'id="shoulderWhiteOffset"'),  # compiled curve by tone sliders
        ):
            with self.subTest(fact=fact_id):
                gap = PAGE[PAGE.index(anchor):PAGE.index('id="%s"' % fact_id)]
                self.assertLess(len(gap), 600, f"{fact_id} not adjacent to {anchor}")
        self.assertNotIn("detectedParams", PAGE)  # the overview card is gone

    def test_support_probe_lines_route_to_their_controls(self) -> None:
        # The probe report is not one block: file identity + Evidence tier by
        # the picker, decoder tiers by the decoder select, priors by the
        # analysis-plates toggle. No consolidated support panel remains.
        for fact_id, anchor in (
            ("fileFact", 'id="filePicker"'),
            ("decodeTierFact", 'id="demosaic"'),
            ("priorsFact", 'id="png"'),
        ):
            with self.subTest(fact=fact_id):
                gap = PAGE[PAGE.index(anchor):PAGE.index('id="%s"' % fact_id)]
                self.assertLess(len(gap), 700, f"{fact_id} not adjacent to {anchor}")
        self.assertIn("SUPPORT_ROUTE", PAGE)
        self.assertNotIn("decodeSupport", PAGE)

    def test_output_controls_live_in_a_modal_not_the_left_panel(self) -> None:
        control_panel = PAGE[
            PAGE.index('<div class="controlPanel">'):
            PAGE.index('<div class="card previewCard">')
        ]
        dialog = PAGE[
            PAGE.index('<dialog class="outputDialog"'):
            PAGE.index("</dialog>")
        ]
        self.assertNotIn('id="format"', control_panel)
        self.assertNotIn('id="outdir"', control_panel)
        self.assertIn('id="outputDialogTitle">输出参数', dialog)
        for control_id in ("format", "deliveryProfile", "gamut", "quality", "chroma", "outdir", "png"):
            with self.subTest(control=control_id):
                self.assertIn(f'id="{control_id}"', dialog)
        self.assertIn('id="go">导出</button>', PAGE)
        self.assertIn('dialog.showModal()', PAGE)
        self.assertIn('id="exportConfirm"', dialog)

    def test_hdr_delivery_does_not_collapse_realtime_tone_core_choices(self) -> None:
        tone_select = PAGE[
            PAGE.index('<select id="toneCore"'):
            PAGE.index('</select>', PAGE.index('<select id="toneCore"'))
        ]
        for core in ("agx", "gated", "neutral", "lum"):
            with self.subTest(core=core):
                self.assertIn(f'value="{core}"', tone_select)

        format_ui = PAGE[PAGE.index("function updateFormatUi()") :]
        format_ui = format_ui[: format_ui.index("async function checkHdrBackend()")]
        self.assertNotIn('$("#toneCore").value="agx"', format_ui)
        self.assertNotIn('$("#toneCore").disabled=hdr', format_ui)
        self.assertIn("updateToneCoreExportUi()", format_ui)
        self.assertIn('id="toneCoreExportHint"', PAGE)


class RealtimeHistogramPageTests(unittest.TestCase):
    """Two histograms, each glued to the function it describes, display-only."""

    def test_scene_histogram_sits_in_the_exposure_card_below_ev_fact(self) -> None:
        exposure_card = PAGE[
            PAGE.index('<div class="secTitle">曝光</div>'):
            PAGE.index('<div class="card" id="toneAdjustCard"')
        ]
        self.assertIn('id="sceneHist"', exposure_card)
        self.assertLess(
            exposure_card.index('id="evFact"'), exposure_card.index('id="sceneHist"')
        )
        gap = PAGE[PAGE.index('id="evFact"'):PAGE.index('id="sceneHist"')]
        self.assertLess(len(gap), 200, "scene histogram not adjacent to #evFact")

    def test_display_histogram_sits_in_the_preview_card_below_preview_wrap(self) -> None:
        preview_card = PAGE[
            PAGE.index('<div class="card previewCard">'):
            PAGE.index('<dialog class="outputDialog"')
        ]
        self.assertIn('id="displayHist"', preview_card)
        self.assertLess(
            preview_card.index('id="previewWrap"'), preview_card.index('id="displayHist"')
        )
        gap = PAGE[PAGE.index('id="previewWrap"'):PAGE.index('id="displayHist"')]
        self.assertLess(len(gap), 300, "display histogram not adjacent to #previewWrap")

    def test_histograms_render_from_the_preview_response_only(self) -> None:
        self.assertIn("function renderSceneHistogram(", PAGE)
        self.assertIn("function renderDisplayHistogram(", PAGE)
        handle = PAGE[PAGE.index("function handleJobResult(") :]
        handle = handle[: handle.index("\n}")]
        self.assertIn("renderSceneHistogram(j.scene_histogram)", handle)
        self.assertIn(
            "renderDisplayHistogram(j.display_histogram,j.hdr_earned_ev)", handle
        )

    def test_histograms_are_display_only_no_interaction(self) -> None:
        # First version is deliberately zero-interaction: no hover, no range
        # selection, no listeners of any kind on either canvas.
        for canvas in ("sceneHist", "displayHist"):
            with self.subTest(canvas=canvas):
                self.assertNotIn(f'#{canvas}").addEventListener', PAGE)
                self.assertNotIn(f'#{canvas}").on', PAGE)
                tag_start = PAGE.index(f'id="{canvas}"')
                tag = PAGE[PAGE.rindex("<canvas", 0, tag_start):PAGE.index(">", tag_start)]
                self.assertNotIn("onclick", tag)
                self.assertNotIn("onmouse", tag)

    def test_hdr_note_uses_served_scalar_and_draws_nothing_without_it(self) -> None:
        renderer = PAGE[PAGE.index("function renderDisplayHistogram(") :]
        renderer = renderer[: renderer.index("\n}")]
        self.assertIn("earnedEv!=null", renderer)
        self.assertIn("HDR 已挣余量", renderer)


class FilmModePlacementTests(unittest.TestCase):
    """Batch-6 P1 regression: colorHeadBlock's unclosed divs swallowed
    filmModeRow into the hidden colour-head region, so hiding the head (a
    reversal preset) also hid the develop-split selector, and switching to
    full left stale non-zero filtration to be rejected server-side."""

    def test_film_mode_row_is_not_nested_inside_the_colour_head_block(self) -> None:
        from html.parser import HTMLParser

        class Depths(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.depth = 0
                self.found: dict[str, int] = {}

            def handle_starttag(self, tag, attrs):
                if tag != "div":
                    return
                ids = dict(attrs).get("id")
                if ids in ("colorHeadBlock", "filmModeRow"):
                    self.found[ids] = self.depth
                self.depth += 1

            def handle_endtag(self, tag):
                if tag == "div":
                    self.depth -= 1

        parser = Depths()
        parser.parse_error = None
        parser.feed(PAGE)
        self.assertIn("colorHeadBlock", parser.found)
        self.assertIn("filmModeRow", parser.found)
        # Siblings: identical depth, and the document's divs are balanced.
        self.assertEqual(
            parser.found["colorHeadBlock"], parser.found["filmModeRow"]
        )
        self.assertEqual(parser.depth, 0)

    def test_colour_head_is_always_visible_and_disables_with_a_reason(self) -> None:
        """Discoverability (user report): hiding the block taught users the
        feature was gone. It now stays visible in every state; the dials
        disable with the reason on screen (no preset / reversal / full mode)
        and reset to zero whenever disabled, and switching film mode still
        refreshes the block."""
        head_ui = PAGE[PAGE.index("function updateColorHeadUi(") :]
        head_ui = head_ui[: head_ui.index("\n}")]
        self.assertNotIn('style.display="none"', head_ui.replace(
            'hint.style.display=enabled?"none":""', ""
        ))
        self.assertIn('isFull=$("#filmMode").value==="full"', head_ui)
        self.assertIn("负片曲线预设", head_ui)
        self.assertIn("反转片没有印相色头", head_ui)
        self.assertIn("0CC", head_ui)
        self.assertIn("el.disabled=!enabled", head_ui)
        self.assertIn("el.value=0", head_ui)
        self.assertIn('id="colorHeadHint"', PAGE)
        listener = '$("#filmMode").addEventListener'
        line = PAGE[PAGE.index(listener) : PAGE.index("\n", PAGE.index(listener))]
        self.assertIn("updateColorHeadUi()", line)
