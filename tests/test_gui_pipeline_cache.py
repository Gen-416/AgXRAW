# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded GUI cache and scheduling contracts with tiny deterministic fixtures."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import weakref
from unittest import mock

from dngscan.gui import preview_cache as pc
from dngscan.gui import service
from dngscan.gui.preview_scheduler import PreviewCoordinator, PreviewSuperseded
from tests.test_preview_cache import _analysis, _bundle


class EntryFlightTests(unittest.TestCase):
    def test_same_key_ten_requests_share_one_compute_without_blocking_other_caches(self):
        entry = pc.PreviewEntry(_bundle(), _analysis())
        entered, release = threading.Event(), threading.Event()
        value = object()

        def build():
            entered.set()
            self.assertTrue(release.wait(3))
            return value

        builder = mock.Mock(side_effect=build)
        with ThreadPoolExecutor(max_workers=11) as pool:
            work = [pool.submit(entry.get_or_build_auto_ev, "same", builder) for _ in range(10)]
            self.assertTrue(entered.wait(3))
            unrelated = pool.submit(entry.get_or_build_plan, "other", lambda: 42)
            try:
                self.assertEqual(unrelated.result(timeout=1), 42)
                entry.put_frame("frame", {"metrics": {"x": 1}})
                self.assertEqual(entry.get_frame("frame")["metrics"]["x"], 1)
            finally:
                release.set()
            self.assertTrue(all(f.result(timeout=3) is value for f in work))
        builder.assert_called_once()

    def test_plan_and_balance_builders_execute_outside_runtime_lock(self):
        entry = pc.PreviewEntry(_bundle(), _analysis())
        for kind, key in (("plan", "p"), ("balance", "daylight")):
            def build():
                acquired = entry._runtime_cache_lock.acquire(blocking=False)
                self.assertTrue(acquired)
                if acquired:
                    entry._runtime_cache_lock.release()
                return object()
            first = entry.get_or_compute(kind, key, build)
            self.assertIs(entry.get_or_compute(kind, key, lambda: self.fail("cache miss")), first)

    def test_failed_flight_is_removed_and_retry_can_succeed(self):
        entry = pc.PreviewEntry(_bundle(), _analysis())
        error = ValueError("build failed")
        with self.assertRaises(ValueError) as raised:
            entry.get_or_build_plan("bad", lambda: (_ for _ in ()).throw(error))
        self.assertIs(raised.exception, error)
        self.assertEqual(entry._runtime_inflight, {})
        self.assertEqual(entry.get_or_build_plan("bad", lambda: 3), 3)

    def test_auto_ev_key_uses_parsed_defaults_not_request_spelling(self):
        from types import SimpleNamespace
        entry = pc.PreviewEntry(_bundle(), _analysis())
        with tempfile.NamedTemporaryFile(suffix=".dng") as source, \
             mock.patch.object(service.PREVIEW_STORE, "get", return_value=entry), \
             mock.patch.object(service, "export_preview_jpeg", return_value={"ok": True}), \
             mock.patch.object(service.dg, "compute_auto_ev", return_value=SimpleNamespace(ev=0.)) as compute:
            params = {"input": source.name, "evAuto": True}
            service.run_preview(params)
            service.run_preview({**params, "wb": "camera", "gamut": "srgb", "quality": 99,
                                 "previewClient": "new-client", "selectionEpoch": 1,
                                 "previewSession": "new-session", "generation": 1})
        compute.assert_called_once()


class SelectionTests(unittest.TestCase):
    def test_scheduler_observes_queue_and_execution_separately_including_failure(self):
        from dngscan.gui import scheduler
        slots = scheduler.RenderScheduler()
        with mock.patch.object(scheduler.time, "monotonic", side_effect=[1., 3., 8., 10., 11., 15.]):
            with slots.slot("preview"):
                self.assertEqual(slots.snapshot()["active"]["preview"], 1)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with slots.slot("prepare"):
                    raise RuntimeError("injected")
        state = slots.snapshot()
        self.assertEqual(state["queue_seconds"], {"preview": 2., "prepare": 1., "export": 0.})
        self.assertEqual(state["execute_seconds"], {"preview": 5., "prepare": 4., "export": 0.})
        self.assertEqual(state["active"], {"preview": 0, "prepare": 0, "export": 0})

    def test_obsolete_queued_flight_skips_decode_then_new_selection_can_retry(self):
        cache = pc.PreviewCache()
        current = threading.Event()
        current.set()
        for _ in range(cache.MAX_CONCURRENT_BUILDS):
            cache._build_slots.acquire()
        with mock.patch.object(pc, "_cache_identity", return_value=(("queued",), "queued")), \
             mock.patch.object(pc.dg, "load_raw") as decode, \
             ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(cache.get, Path("queued.dng"), "clip", "camera", _is_current=current.is_set)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with cache.lock:
                    attached = bool(cache._inflight)
                if attached:
                    break
                time.sleep(.001)
            current.clear()
            for _ in range(cache.MAX_CONCURRENT_BUILDS):
                cache._build_slots.release()
            self.assertTrue(attached)
            with self.assertRaises(PreviewSuperseded):
                old.result(timeout=2)
            self.assertEqual(cache._inflight, {})
            entry = pc.PreviewEntry(_bundle(), _analysis())
            with mock.patch.object(pc, "_read_disk_entry", return_value=entry):
                self.assertIs(cache.get(Path("queued.dng"), "clip", "camera"), entry)
        decode.assert_not_called()

    def test_stable_client_epoch_supersedes_sessions_without_canceling_another_client(self):
        c = PreviewCoordinator()
        self.assertTrue(c.register_selection("one", 1))
        self.assertTrue(c.register_selection("two", 1))
        self.assertTrue(c.register_selection("one", 2))
        self.assertFalse(c.selection_is_current("one", 1))
        self.assertTrue(c.selection_is_current("two", 1))
        self.assertFalse(c.register_selection("one", 1))
        self.assertTrue(c.selection_is_current("", 0))

    def test_prepare_rechecks_selection_after_waiting_for_slot(self):
        coordinator = PreviewCoordinator()
        registered = threading.Event()
        original = coordinator.register_selection
        def register(client, epoch):
            result = original(client, epoch)
            registered.set()
            return result
        with mock.patch.object(service, "PREVIEW_COORDINATOR", coordinator), \
             mock.patch.object(coordinator, "register_selection", side_effect=register), \
             mock.patch.object(service, "_prepare_preview_current") as prepare, \
             ThreadPoolExecutor(max_workers=1) as pool:
            with service.SCHEDULER.slot("prepare"):
                job = pool.submit(service.prepare_preview, {"previewClient": "same", "selectionEpoch": 1})
                self.assertTrue(registered.wait(2))
                coordinator.register_selection("same", 2)
            self.assertTrue(job.result(timeout=2)["superseded"])
        prepare.assert_not_called()

    def test_shared_raw_flight_survives_stale_owner_if_another_subscriber_needs_it(self):
        cache = pc.PreviewCache()
        # Publication can immediately evict an entry bigger than the budget;
        # subscribers still own this flight's result and must not decode again.
        cache.max_memory_bytes = 1
        entry = pc.PreviewEntry(_bundle(), _analysis())
        entered, release, owner_current = threading.Event(), threading.Event(), threading.Event()
        owner_current.set()
        def load(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return entry.bundle
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(pc, "_cache_identity", return_value=(("shared",), "shared")))
            stack.enter_context(mock.patch.object(pc, "_read_disk_entry", return_value=None))
            stack.enter_context(mock.patch.object(pc, "build_proxy_entry", return_value=entry))
            stack.enter_context(mock.patch.object(pc.DISK_WRITER, "submit"))
            decode = stack.enter_context(mock.patch.object(pc.dg, "load_raw", side_effect=load))
            analyze = stack.enter_context(mock.patch.object(pc.dg, "analyze", return_value=(entry.analysis, None, None)))
            with ThreadPoolExecutor(max_workers=2) as pool:
                old = pool.submit(cache.get, Path("shared.dng"), "clip", "camera", _is_current=owner_current.is_set)
                self.assertTrue(entered.wait(2))
                owner_current.clear()
                current = pool.submit(cache.get, Path("shared.dng"), "clip", "camera", _is_current=lambda: True)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    with cache.lock:
                        attached = len(cache._inflight[("shared",)].interests) == 2
                    if attached:
                        break
                    time.sleep(.001)
                release.set()
                self.assertTrue(attached)
                with self.assertRaises(PreviewSuperseded):
                    old.result(timeout=2)
                self.assertIs(current.result(timeout=2), entry)
            decode.assert_called_once()
            analyze.assert_called_once()
            self.assertEqual(cache.entries, {})
            # A later independent request may rebuild the evicted entry.
            self.assertIs(cache.get(Path("shared.dng"), "clip", "camera"), entry)
        self.assertEqual(decode.call_count, 2)
        self.assertEqual(analyze.call_count, 2)


class AnalysisEnvelopeTests(unittest.TestCase):
    def test_cached_export_refreshes_masks_releases_analysis_planes_and_reuses_auto_plan(self):
        from types import SimpleNamespace
        from dngscan import raw_io
        plan = object()
        def auto(*args, **kwargs):
            kwargs["_plan_sink"].append(plan)
            return SimpleNamespace(ev=.25)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.dng"
            source.write_bytes(b"tiny fixture")
            bundle = _bundle()
            bundle._analysis_y_render = bundle.xyz_render[..., 1]
            with mock.patch.object(service.dg, "require_dependencies"), \
                 mock.patch.object(service.PREVIEW_STORE, "peek", return_value=None), \
                 mock.patch.object(service, "_load_export_scene", return_value=bundle) as load, \
                 mock.patch.object(service, "_cached_full_analysis", return_value=_analysis()), \
                 mock.patch.object(raw_io, "refresh_clip_masks_from_fullwell") as masks, \
                 mock.patch.object(service.dg, "compute_auto_ev", side_effect=auto), \
                 mock.patch.object(service.dg, "build_render_plan") as compile_plan, \
                 mock.patch.object(service.dg, "analyze") as analyze, \
                 mock.patch.object(service.dg, "export_jpeg", side_effect=RuntimeError("capture export")) as export:
                with self.assertRaisesRegex(RuntimeError, "capture export"):
                    service.run_export({"input": str(source), "outdir": directory, "evAuto": True})
            masks.assert_called_once_with(bundle, _analysis().channel_fullwell)
            compile_plan.assert_not_called()
            analyze.assert_not_called()
            self.assertTrue(load.call_args.kwargs["analysis_luminance_only"])
            args = export.call_args.kwargs
            self.assertIs(args["tone_plan"], plan)
            self.assertIsNone(args["bundle"].xyz_render)
            self.assertIsNone(args["bundle"]._analysis_y_render)
            self.assertTrue(args["return_rgb"])

    def test_immediate_export_uses_exact_analysis_before_disk_write(self):
        source = _bundle()
        entry = pc.PreviewEntry(source, _analysis(), source_metadata=pc._bundle_metadata(source), cache_digest="fresh")
        envelope = service._preview_analysis_envelope(entry)
        self.assertIsInstance(envelope, str)
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(pc, "_cache_identity", return_value=(("fresh",), "fresh")), \
             mock.patch.object(pc, "_cache_dir", return_value=Path(directory)):
            loaded = service._cached_full_analysis(source.path, "clip", "camera", "libraw", "auto", "auto",
                                                  decoded_bundle=source, envelope=envelope)
            self.assertEqual(json.dumps(pc._analysis_to_json(loaded), sort_keys=True),
                             json.dumps(pc._analysis_to_json(entry.analysis), sort_keys=True))
            original_scene = source.scene_rec2020_render
            source.scene_rec2020_render = original_scene.astype(pc.dg.np.float32)
            self.assertIsNone(service._cached_full_analysis(source.path, "clip", "camera", "libraw", "auto", "auto",
                                                           decoded_bundle=source, envelope=envelope))
            source.scene_rec2020_render = original_scene
            for field, value in (("digest", "other"), ("native_abi", -1), ("cache_version", -1)):
                changed = json.loads(envelope)
                changed[field] = value
                self.assertIsNone(service._cached_full_analysis(source.path, "clip", "camera", "libraw", "auto", "auto",
                                                               decoded_bundle=source, envelope=json.dumps(changed)))
            source.scene_scale *= 2
            self.assertIsNone(service._cached_full_analysis(source.path, "clip", "camera", "libraw", "auto", "auto",
                                                           decoded_bundle=source, envelope=envelope))
            self.assertIsNone(service._cached_full_analysis(source.path, "clip", "daylight", "libraw", "auto", "auto",
                                                           decoded_bundle=source, envelope=envelope))

    def test_source_geometry_is_preserved_for_disk_and_envelope_and_v19_is_rejected(self):
        source = _bundle()
        with mock.patch.object(pc, "PROXY_LONG_EDGE", 4):
            entry = pc.build_proxy_entry(source, _analysis())
        entry.cache_digest = "geometry"
        self.assertEqual(entry.bundle.scene_rec2020_render.shape, (4, 4, 3))
        self.assertEqual(entry.source_metadata["scene_shape"], [8, 8, 3])
        self.assertEqual(entry.source_metadata["scene_dtype"], "uint16")
        self.assertEqual(pc._bundle_metadata(entry.bundle)["scene_dtype"], "float32")
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(pc, "_cache_identity", return_value=(("geometry",), "geometry")), \
             mock.patch.object(pc, "_cache_dir", return_value=Path(directory)):
            path = Path(directory) / "geometry.npz"
            pc._write_disk_entry(path, entry)
            restored = pc._read_disk_entry(path, source.path, False)
            self.assertEqual(restored.source_metadata, entry.source_metadata)
            self.assertEqual(restored.bundle.scene_rec2020_render.shape, (4, 4, 3))
            for envelope in (None, service._preview_analysis_envelope(entry)):
                self.assertIsNotNone(service._cached_full_analysis(source.path, "clip", "camera", "libraw", "auto", "auto",
                                                                  decoded_bundle=source, envelope=envelope))
            # The old schema cannot authorize full-resolution analysis reuse.
            with mock.patch.object(pc, "PREVIEW_CACHE_VERSION", 19):
                pc._write_disk_entry(path, entry)
                old_envelope = service._preview_analysis_envelope(entry)
            self.assertIsNone(pc._read_disk_entry(path, source.path, False))
            self.assertIsNone(service._cached_full_analysis(source.path, "clip", "camera", "libraw", "auto", "auto",
                                                           decoded_bundle=source, envelope=old_envelope))

    def test_spawn_strips_untrusted_envelope_and_carries_only_parent_cache_snapshot(self):
        captured = []
        class Process:
            def __init__(self, target, args, name):
                captured.append(args[0])
            def start(self):
                raise RuntimeError("captured payload")
        context = mock.Mock()
        context.Process = Process
        source = _bundle()
        entry = pc.PreviewEntry(source, _analysis(), source_metadata=pc._bundle_metadata(source), cache_digest="trusted")
        with tempfile.NamedTemporaryFile(suffix=".dng") as file, \
             mock.patch.object(service.mp, "get_context", return_value=context), \
             mock.patch.object(service.PREVIEW_STORE, "peek", side_effect=[None, entry]):
            params = {"input": file.name, "filmOpticsSeed": 1,
                      "_previewAnalysis": "forged", "_previewDecode": {"forged": True}}
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "captured payload"):
                    service.run_export_isolated(params)
        self.assertNotIn("_previewAnalysis", captured[0])
        self.assertNotIn("_previewDecode", captured[0])
        self.assertEqual(captured[1]["_previewAnalysis"], service._preview_analysis_envelope(entry))
        self.assertEqual(captured[1]["_previewDecode"], service._preview_decode_contract(source))
        self.assertEqual(params["_previewAnalysis"], "forged")


class DiskAndMemoryTests(unittest.TestCase):
    def test_byte_accounting_skips_only_plain_scalars_and_keeps_nested_payloads(self):
        from dataclasses import dataclass
        np = pc.dg.np

        @dataclass
        class Holder:
            value: object

        @dataclass
        class ScalarWithPayload(int):
            payload: object = None

        array = np.zeros((2, 3), np.float32)
        text, binary = "a separately owned text payload", bytearray(b"binary payload")
        scalar = ScalarWithPayload(7)
        scalar.payload = Holder([array[:, ::-1], text])
        payload = {"dict keys stay outside the byte contract": [None, True, 5000,
                   2.5, 3j, np.float64(2), Holder((array, scalar, binary))]}
        self.assertEqual(pc._owned_bytes(payload), array.nbytes + len(text) + len(binary))

    def test_cold_memory_disk_paths_preserve_pixels_and_analysis(self):
        cache, writer = pc.PreviewCache(), pc.DiskWriter()
        source = _bundle()
        source._analysis_y_render = source.xyz_render[..., 1]
        with tempfile.TemporaryDirectory() as directory, tempfile.NamedTemporaryFile(suffix=".dng") as file, \
             mock.patch.object(pc, "_cache_dir", return_value=Path(directory)), \
             mock.patch.object(pc, "DISK_WRITER", writer), \
             mock.patch.object(pc.dg, "load_raw", return_value=source) as decode, \
             mock.patch.object(pc.dg, "analyze", return_value=(_analysis(), None, None)) as analyze:
            first = cache.get(Path(file.name), "clip", "camera")
            self.assertIs(cache.get(Path(file.name), "clip", "camera"), first)
            self.assertTrue(writer.flush(2))
            cache.clear_memory()
            disk = cache.get(Path(file.name), "clip", "camera")
        decode.assert_called_once()
        analyze.assert_called_once()
        self.assertTrue(decode.call_args.kwargs["_analysis_luminance_only"])
        self.assertFalse(analyze.call_args.kwargs["_return_planes"])
        self.assertIsNone(first.bundle.xyz_render)
        self.assertIsNone(first.bundle._analysis_y_render)
        pc.dg.np.testing.assert_array_equal(first.bundle.scene_rec2020_render, disk.bundle.scene_rec2020_render)
        self.assertEqual(json.dumps(pc._analysis_to_json(first.analysis), sort_keys=True),
                         json.dumps(pc._analysis_to_json(disk.analysis), sort_keys=True))
        self.assertEqual(first.source_metadata, disk.source_metadata)

    def test_byte_eviction_recomputes_optional_plan_without_losing_active_result(self):
        cache = pc.PreviewCache()
        entry = pc.PreviewEntry(_bundle(), _analysis())
        cache.entries[("one",)] = entry
        cache.max_memory_bytes = pc._owned_bytes(entry)
        entry._memory_notify = weakref.WeakMethod(cache._trim_memory)
        build = mock.Mock(side_effect=lambda: pc.dg.np.zeros((100,), pc.dg.np.float32))
        first = entry.get_or_build_plan("same", build)
        self.assertEqual(entry._plan_cache, {})
        self.assertEqual(first.shape, (100,))
        second = entry.get_or_build_plan("same", build)
        self.assertEqual(build.call_count, 2)
        self.assertIsNot(first, second)
        self.assertLessEqual(cache.memory_snapshot()["bytes"], cache.max_memory_bytes)

    def test_writer_byte_limit_skips_oversized_snapshot(self):
        writer = pc.DiskWriter(max_bytes=1)
        with mock.patch.object(pc, "_write_disk_entry") as write:
            writer.submit(Path("large.npz"), pc.PreviewEntry(_bundle(), _analysis()))
            self.assertTrue(writer.flush(1))
        write.assert_not_called()
        self.assertEqual(writer.snapshot()["pending_items"], 0)

    def test_slow_writer_does_not_delay_submit_and_queue_drops_old_pending_entries(self):
        writer = pc.DiskWriter(max_items=1)
        entry = pc.PreviewEntry(_bundle(), _analysis())
        entered, release = threading.Event(), threading.Event()
        paths = []
        def write(path, snapshot):
            paths.append(path.name)
            self.assertEqual(snapshot._pixel_cache, {})
            if path.name == "first.npz":
                entered.set()
                self.assertTrue(release.wait(3))
        with mock.patch.object(pc, "_write_disk_entry", side_effect=write):
            writer.submit(Path("first.npz"), entry)
            self.assertTrue(entered.wait(2))
            writer.submit(Path("old.npz"), entry)
            writer.submit(Path("new.npz"), entry)
            release.set()
            self.assertTrue(writer.flush(3))
        self.assertEqual(paths, ["first.npz", "new.npz"])

    def test_writer_failure_does_not_poison_the_next_write(self):
        writer = pc.DiskWriter()
        entry = pc.PreviewEntry(_bundle(), _analysis())
        with mock.patch.object(pc, "_write_disk_entry", side_effect=[ValueError("injected"), None]) as write, \
             self.assertLogs("dngscan.gui.preview_cache", level="WARNING"):
            writer.submit(Path("bad.npz"), entry)
            self.assertTrue(writer.flush(2))
            writer.submit(Path("good.npz"), entry)
            self.assertTrue(writer.flush(2))
        self.assertEqual(write.call_count, 2)

    def test_white_balance_children_share_identical_immutable_dither_owner(self):
        base = pc.PreviewEntry(_bundle(), _analysis())
        child = pc.PreviewCache._build_balance(base, "daylight")
        self.assertIsNone(child.bundle.xyz_render)
        self.assertIsNone(child.bundle._analysis_y_render)
        with mock.patch.object(pc.dg, "deterministic_dither_planes", wraps=pc.dg.deterministic_dither_planes) as build:
            first, second = base.get_or_build_dither_noise(), child.get_or_build_dither_noise()
        self.assertIs(first, second)
        build.assert_called_once()
        self.assertTrue(all(not plane.flags.writeable for plane in first))
        self.assertEqual(pc._owned_bytes([first, second]), sum(plane.nbytes for plane in first))

    def test_dither_notifies_on_first_shared_owner_attachment_only(self):
        base = pc.PreviewEntry(_bundle(), _analysis())
        child = pc.PreviewCache._build_balance(base, "daylight")
        with mock.patch.object(base, "_changed") as base_changed, \
             mock.patch.object(child, "_changed") as child_changed:
            noise = base.get_or_build_dither_noise()
            self.assertIs(child.get_or_build_dither_noise(), noise)
            for _ in range(3):
                self.assertIs(base.get_or_build_dither_noise(), noise)
                self.assertIs(child.get_or_build_dither_noise(), noise)
        base_changed.assert_called_once()
        child_changed.assert_called_once()

    def test_buffer_accounting_deduplicates_owners_and_eviction_keeps_active_values_alive(self):
        np = pc.dg.np
        owner = np.zeros((12, 13, 3), np.uint8)
        self.assertEqual(pc._owned_bytes([owner, owner[::-1], owner[:, ::-1]]), owner.nbytes)
        cache = pc.PreviewCache()
        first, second = pc.PreviewEntry(_bundle(), _analysis()), pc.PreviewEntry(_bundle(), _analysis())
        cache.entries[("old",)] = first
        cache.entries[("new",)] = second
        cache.max_memory_bytes = pc._owned_bytes(second)
        cache._trim_memory()
        self.assertEqual(list(cache.entries), [("new",)])
        self.assertLessEqual(cache.memory_snapshot()["bytes"], cache.max_memory_bytes)
        self.assertIsNotNone(first.bundle.scene_rec2020_render)

    def test_internal_pixel_transfer_avoids_copy_while_public_insertion_is_defensive(self):
        np = pc.dg.np
        entry = pc.PreviewEntry(_bundle(), _analysis())
        pixels = np.zeros((2, 3, 3), np.uint8)
        self.assertIs(entry.put_pixels("owned", pixels, _take_ownership=True), pixels)
        self.assertFalse(pixels.flags.writeable)
        external = np.zeros((2, 3, 3), np.uint8)
        stored = entry.put_pixels("external", external)
        external.fill(255)
        self.assertEqual(int(stored.max()), 0)


if __name__ == "__main__":
    unittest.main()
