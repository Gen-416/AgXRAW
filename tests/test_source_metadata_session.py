# SPDX-License-Identifier: GPL-3.0-or-later
"""Metadata reuse is task-local, source-validated and independent of mutable plans."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import coreimage_decode, dng_opcodes, metadata
from dngscan.source_metadata import SourceMetadataSession, cached_source_metadata, source_metadata_session


class SourceMetadataSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "source.dng"
        self.path.write_bytes(b"II" + struct.pack("<HIHI", 42, 8, 0, 0))

    def test_outside_scope_always_reads_and_new_tasks_do_not_share(self):
        calls = []

        @cached_source_metadata
        def reader(path):
            calls.append(path)
            return {"value": [1]}

        reader(self.path)
        reader(self.path)
        self.assertEqual(len(calls), 2)
        with source_metadata_session(self.path):
            reader(self.path)
            reader(self.path)
        with source_metadata_session(self.path):
            reader(self.path)
        self.assertEqual(len(calls), 4)

    def test_nested_same_source_scope_reuses_identity_and_copies_results(self):
        calls = []

        @cached_source_metadata
        def reader(path):
            calls.append(path)
            return {"nested": [1, {"value": 2}]}

        with source_metadata_session(self.path) as outer:
            first = reader(self.path)
            first["nested"][1]["value"] = 99
            with source_metadata_session(self.path) as nested:
                self.assertIs(nested, outer)
                self.assertTrue(nested.is_current())
                second = reader(self.path)
                self.assertEqual(second["nested"][1]["value"], 2)
            second["nested"][0] = 100
            self.assertEqual(reader(self.path)["nested"][0], 1)
            self.assertEqual(len(calls), 1)
        self.assertEqual(outer._memo, {})

    def test_mutable_opcode_plan_and_numpy_payload_are_isolated(self):
        @cached_source_metadata
        def recipe(_path):
            plan = dng_opcodes.OpcodePlan()
            plan.names.append("source")
            plan.post.append({"array": np.arange(4, dtype=np.float32)})
            return plan

        with source_metadata_session(self.path):
            first = recipe(self.path)
            first.names.append("runtime")
            first.post[0]["array"][0] = 99
            second = recipe(self.path)
            self.assertEqual(second.names, ["source"])
            np.testing.assert_array_equal(second.post[0]["array"], np.arange(4, dtype=np.float32))
            self.assertFalse(np.shares_memory(first.post[0]["array"], second.post[0]["array"]))

    def test_replacement_invalidates_session_permanently_including_nested_scope(self):
        calls = []

        @cached_source_metadata
        def reader(path):
            calls.append(path)
            return path.read_bytes()

        with source_metadata_session(self.path) as session:
            original = reader(self.path)
            backup = self.path.with_suffix(".old")
            self.path.rename(backup)
            self.path.write_bytes(original)
            self.assertFalse(session.is_current())
            self.assertEqual(session._memo, {})
            self.path.unlink()
            backup.rename(self.path)
            self.assertFalse(session.is_current())
            with source_metadata_session(self.path) as nested:
                self.assertIs(nested, session)
                reader(self.path)
                reader(self.path)
        self.assertEqual(len(calls), 3)

    def test_in_place_change_with_restored_mtime_is_not_reused(self):
        with source_metadata_session(self.path) as session:
            before = self.path.stat()
            self.path.write_bytes(b"MM" + self.path.read_bytes()[2:])
            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertFalse(session.is_current())

    def test_source_change_during_reader_is_never_published(self):
        @cached_source_metadata
        def reader(path):
            path.write_bytes(b"new source")
            return {"from": "old source"}

        with source_metadata_session(self.path) as session:
            self.assertEqual(reader(self.path), {"from": "old source"})
            self.assertFalse(session.is_current())
            self.assertEqual(session._memo, {})

    def test_foreign_path_bypasses_cache_without_invalidating_owner(self):
        other = self.path.with_name("other.dng")
        other.write_bytes(b"other")
        calls = []

        @cached_source_metadata
        def reader(path):
            calls.append(path)
            return path.read_bytes()

        with source_metadata_session(self.path) as session:
            reader(self.path)
            reader(other)
            reader(other)
            reader(self.path)
            self.assertTrue(session.is_current())
        self.assertEqual(calls, [self.path, other, other])

    def test_errors_and_explicit_unavailable_values_are_not_cached(self):
        calls = []

        @cached_source_metadata(cacheable=lambda value: value is not None)
        def reader(_path):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("failed read")
            return None if len(calls) == 2 else {"valid": True}

        with source_metadata_session(self.path):
            with self.assertRaisesRegex(ValueError, "failed read"):
                reader(self.path)
            self.assertIsNone(reader(self.path))
            self.assertEqual(reader(self.path), {"valid": True})
            self.assertEqual(reader(self.path), {"valid": True})
        self.assertEqual(len(calls), 3)

    def test_missing_identity_leaves_original_reader_available(self):
        missing = self.path.with_name("missing.dng")
        session = SourceMetadataSession(missing)
        self.assertIsNone(session.identity)
        self.assertFalse(session.is_current())
        self.assertEqual(session.read(lambda _: "original fallback", missing), "original fallback")

    def test_task_context_is_not_implicitly_shared_with_worker_threads(self):
        calls = []

        @cached_source_metadata
        def reader(path):
            calls.append(path)
            return {"value": 1}

        with source_metadata_session(self.path):
            reader(self.path)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(reader, self.path).result()
                pool.submit(reader, self.path).result()
            reader(self.path)
        self.assertEqual(len(calls), 3)

    def test_reader_identity_and_extra_arguments_do_not_alias_memo(self):
        @cached_source_metadata
        def reader(_path, value=1):
            return {"value": value}

        @cached_source_metadata
        def other_reader(_path):
            return {"value": 2}

        with source_metadata_session(self.path):
            self.assertEqual(reader(self.path), {"value": 1})
            self.assertEqual(reader(self.path, value=3), {"value": 3})
            self.assertEqual(other_reader(self.path), {"value": 2})
            self.assertEqual(reader(self.path), {"value": 1})

    def test_real_opcode_summary_and_shot_readers_reuse_inside_scope_only(self):
        # A minimal TIFF with an inline ISO suffices to make shot metadata valid.
        self.path.write_bytes(b"II" + struct.pack("<HIHHHIII", 42, 8, 1, metadata.TAG_ISO, 4, 1, 400, 0))
        with patch.object(coreimage_decode, "_opcode_summary_ids", wraps=coreimage_decode._opcode_summary_ids) as opcodes, \
             patch.object(metadata, "_parse_tiff_shot_info", wraps=metadata._parse_tiff_shot_info) as shot:
            with source_metadata_session(self.path):
                for _ in range(2):
                    self.assertTrue(coreimage_decode.read_dng_opcodes(self.path)["parsed"])
                    self.assertEqual(metadata.read_dng_shot_info(self.path).iso, 400)
            self.assertEqual(opcodes.call_count, 1)
            self.assertEqual(shot.call_count, 1)
            coreimage_decode.read_dng_opcodes(self.path)
            metadata.read_dng_shot_info(self.path)
            self.assertEqual(opcodes.call_count, 2)
            self.assertEqual(shot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
