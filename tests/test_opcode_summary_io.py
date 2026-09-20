# SPDX-License-Identifier: GPL-3.0-or-later
"""Apple opcode diagnostics preserve their tolerant policy with bounded reads."""
import io
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

from dngscan import coreimage_decode as ci


class Tiff:
    def __init__(self, order="<", magic=42):
        self.order = order
        self.data = bytearray((b"II" if order == "<" else b"MM") + struct.pack(order + "HI", magic, 8))

    def ifd(self, entries, next_ifd=0):
        offset = len(self.data)
        self.data += struct.pack(self.order + "H", len(entries))
        for tag, typ, count, value in entries:
            self.data += struct.pack(self.order + "HHII", tag, typ, count, value)
        self.data += struct.pack(self.order + "I", next_ifd)
        return offset

    def opcodes(self, ids):
        offset = len(self.data)
        self.data += struct.pack(">I", len(ids))
        for oid in ids:
            # Unknown versions/required flags are diagnostic data, not rejection.
            self.data += struct.pack(">IIII", oid, 0x7FFFFFFF, 0, 0)
        return offset

    def value(self, ifd, index, value):
        struct.pack_into(self.order + "I", self.data, ifd + 2 + index * 12 + 8, value)


class SparseReader:
    """Seekable multi-gigabyte logical file, allocating only requested metadata."""
    def __init__(self, length, pieces):
        self.length, self.pieces, self.position = length, pieces, 0
        self.reads = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.length if whence == 2 else self.position if whence == 1 else 0)
        return self.position

    def read(self, size=-1):
        if not 0 <= size <= 65535 * 12:
            raise AssertionError(f"unbounded read: {size}")
        self.reads.append((self.position, size))
        out = bytearray(min(size, max(0, self.length - self.position)))
        for start, content in self.pieces:
            lo, hi = max(start, self.position), min(start + len(content), self.position + len(out))
            if hi > lo:
                out[lo - self.position:hi - self.position] = content[lo - start:hi - start]
        self.position += len(out)
        return bytes(out)


class OpcodeSummaryIoTests(unittest.TestCase):
    def read(self, data):
        with patch.object(Path, "open", return_value=io.BytesIO(data)), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("whole RAW read")):
            return ci.read_dng_opcodes(Path("fixture.dng"))

    def test_both_endians_follow_single_inline_subifd_and_unknown_opcode(self):
        for order in ("<", ">"):
            for typ in (4, 13):
                with self.subTest(order=order, typ=typ):
                    t = Tiff(order)
                    main = t.ifd([(330, typ, 1, 0)])
                    sub = t.ifd([(0xC741, 7, 36, 0)])
                    t.value(main, 0, sub)
                    t.value(sub, 0, t.opcodes([1, 123456]))
                    result = self.read(t.data)
                    self.assertTrue(result["parsed"])
                    self.assertEqual(result["ids"], (1, 123456))
                    self.assertTrue(result["geometry"])
                    self.assertIn("opcode123456", result["names"])

    def test_all_opcode_tags_next_ifd_and_duplicates_remain_visible(self):
        t = Tiff()
        first = t.ifd([(0xC740, 7, 20, 0), (0xC741, 1, 20, 0)])
        second = t.ifd([(0xC74E, 7, 36, 0)])
        struct.pack_into("<I", t.data, first + 2 + 2 * 12, second)
        for owner, index, ids in ((first, 0, [9]), (first, 1, [1]), (second, 0, [9, 14])):
            t.value(owner, index, t.opcodes(ids))
        self.assertEqual(self.read(t.data)["ids"], (1, 9, 14))

    def test_subifd_list_retains_eight_entry_limit(self):
        t = Tiff()
        main = t.ifd([(330, 4, 10, 0)])
        offsets = [t.ifd([(0xC741, 7, 20, 0)]) for _ in range(10)]
        for index, offset in enumerate(offsets):
            t.value(offset, 0, t.opcodes([100 + index]))
        t.value(main, 0, len(t.data))
        t.data += struct.pack("<10I", *offsets)
        self.assertEqual(self.read(t.data)["ids"], tuple(range(100, 108)))

    def test_depth_and_cycle_limits_preserve_diagnostic_traversal(self):
        t = Tiff()
        offsets = [t.ifd([(0xC741, 7, 20, 0)]) for _ in range(5)]
        for index, offset in enumerate(offsets):
            t.value(offset, 0, t.opcodes([100 + index]))
            struct.pack_into("<I", t.data, offset + 14, offsets[(index + 1) % len(offsets)])
        self.assertEqual(self.read(t.data)["ids"], (100, 101, 102, 103))
        struct.pack_into("<I", t.data, offsets[1] + 14, offsets[0])
        self.assertEqual(self.read(t.data)["ids"], (100, 101))

    def test_inline_empty_lists_and_legacy_magic_tolerance(self):
        for typ in (1, 2, 7):
            t = Tiff(magic=99)
            t.ifd([(0xC741, typ, 4, 0)])
            self.assertEqual(self.read(t.data), {
                "ids": (), "names": (), "geometry": False, "parsed": True})

    def test_opcode_limit_and_incomplete_payload_still_report_readable_headers(self):
        t = Tiff()
        main = t.ifd([(0xC741, 7, 1000, 0)])
        payload = t.opcodes(list(range(100, 120)))
        t.value(main, 0, payload)
        self.assertEqual(self.read(t.data)["ids"], tuple(range(100, 116)))
        # Tag count does not constrain this best-effort diagnostic walker.
        struct.pack_into("<I", t.data, main + 2 + 4, 5)
        struct.pack_into(">I", t.data, payload + 4 + 12, 0xFFFFFFFF)
        self.assertEqual(self.read(t.data)["ids"], (100,))

    def test_malformed_ranges_and_non_tiff_are_nonfatal(self):
        for data, parsed in ((b"", False), (b"not-tiff", False),
                             (b"II" + struct.pack("<HI", 42, 0xFFFFFFFF), True),
                             (b"II" + struct.pack("<HI", 42, 8) + b"\xff\xff", True)):
            with self.subTest(data=data):
                result = self.read(data)
                self.assertEqual(result["ids"], ())
                self.assertEqual(result["parsed"], parsed)
        with patch.object(Path, "open", side_effect=OSError("unavailable")):
            self.assertFalse(ci.read_dng_opcodes(Path("fixture.dng"))["parsed"])

    def test_short_read_after_size_check_discards_partial_result(self):
        class ShortReader(io.BytesIO):
            def read(self, size=-1):
                result = super().read(size)
                return result[:-1] if self.tell() > 8 else result

        t = Tiff()
        t.ifd([])
        with patch.object(Path, "open", return_value=ShortReader(t.data)):
            self.assertFalse(ci.read_dng_opcodes(Path("fixture.dng"))["parsed"])

    def test_sparse_large_payload_is_skipped_without_large_reads(self):
        payload = 512 * 1024 * 1024
        skip = 1024 * 1024 * 1024
        header = b"II" + struct.pack("<HI", 42, 8)
        table = struct.pack("<HHHIII", 1, 0xC741, 7, skip + 36, payload, 0)
        source = SparseReader(2 * 1024 * 1024 * 1024, [
            (0, header + table), (payload, struct.pack(">IIIII", 2, 1, 1, 0, skip)),
            (payload + 20 + skip, struct.pack(">IIII", 999, 1, 0, 0))])
        with patch.object(Path, "open", return_value=source), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("whole RAW read")):
            result = ci.read_dng_opcodes(Path("large.dng"))
        self.assertEqual(result["ids"], (1, 999))
        self.assertLess(sum(size for _, size in source.reads), 128)
        self.assertLessEqual(max(size for _, size in source.reads), 16)


if __name__ == "__main__":
    unittest.main()
