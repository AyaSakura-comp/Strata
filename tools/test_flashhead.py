"""tools/test_flashhead.py - the FlashHead table file and the static set's rounding (no GPU, no model, no torch).

    python tools/test_flashhead.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flashhead as FH  # noqa: E402


class Tables(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        n_in, c, b = 64, 5, 4
        cent = rng.integers(0, 256, (c, n_in // 32 * 18), dtype=np.uint8)
        c2t = rng.permutation(np.arange(10, 10 + c * b, dtype=np.int32)).reshape(c, b)
        static = np.arange(10, dtype=np.int32)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "t.bin")
            FH.write_tables(path, cent, c2t, static, n_in, 30)
            t = FH.read_tables(path)
            raw = Path(path).read_bytes()
        self.assertEqual((t["n_in"], t["n_vocab"]), (n_in, 30))
        self.assertTrue(np.array_equal(t["centroids"], cent))
        self.assertTrue(np.array_equal(t["c2t"], c2t))
        self.assertTrue(np.array_equal(t["static"], static))
        # the engine's header: "SFH1", then version, n_in, n_vocab, clusters, cluster size, static count
        self.assertEqual(raw[:4], b"SFH1")
        self.assertEqual(list(np.frombuffer(raw[4:28], "<u4")), [1, n_in, 30, c, b, 10])
        self.assertEqual(len(raw), 28 + cent.size + c2t.size * 4 + static.size * 4)

    def test_a_truncated_file_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "t.bin")
            FH.write_tables(path, np.zeros((2, 36), np.uint8), np.zeros((2, 2), np.int32), np.zeros(3, np.int32), 64,
                            10)
            Path(path).write_bytes(Path(path).read_bytes()[:-4])
            with self.assertRaises(Exception):
                FH.read_tables(path)


class StaticSet(unittest.TestCase):
    def test_most_frequent_then_rounded_for_equal_clusters(self):
        counts = Counter({5: 10, 3: 9, 7: 8, 1: 1})
        s = FH.static_set(counts, 3, 20, 4)       # 3 picked, 17 left: one more so 16 divide into clusters of 4
        self.assertEqual(s.size, 4)
        self.assertTrue({5, 3, 7} <= set(s.tolist()))
        self.assertEqual((20 - s.size) % 4, 0)
        self.assertTrue(np.all(np.diff(s) > 0))   # ascending

    def test_ids_outside_the_vocabulary_are_skipped(self):
        s = FH.static_set(Counter({99: 50, 2: 3}), 2, 10, 2)
        self.assertNotIn(99, s.tolist())
        self.assertEqual((10 - s.size) % 2, 0)


if __name__ == "__main__":
    unittest.main()
