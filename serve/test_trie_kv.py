"""serve/test_trie_kv.py - the prefix-tree KV cache (serve/trie_kv.py): the tree, the engine's checkpoint rules, the
branch rule and the quota offline, then the cache on the chat path of a server over a fake hybrid engine (no GPU).

    python -m unittest serve.test_trie_kv -v
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, SessionRefused, serve  # noqa: E402
from serve.trie_kv import (IMAGE_PAD, LIVE_IDS_AT, STRATA_MAGIC, VISION_START, EngineView, Store, Trie,  # noqa: E402
                           TrieCache, boundary, branch_point, collapse, file_end, first_turn, fold_position, lcp,
                           legacy_namespace, namespace, node_cuts, read_points, to_template)

ROOT = Path(__file__).resolve().parents[1]

S, SYS, ASSISTANT, USER = 248045, 8678, 77091, 846     # <|im_start|>, "system" (effort turn), roles


def turn(role, body):
    return [S, role] + body + [248046, 198]


def prompt(system_len, *turns, effort=True):
    ids = turn(SYS, list(range(1000, 1000 + system_len)))
    for t in turns:
        ids += t
    if effort:
        ids += turn(SYS, [5, 6, 7])                    # the trailing effort turn (effort_position end)
    return np.asarray(ids + [S, ASSISTANT, 198], np.int32)


class TrieTest(unittest.TestCase):
    def test_match_uses_points_inside_edges(self):
        t = Trie()
        a = np.arange(100, dtype=np.int32)
        b = np.concatenate([a[:60], np.arange(500, 540, dtype=np.int32)])
        t.insert(a, [50, 100], "A")
        t.insert(b, [100], "B")                         # splits A's edge at 60: A's point 50 must survive
        best, common = t.match(np.concatenate([a[:70], [9, 9]]).astype(np.int32))
        self.assertEqual((best, common), (("A", 50), 70))
        best, common = t.match(b)
        self.assertEqual(best, ("B", 100))
    def test_match_short_prefix(self):
        t = Trie()
        a = np.arange(100, dtype=np.int32)
        t.insert(a, [50], "A")
        self.assertEqual(t.match(a[:55])[0], ("A", 50))
        self.assertIsNone(t.match(a[:49])[0])


class RulesTest(unittest.TestCase):
    def test_boundary_skips_the_effort_turn(self):
        p = prompt(3000, turn(USER, [1, 2, 3]))
        at = boundary(p, S, SYS)
        self.assertEqual(p[at], S)
        self.assertEqual(p[at + 1], SYS)                # the effort turn's start, not the assistant header
        self.assertEqual(boundary(p, S, -1), len(p) - 3)

    def test_engine_view_root_and_turn(self):
        a1 = prompt(3000, turn(USER, [1, 2, 3]))
        v = EngineView()
        v.read("leaf_a", a1, 0, S, SYS, 2048, 16384)
        root = 3000 + 4
        self.assertIn(root, v.points)                   # first turn boundary after a read from 0
        self.assertIn(boundary(a1, S, SYS), v.points)
        b1 = prompt(3000, turn(USER, [4, 5]))
        self.assertEqual(v.resume(b1, "leaf_a"), root)  # a new chat mounts through the root
        self.assertEqual(v.resume(b1, "someone_else"), 0)

    def test_branch_rule(self):
        a1 = prompt(3000, turn(USER, [1, 2, 3]))
        a2 = prompt(3000, turn(USER, [1, 2, 3]), turn(ASSISTANT, [9, 9]), turn(USER, [4]))
        b1 = prompt(3000, turn(USER, [7, 7]))
        seen = [(a1, boundary(a1, S, SYS))]
        self.assertEqual(branch_point(a2, seen, S), -1)     # the next turn of A: a continuation, no node
        self.assertEqual(branch_point(b1, seen, S), 3004)   # a new chat: a node at the end of the system prompt
        fork = prompt(3000, turn(USER, [1, 2, 3]), turn(ASSISTANT, [9, 9]), turn(USER, [8]))
        self.assertEqual(branch_point(fork, [(a2, boundary(a2, S, SYS))], S), 3004 + 7 + 6)  # A's 2nd user turn

    def test_golden_base_on_first_sight(self):
        a1 = prompt(3000, turn(USER, [1, 2, 3]))
        root, rules = 3004, (S, 2048, 2048)
        self.assertEqual(node_cuts(a1, 0, 0, -1, *rules), [(root, "golden")])         # first sight: read anyway, kept
        self.assertEqual(node_cuts(a1, 0, root, -1, *rules), [])                      # a file holds it already
        a2 = prompt(3000, turn(USER, [1, 2, 3]), turn(ASSISTANT, [9] * 3000), turn(USER, [4]))
        self.assertEqual(node_cuts(a2, len(a2) - 10, 0, -1, *rules), [])             # the engine goes on past it
        self.assertEqual(node_cuts(a1, root, 0, root, *rules), [(root, "golden")])    # a new chat on the live root: one
        self.assertEqual(node_cuts(prompt(1000, turn(USER, [1])), 0, 0, -1, *rules), [])   # cheaper to read again
        q = list(range(5000, 8000))                     # a long first question another path shares: both nodes
        b = prompt(3000, turn(USER, q), turn(ASSISTANT, [9]), turn(USER, [4]))
        cut = root + len(q) + 4
        self.assertEqual(node_cuts(b, 0, 0, cut, *rules), [(root, "golden"), (cut, "branch")])
        img = prompt(3000, turn(USER, [1]))
        img[10] = IMAGE_PAD
        self.assertEqual(node_cuts(img, 0, 0, -1, *rules), [])                        # /completion takes no images

    def test_file_end(self):
        a1 = prompt(3000, turn(USER, [1, 2, 3]))
        live = np.concatenate([a1, [11, 12, 13]]).astype(np.int32)   # prompt + generated
        at = boundary(a1, S, SYS)
        self.assertEqual(file_end(live, [at, len(live)], S), at)
        node = a1[:3004]
        self.assertEqual(file_end(node, [3003, 3004], S), 3004)


class ImageTest(unittest.TestCase):
    def test_collapse_folds_image_tokens_and_moves_points(self):
        template = np.asarray([1, 2, VISION_START, IMAGE_PAD, 248054, 3, 4, S, 5], np.int32)
        engine = np.asarray([1, 2, VISION_START] + [IMAGE_PAD] * 5 + [248054, 3, 4, S, 5], np.int32)
        ids, points = collapse(engine, [2, 5, 11, 13])           # 5 is inside the image: no state there
        self.assertTrue(np.array_equal(ids, template))
        self.assertEqual(points, [2, 7, 9])
        self.assertEqual(to_template(11, len(engine), template), 7)
        self.assertEqual(to_template(2, len(engine), template), 2)


BASE = np.arange(3000, dtype=np.int32)


def ext(base, k):
    return np.concatenate([base, [k, k]]).astype(np.int32)


class QuotaTest(unittest.TestCase):
    def test_least_recently_used_leaves_go_first(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(d, "ns", max_bytes=250)
            for name, kind, used, k in (("leaf_old", "leaf", 1, 1), ("node_old", "node", 0, None),
                                        ("leaf_new", "leaf", 5, 2), ("leaf_live", "leaf", 2, 3)):
                fn = f"trie_ns_{name}.bin"
                Path(os.path.join(d, fn)).write_bytes(b"x" * 100)
                ids, points = (BASE, [2999, 3000]) if k is None else (ext(BASE, k), [3002])
                st.files[fn] = dict(size=100, kind=kind, ids=ids, points=points)
                st.used[fn] = used
            dropped = st.enforce_quota({"trie_ns_leaf_live.bin"})
            # 400 bytes > 250: the oldest leaf, then the next oldest leaf that is not in use; the node they resume
            # from stays (its conversations were used at 5)
            self.assertEqual(dropped, ["trie_ns_leaf_old.bin", "trie_ns_leaf_new.bin"])
            self.assertEqual(sorted(st.files), ["trie_ns_leaf_live.bin", "trie_ns_node_old.bin"])

    def test_a_node_lives_as_long_as_its_conversations(self):
        # root A: restored long ago, its conversations used since; root B: an old system prompt nobody reaches
        b = np.arange(5000, 8000, dtype=np.int32)
        with tempfile.TemporaryDirectory() as d:
            st = Store(d, "ns", max_bytes=250)
            for name, kind, ids, points, used in (
                    ("node_a", "node", BASE, [2999, 3000], 0), ("leaf_a1", "leaf", ext(BASE, 1), [3002], 2),
                    ("leaf_a2", "leaf", ext(BASE, 2), [3002], 9), ("node_b", "node", b, [2999, 3000], 1),
                    ("leaf_b1", "leaf", ext(b, 3), [3002], 1)):
                fn = f"trie_ns_{name}.bin"
                Path(os.path.join(d, fn)).write_bytes(b"x" * 100)
                st.files[fn] = dict(size=100, kind=kind, ids=ids, points=points)
                st.used[fn] = used
            last = st.last_used()
            self.assertEqual((last["trie_ns_node_a.bin"], last["trie_ns_node_b.bin"]), (9, 1))
            # 500 bytes > 250: B's conversation, then B's node (a leaf before its node), then A's older conversation
            self.assertEqual(st.enforce_quota(set()), ["trie_ns_leaf_b1.bin", "trie_ns_node_b.bin", "trie_ns_leaf_a1.bin"])

    def test_legacy_snapshots_go_before_ours_with_their_sidecars(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(d, "ns", max_bytes=200)              # 360 bytes: both leftovers go, ours stays
            def put(n, size, mtime):
                path = os.path.join(d, n)
                Path(path).write_bytes(b"x" * size)
                os.utime(path, (mtime, mtime))
            put("kv2_abc_snap_old.bin", 100, 1); put("kv2_abc_snap_old.meta.json", 10, 1)
            put("golden_h_base_system_prompt.bin", 100, 2); put("golden_h_base_system_prompt.bin.ckpt", 50, 2)
            put("trie_ns_leaf_a.bin", 100, 3)
            st.files["trie_ns_leaf_a.bin"] = dict(size=100, kind="leaf")
            self.assertEqual(st.enforce_quota(set(), dry_run=True), ["kv2_abc_snap_old.bin", "golden_h_base_system_prompt.bin"])
            self.assertEqual(len(os.listdir(d)), 5)          # a dry run deletes nothing
            st.enforce_quota(set())
            self.assertEqual(sorted(n for n in os.listdir(d) if n != "trie_ns_state.json"), ["trie_ns_leaf_a.bin"])

    def test_older_namespace_goes_first(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(d, "ns", max_bytes=150)
            for fn in ("trie_old_leaf_x.bin", "trie_ns_leaf_a.bin"):
                Path(os.path.join(d, fn)).write_bytes(b"x" * 100)
            st.files["trie_ns_leaf_a.bin"] = dict(size=100, kind="leaf")
            self.assertEqual(st.enforce_quota(set()), ["trie_old_leaf_x.bin"])
            self.assertEqual(sorted(n for n in os.listdir(d) if n.endswith(".bin")), ["trie_ns_leaf_a.bin"])



class InServerHelpersTest(unittest.TestCase):
    def test_fold_position_maps_engine_positions(self):
        engine = np.asarray([1, 2, VISION_START] + [IMAGE_PAD] * 5 + [248054, 3, 4, S, 5], np.int32)
        self.assertEqual(fold_position(engine, 2), 2)
        self.assertEqual(fold_position(engine, 11), 7)     # past the image: its 4 extra tokens come off
        self.assertIsNone(fold_position(engine, 5))        # inside the image: no state there
        self.assertEqual(fold_position(engine, 20), 16)    # past the prompt (a budget's second pass)

    def test_namespace_follows_the_model_not_the_binary_or_env(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.gguf")
            Path(model).write_bytes(b"x" * 10)
            args = ["--serve", "--native", model, "--control-vector-scaled", model + ":1.0", "--kv", "int8"]
            ns = namespace(args)
            self.assertEqual(ns, namespace(list(args)))
            self.assertNotEqual(ns, namespace(args[:-1] + ["q8"]))          # another engine option
            os.utime(model, (1, 1))                                          # the model written again
            self.assertNotEqual(ns, namespace(args))

    def test_adopt_takes_over_the_proxys_files(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "strata-x.json")
            Path(cfg).write_text('{"exe": "nowhere"}')
            old = legacy_namespace(cfg, "nowhere")
            write_session(os.path.join(d, f"trie_{old}_node_abc.bin"), list(range(3000)), [list(range(2999))])
            Path(d, f"trie_{old}_state.json").write_text(json.dumps({f"trie_{old}_node_abc.bin": 123.0}))
            svc = type("S", (), {"engine": None})()
            cache = TrieCache(svc, d, "new", 2**30)
            self.assertEqual(cache.adopt(old), 1)
            self.assertEqual(sorted(n for n in os.listdir(d) if n.startswith("trie_")),
                             ["trie_new_node_abc.bin", "trie_new_state.json"])
            self.assertEqual(cache.store.used["trie_new_node_abc.bin"], 123.0)
            self.assertEqual(cache.store.files["trie_new_node_abc.bin"]["points"], [2999, 3000])


# ---------------------------------------------------------------------------------------- in the server
def write_session(path, live, checkpoints):
    """A session file as the engine writes it, as far as read_points reads it: the live state's ids, then each
    checkpoint's (their K/V and other parts empty here)."""
    def part(ids):
        out = struct.pack("<Q", len(ids)) + np.asarray(ids, "<i4").tobytes() + struct.pack("<Q", 0)
        return out + struct.pack("<Q", 0) * 5 + struct.pack("<Q", 0)
    data = STRATA_MAGIC + b"\0" * (LIVE_IDS_AT - len(STRATA_MAGIC)) + part(live)
    data += struct.pack("<Q", len(checkpoints)) + b"".join(part(c) for c in checkpoints)
    Path(path).write_bytes(data)


TOK = ByteTokenizer()
IM_START = TOK.encode("<|im_start|>", parse_special=True)[0]


class HybridEngine(MockEngine):
    """One slot that holds a state only where it checkpointed (a Gated DeltaNet model): the end of what it read last
    and the turn boundaries its rules keep; it reuses the deepest one the next prompt starts with and must read at
    least one token.  SAVE / RESTORE write and read real session files."""

    def __init__(self):
        super().__init__(TOK, "Hmm.</think>\n\nFine, thanks.")
        self.spawn = (None, ["--turn-token", str(IM_START)])
        self.held, self.points, self.last, self.reads, self.refuse = [], set(), {}, [], None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        ids = list(ids)
        live = len(self.held) if self.held and ids[:len(self.held)] == self.held else 0
        common = lcp(np.asarray(self.held, np.int32), np.asarray(ids, np.int32))
        reused = max([p for p in self.points | {live} if p <= common and p < len(ids)], default=0)
        arr = np.asarray(ids, np.int32)
        keep = [p for p in self.points if p <= reused] + ([reused] if reused else [])
        at = boundary(arr, IM_START, -1, reused)
        first = first_turn(arr, IM_START)
        if reused == 0 and 2048 <= first < at:
            keep.append(first)                          # the root, after a read from the start
        if at > 0:
            keep.append(at)
        self.points = set(sorted(set(keep))[-6:])
        out = list(super().generate(ids, max_new, sampling, cancel))
        self.held = ids + out[:-1]                      # the last token written is not read yet
        self.last = {"generated": len(out), "prompt_tokens": len(ids), "prompt_ms": 1.0, "decode_ms": 1.0,
                     "finish": "stop", "reused": reused, "prompt_read": len(ids) - reused}
        self.reads.append((len(ids), reused))
        yield from out

    def session_file(self, action, path):
        if self.refuse:
            raise SessionRefused(self.refuse, "session file: not enough RAM to read it (test)")
        if action == "save":
            write_session(path, self.held, [self.held[:p] for p in sorted(self.points) if p < len(self.held)])
        else:
            live, points = read_points(path)
            self.held, self.points = [int(t) for t in live], set(points)
        return {"tokens": len(self.held), "bytes": os.path.getsize(path), "ms": 1.0}


SYSTEM_A = "You are pi, a coding agent. " + "Tools and skills follow. " * 120      # ~3,000 tokens
SYSTEM_B = "You are Hay, a portfolio assistant. " + "Holdings and rules follow. " * 110


class CacheOnTheChatPath(unittest.TestCase):
    """The cache inside the server, over HybridEngine: what each request restores, saves and reuses."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.engine = HybridEngine()
        self.svc = Service(self.engine, TOK, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.log = os.path.join(self.dir.name, "trie-kv.jsonl")
        self.svc.trie = TrieCache(self.svc, self.dir.name, "t", 2**34, self.log)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def chat(self, system, *turns, stream=False, **extra):
        msgs = [{"role": "system", "content": system}]
        for i, t in enumerate(turns):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
        body = {"messages": msgs, "max_tokens": 32, "stream": stream, **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        return raw if stream else json.loads(raw)

    def decisions(self):
        with open(self.log, encoding="utf-8") as f:
            return [json.loads(x) for x in f]

    def files(self, kind):
        return sorted(n for n in os.listdir(self.dir.name) if n.startswith(f"trie_t_{kind}_"))

    def test_golden_base_then_next_turn_then_resume_after_another_client(self):
        r = self.chat(SYSTEM_A, "hello")
        self.assertEqual(r["choices"][0]["message"]["content"], "Fine, thanks.")
        d = self.decisions()[-1]
        self.assertIn("golden", d)                       # first sight: the system turn read on its own and saved
        self.assertEqual(self.files("node"), ["trie_t_node_" + d["golden"][len("node_"):]])
        self.assertIn(d["engine_reused"], (d["golden_at"], d["golden_at"] + 1))   # went on from the node (its live end)
        leaf_a, base = d["owner"], d["golden_at"]

        self.chat(SYSTEM_A, "hello", "Fine, thanks.", "and now?")
        d = self.decisions()[-1]
        self.assertEqual(d["owner"], leaf_a)             # the next turn keeps the conversation's file
        self.assertNotIn("action", d)                    # the engine still holds it: nothing restored
        self.assertGreater(d["engine_reused"], base)
        self.assertEqual(d["engine_reused"], d["live"])  # what the cache expected the engine to reuse

        self.chat(SYSTEM_B, "what do I hold?")           # another client: its own base; A's turn saved first
        d = self.decisions()[-1]
        self.assertIn("golden", d)
        self.assertIn("trie_t_" + leaf_a, self.files("leaf"))

        self.chat(SYSTEM_A, "hello", "Fine, thanks.", "and now?", "Fine, thanks.", "third")
        d = self.decisions()[-1]
        self.assertEqual((d["action"], d["owner"]), ("restore", leaf_a))
        self.assertEqual(d["engine_reused"], d["expect"])        # resumed where the file ends, not cold
        self.assertGreater(d["engine_reused"], base)

        self.chat(SYSTEM_B, "and cash?")                 # B again, then a new chat of A: restored at A's base
        self.chat(SYSTEM_A, "a new question")
        d = self.decisions()[-1]
        self.assertEqual(d["action"], "restore")
        self.assertNotEqual(d["owner"], leaf_a)          # a new conversation, a new file
        self.assertGreaterEqual(d["engine_reused"], base)
        self.assertFalse(self.svc.trie.lock.locked())

    def test_a_warm_before_the_first_chat_request_is_continued_not_restored_over(self):
        """Research first: a client warms its next prompt (/completion) before any chat request of the turn. The warm
        leaves another conversation's slot ownerless; the chat request that extends it goes on from it."""
        self.chat(SYSTEM_A, "hello")                     # golden base + A's leaf
        self.chat(SYSTEM_B, "what do I hold?")           # another client has the slot
        question = "new question " + "with evidence words " * 60
        msgs = [{"role": "system", "content": SYSTEM_A}, {"role": "user", "content": question}]
        req = urllib.request.Request(self.base + "/apply-template", data=json.dumps(
            {"messages": msgs, "add_generation_prompt": False}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            prefix = json.loads(r.read())["prompt"]
        req = urllib.request.Request(self.base + "/completion", data=json.dumps(
            {"prompt": prefix, "n_predict": 0}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            json.loads(r.read())
        warmed = len(self.svc.slot_held)
        self.assertTrue(self.svc.slot_held_warm)
        self.assertEqual(self.svc.slot_owner_now(), "")  # it left B's conversation
        self.chat(SYSTEM_A, question)
        d = self.decisions()[-1]
        self.assertNotIn("action", d)                    # nothing restored over the warm
        self.assertEqual(d["live_warm"], warmed)
        self.assertEqual(d["engine_reused"], warmed)     # the engine went on from the warm's end
        self.assertFalse(self.svc.slot_held_warm)

    def test_streamed_request_is_planned_too(self):
        raw = self.chat(SYSTEM_A, "hello", stream=True)
        self.assertIn(b"[DONE]", raw)
        d = self.decisions()[-1]
        self.assertIn("golden", d)
        self.assertIsNotNone(d["ttft_ms"])
        self.assertFalse(self.svc.trie.lock.locked())

    def test_refused_restore_still_answers_and_keeps_a_file(self):
        self.chat(SYSTEM_A, "hello")
        self.chat(SYSTEM_B, "what do I hold?")
        self.engine.refuse = "memory"                    # not enough RAM: restore and save are refused
        r = self.chat(SYSTEM_A, "hello", "Fine, thanks.", "and now?")
        self.assertEqual(r["choices"][0]["message"]["content"], "Fine, thanks.")
        d = self.decisions()[-1]
        self.assertNotIn("plan_error", d)
        self.assertEqual(d["action"], "restore_refused")
        self.assertTrue(d["owner"].startswith("leaf_"))  # the conversation still has its own file
        self.assertEqual(d["engine_reused"], 0)          # read cold, as without the cache

    def test_own_slot_snapshot_skips_the_cache(self):
        self.chat(SYSTEM_A, "hello", slot_snapshot="mine.bin")
        self.assertFalse(os.path.exists(self.log) and self.decisions())
        self.assertEqual(self.svc.slot_owner_now(), "mine.bin")

    def test_warm_leaves_the_requests_image_file_alone(self):
        self.svc.embeddings.path = "/tmp/the-requests-images.bin"
        self.svc.request_trace.record = {"marker": 1}
        ids = TOK.encode("<|im_start|>system\n" + SYSTEM_A + "<|im_end|>\n<|im_start|>", parse_special=True)
        self.svc.trie.warm(ids, "trie_t_node_x.bin")
        self.assertEqual(self.svc.embeddings.path, "/tmp/the-requests-images.bin")
        self.assertEqual(self.svc.request_trace.record, {"marker": 1})
        self.assertEqual(self.engine.reads[-1], (len(ids), 0))
