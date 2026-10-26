"""serve/test_slots.py - llama-server compatible /slots/0?action=save|restore over the engine's SAVE/RESTORE lines
(mock engine: no GPU, no pack).

    python -m unittest serve.test_slots -v
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, Service, StrataEngine, serve, slot_filename_problem,  # noqa: E402
                          slot_save_dir)

ROOT = Path(__file__).resolve().parents[1]


class FakeProc:
    """Stands in for the engine process: answers SAVE / RESTORE lines like `strata --serve` does."""

    def __init__(self, engine):
        self.engine = engine
        self.stdin = self
        self.sent: list[str] = []

    def write(self, s):
        self.sent.append(s)
        cmd, _, path = s.strip().partition(" ")
        script = self.engine.script
        if script is not None:                      # a scripted answer: a list of lines, None = the process ends
            # played by its own thread, like the real engine's output: write() returns at once, so the server's
            # wait (and its deadline) is already running when a delayed line arrives
            def play():
                for line in script:
                    if self.killed:
                        return
                    if isinstance(line, float):
                        time.sleep(line)
                    else:
                        self.engine.lines.put(line)
            self.player = threading.Thread(target=play, daemon=True)
            self.player.start()
            return
        if cmd == "SAVE":
            if self.engine.fail:
                self.engine.lines.put("SERR invalid 0 no complete session to save\n")
            else:
                Path(path).write_bytes(b"x" * 1234)
                self.engine.lines.put("SAVED 62993 1234 401.5\n")
        elif cmd == "RESTORE":
            if not Path(path).exists() or self.engine.fail:
                self.engine.lines.put("SERR invalid 0 session file: saved with another model (model fingerprint differs)\n")
            else:
                self.engine.lines.put("RESTORED 62993 1234 560.2\n")
        elif cmd == "DIE":
            self.engine.lines.put(None)

    def flush(self):
        pass

    def poll(self):
        return -9 if self.killed else None

    killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


class FakeEngine(StrataEngine):
    def __init__(self):
        super().__init__("strata", ["--max-context", "65536"], lazy=True)
        self.max_context = 65536
        self.unloaded, self.ended = False, False
        self.lines = queue.Queue()
        self.fail = False
        self.script = None
        self.proc = FakeProc(self)


class SlotServer(unittest.TestCase):
    """The server over a FakeEngine, its slot save path a temporary folder."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.engine = FakeEngine()
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def setUp(self):
        self.engine.fail = False
        self.engine.script = None
        self.engine.silence_s = 300.0
        self.fresh_engine()
        self.svc.slot_save_path = self.dir.name

    def fresh_engine(self):
        self.engine.ended = False
        old = getattr(self.engine.proc, "player", None)
        if old is not None:                          # the previous scripted engine has finished...
            self.engine.proc.killed = True
            old.join(timeout=5)
        while not self.engine.lines.empty():         # ...and left no line for the next exchange
            self.engine.lines.get_nowait()
        self.engine.proc = FakeProc(self.engine)

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())


class Slots(SlotServer):
    def test_a_foreign_page_or_a_form_never_reaches_the_engine(self):
        for action in ("save", "restore"):
            for headers, code in (({"Content-Type": "application/json", "Origin": "http://evil.example.com"}, 403),
                                  ({"Content-Type": "text/plain"}, 415),
                                  ({"Content-Type": "application/x-www-form-urlencoded"}, 415)):
                s, b = self.post(f"/slots/0?action={action}", {"filename": "x.bin"}, headers)
                self.assertEqual(s, code, (action, headers, b))
        self.assertEqual(self.engine.proc.sent, [])
        self.assertFalse(Path(self.dir.name, "x.bin").exists())

    def test_save_writes_into_the_slot_save_path(self):
        s, b = self.post("/slots/0?action=save", {"filename": "a.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.proc.sent, [f"SAVE {Path(self.dir.name, 'a.bin')}\n"])
        self.assertEqual((b["id_slot"], b["filename"], b["n_saved"], b["n_written"]), (0, "a.bin", 62993, 1234))
        self.assertAlmostEqual(b["timings"]["save_ms"], 401.5)

    def test_restore_answers_like_llama_server(self):
        self.post("/slots/0?action=save", {"filename": "b.bin"})
        s, b = self.post("/slots/0?action=restore", {"filename": "b.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual((b["id_slot"], b["filename"], b["n_restored"], b["n_read"]), (0, "b.bin", 62993, 1234))
        self.assertAlmostEqual(b["timings"]["restore_ms"], 560.2)

    def test_refused_file_is_an_error_not_a_success(self):
        self.post("/slots/0?action=save", {"filename": "c.bin"})
        self.engine.fail = True
        s, b = self.post("/slots/0?action=restore", {"filename": "c.bin"})
        self.assertEqual(s, 400, b)
        self.assertIn("fingerprint", b["error"]["message"])

    def test_missing_file_is_not_found(self):
        s, b = self.post("/slots/0?action=restore", {"filename": "never-saved.bin"})
        self.assertEqual(s, 404, b)
        self.assertEqual(self.engine.proc.sent, [])

    def test_filename_cannot_leave_the_directory(self):
        for bad in ("../x.bin", "/etc/passwd", "a/b.bin", "", ".", "..", "x\ny.bin",
                    # Windows: another drive, a stream, devices with and without an extension, names Windows trims
                    "D:escape.bin", "C:\\x.bin", "a.bin:stream", "NUL", "nul.txt", "CON.bin", "com1.x", "LPT9",
                    "a.bin.", "a.bin ", ".hidden", ".a.bin.0123456789abcdef.tmp", "x\x7fy", "a*b", "a?b", 'a"b',
                    "a<b", "a>b", "a|b", "x" * 201, None, 5, ["a"]):
            s, b = self.post("/slots/0?action=save", {"filename": bad})
            self.assertEqual(s, 400, (bad, b))
        self.assertEqual(self.engine.proc.sent, [])

    def test_ordinary_names_are_accepted(self):
        for good in ("a.bin", "chat-2026-10-03.bin", "a.bin.tmp", "consul.bin", "nullable", "caf\u00e9.bin",
                     "\u4f1a\u8bdd.bin", "x" * 200):
            self.assertIsNone(slot_filename_problem(good), good)

    def test_relative_save_dir_becomes_absolute(self):
        with tempfile.TemporaryDirectory() as base:
            got = slot_save_dir("sessions", base)
            self.assertEqual(got, os.path.join(base, "sessions"))
            self.assertTrue(os.path.isabs(got) and os.path.isdir(got))
            if os.name != "nt":
                self.assertEqual(os.stat(got).st_mode & 0o777, 0o700)
            self.assertEqual(slot_save_dir(got, "/elsewhere"), got)     # an absolute path stays as it is
            cwd = os.getcwd()
            try:
                os.chdir(base)
                self.assertEqual(slot_save_dir("rel"), os.path.join(os.path.realpath(base), "rel"))
            finally:
                os.chdir(cwd)
            for bad in ("", "  ", None, 3, "a\nb"):
                with self.assertRaises(ValueError):
                    slot_save_dir(bad, base)

    def test_engine_gets_an_absolute_path_on_one_line(self):
        self.post("/slots/0?action=save", {"filename": "abs.bin"})
        line = self.engine.proc.sent[0]
        self.assertTrue(line.endswith("\n") and line.count("\n") == 1)
        self.assertTrue(os.path.isabs(line[len("SAVE "):-1]))

    def test_restore_of_a_link_reaches_the_engine_which_decides(self):
        """The server only checks the name exists (lstat); the engine opens it without following a link."""
        if os.name == "nt":
            self.skipTest("symlinks need privileges on Windows")
        target = Path(self.dir.name, "real.bin")
        target.write_bytes(b"x")
        link = Path(self.dir.name, "link.bin")
        link.symlink_to(target)
        dangling = Path(self.dir.name, "dangling.bin")
        dangling.symlink_to(Path(self.dir.name, "nothing"))
        self.engine.script = ["SERR invalid 0 session file: x: a symbolic link, not a file\n"]
        for name in ("link.bin", "dangling.bin"):
            s, b = self.post("/slots/0?action=restore", {"filename": name})
            self.assertEqual(s, 400, b)
            self.assertIn("symbolic link", b["error"]["message"])

    def test_progress_lines_keep_a_long_save_alive(self):
        self.engine.silence_s = 0.3
        self.engine.script = [0.2, "SESSION 268435456 1198691396\n", 0.2, "SESSION 536870912 1198691396\n", 0.2,
                              "SAVED 63000 1198691396 2000.0\n"]
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "long.bin"})
        self.assertEqual(s, 200, b)
        self.assertGreater(time.monotonic() - t0, 0.5)       # the lines arrived during the wait, past its 0.3 s
        self.assertEqual(b["n_written"], 1198691396)
        self.assertFalse(self.engine.ended)

    def test_a_small_save_reports_its_only_partial_block(self):
        """B: a file below one 16 MiB block still reports (the start, then its last partial block)."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SESSION 0 56834\n", 0.2, "SESSION 56834 56834\n", 0.2, "SAVED 120 56834 410.0\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "small.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual(b["n_written"], 56834)
        self.assertFalse(self.engine.ended)

    def test_a_blocking_step_gets_its_announced_allowance(self):
        """B: SWAIT gives one non-streaming step (a flush, the device transfer) its own bounded wait."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SESSION 1198691396 1198691396\n", "SWAIT flush 1\n", 0.7, "SWAIT publish 1\n", 0.7,
                              "SAVED 63000 1198691396 2000.0\n"]
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "flush.bin"})
        self.assertEqual(s, 200, b)
        self.assertGreater(time.monotonic() - t0, 1.3)       # longer than the silence limit, twice
        self.assertFalse(self.engine.ended)
        Path(self.dir.name, "flush.bin").write_bytes(b"x")    # the scripted engine wrote nothing; restore needs a file
        self.engine.script = ["SESSION 16777216 1198691396\n", 0.2, "SWAIT validate 1\n", 0.7, "SWAIT transfer 1\n", 0.7,
                              "RESTORED 63000 1198691396 900.0\n"]
        s, b = self.post("/slots/0?action=restore", {"filename": "flush.bin"})
        self.assertEqual(s, 200, b)

    def test_a_blocking_step_that_never_ends_is_ended(self):
        """B: the allowance is a bound, not a heartbeat: a step that outlives it ends the engine."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SWAIT flush 1\n"]               # then nothing
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "stuck.bin"})
        took = time.monotonic() - t0
        self.assertEqual(s, 500, b)
        self.assertGreater(took, 0.9)
        self.assertLess(took, 5)
        self.assertIn("blocking step allowed 1 s", b["error"]["message"])
        self.assertTrue(self.engine.ended and self.engine.proc.killed)

    def test_progress_ends_the_allowance(self):
        """B: after a SESSION line the ordinary silence limit applies again (the allowance was for one step)."""
        self.engine.silence_s = 0.3
        self.engine.script = ["SWAIT transfer 2\n", "SESSION 1 2\n", 0.8, "SAVED 1 2 1.0\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "reset.bin"})
        self.assertEqual(s, 500, b)
        self.assertIn("said nothing", b["error"]["message"])
        self.assertNotIn("blocking step", b["error"]["message"])
        self.assertTrue(self.engine.ended)

    def test_malformed_allowances_end_the_engine(self):
        for script in (["SWAIT flush 0\n"], ["SWAIT flush 3601\n"], ["SWAIT 5\n"], ["SWAIT flush x\n"],
                       ["SWAIT flush -1\n"], ["SWAIT fl:sh 5\n"], ["SWAIT flush 5 extra\n"]):
            with self.subTest(script=script):
                self.fresh_engine()
                self.engine.script = script + [0.1, "SAVED 1 2 1.0\n"]
                s, b = self.post("/slots/0?action=save", {"filename": "bad.bin"})
                self.assertEqual(s, 500, b)
                self.assertTrue(self.engine.ended and self.engine.proc.killed)

    def test_a_silent_engine_is_ended_not_waited_for(self):
        self.engine.silence_s = 0.3
        self.engine.script = []                      # the engine never answers
        t0 = time.monotonic()
        s, b = self.post("/slots/0?action=save", {"filename": "silent.bin"})
        self.assertEqual(s, 500, b)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertIn("said nothing", b["error"]["message"])
        self.assertTrue(self.engine.ended and self.engine.proc.killed)
        self.assertTrue(self.svc.fifo.acquire(blocking=False))   # the FIFO was released
        self.svc.fifo.release()
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.status["busy"])

    def test_engine_death_is_a_server_error(self):
        self.engine.script = [None]
        s, b = self.post("/slots/0?action=save", {"filename": "dead.bin"})
        self.assertEqual(s, 500, b)

    def test_failed_restore_transfer_is_a_server_error(self):
        Path(self.dir.name, "t.bin").write_bytes(b"x")
        self.engine.script = ["SESSION 268435456 1198691396\n",
                              "FATAL restoring the session file failed after the device state was changed: copy\n"]
        s, b = self.post("/slots/0?action=restore", {"filename": "t.bin"})
        self.assertEqual(s, 500, b)
        self.assertIn("device state", b["error"]["message"])

    def test_malformed_or_mismatched_reply_is_a_server_error(self):
        for script in (["RESTORED 1 2 3\n"], ["SAVED x y z\n"], ["SAVED 1 2\n"], ["T 5\n"]):
            self.engine.script = script
            self.engine.ended = False
            self.engine.proc = FakeProc(self.engine)
            s, b = self.post("/slots/0?action=save", {"filename": "m.bin"})
            self.assertEqual(s, 500, (script, b))
            self.assertTrue(self.engine.ended, script)           # out of step: the engine is ended

    def test_out_of_protocol_lines_end_the_engine(self):
        """R4: a line this exchange does not allow is not a heartbeat, and an answer must hold valid numbers."""
        for script in (["SESSION nonsense\n"], ["SESSION 5\n"], ["SESSION -1 10\n"], ["SESSION 11 10\n"],
                       ["SESSION 8 10\n", "SESSION 4 10\n"],      # progress going back
                       ["SAVED -1 -2 nan\n"], ["SAVED 1 2 nan\n"], ["SAVED 1 2 inf\n"], ["SAVED 1 2 -5\n"],
                       ["SAVED +1 2 3\n"], ["ERR bare refusal\n"], ["SERR weird 0 x\n"], ["SERR io 2 x\n"],
                       ["SERR io\n"], ["garbage\n"]):
            self.engine.script = script
            self.engine.ended = False
            self.engine.proc = FakeProc(self.engine)
            s, b = self.post("/slots/0?action=save", {"filename": "p.bin"})
            self.assertEqual(s, 500, (script, b))
            self.assertTrue(self.engine.ended and self.engine.proc.killed, script)
            json.dumps(b, allow_nan=False)                        # standard JSON

    def test_log_lines_are_not_answers(self):
        self.engine.script = ["INFO something\n", "SAVED 1 2 3.0\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "i.bin"})
        self.assertEqual(s, 200, b)
        self.assertFalse(self.engine.ended)

    def test_typed_refusals_have_their_own_codes(self):
        """R5: the engine's category decides the status, never the words of the message."""
        cases = [("storage", "No space left on device", 507), ("storage", "espace insuffisant", 507),
                 ("memory", "out of memory for the write buffer", 503), ("invalid", "not enough disk space", 400),
                 ("invalid", "saved with another model", 400), ("io", "Permission denied", 500),
                 ("io", "RAM", 500)]
        for kind, msg, code in cases:
            self.engine.script = [f"SERR {kind} 0 {msg}\n"]
            s, b = self.post("/slots/0?action=save", {"filename": "k.bin"})
            self.assertEqual(s, code, (kind, msg, b))
            self.assertEqual(b["error"]["kind"], kind)
            self.assertEqual(b["error"]["message"], msg)
            self.assertNotIn("published", b["error"])
            self.assertFalse(self.engine.ended)                   # a refusal keeps the engine

    def test_published_failure_is_reported(self):
        """R3: the folder flush failed after the rename - the new file replaced the old one; the answer says so."""
        self.engine.script = ["SERR io 1 flushing the directory: Input/output error (the new file has already "
                              "replaced the old one)\n"]
        s, b = self.post("/slots/0?action=save", {"filename": "pub.bin"})
        self.assertEqual(s, 500, b)
        self.assertTrue(b["error"]["published"])
        self.assertIn("replaced", b["error"]["message"])

    def test_disk_and_ram_refusals_have_their_own_codes(self):
        self.engine.script = ["SERR storage 0 session file: not enough disk space (1143 MiB plus a reserve of 4096 "
                              "MiB needed, 10 MiB free)\n"]
        self.assertEqual(self.post("/slots/0?action=save", {"filename": "d.bin"})[0], 507)
        Path(self.dir.name, "r.bin").write_bytes(b"x")
        self.engine.script = ["SERR memory 0 session file: not enough RAM to read it (1143 MiB plus a floor of 2560 "
                              "MiB needed, 900 MiB available)\n"]
        self.assertEqual(self.post("/slots/0?action=restore", {"filename": "r.bin"})[0], 503)

    def test_slot_work_counts_as_activity_and_shows_in_status(self):
        seen = {}

        def during():
            time.sleep(0.15)
            with self.svc.status_lock:
                seen.update(self.svc.status)
        self.engine.script = [0.4, "SAVED 1 2 3.0\n"]
        self.svc.last_request_at = 0
        t = threading.Thread(target=during)
        t.start()
        s, _ = self.post("/slots/0?action=save", {"filename": "act.bin"})
        t.join()
        self.assertEqual(s, 200)
        self.assertTrue(seen.get("busy"))
        self.assertEqual(seen.get("phase"), "saving the session")
        self.assertGreater(self.svc.last_request_at, time.time() - 5)
        self.assertFalse(self.svc.status["busy"])
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertEqual(self.svc.unload(idle_for=60), "busy")        # just active: the idle unload waits

    def test_only_slot_zero_and_known_actions(self):
        self.assertEqual(self.post("/slots/1?action=save", {"filename": "d.bin"})[0], 400)
        self.assertEqual(self.post("/slots/0?action=erase_all", {"filename": "d.bin"})[0], 400)
        self.assertEqual(self.engine.proc.sent, [])

    def test_disabled_without_a_slot_save_path(self):
        saved = self.svc.slot_save_path
        self.svc.slot_save_path = None
        try:
            s, b = self.post("/slots/0?action=save", {"filename": "e.bin"})
        finally:
            self.svc.slot_save_path = saved
        self.assertEqual(s, 501, b)

    def test_refused_with_parallel_requests(self):
        """#465: batch slots do not hold the FIFO, so a session file is refused before anything reaches the engine."""
        self.engine.batch = 2
        try:
            s, b = self.post("/slots/0?action=save", {"filename": "p.bin"})
        finally:
            self.engine.batch = 0
        self.assertEqual(s, 501, b)
        self.assertIn("parallel", b["error"]["message"])
        self.assertEqual(self.engine.proc.sent, [])

    def test_waits_for_a_running_request(self):
        """A save never interleaves with a generation: it takes the same FIFO."""
        self.svc.fifo.acquire()
        out = {}
        t = threading.Thread(target=lambda: out.update(r=self.post("/slots/0?action=save", {"filename": "f.bin"})))
        t.start()
        t.join(0.5)
        self.assertTrue(t.is_alive())
        self.assertEqual(self.engine.proc.sent, [])
        self.svc.fifo.release()
        t.join(10)
        self.assertEqual(out["r"][0], 200)



class AnsweringEngine(FakeEngine):
    """FakeEngine that also answers requests ("ok", then the end of the turn) and keeps their prompts."""

    def __init__(self):
        super().__init__()
        self.prompts: list[list[int]] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        tok = ByteTokenizer()
        yield from (tok.encode("ok") + tok.encode("<|im_end|>", parse_special=True))[:max_new]


class SlotOwners(SlotServer):
    """llama-server's slot_snapshot (pi-kv-cache-manager): whose the live conversation is, saved for its owner before
    another conversation replaces it; erase; /apply-template, /completion and /lora-adapters for the prompt cache."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.tok = ByteTokenizer()
        cls.engine = AnsweringEngine()
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.slot_save_path = cls.dir.name
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    def setUp(self):
        super().setUp()
        self.svc.slot_owner, self.svc.slot_dirty, self.svc.slot_proc = "", False, None
        self.svc.effort_end = False
        self.engine.prompts.clear()

    def chat(self, owner=None, **extra):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **extra}
        if owner is not None:
            body["slot_snapshot"] = owner
        return self.post("/v1/chat/completions", body)

    def saves(self):
        return [line for line in self.engine.proc.sent if line.startswith("SAVE ")]

    def slot(self):
        with urllib.request.urlopen(self.base + "/slots", timeout=10) as r:
            return json.loads(r.read())[0]

    def save_line(self, name):
        return f"SAVE {Path(self.dir.name, name)}\n"

    def test_the_owner_is_saved_before_an_unowned_request(self):
        self.assertEqual(self.chat("a.bin")[0], 200)
        self.assertEqual(self.saves(), [])                     # nobody owned what was there before
        self.assertEqual(self.slot()["snapshot_filename"], "a.bin")
        self.assertGreater(self.slot()["t_last_used"], 0)
        self.assertEqual(self.chat()[0], 200)                  # a planner call without a session
        self.assertEqual(self.saves(), [self.save_line("a.bin")])
        self.assertEqual(self.slot()["snapshot_filename"], "")
        self.assertEqual(self.chat()[0], 200)                  # unowned after unowned: nothing to keep
        self.assertEqual(len(self.saves()), 1)

    def test_the_same_owner_continues_without_a_save(self):
        for _ in range(3):
            self.assertEqual(self.chat("a.bin")[0], 200)
        self.assertEqual(self.saves(), [])

    def test_another_owner_saves_the_first_one(self):
        self.chat("a.bin")
        self.chat("b.bin")
        self.assertEqual(self.saves(), [self.save_line("a.bin")])
        self.assertEqual(self.slot()["snapshot_filename"], "b.bin")

    def test_a_saved_conversation_is_not_saved_again(self):
        self.chat("a.bin")
        self.assertEqual(self.post("/slots/0?action=save", {"filename": "a.bin"})[0], 200)
        self.chat("b.bin")
        self.assertEqual(self.saves(), [self.save_line("a.bin")])   # only the explicit save

    def test_a_restore_saves_the_outgoing_owner_first(self):
        Path(self.dir.name, "b.bin").write_bytes(b"x")
        self.chat("a.bin")
        s, b = self.post("/slots/0?action=restore", {"filename": "b.bin"})
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.proc.sent, [self.save_line("a.bin"), f"RESTORE {Path(self.dir.name, 'b.bin')}\n"])
        self.assertEqual(self.slot()["snapshot_filename"], "b.bin")
        self.chat("b.bin")                                     # continues the restored file: nothing to save
        self.assertEqual(len(self.saves()), 1)

    def test_a_refused_save_does_not_fail_the_next_request(self):
        self.chat("a.bin")
        self.engine.fail = True
        s, b = self.chat("b.bin")
        self.assertEqual(s, 200, b)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_erase_saves_a_changed_owner_and_forgets_it(self):
        self.chat("a.bin")
        s, b = self.post("/slots/0?action=erase", {})
        self.assertEqual((s, b), (200, {"id_slot": 0, "n_erased": 0}))
        self.assertEqual(self.saves(), [self.save_line("a.bin")])
        self.assertEqual(self.slot()["snapshot_filename"], "")
        self.svc.fifo.acquire()                                # nothing to save: no wait for a running request
        try:
            self.assertEqual(self.post("/slots/0?action=erase", {})[0], 200)
        finally:
            self.svc.fifo.release()
        self.assertEqual(len(self.saves()), 1)

    def test_a_restarted_engine_holds_no_owner(self):
        self.chat("a.bin")
        self.fresh_engine()                                    # a new engine process
        self.assertEqual(self.slot()["snapshot_filename"], "")
        self.chat("b.bin")
        self.assertEqual(self.saves(), [])

    def test_a_bad_slot_snapshot_is_refused(self):
        for name in ("../a.bin", "a/b", ".hidden", 5):
            s, b = self.chat(name)
            self.assertEqual(s, 400, (name, b))
            self.assertIn("slot_snapshot", b["error"]["message"])
        self.assertEqual(self.engine.prompts, [])

    def test_apply_template_is_the_start_of_the_chat_prompt(self):
        system = [{"role": "system", "content": "You are pi. A literal <think> stays text."}]
        tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        for effort_end in (False, True):
            self.svc.effort_end = effort_end
            for effort in ("xhigh", "low", "medium", "none"):
                s, t = self.post("/apply-template", {"messages": system, "tools": tools, "reasoning_effort": effort,
                                                     "add_generation_prompt": False})
                self.assertEqual(s, 200, t)
                self.assertTrue(t["prompt"].endswith("<|im_end|>\n"))
                self.engine.prompts.clear()
                self.chat(messages=system + [{"role": "user", "content": "hi"}], tools=tools, reasoning_effort=effort)
                self.assertEqual(self.engine.prompts[0][:len(t["tokens"])], t["tokens"], (effort_end, effort))
                if effort_end:                                 # one prefix for every effort
                    s, x = self.post("/apply-template", {"messages": system, "tools": tools,
                                                         "add_generation_prompt": False})
                    self.assertEqual(x["tokens"], t["tokens"])
        s, t = self.post("/apply-template", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(t["prompt"].endswith("<|im_start|>assistant\n<think>\n"))   # the chat path's prompt

    def test_completion_can_only_read_the_prompt(self):
        ids = self.tok.encode("<|im_start|>system\nhello<|im_end|>\n", parse_special=True)
        s, b = self.post("/completion", {"prompt": ids, "n_predict": 0, "id_slot": 0, "cache_prompt": True})
        self.assertEqual(s, 200, b)
        self.assertEqual((b["content"], b["tokens_evaluated"], b["tokens_predicted"]), ("", len(ids), 0))
        self.assertEqual(self.engine.prompts, [ids])
        s, b = self.post("/completion", {"prompt": "<|im_start|>system\nhello<|im_end|>\n", "n_predict": 4})
        self.assertEqual((s, b["content"]), (200, "ok"))
        self.assertEqual(self.engine.prompts[1], ids)          # the text is read with its control tokens
        for bad in ({"prompt": ""}, {"prompt": [1, -2]}, {"prompt": "x", "n_predict": "0"}, {"prompt": "x", "stream": True},
                    {"prompt": "x", "slot_snapshot": "../a"}):
            self.assertEqual(self.post("/completion", bad)[0], 400, bad)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_no_lora_adapters(self):
        with urllib.request.urlopen(self.base + "/lora-adapters", timeout=10) as r:
            self.assertEqual(json.loads(r.read()), [])


if __name__ == "__main__":
    unittest.main()
