"""serve/trie_kv.py - a passive prefix-tree KV cache for the engine's one slot (--trie-kv, or "trie_kv" in the config).

It was strata-trie-kv, a proxy in front of this server; it now runs inside it, on the chat path. No session ids:
every POST /v1/chat/completions without a slot_snapshot of its own is matched against a prefix tree of what is on
disk, and the engine is given the deepest saved state that request can continue from:

- the tree holds only token ids (4 bytes a token, read from the session files' heads), never the K/V;
- a node is a point where a session file holds a full running state: its end, or its checkpoint. The models are
  hybrid (Gated DeltaNet), so a state exists only where one was saved, not at every common prefix;
- leaves come for free: each request is given a file of its own (slot_snapshot) and the server saves the
  conversation there before anything else replaces it (Service.slot_handover, save-before-evict);
- the golden base is the root of each client: the first time a system turn (system prompt, tools, skill list) is
  read, it is read on its own (a warm, as POST /completion with n_predict 0) and saved as a node before the request
  goes on, so every later conversation of that client resumes there. A conversation's file cannot stand in: the
  engine saves only its deepest checkpoint, and a Gated DeltaNet state cannot be cut back to an earlier point;
- inner nodes are made passively: when a request leaves a path another request already took, at a turn boundary
  no file holds, the shared part is read on its own and saved as a node;
- files are dropped least recently used first once the folder passes its quota.

Inside the server (rather than in front of it) the prompt is tokenized once (the chat path's own ids), the engine's
reuse comes from its DONE line instead of its log, a restore or save is a call instead of an HTTP round trip, and
the cache keeps its files across changes that do not change the session files: the namespace is the model and the
engine's arguments, not the run config's bytes (a new STRATA_* variable or engine binary keeps the cache; the
engine refuses a file of another model, and the cache then stops offering it).

    python3 serve/trie_kv.py --tree              # the snapshot tree as it stands
    python3 serve/trie_kv.py --dry-run-quota     # what the quota would delete now
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import threading
import time
import uuid

import numpy as np

STRATA_MAGIC = b"STRSESS\x01"
LIVE_IDS_AT = 64 + 18 * 8 + 3 * 8          # header, geometry[18], layer range + cvec: the live checkpoint's ids
VISION_START, IMAGE_PAD = 248053, 248056    # an image: <|vision_start|> + one <|image_pad|> per image token
DECISIONS = "trie-kv.jsonl"                 # one line per request, in the slot save folder


def lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    d = np.flatnonzero(a[:n] != b[:n])
    return int(d[0]) if len(d) else n


def boundary(ids: np.ndarray, turn_token: int, tail_role: int, after: int = 0) -> int:
    """Where the engine checkpoints a prompt (generate.cpp turn_at): its last turn boundary past `after`, moved to the
    turn before it when that one is a trailing effort turn (--tail-role-token). -1: none."""
    turns = np.flatnonzero(ids == turn_token)
    turns = turns[turns > after]
    if not len(turns):
        return -1
    at = int(turns[-1])
    if tail_role >= 0 and len(turns) > 1 and int(turns[-2]) + 1 < len(ids) and ids[int(turns[-2]) + 1] == tail_role:
        at = int(turns[-2])
    return at


def file_end(ids: np.ndarray, points, turn_token: int) -> int:
    """How far a file's conversation goes: its deepest resume point on a turn boundary (a leaf's checkpoint), or its
    end when none is (a node: it ends where the next turn would open)."""
    on_turn = [p for p in points if p < len(ids) and ids[p] == turn_token]
    return max(on_turn) if on_turn else len(ids)


def branch_point(ids: np.ndarray, paths, turn_token: int) -> int:
    """Where this prompt leaves another path - (tokens, how far that conversation goes) - that goes on with a turn of
    its own past that point: the last turn boundary they share. Not where it simply continues a conversation that
    ended there (the next turn of a chat). -1: no branch."""
    cut = -1
    for other, end in paths:
        c = lcp(other, ids)
        turns = np.flatnonzero(ids[: c + 1] == turn_token)
        if len(turns) and int(turns[-1]) < end:
            cut = max(cut, int(turns[-1]))
    return cut


def first_turn(ids: np.ndarray, turn_token: int) -> int:
    """Where the prompt's first turn ends: for a chat, its system turn (system prompt, tools, skill list), the golden
    base. -1: none."""
    turns = np.flatnonzero(ids == turn_token)
    return next((int(i) for i in turns if i >= 1), -1)


def node_cuts(ids: np.ndarray, start: int, r_disk: int, branch: int, turn_token: int, root_min: int,
              split_min: int) -> list[tuple[int, str]]:
    """Where to save a node before this request runs, as (position, kind), shallowest first:
    - "golden": the end of the system turn when no file holds a state there and the engine will not resume past it
      (a prompt seen for the first time, or whose files all went). It is read anyway; saved, every later conversation
      of that client resumes there, also once another client has had the engine. A conversation's own file cannot
      stand in: Strata saves only its deepest checkpoint, and a Gated DeltaNet state cannot be cut back;
    - "branch": where this prompt leaves another path at a turn boundary no file holds.
    Each is root_min long, split_min past the deepest saved state, short of the prompt's end, and has no image before
    it (/completion takes none)."""
    out, have = [], max(r_disk, 0)
    for at, kind in ((first_turn(ids, turn_token), "golden"), (branch, "branch")):
        if kind == "golden" and start > at:
            continue                                     # the engine goes on past it: a conversation's next turn
        if (at >= root_min and at - have >= split_min and at < len(ids) - 1
                and not (ids[: at + 1] == IMAGE_PAD).any()):
            out.append((at, kind))
            have = at
    return out


def collapse(ids: np.ndarray, points) -> tuple[np.ndarray, list[int]]:
    """The engine's tokens -> the template's: an image is one <|image_pad|> in /apply-template but one per image
    token in the engine (serve/server.py prepare), so a saved file is folded the same way before it is compared, and
    its resume points move with it (one inside an image has no state and is dropped)."""
    pad = ids == IMAGE_PAD
    extra = pad & np.concatenate([[False], pad[:-1]])
    if not extra.any():
        return ids, list(points)
    before = np.concatenate([[0], np.cumsum(extra)])     # before[p]: extra pads among ids[:p]
    moved = [int(p - before[p]) for p in points if not (p < len(ids) and extra[p])]
    return ids[~extra], moved


def to_template(reused: int, n_engine: int, ids: np.ndarray) -> int | None:
    """An engine position of this prompt -> its /apply-template position: the same before the first image, minus
    every image's extra tokens past the last one; None in between (each image's size is not known here)."""
    extra = n_engine - len(ids)
    if extra <= 0:
        return reused
    pads = np.flatnonzero(ids == IMAGE_PAD)
    if not len(pads) or reused <= int(pads[0]):
        return reused
    if reused >= int(pads[-1]) + 1 + extra:
        return reused - extra
    return None


def read_points(path: str) -> tuple[np.ndarray, list[int]]:
    """A Strata session file's tokens and the prefix lengths it can resume at (its end, its checkpoints), read
    without its K/V: the ids sit at fixed offsets, the states between them are skipped by seeking."""
    with open(path, "rb") as f:
        if f.read(8) != STRATA_MAGIC:
            raise ValueError("not a Strata session file")
        f.seek(LIVE_IDS_AT)

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def checkpoint():
            ids = np.frombuffer(f.read(4 * u64()), "<i4").astype(np.int32)
            f.seek(16 * u64(), 1)                        # image keys
            for _ in range(5):                           # gdn, ple, tails, dead, block_pos
                f.seek(u64(), 1)
            u64()                                        # LRU stamp
            return ids
        live = checkpoint()
        points = {len(live)}
        for _ in range(u64()):
            points.add(len(checkpoint()))
    return live, sorted(points)


class Trie:
    """A radix tree over token ids. A path's resume points sit on the node whose edge holds them."""

    class Node:
        __slots__ = ("edge", "kids", "depth", "points")

        def __init__(self, edge, depth):
            self.edge, self.kids, self.depth, self.points = edge, {}, depth, []

    def __init__(self):
        self.root = self.Node(np.empty(0, np.int32), 0)

    def insert(self, ids: np.ndarray, points=(), ref=None):
        node, i, path = self.root, 0, [self.root]
        while i < len(ids):
            kid = node.kids.get(int(ids[i]))
            if kid is None:
                kid = self.Node(ids[i:], i)
                node.kids[int(ids[i])] = kid
                path.append(kid)
                break
            k = lcp(kid.edge, ids[i:])
            if k < len(kid.edge):                        # split the edge at the common prefix: a new inner node
                mid = self.Node(kid.edge[:k], i)
                kid.edge, kid.depth = kid.edge[k:], i + k
                moved = [(r, p) for r, p in kid.points if p <= i + k]
                kid.points = [(r, p) for r, p in kid.points if p > i + k]
                mid.points = moved
                mid.kids[int(kid.edge[0])] = kid
                node.kids[int(ids[i])] = mid
                kid = mid
            node, i = kid, i + k
            path.append(node)
        for p in points:
            holder = next((n for n in path if n.depth < p <= n.depth + len(n.edge)), self.root)
            holder.points.append((ref, p))

    def match(self, ids: np.ndarray):
        """-> (deepest resume point any path offers this prompt: (ref, length) or None, the longest common prefix
        with any path)."""
        node, i, best = self.root, 0, None
        while True:
            reach = i if node is self.root else node.depth + len(node.edge)
            for r, p in node.points:
                if p <= reach and (best is None or p > best[1]):
                    best = (r, p)
            if i >= len(ids):
                return best, i
            kid = node.kids.get(int(ids[i]))
            if kid is None:
                return best, i
            k = lcp(kid.edge, ids[i:])
            if k < len(kid.edge):
                for r, p in kid.points:
                    if p <= i + k and (best is None or p > best[1]):
                        best = (r, p)
                return best, i + k
            node, i = kid, i + k


LEGACY = ("golden_", "kv2_")       # pi-kv-cache-manager's files (uninstalled 2026-10-31): nothing reads them now


class Store:
    """The snapshot files this cache owns (trie_<ns>_*.bin) and when each was last used."""

    def __init__(self, folder: str, ns: str, max_bytes: int):
        self.folder, self.ns, self.max_bytes = folder, ns, max_bytes
        self.prefix = f"trie_{ns}_"
        self.files: dict[str, dict] = {}                 # name -> {ids, points, size, mtime, kind}
        self.state_path = os.path.join(folder, f"{self.prefix}state.json")
        try:
            self.used = json.load(open(self.state_path))
        except (OSError, ValueError):
            self.used = {}

    def refresh(self):
        seen = set()
        for e in os.scandir(self.folder):
            if not (e.name.startswith(self.prefix) and e.name.endswith(".bin")):
                continue
            st = e.stat()
            seen.add(e.name)
            old = self.files.get(e.name)
            if old and old["size"] == st.st_size and old["mtime"] == st.st_mtime:
                continue
            try:
                ids, points = collapse(*read_points(e.path))
            except (OSError, ValueError, struct.error):
                continue                                 # being written, or not ours to read
            self.files[e.name] = dict(ids=ids, points=points, size=st.st_size, mtime=st.st_mtime,
                                      kind="node" if e.name.startswith(self.prefix + "node_") else "leaf")
            self.used.setdefault(e.name, st.st_mtime)
        for gone in set(self.files) - seen:
            del self.files[gone]

    def touch(self, name: str):
        self.used[name] = time.time()
        self.save_state()

    def save_state(self):
        live = {k: v for k, v in self.used.items() if k in self.files}
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(live, f)
        os.replace(tmp, self.state_path)

    def others(self) -> list[tuple[int, float, list[str], int]]:
        """The folder's other snapshots, each with its sidecars, as (order, mtime, files, bytes): another run config's
        files of ours (the engine refuses them) first, then pi-kv-cache-manager's leftovers."""
        names = os.listdir(self.folder)
        out = []
        for name in names:
            if not name.endswith(".bin") or name.startswith(self.prefix):
                continue
            if name.startswith("trie_"):
                order, files = 0, [name]
            elif name.startswith(LEGACY):
                stem = name[:-4]               # .bin.ckpt, .bin.dft, .bin.media.json, .meta.json
                order, files = 1, [name] + [n for n in names if n != name and n.startswith(stem + ".")]
            else:
                continue
            try:
                stats = [os.stat(os.path.join(self.folder, n)) for n in files]
            except FileNotFoundError:
                continue
            out.append((order, os.stat(os.path.join(self.folder, name)).st_mtime, files, sum(x.st_size for x in stats)))
        return sorted(out)

    def last_used(self) -> dict[str, float]:
        """When each file was last used. A node counts as used whenever a file that resumes from it was: the node of a
        system prompt still in use lives as long as its conversations, and one nobody reaches any more (an old system
        prompt) goes with its own."""
        last = {n: self.used.get(n, 0) for n in self.files}
        for n, f in self.files.items():
            if f["kind"] != "node":
                continue
            at = min(f["points"])
            for m, g in self.files.items():
                if m != n and len(g["ids"]) >= at and lcp(g["ids"], f["ids"]) >= at:
                    last[n] = max(last[n], self.used.get(m, 0))
        return last

    def enforce_quota(self, keep: set[str], dry_run: bool = False) -> list[str]:
        """Keep the whole folder under its quota ("trie_kv": {"max_gb": ...} in the config, or --trie-kv-max-gb; 40 GB). Other snapshots go first
        (others()), then ours least recently used (last_used()), a leaf before its node. The live file is kept."""
        others = self.others()
        total = sum(f["size"] for f in self.files.values()) + sum(o[3] for o in others)
        dropped = []
        for _, _, files, size in others:
            if total <= self.max_bytes:
                break
            for n in files:
                if not dry_run:
                    try:
                        os.unlink(os.path.join(self.folder, n))
                    except FileNotFoundError:
                        pass
            total -= size
            dropped.append(files[0])
        last = self.last_used()
        order = sorted(self.files, key=lambda n: (last[n], self.files[n]["kind"] == "node"))
        for name in order:
            if total <= self.max_bytes:
                break
            if name in keep:
                continue
            if not dry_run:
                try:
                    os.unlink(os.path.join(self.folder, name))
                except FileNotFoundError:
                    pass
                self.used.pop(name, None)
            total -= self.files[name]["size"]
            if not dry_run:
                self.files.pop(name)
            dropped.append(name)
        if dropped and not dry_run:
            self.save_state()
        return dropped


class EngineView:
    """What the cache knows of the engine's one slot: the tokens of the conversation it holds (the last request this
    cache planned, or the file it restored) and the prefix lengths it holds a state for. Valid only while Strata's
    /slots still names this cache's file as the slot's owner - anything else (another client) means unknown."""

    def __init__(self):
        self.owner, self.tokens, self.points, self.end = None, np.empty(0, np.int32), [], 0

    def resume(self, ids: np.ndarray, slot_owner: str) -> int:
        if not self.owner or slot_owner != self.owner:
            return 0
        common = lcp(self.tokens, ids)
        return max([p for p in self.points if p <= common], default=0)

    def restored(self, name, ids, points, end):
        self.owner, self.tokens, self.points, self.end = name, ids, list(points), end

    def read(self, owner, ids, reused, turn_token, tail_role, root_min, every):
        """After the engine read `ids` from `reused`: the checkpoints it keeps (generate.cpp): the prompt's first turn
        boundary when it read from token 0 (the root, pinned), its last turn boundary (before a trailing effort
        turn), the end of a long batched read; older ones that are still a prefix of this prompt stay."""
        common = lcp(self.tokens, ids) if self.owner else 0
        keep = [p for p in self.points if p <= common and p <= reused] if reused else []
        if reused:
            keep.append(reused)                          # it resumed there, so it holds a state there
        n = len(ids)
        at = boundary(ids, turn_token, tail_role, reused)
        if reused == 0:
            first = first_turn(ids, turn_token)
            if root_min <= first < at:
                keep.append(first)
        if at > 0:
            keep.append(at)
        if n - 1 - reused >= every:
            keep.append(n - 1)
        self.owner, self.tokens, self.points = owner, ids, sorted(set(keep))[-6:]
        self.end = boundary(ids, turn_token, tail_role)


def print_tree(store: "Store", tokenizer_path: str | None):
    """The snapshot tree as it stands: each branch with the tokens that start it, and the files that can resume
    on it (where, how big, when last used)."""
    # byte-level BPE (vocab.json): each token's characters stand for bytes (GPT-2's table)
    try:
        vocab = {i: t for t, i in json.load(open(os.path.join(tokenizer_path, "vocab.json"))).items()}
    except (OSError, ValueError, TypeError):
        vocab = {}
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    to_byte = {chr(c): b for b, c in zip(bs, cs)}

    def text(ids):
        out = bytearray()
        for i in ids:
            t = vocab.get(int(i), "")
            if t.startswith("<|") and t.endswith("|>"):
                out += {"<|im_start|>": b"<s>", "<|im_end|>": b"</s>", "<|image_pad|>": b"[img]"}.get(t, t.encode())
            else:
                out += bytes(to_byte.get(ch, 63) for ch in t)
        return out.decode("utf-8", "replace").replace("\n", " ")[:60]

    trie = Trie()
    for name, f in store.files.items():
        trie.insert(f["ids"], f["points"], name)

    def walk(node, indent):
        end = node.depth + len(node.edge)
        files = {}
        for ref, p in node.points:
            files.setdefault(ref, []).append(p)
        label = f"[{node.depth:>6}..{end:<6}] {text(node.edge[:24])}" if node is not trie.root else "(root of the tree)"
        print(f"{indent}{label}")
        for ref, ps in sorted(files.items()):
            f = store.files[ref]
            used = time.strftime("%m-%d %H:%M", time.localtime(store.used.get(ref, f["mtime"])))
            print(f"{indent}    * {f['kind']:4} {ref[len(store.prefix):][:30]:30} resume at {sorted(ps)}  "
                  f"{f['size'] / 2**20:5.0f} MiB  used {used}")
        for kid in sorted(node.kids.values(), key=lambda k: k.depth):
            walk(kid, indent + "  ")
    walk(trie.root, "")


def fold_position(engine_ids: np.ndarray, p: int) -> int | None:
    """A position of the engine's prompt (one <|image_pad|> per image token) -> the folded prompt's (collapse: one per
    image).  None inside an image: no state there.  Past the prompt's end (a thinking budget's second pass reads the
    prompt plus what was written) every image's extra tokens come off."""
    pad = engine_ids == IMAGE_PAD
    extra = pad & np.concatenate([[False], pad[:-1]])
    if p < len(engine_ids) and extra[p]:
        return None
    return int(p - int(extra[:p].sum()))


def namespace(args) -> str:
    """Which session files this engine can take: its arguments, each file one names by its size and time (a model,
    a control vector or a profile written again is another model).  Not the engine binary, its environment or the
    server's settings: those leave the session files as they are, and a file the engine cannot read after all is
    refused (kind invalid: another model) and no longer offered."""
    parts = []
    for a in args or []:
        a = str(a)
        parts.append(a)
        for path in (a, a.rsplit(":", 1)[0]):          # --control-vector-scaled FILE:SCALE
            if os.path.isfile(path):
                st = os.stat(path)
                parts.append(f"{st.st_size}:{int(st.st_mtime)}")
                break
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()[:12]


def legacy_namespace(config_path: str, exe: str | None) -> str:
    """strata-trie-kv's (the proxy's) namespace: the run config's bytes and the engine binary's size and time."""
    st = os.stat(exe) if exe and os.path.exists(exe) else None
    with open(config_path, "rb") as f:
        return hashlib.sha1(f.read() + repr((st.st_size, st.st_mtime) if st else None).encode()).hexdigest()[:12]


class SlotRefused(RuntimeError):
    def __init__(self, status: int, kind: str | None, message: str):
        super().__init__(f"HTTP {status} {kind or ''}: {message}")
        self.status, self.kind = status, kind


class TrieCache:
    """The cache inside the server: plan() before a chat request runs, finish() after it, around() for both."""

    def __init__(self, svc, folder: str, ns: str, max_bytes: int, decisions: str | None = None,
                 min_gain: int = 2048, split_min: int = 2048, root_min: int = 2048):
        self.svc, self.folder = svc, folder
        self.store = Store(folder, ns, max_bytes)
        self.view = EngineView()
        self.lock = threading.Lock()                     # the engine has one slot: plan + request stay together
        self.recent: list = []                           # the last prompts: where a new one leaves them = a node
        self.bad: set[str] = set()                       # files the engine refused as another model's
        self.min_gain, self.split_min, self.root_min = min_gain, split_min, root_min
        self.turn_token, self.tail_role, self.every = 248045, -1, 16384
        self.decisions = open(decisions, "a", buffering=1, encoding="utf-8") if decisions else None
        self.engine_rules((getattr(svc.engine, "spawn", None) or (None, []))[1])

    def engine_rules(self, args):
        """The engine's checkpoint rules come from its command line."""
        args = [str(a) for a in args or []]
        for flag, attr in (("--turn-token", "turn_token"), ("--tail-role-token", "tail_role"),
                           ("--prompt-cache-root", "root_min"), ("--prompt-cache-every", "every")):
            if flag in args[:-1]:
                try:
                    setattr(self, attr, int(args[args.index(flag) + 1]))
                except ValueError:
                    pass

    def adopt(self, old_ns: str) -> int:
        """Take over another namespace's files (strata-trie-kv's, for the same config and engine): renamed, with
        when each was last used.  -> how many."""
        old, new = f"trie_{old_ns}_", self.store.prefix
        if old == new:
            return 0
        n = 0
        for name in os.listdir(self.folder):
            if name.startswith(old) and name.endswith(".bin"):
                os.replace(os.path.join(self.folder, name), os.path.join(self.folder, new + name[len(old):]))
                n += 1
        try:
            with open(os.path.join(self.folder, f"{old}state.json"), encoding="utf-8") as f:
                used = json.load(f)
            for k, v in used.items():
                if k.startswith(old):
                    self.store.used.setdefault(new + k[len(old):], v)
            os.unlink(os.path.join(self.folder, f"{old}state.json"))
        except (OSError, ValueError):
            pass
        self.store.refresh()
        self.store.save_state()
        return n

    def log(self, rec: dict):
        if self.decisions is not None:
            try:
                self.decisions.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except (OSError, ValueError):
                pass

    # ---- the engine's slot, in-process (the proxy's HTTP calls)
    def slot_owner(self) -> str:
        return self.svc.slot_owner_now()

    def holds(self, name: str) -> bool:
        """The slot still holds `name` and nobody runs: no request that skips the cache (one with its own
        slot_snapshot, /completion, /v1/messages) took it right after our restore or save."""
        s = (self.svc.slots_view() or [{}])[0]
        return s.get("snapshot_filename") == name and not s.get("is_processing")

    def slot_action(self, action: str, name: str) -> dict:
        status, body = self.svc.slot_action("0", action, name)
        if status >= 400:
            err = (body or {}).get("error") or {}
            raise SlotRefused(status, err.get("kind"), str(err.get("message")))
        return body

    def restore(self, name: str, rec: dict) -> dict:
        """Restore `name`, again when another request took the slot in the meantime (the restore waits for it)."""
        r = {}
        for attempt in range(4):
            r = self.slot_action("restore", name)
            if self.holds(name):
                break
            rec["stolen"] = attempt + 1
        return r

    def warm(self, ids, owner: str) -> dict:
        """Read `ids` into the slot as `owner`'s conversation, as POST /completion with n_predict 0 does (the engine
        writes one token, not used).  The chat request's own image file and API-monitor record (per thread) are set
        aside meanwhile: the warm must neither use nor delete them.  -> the engine's figures for it."""
        svc = self.svc
        emb, trace = getattr(svc.embeddings, "path", None), getattr(svc.request_trace, "record", None)
        svc.embeddings.path, svc.request_trace.record = None, None
        before = svc.engine.last
        try:
            for _ in svc.run([int(t) for t in ids], False, None, 1, {"slot_snapshot": owner}, threading.Event(),
                             warm=True):
                pass
        finally:
            svc.embeddings.path, svc.request_trace.record = emb, trace
        last = svc.engine.last
        return dict(last) if last is not before and isinstance(last, dict) else {}

    # ---- the decision
    def plan(self, engine_ids):
        """Before the chat request runs: restore the deepest file it continues from when that beats what the engine
        holds, cut a node where it leaves another prompt's path, and name its own file.  -> (folded ids, owner,
        decision record)."""
        rec = {"t": time.strftime("%H:%M:%S")}
        ids, _ = collapse(np.asarray(engine_ids, np.int32), [])
        rec["tokens"] = int(len(ids))
        t1 = time.perf_counter()
        self.store.refresh()
        trie = Trie()                                    # rebuilt per request: a few ms at this folder's size
        ends = {}
        for name, f in self.store.files.items():
            if name in self.bad:
                continue
            trie.insert(f["ids"], f["points"], name)
            ends[name] = file_end(f["ids"], f["points"], self.turn_token)
        best, common = trie.match(ids)
        cut = branch_point(ids, [(f["ids"], ends[n]) for n, f in self.store.files.items() if n in ends] + self.recent,
                           self.turn_token)
        slot = self.slot_owner()
        r_live = self.view.resume(ids, slot)
        # The engine also goes on from the end of a warm (/completion, which this cache does not plan) that this
        # prompt extends exactly: the evidence a client prefilled before its first chat request of the turn
        # (research first). Such a warm left the slot without an owner when it left the slot's conversation, so the
        # view above cannot see it, and the deepest file would be restored over the warm.
        held = self.svc.slot_held if getattr(self.svc, "slot_held_warm", False) else []
        engine = np.asarray(engine_ids, np.int32)
        warm_live = (len(held) if len(ids) == len(engine) and r_live < len(held) < len(engine)
                     and np.array_equal(engine[:len(held)], np.asarray(held, np.int32)) else 0)
        if warm_live:
            r_live = warm_live
            rec["live_warm"] = warm_live
        r_disk = best[1] if best else 0
        rec.update(trie_ms=round((time.perf_counter() - t1) * 1000, 2), files=len(self.store.files),
                   slot_owner=slot[:40], live=r_live, disk=r_disk, disk_file=best[0] if best else None,
                   common=common, branch=cut)
        start, restored = r_live, None
        if best and r_disk >= r_live + self.min_gain:
            t2 = time.perf_counter()
            try:
                r = self.restore(best[0], rec)
            except SlotRefused as e:                     # not enough RAM, another model's file: read without it,
                rec.update(action="restore_refused", restore_error=str(e)[:200])   # the request keeps its own file
                if e.kind == "invalid":
                    self.bad.add(best[0])
            else:
                f = self.store.files[best[0]]
                self.view.restored(best[0], f["ids"], f["points"], ends[best[0]])
                self.store.touch(best[0])
                start, restored = r_disk, best[0]
                rec.update(action="restore", restore_ms=round((time.perf_counter() - t2) * 1000),
                           restored_tokens=r.get("n_restored"))
        elif best and r_disk and r_disk >= r_live:
            self.store.touch(best[0])                    # the engine still holds it: in use all the same
        # nodes (node_cuts): the system turn the first time it is seen (the golden base), and where this prompt leaves
        # another path; each part read on its own and saved, then the request goes on from it
        node = None
        for at, kind in node_cuts(ids, start, r_disk, cut, self.turn_token, self.root_min, self.split_min):
            if warm_live and at < start:
                continue                                 # behind the warm the engine holds: no state there to save
            name = f"{self.store.prefix}node_{hashlib.sha1(ids[:at].tobytes()).hexdigest()[:16]}.bin"
            key = "golden" if kind == "golden" else "node"
            t3 = time.perf_counter()
            # through the next turn's <|im_start|>: the engine must read at least one token, so a prompt ending
            # exactly on a state it holds (the root it keeps in RAM) would be read again from the start
            got = self.warm(ids[: at + 1], name)
            node, start = name, at
            try:
                s = self.slot_action("save", name)
            except SlotRefused as e:                     # read all the same: the request goes on from it
                rec.update({key: name[len(self.store.prefix):], f"{key}_save_error": str(e)[:200]})
                continue
            if not self.holds(name):                     # taken right after the save: the node is on disk now
                rec["node_stolen"] = True
                self.restore(name, rec)
            self.store.refresh()
            f = self.store.files.get(name)
            if f:
                self.view.restored(name, f["ids"], f["points"], file_end(f["ids"], f["points"], self.turn_token))
            self.store.touch(name)
            read = got.get("prompt_read", (got.get("prompt_tokens") or 0) - (got.get("reused") or 0)) if got else None
            rec.update({key: name[len(self.store.prefix):], f"{key}_at": at,
                        f"{key}_ms": round((time.perf_counter() - t3) * 1000), f"{key}_read": read,
                        f"{key}_save_ms": (s.get("timings") or {}).get("save_ms")})
        # its own file: the leaf whose whole conversation this prompt continues (restored, or what the engine holds),
        # so a conversation keeps one file; a new conversation, a fork or a node gets a new leaf
        owner = None
        if node is None and best and self.store.files[best[0]]["kind"] == "leaf" and r_disk >= ends[best[0]]:
            owner = best[0]                              # restored or not (the engine held enough): its next turn
        elif (node is None and not restored and r_live and slot == self.view.owner and "_leaf_" in slot
              and r_live >= self.view.end):
            owner = slot
        if owner is None:
            owner = f"{self.store.prefix}leaf_{time.strftime('%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.bin"
        rec.update(expect=start, owner=owner[len(self.store.prefix):])
        self.recent = (self.recent + [(ids, boundary(ids, self.turn_token, self.tail_role))])[-16:]
        return ids, owner, rec

    def finish(self, engine_ids, ids, owner, rec, before):
        """After the request: what the engine holds now (its DONE line: how much it reused), the owner's use, the
        quota, the decision line."""
        last = self.svc.engine.last
        if last is not before and isinstance(last, dict) and last.get("reused") is not None:
            n, reused = last.get("prompt_tokens"), last["reused"]
            rec.update(engine_reused=reused, engine_read=last.get("prompt_read", (n or 0) - reused),
                       engine_read_ms=last.get("prompt_ms"))
            if n != len(ids):
                rec["engine_tokens"] = n                 # images: the engine's count has every image token
            at = fold_position(np.asarray(engine_ids, np.int32), reused)
            self.view.read(owner, ids, at if at is not None else 0, self.turn_token, self.tail_role, self.root_min,
                           self.every)
        else:                                            # no DONE (an error, the client left): as planned
            self.view.read(owner, ids, rec.get("expect", 0), self.turn_token, self.tail_role, self.root_min, self.every)
        self.store.refresh()
        self.store.touch(owner)                          # used now: the engine writes its file only when evicted
        dropped = self.store.enforce_quota({owner})
        if dropped:
            rec["dropped"] = dropped
        self.log(rec)

    def around(self, engine_ids, req: dict, chunks):
        """The chat request's chunks, planned: the plan runs when the first chunk is asked for (before the engine
        starts: Service.run reads req's slot_snapshot then), the request runs under the cache's lock, and finish()
        once it ends - also when its client leaves.  The cache never fails a request: a plan that raises is logged
        and the request runs as it came."""
        with self.lock:
            t0 = time.perf_counter()
            plan = None
            try:
                plan = self.plan(engine_ids)
                req["slot_snapshot"] = plan[1]
                plan[2]["plan_ms"] = round((time.perf_counter() - t0) * 1000)
            except Exception as e:                       # noqa: BLE001 - never fail the request over the cache
                self.log({"t": time.strftime("%H:%M:%S"), "plan_error": repr(e)[:300]})
            before, first = self.svc.engine.last, None
            try:
                for c in chunks:
                    if first is None and isinstance(c, dict) and any(
                            (ch.get("delta") or {}).get(k) for ch in c.get("choices") or []
                            for k in ("content", "reasoning_content", "tool_calls")):
                        first = time.perf_counter()
                    yield c
            finally:
                if plan is not None:
                    ids, owner, rec = plan
                    rec["ttft_ms"] = round((first - t0) * 1000) if first else None
                    rec["total_ms"] = round((time.perf_counter() - t0) * 1000)
                    try:
                        self.finish(engine_ids, ids, owner, rec, before)
                    except Exception as e:               # noqa: BLE001
                        self.log({"finish_error": repr(e)[:300], **rec})


def from_config(svc, cfg: dict, config_path: str | None, max_gb: float | None = None) -> TrieCache:
    """The cache for this server: its files in the slot save folder, its namespace from the engine's arguments; the
    proxy's files for this same config and engine are taken over."""
    opts = cfg.get("trie_kv") if isinstance(cfg.get("trie_kv"), dict) else {}
    if max_gb is None:
        max_gb = float(opts.get("max_gb", 40))
    folder = svc.slot_save_path
    args = (getattr(svc.engine, "spawn", None) or (None, []))[1]
    cache = TrieCache(svc, folder, namespace(args), int(max_gb * 2**30), os.path.join(folder, DECISIONS),
                      **{k: int(opts[k]) for k in ("min_gain", "split_min", "root_min") if k in opts})
    if config_path:
        try:
            n = cache.adopt(legacy_namespace(config_path, cfg.get("exe")))
            if n:
                print(f"[strata] trie-kv: took over {n} files of strata-trie-kv", flush=True)
        except OSError as e:
            print(f"[strata] trie-kv: could not take over strata-trie-kv's files: {e}", flush=True)
    cache.store.refresh()
    dropped = cache.store.enforce_quota(set())
    if dropped:
        print(f"[strata] trie-kv: quota {max_gb:g} GB dropped {len(dropped)} files", flush=True)
    print(f"[strata] trie-kv on: {len(cache.store.files)} files {cache.store.prefix}*.bin in {folder}, quota "
          f"{max_gb:g} GB for the folder; turn {cache.turn_token}, tail role {cache.tail_role}, root >= "
          f"{cache.root_min}", flush=True)
    return cache


def main():
    ap = argparse.ArgumentParser(description="Look at the trie-kv cache's files (the server runs the cache itself).")
    ap.add_argument("--folder", default=os.path.expanduser("~/.cache/llama-slots"), help="the slot save folder")
    ap.add_argument("--ns", help="the namespace (default: the one whose state file changed last)")
    ap.add_argument("--tokenizer", help="a tokenizer folder with vocab.json, for the tree's labels")
    ap.add_argument("--max-gb", type=float, default=40.0)
    ap.add_argument("--tree", action="store_true", help="print the snapshot tree")
    ap.add_argument("--dry-run-quota", action="store_true", help="print what the quota would delete now")
    a = ap.parse_args()
    ns = a.ns
    if ns is None:
        states = [e for e in os.scandir(a.folder) if e.name.startswith("trie_") and e.name.endswith("_state.json")]
        if not states:
            raise SystemExit(f"no trie-kv files in {a.folder}")
        ns = max(states, key=lambda e: e.stat().st_mtime).name[len("trie_"):-len("_state.json")]
    store = Store(a.folder, ns, int(a.max_gb * 2**30))
    store.refresh()
    if a.dry_run_quota:
        for n in store.enforce_quota(set(), dry_run=True):
            print(n)
    else:
        print(f"namespace {ns}: {len(store.files)} files")
        print_tree(store, a.tokenizer)


if __name__ == "__main__":
    main()
