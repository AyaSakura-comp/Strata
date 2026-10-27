"""The MTP draft head as a retrieval head (FlashHead): the tables the engine reads from --mtp-flashhead.

The draft layer proposes tokens; the target verifies every one of them with its own full head, so the draft head may
be approximate - a worse proposal costs acceptance, never output.  A token subset (tools/draft_vocab.py) makes the
draft head smaller but can never propose a token outside it (an English subset drafts almost nothing of a Chinese or
Japanese answer).  FlashHead (arXiv 2603.14591) keeps the whole vocabulary and reads only a small part of it per
draft token:

    stage 1: h . centroid[c] for every cluster c (Q4_0, a few MB)  -> the `probes` best clusters
    stage 2: the exact head rows of the static tokens + every token of those clusters -> argmax

This script writes the tables, from the model's own head:

    python tools/flashhead.py --gguf <model>-00001-of-0000N.gguf [--gguf <model>-00002-...] \
        --corpus agent.txt zh.txt ja.txt en.txt --out data/flashhead.bin

- static: the --static most frequent tokens of the corpus (the head of what the model writes: misses concentrate
  there), always scored.  Rounded up so the rest of the vocabulary divides into equal clusters.
- clusters: balanced spherical k-means over the remaining head rows, --cluster-size tokens each (equal sizes: the
  engine's stage 2 gathers cluster c's rows as c2t[c * b .. c * b + b), no ragged lists).  Static tokens are in no
  cluster, so a candidate list never holds a token twice.
- centroids: Q4_0 rows.  Stage 1 only chooses which clusters to read; stage 2 recomputes exact logits for them.

File layout (little-endian): "SFH1", u32 version (1), u32 n_in, u32 n_vocab, u32 clusters, u32 cluster_size,
u32 n_static, then the Q4_0 centroid rows (clusters x n_in / 32 blocks of 18 bytes), int32 c2t (clusters x
cluster_size), int32 static (n_static, ascending).

Needs torch (a GPU makes the clustering minutes instead of hours), numpy and gguf-py's quants (for the
Q6_K dequantisation and Q4_0 quantisation; --gguf-py names its folder when it is not importable).
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_reader  # noqa: E402

MAGIC = b"SFH1"


def head_rows(paths: list[str]) -> tuple[np.ndarray, int, int, int]:
    """The raw rows of output.weight from whichever shard holds it -> (uint8 [n_vocab, row_bytes], type, n_in, n_vocab)."""
    for p in paths:
        g = gguf_reader.GGUFFile(Path(p))
        for t in g.tensors:
            if t.name == "output.weight":
                n_in, n_vocab = t.shape[0], t.shape[1]
                nbytes = t.expected_bytes()
                if nbytes is None:
                    raise SystemExit(f"output.weight is {t.type_name}, which this script cannot size")
                rows = np.memmap(p, dtype=np.uint8, mode="r", offset=g.data_start + t.offset,
                                 shape=(n_vocab, nbytes // n_vocab))
                print(f"[head] {Path(p).name}: output.weight {t.type_name} [{n_in}, {n_vocab}]", flush=True)
                return rows, t.type_id, n_in, n_vocab
    raise SystemExit("output.weight is in none of the --gguf files")


def corpus_counts(gguf: str, paths: list[str]) -> Counter:
    """Token occurrences over the corpus files, tokenized as plain text (no special tokens) as draft_vocab.py does."""
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_gguf(gguf)
    counts: Counter = Counter()
    for p in paths:
        ids = tok.encode(Path(p).read_text(encoding="utf-8", errors="replace"))
        counts.update(ids)
        print(f"[corpus] {Path(p).name}: {len(ids)} tokens", flush=True)
    return counts


def static_set(counts: Counter, n: int, n_vocab: int, b: int) -> np.ndarray:
    picked = [t for t, _ in counts.most_common() if 0 <= t < n_vocab][:n]
    extra = (n_vocab - len(picked)) % b                 # the rest must divide into clusters of b
    if extra:
        have = set(picked)
        picked += [t for t in range(n_vocab) if t not in have][:extra]
    total = sum(counts.values()) or 1
    print(f"[static] {len(picked)} tokens, {sum(counts[t] for t in picked) / total:.1%} of the corpus' occurrences",
          flush=True)
    return np.sort(np.array(picked, dtype=np.int32))


def dequant(rows: np.ndarray, type_id: int, n_in: int, ids: np.ndarray, gguf_py: str | None):
    if gguf_py:
        sys.path.insert(0, gguf_py)
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    out = np.empty((ids.size, n_in), dtype=np.float16)
    for s in range(0, ids.size, 16384):
        chunk = np.asarray(rows[ids[s:s + 16384]])
        out[s:s + 16384] = np.asarray(dequantize(chunk, GGMLQuantizationType(type_id)),
                                      dtype=np.float32).reshape(-1, n_in)
    return out


def capacity_fill(cand, cand_sim, c: int, b: int):
    """Each token to its most similar cluster that still has room; tokens with the strongest preference first."""
    import torch
    order = torch.argsort(cand_sim[:, 0] - cand_sim[:, -1], descending=True).cpu().numpy()
    cand_np = cand.cpu().numpy()
    assign = np.full(cand_np.shape[0], -1, dtype=np.int64)
    free = np.full(c, b, dtype=np.int64)
    for i in order:
        for k in cand_np[i]:
            if free[k] > 0:
                assign[i] = k
                free[k] -= 1
                break
    left = np.nonzero(assign < 0)[0]
    if left.size:
        slots = np.repeat(np.nonzero(free > 0)[0], free[free > 0])
        assign[left] = slots[:left.size]
    return torch.from_numpy(assign).to(cand.device)


def balanced_kmeans(E, c: int, iters: int, seed: int = 0):
    """Balanced spherical k-means: E [v, d] L2-normalised (fp16 on the device) -> (centroids [c, d] fp32, assign [v])."""
    import torch
    v, _ = E.shape
    b = v // c
    g = torch.Generator(device=E.device).manual_seed(seed)
    cent = E[torch.randperm(v, generator=g, device=E.device)[:c]].float().clone()
    K = 32                                               # candidate clusters per token for the capacity fill
    for it in range(iters):
        cand = torch.empty((v, K), dtype=torch.int64, device=E.device)
        cand_sim = torch.empty((v, K), dtype=torch.float32, device=E.device)
        ch = cent.half()
        for s in range(0, v, 8192):
            m, a = (E[s:s + 8192] @ ch.T).float().topk(K, dim=1)
            cand_sim[s:s + 8192], cand[s:s + 8192] = m, a
        assign = capacity_fill(cand, cand_sim, c, b)
        members = torch.argsort(assign, stable=True).view(c, b)
        cent = torch.nn.functional.normalize(E[members].float().sum(dim=1), dim=1)
        if it % 5 == 0 or it == iters - 1:
            obj = (E.float() * cent[assign]).sum(dim=1).mean().item()
            print(f"[kmeans] iter {it:3d}  mean cos {obj:.5f}", flush=True)
    return cent, assign


def quantize_q4_0(x: np.ndarray, gguf_py: str | None) -> np.ndarray:
    if gguf_py:
        sys.path.insert(0, gguf_py)
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize
    q = np.asarray(quantize(x.astype(np.float32), GGMLQuantizationType.Q4_0), dtype=np.uint8)
    assert q.shape == (x.shape[0], x.shape[1] // 32 * 18), q.shape
    return q


def write_tables(path: str, centroids_q4: np.ndarray, c2t: np.ndarray, static: np.ndarray, n_in: int,
                 n_vocab: int) -> None:
    """The file the engine reads (layout in the module docstring)."""
    c, b = c2t.shape
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<6I", 1, n_in, n_vocab, c, b, static.size))
        f.write(np.ascontiguousarray(centroids_q4, dtype=np.uint8).tobytes())
        f.write(np.ascontiguousarray(c2t, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(static, dtype=np.int32).tobytes())


def read_tables(path: str) -> dict:
    """write_tables' file back: n_in, n_vocab, centroids (raw Q4_0 rows), c2t [clusters, b], static."""
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path} is not a FlashHead table file")
    version, n_in, n_vocab, c, b, n_static = struct.unpack("<6I", raw[4:28])
    if version != 1:
        raise ValueError(f"{path}: version {version}")
    cent = c * (n_in // 32) * 18
    off = 28
    out = {"n_in": n_in, "n_vocab": n_vocab,
           "centroids": np.frombuffer(raw, np.uint8, cent, off).reshape(c, -1)}
    off += cent
    out["c2t"] = np.frombuffer(raw, np.int32, c * b, off).reshape(c, b)
    off += c * b * 4
    out["static"] = np.frombuffer(raw, np.int32, n_static, off)
    if off + n_static * 4 != len(raw):
        raise ValueError(f"{path}: {len(raw) - off - n_static * 4} trailing bytes")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", action="append", required=True, help="a model GGUF shard (repeat for each)")
    ap.add_argument("--corpus", nargs="+", required=True, help="UTF-8 text the model writes (its static set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--static", type=int, default=4096)
    ap.add_argument("--cluster-size", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30, help="k-means iterations (it is flat by ~25)")
    ap.add_argument("--gguf-py", default=None, help="folder holding gguf-py's `gguf` package")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    rows, type_id, n_in, n_vocab = head_rows(a.gguf)
    b = a.cluster_size
    static = static_set(corpus_counts(a.gguf[0], a.corpus), a.static, n_vocab, b)
    mask = np.ones(n_vocab, dtype=bool)
    mask[static] = False
    rest = np.nonzero(mask)[0].astype(np.int64)
    c = rest.size // b
    assert c * b == rest.size

    t0 = time.time()
    W = dequant(rows, type_id, n_in, rest, a.gguf_py)
    print(f"[head] {rest.size} rows dequantised in {time.time() - t0:.0f} s", flush=True)
    E = torch.nn.functional.normalize(torch.from_numpy(W).to(a.device).float(), dim=1).half()
    del W
    t0 = time.time()
    cent, assign = balanced_kmeans(E, c, a.iters)
    print(f"[kmeans] {c} clusters of {b} in {time.time() - t0:.0f} s", flush=True)
    members = torch.argsort(assign, stable=True).view(c, b).cpu().numpy()
    c2t = rest[members].astype(np.int32)                 # cluster-local index -> token id
    cq = quantize_q4_0(cent.cpu().numpy(), a.gguf_py)

    write_tables(a.out, cq, c2t, static, n_in, n_vocab)
    print(f"[done] {a.out}: {c} clusters x {b} + {static.size} static, centroids {cq.nbytes / 1048576:.1f} MiB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
