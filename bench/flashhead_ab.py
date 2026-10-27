"""bench/flashhead_ab.py - interleaved cold-boot A/B of the MTP draft head (token subsets vs FlashHead), per language.

    python bench/flashhead_ab.py --root DIR --samples 3 --tokens 1024 --variant en --variant cjk --variant fh ...

Every variant runs the SAME engine binary (--exe) with the production config (strata-iq3_s.json) on port 8101; only
the draft head changes:

    en     --mtp-draft-vocab data/draft_vocab_en.bin   (the production subset: English and code, 40,525 tokens)
    cjk    --mtp-draft-vocab data/draft_vocab.bin      (setup's default: + Han, kana, Hangul, CJK punctuation)
    full   STRATA_MTP_FULL_HEAD=1                      (the whole vocabulary, dense: the acceptance reference)
    fh     STRATA_MTP_FLASHHEAD=<--tables>, STRATA_MTP_FLASHHEAD_PROBES=<--probes>
    fh<P>  the same with P probes

Method (the ab-qwen-profiling skill's, on Strata): production strata.service is stopped for the run and always
restarted; each sample is a cold boot; the run order alternates forward / reverse; the GPU must read 0% busy for
ten consecutive seconds before each timed request.  One boot reads the calibrated ~20K-token prompt once (a 1-token
request; the document is the system turn, which Strata keeps as a cache root), then answers it in each language in
turn (zh, en, ja, rotated per sample) with thinking off, temperature 0, seed 42 and `--tokens` output tokens: the
20K prefix is reused from the engine's cache, so the timed part is decode over a 20K context.  The numbers are the
ENGINE's own per-request summary (generated tokens / ms, drafts accepted / proposed), not client timings.  Outputs
are kept: at temperature 0 the target decides every token, so each language's text must be identical across variants
(a check that the draft head only changes speed).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/chihmin/src/Strata")
PROD_CFG = REPO / "strata-iq3_s.json"
PY = REPO / ".venv/bin/python"
PROMPT = Path.home() / ".claude/skills/ab-qwen-profiling/assets/qwen-prompt-20000.txt"
PORT = 8101
ASK = {
    "zh": "請用繁體中文詳細整理上面這份文件的重點，分成很多段落，盡量寫長，全部使用繁體中文，不要使用英文。",
    "en": "Write a long, detailed English summary of the document above, in many paragraphs.",
    "ja": "上の文書の要点を日本語で詳しく、いくつもの段落に分けて、できるだけ長く説明してください。英語は使わないでください。",
}
LINE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), "
                  r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def log(root: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(root / "driver.log", "a") as f:
        f.write(line + "\n")


def http(path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu_idle(max_wait: float = 120) -> bool:
    n, t0 = 0, time.time()
    while time.time() - t0 < max_wait:
        try:
            busy = int(Path("/sys/class/drm/card1/device/gpu_busy_percent").read_text())
        except Exception:
            busy = 100
        n = n + 1 if busy == 0 else 0
        if n >= 10:
            return True
        time.sleep(1)
    return False


def variant_cfg(name: str, a, root: Path, tag: str) -> Path:
    cfg = json.loads(PROD_CFG.read_text())
    cfg["exe"] = a.exe
    cfg["port"] = PORT
    cfg["log"] = str(root / f"{tag}.engine.log")
    cfg.pop("vision", None)                  # the image encoder (CPU) plays no part in decode
    cfg.pop("slot_save_path", None)
    args = [x for x in cfg["args"] if x != "--vision"]
    i = args.index("--mtp-draft-vocab")
    env = dict(cfg.get("env", {}))
    if name == "cjk":
        args[i + 1] = str(REPO / "data/draft_vocab.bin")
    elif name == "full":
        env["STRATA_MTP_FULL_HEAD"] = "1"
    elif name.startswith("fh"):
        env["STRATA_MTP_FLASHHEAD"] = a.tables
        env["STRATA_MTP_FLASHHEAD_PROBES"] = name[2:] or str(a.probes)
    elif name != "en":
        raise SystemExit(f"unknown variant {name}")
    cfg["args"], cfg["env"] = args, env
    path = root / f"{tag}.json"
    path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    return path


def engine_lines(path: Path) -> list[dict]:
    text = path.read_text(errors="replace") if path.exists() else ""
    out = []
    for m in LINE.finditer(text):
        g = m.groups()
        out.append({"prompt": int(g[0]), "reused": int(g[1]), "read": int(g[2]), "prefill_tps": float(g[4]),
                    "generated": int(g[5]), "gen_ms": float(g[6]), "decode_tps": float(g[7]),
                    "accepted": int(g[8]), "proposed": int(g[9])})
    return out


def one_boot(name: str, sample: int, a, root: Path, prefix: str) -> list[dict]:
    tag = f"{name}-{sample}"
    cfg = variant_cfg(name, a, root, tag)
    out = open(root / f"{tag}.server.log", "w")
    proc = subprocess.Popen([str(PY), str(REPO / "serve/server.py"), "--engine", "strata", "--config", str(cfg),
                             "--port", str(PORT)], cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
    rows = []
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                log(root, f"[{tag}] server exited ({proc.returncode})")
                return rows
            try:
                if http("/health", timeout=5).get("loaded"):
                    break
            except Exception:
                pass
            if time.time() - t0 > 900:
                log(root, f"[{tag}] not loaded after 900 s")
                return rows
            time.sleep(3)
        elog = root / f"{tag}.engine.log"
        head = [l for l in elog.read_text(errors="replace").splitlines() if "draft head" in l or "FlashHead" in l]
        log(root, f"[{tag}] loaded after {time.time() - t0:.0f} s: {' | '.join(head) or 'no draft head line'}")
        # the document is the system turn: Strata keeps a long system prompt as a cache root, so each language's
        # request reads only its own question
        system = {"role": "system", "content": prefix}
        http("/v1/chat/completions", {"model": "m", "messages": [system, {"role": "user", "content": "Reply OK."}],
                                      "max_tokens": 1, "temperature": 0, "seed": 42, "reasoning_effort": "none"})
        langs = ["zh", "en", "ja"]
        k = (sample - 1) % 3
        langs = langs[k:] + langs[:k]
        for lang in langs:
            idle = gpu_idle()
            n0 = len(engine_lines(elog))
            try:
                r = http("/v1/chat/completions", {
                    "model": "m", "messages": [system, {"role": "user", "content": ASK[lang]}],
                    "max_tokens": a.tokens, "temperature": 0, "seed": 42, "reasoning_effort": "none"})
            except Exception as e:                       # an engine that stopped (its stall watchdog, a crash)
                log(root, f"[{tag}] {lang}: request failed ({e}); the boot is run again")
                return []
            time.sleep(0.5)
            lines = engine_lines(elog)[n0:]
            if not lines:
                log(root, f"[{tag}] {lang}: no engine summary line")
                continue
            e = lines[-1]
            text = r["choices"][0]["message"].get("content") or ""
            (root / f"{tag}.{lang}.txt").write_text(text)
            e.update({"variant": name, "sample": sample, "lang": lang, "idle": idle,
                      "finish": r["choices"][0].get("finish_reason"), "chars": len(text)})
            rows.append(e)
            log(root, f"[{tag}] {lang}: {e['generated']} tok {e['decode_tps']:.2f} tok/s, drafts {e['accepted']}/"
                      f"{e['proposed']} ({e['accepted'] / max(1, e['proposed']):.3f}), reused {e['reused']}, "
                      f"finish {e['finish']}, idle {idle}")
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        out.close()
        time.sleep(10)


def summarize(root: Path) -> None:
    rows = [json.loads(l) for l in (root / "rows.jsonl").read_text().splitlines() if l.strip()]
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    base = variants[0]
    print(f"\n{'lang':4s} {'variant':8s} {'decode tok/s runs':28s} {'median':>7s} {'vs ' + base:>8s} {'acc/prop':>8s}"
          f" {'acc/tok':>8s}")
    for lang in ("zh", "en", "ja"):
        med = {}
        for v in variants:
            rs = [r for r in rows if r["variant"] == v and r["lang"] == lang]
            if not rs:
                continue
            tps = [r["decode_tps"] for r in rs]
            med[v] = statistics.median(tps)
            acc = sum(r["accepted"] for r in rs) / max(1, sum(r["proposed"] for r in rs))
            per_tok = sum(r["accepted"] for r in rs) / max(1, sum(r["generated"] for r in rs))
            delta = (med[v] / med[base] - 1) * 100 if base in med else 0.0
            print(f"{lang:4s} {v:8s} {' '.join(f'{t:6.2f}' for t in tps):28s} {med[v]:7.2f} {delta:+7.2f}% {acc:8.3f}"
                  f" {per_tok:8.3f}")
    # the target decides every token at temperature 0: the same language's text must match across variants
    for lang in ("zh", "en", "ja"):
        texts = {p.name: p.read_text() for p in sorted(root.glob(f"*.{lang}.txt"))}
        same = len(set(texts.values())) <= 1
        print(f"{lang}: {len(texts)} outputs, {'all identical' if same else 'DIFFER'}")
        if not same:
            ref = next(iter(texts.values()))
            for n, t in texts.items():
                if t != ref:
                    i = next((k for k in range(min(len(t), len(ref))) if t[k] != ref[k]), min(len(t), len(ref)))
                    print(f"   {n} differs from the first at char {i}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--exe", default="/home/chihmin/src/Strata-fh/build-hip/strata")
    ap.add_argument("--tables", default="")
    ap.add_argument("--probes", type=int, default=256)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep the complete boots of an earlier run in --root")
    a = ap.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize(root)
        return 0
    prefix = PROMPT.read_text()
    log(root, f"variants {a.variant} samples {a.samples} tokens {a.tokens} exe {a.exe} tables {a.tables}")
    # --resume: keep the boots that answered every language, drop the partial ones (they run again)
    rows_path = root / "rows.jsonl"
    done: set[tuple[str, int]] = set()
    if a.resume and rows_path.exists():
        kept = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        langs = {}
        for r in kept:
            langs.setdefault((r["variant"], r["sample"]), set()).add(r["lang"])
        done = {k for k, v in langs.items() if len(v) == 3}
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in kept if (r["variant"], r["sample"]) in done))
        log(root, f"resume: {len(done)} complete boots kept")
    subprocess.run(["systemctl", "--user", "stop", "strata"], check=False)
    try:
        for s in range(1, a.samples + 1):
            order = a.variant if s % 2 == 1 else list(reversed(a.variant))
            for v in order:
                if (v, s) in done:
                    continue
                for attempt in range(2):                 # one more boot when the first one stopped mid-way
                    rows = one_boot(v, s, a, root, prefix)
                    if len(rows) == 3:
                        break
                with open(rows_path, "a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
    finally:
        subprocess.run(["systemctl", "--user", "start", "strata"], check=False)
        log(root, "production strata.service started again")
    summarize(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
