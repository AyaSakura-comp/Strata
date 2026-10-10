"""bench/ab_bench.py — the actual A/B GPU-exclusive probe for the prefill-20k / decode-128 run.

Builds on bench/decode_bench.py (the library that owns the boot/stop/restore
machinery, the HTTP generation helper and the engine-log parsing) but owns the
main loop so the completion policy and the path discipline match how EVO runs
measure.

Workload (this run's fitness function):
- Prompt = decode_bench.BASE repeated 192 times (~52 KB, ~20k tokens; the
  engine's own fresh-token count is reported as prompt_fresh_tokens).
- One streamed request per boot with max_tokens = 128 (decode_bench.GEN_TOKENS
  overridden below, BEFORE any generation happens; decode_bench's functions
  read these module globals at call time).
- completion_tokens_median is reported so every reading shows how many tokens
  the generations actually produced (128 = the declared workload; materially
  more means the probe is not measuring what the spec declares).

Path discipline (preflight runs in the real repo; a RUN measures a git worktree
under ~/.evo/runs/... that contains COMMITTED files only):
- The candidate spec is the worktree's spec/strata-iq3_s.json — a committed
  copy of the production config, the file the worker edits. (The production
  /strata-iq3_s.json at the repo root is gitignored on purpose.)
- Everything else — .venv python, serve/server.py, the engine binary, model
  packs, tuning tables — comes from the REAL repo at REAL_REPO: the JSON names
  all of them by absolute path, so booting the production stack unchanged with
  the candidate JSON measures exactly the candidate's effect.
- ALL probe artifacts (spec-boot<N>.json copies, server/engine logs, live
  state) go to the REAL repo's bench/tmp, NEVER the measured worktree: an
  untracked file appearing in the worktree outside the editable region makes
  the framework refuse the round as region_violated (observed on this run:
  the spec-boot<N>.json copies written by a bench invocation from inside the
  worktree refused two consecutive rounds).
- A startup guard refuses to run (exit 1, full key set, production untouched)
  if the artifact dir would ever resolve inside the measured tree, and stale
  artifacts from earlier invocations are removed before the first boot so
  parsing never reads a crashed run's spec or logs.

Metric definitions (all reported per reading, median + worst of the boots):
- prefill_tps_engine / decode_tps_engine — the ENGINE's own rates, parsed from
  the per-boot engine log's per-request summary line. prefill = 1000 *
  fresh_tokens / prompt_ms (cache-reused prompt tokens excluded, no HTTP or
  queueing latency). This is the authoritative prefill gate for the run.
- prefill_tps / decode_tps — the client view from stream timings (prefill =
  prompt_tokens / TTFT, so connection, queueing and the first decoded token are
  included). Kept as an end-to-end reference only.

Completion policy:
- Requests 128 tokens per boot.
- A boot counts when its generation completed (DONE / finish_reason) with
  completion_tokens >= MIN_COMPLETION (96 = 75% of the request) — enough
  decoded tokens for a stable rate at this workload.

GPU exclusivity: production strata.service is stopped before measuring and
ALWAYS restored after (finally block), exactly as decode_bench.py documents.
A GPU that never goes idle after the stop is an instrument failure, not a
(contended) reading.
"""

from __future__ import annotations

import json
import signal
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decode_bench as db  # noqa: E402

MIN_COMPLETION = 96  # tokens; 75% of the 128-token request

# Workload override for THIS probe (module globals, read at call time by
# decode_bench's generate_once/payload construction): 128-token generations
# against a ~20k-token prompt.
db.GEN_TOKENS = 128
db.PROMPT = db.BASE * 192  # ~52 KB; ~20k tokens by the engine's own count
# One extra reported key so a reading proves its own generation length.
db.KEYS = db.KEYS + ["completion_tokens_median"]

WORKTREE = Path(__file__).resolve().parents[1]
REAL_REPO = Path("/home/chihmin/src/Strata")

# Candidate = the tree being measured (worktree in a run, real repo in preflight).
db.SPEC_PATH = WORKTREE / "spec" / "strata-iq3_s.json"
# Artifacts = the REAL repo, always. Writing them into the measured worktree
# shows up as an out-of-region tree change and the round is refused.
db.TMP = REAL_REPO / "bench" / "tmp"
# Runtime stack = the real repo (its .venv and engine exist only there).
db.PYTHON = REAL_REPO / ".venv" / "bin" / "python"
db.REPO = REAL_REPO


def main() -> int:
    def _on_signal(sig, _frame):
        raise SystemExit(128 + sig)

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, _on_signal)
        except Exception:
            pass

    db.TMP.mkdir(parents=True, exist_ok=True)
    metrics = {k: 0.0 for k in db.KEYS}
    instrument_error = None

    if not db.SPEC_PATH.exists():
        print(f"candidate spec missing: {db.SPEC_PATH}", file=sys.stderr)
        db.emit(metrics)
        return 1
    try:
        model = json.loads(db.SPEC_PATH.read_text(encoding="utf-8-sig")).get("model_name") or "default"
    except Exception:
        model = "default"

    # Region guard: in a run the measured tree is a git worktree of committed
    # files (WORKTREE != REAL_REPO); probe artifacts must never land inside it,
    # or every later round is refused region_violated (rounds 1-5 of the first
    # decode-80 run died exactly this way). Fail BEFORE stopping production.
    if WORKTREE.resolve() != REAL_REPO.resolve():
        try:
            tmp_r, wt_r = db.TMP.resolve(), WORKTREE.resolve()
            if tmp_r == wt_r or wt_r in tmp_r.parents:
                instrument_error = (
                    f"probe artifact dir {tmp_r} lies inside the measured tree "
                    f"{wt_r}; refusing to pollute it"
                )
        except Exception:
            pass  # a path-resolution failure here must not block measuring

    # Stale-artifact cleanup: drop only our own known patterns under db.TMP so
    # per-boot parsing never reads a previous (possibly crashed) invocation's
    # spec copy or logs.
    if instrument_error is None:
        for _pat in ("spec-boot*.json", "server-boot*.log", "engine-boot*.log"):
            for _f in db.TMP.glob(_pat):
                try:
                    _f.unlink()
                except OSError:
                    pass

    # GPU exclusivity: take production down (only when the guard above passed).
    if instrument_error is None and not db.service_ctl("stop"):
        instrument_error = "could not stop strata.service (no working systemctl path)"
    if instrument_error is None:
        db.cleanup_leftovers()
        if not db.wait_gpu_free():
            instrument_error = "GPU never went idle after stopping production"

    results = []
    try:
        if instrument_error is None:
            for i in range(db.BOOTS):
                r = db.boot_and_measure(i, model)
                if r is not None and r["completion_tokens"] >= MIN_COMPLETION:
                    results.append(r)
    finally:
        # Production restore ALWAYS runs, whatever happened above.
        restored = False
        if db.service_ctl("start"):
            t0 = time.time()
            while time.time() - t0 < db.MODEL_READY_TIMEOUT_S:
                if db.http_json(f"http://127.0.0.1:{db.PROD_PORT}/v1/models", timeout=10.0) is not None:
                    restored = True
                    break
                time.sleep(10)
        if not restored:
            print("PRODUCTION RESTORE FAILED: strata.service is not answering on 8090",
                  file=sys.stderr)
            instrument_error = instrument_error or "production restore failed"

    if results:
        db.aggregate(metrics, results)
        metrics["completion_tokens_median"] = statistics.median(
            r["completion_tokens"] for r in results
        )
    metrics["runs_complete"] = len(results) / db.BOOTS

    db.emit(metrics)
    if instrument_error:
        print(instrument_error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
