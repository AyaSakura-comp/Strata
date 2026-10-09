"""bench/ab_bench.py — the actual A/B GPU-exclusive probe for the decode-80 run.

Builds on bench/decode_bench.py (the library that owns the boot/stop/restore
machinery and the HTTP generation helper) but owns the main loop so the
completion policy matches what the model really does:

- Requests 1024 tokens (a request for less than ~that can make this server
  finish immediately with zero content — observed with max_tokens=512).
- Under greedy decoding the model emits its natural stop at ~902 tokens, so a
  run cannot "complete all 1024". A boot therefore counts when its generation
  completed (DONE / finish_reason) with completion_tokens >= MIN_COMPLETION
  (512) — enough decoded tokens for a stable rate.

Everything else — 3 cold boots of strata-iq3_s.json on port 8101, production
strata.service stopped before measuring and ALWAYS restored after (finally
block), median decode/prefill TPS over the completed runs — follows
decode_bench.py's protocol and helpers.
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

MIN_COMPLETION = 512  # tokens; below the model's ~902-token natural stop


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

    # GPU exclusivity: take production down.
    if not db.service_ctl("stop"):
        instrument_error = "could not stop strata.service (no working systemctl path)"
    if instrument_error is None:
        db.cleanup_leftovers()
        db.wait_gpu_free()

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
        dec = [r["decode_tps"] for r in results]
        pf = [r["prefill_tps"] for r in results]
        metrics["decode_tps"] = statistics.median(dec)
        metrics["prefill_tps"] = statistics.median(pf)
        metrics["decode_tps_min"] = min(dec)
        metrics["prefill_tps_min"] = min(pf)
        metrics["prompt_tokens"] = results[0]["prompt_tokens"]
        metrics["prompt_tokens_estimated"] = results[0]["estimated"]
    metrics["runs_complete"] = len(results) / db.BOOTS

    db.emit(metrics)
    if instrument_error:
        print(instrument_error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
