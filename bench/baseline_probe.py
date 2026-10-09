"""bench/baseline_probe.py — read-only baseline reading from the PRODUCTION server.

Sends the exact bench prompt (from bench/decode_bench.py, so the two can never
drift) as one streaming request to the production server on port 8090 and times
it. Stops nothing, changes nothing: production keeps serving.

Generation length is 512 tokens (not 1024): under greedy decoding the model's
natural stop is ~902 tokens, so a 1024-token request can never complete; 512 is
well below it, so every reading completes the full request. This matches the
A/B probe bench/ab_bench.py.

The number this yields is measured under production conditions (warm resident
model), not under the A/B cold-boot protocol of ab_bench.py; it is a reference
for setting the prefill floor in task-decode80.yaml.

Protocol: exit 0 + JSON on a completed generation; exit 1 (full key set still
printed as zeros) if production could not be measured.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decode_bench as db  # noqa: E402

db.GEN_TOKENS = 512  # request length; below the model's ~902-token natural stop

PROD_PORT = 8090

KEYS = [
    "decode_tps",               # decode tok/s on the production server
    "prefill_tps",              # prompt tokens / time-to-first-token
    "completion_tokens",        # tokens the server said it produced (or chunks)
    "prompt_tokens",            # as reported, else chars/4 estimate
    "prompt_tokens_estimated",  # 1.0 when prompt_tokens is an estimate
    "ttft_s",                   # request start -> first content token
    "gen_s",                    # first -> last content token
]


def emit(metrics: dict) -> None:
    print(json.dumps({k: float(metrics.get(k, 0.0)) for k in KEYS}))


def main() -> int:
    metrics = {k: 0.0 for k in KEYS}
    try:
        model = json.loads(db.SPEC_PATH.read_text(encoding="utf-8-sig")).get("model_name") or "default"
    except Exception:
        model = "default"

    t0 = time.perf_counter()
    r = db.generate_once(PROD_PORT, model)
    total = time.perf_counter() - t0
    if r is None or r["completion_tokens"] < db.GEN_TOKENS:
        print("baseline generation failed or incomplete on production", file=sys.stderr)
        emit(metrics)
        return 1

    metrics["decode_tps"] = r["decode_tps"]
    metrics["prefill_tps"] = r["prefill_tps"]
    metrics["completion_tokens"] = r["completion_tokens"]
    metrics["prompt_tokens"] = r["prompt_tokens"]
    metrics["prompt_tokens_estimated"] = r["estimated"]
    if r["prefill_tps"] > 0:
        ttft = r["prompt_tokens"] / r["prefill_tps"]
        metrics["ttft_s"] = ttft
        metrics["gen_s"] = max(total - ttft, 0.0)
    emit(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
