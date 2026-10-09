"""bench/diag_probe.py — why does the baseline reading fail on production?

Sends the EXACT baseline request (bench prompt, 1024 max_tokens, stream,
stream_options) to the production server on 8090 and reports every interior
detail of the exchange (chunk counts, usage, finish_reason, timings, content
size) so the operator can see where baseline_probe.py's generate_once fails.
Read-only: changes nothing on the server.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_bench import GEN_TOKENS, PROMPT, SPEC_PATH  # noqa: E402

KEYS = [
    "http_status",        # 200 / HTTP code / -1 transport error
    "got_done",           # 1 if the stream delivered [DONE]
    "chunks",             # SSE chunks carrying content
    "completion_tokens",  # usage.completion_tokens (0 if no usage)
    "prompt_tokens",      # usage.prompt_tokens (0 if no usage)
    "usage_present",      # 1 if any usage object arrived
    "finish_stop",        # 1 if finish_reason=stop seen
    "finish_length",      # 1 if finish_reason=length seen
    "ttft_s",             # request start -> first content token
    "gen_s",              # first -> last content token
    "content_chars",      # total characters of generated content
]


def emit(m):
    print(json.dumps({k: float(m.get(k, 0.0)) for k in KEYS}))


def main() -> int:
    m = {k: 0.0 for k in KEYS}
    try:
        model = json.loads(SPEC_PATH.read_text(encoding="utf-8-sig")).get("model_name") or "default"
    except Exception:
        model = "default"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": GEN_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8090/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.perf_counter()
    t_first = None
    t_last = None
    err = None
    first_content = ""
    last_content = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            m["http_status"] = float(resp.status)
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    m["got_done"] = 1.0
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                if obj.get("usage"):
                    m["usage_present"] = 1.0
                    u = obj["usage"]
                    m["completion_tokens"] = float(u.get("completion_tokens") or 0)
                    m["prompt_tokens"] = float(u.get("prompt_tokens") or 0)
                for ch in obj.get("choices") or []:
                    piece = ((ch.get("delta") or {}).get("content")
                             or (ch.get("message") or {}).get("content") or "")
                    if piece:
                        m["chunks"] += 1
                        m["content_chars"] += len(piece)
                        if t_first is None:
                            t_first = now
                            first_content = piece[:120]
                        t_last = now
                        last_content = piece[-120:]
                    fr = ch.get("finish_reason")
                    if fr:
                        t_last = now
                        if fr == "stop":
                            m["finish_stop"] = 1.0
                        elif fr == "length":
                            m["finish_length"] = 1.0
    except urllib.error.HTTPError as e:
        m["http_status"] = float(e.code)
        try:
            err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
        except Exception:
            err = f"HTTP {e.code}"
    except Exception as e:
        m["http_status"] = -1.0
        err = repr(e)

    if t_first is not None:
        m["ttft_s"] = t_first - t_start
        m["gen_s"] = max((t_last or t_first) - t_first, 0.0)

    summary = {
        "err": err,
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "metrics": {k: m[k] for k in KEYS},
        "first_content": first_content,
        "last_content": last_content,
        "prompt_chars": len(PROMPT),
    }
    print("[diag2] " + json.dumps(summary), file=sys.stderr)
    emit(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
