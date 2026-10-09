"""bench/ab_bench.py — the actual A/B GPU-exclusive probe for the decode-80 run.

Thin wrapper over bench/decode_bench.py (the library that owns the boot/stop/
restore machinery). Why it exists: under greedy decoding (temperature 0) the
model emits its natural stop at ~902 tokens (measured: usage.completion_tokens
= 902, finish_reason=stop against production), so decode_bench's request for
1024 tokens can never "complete" and its runs_complete gate would be 0 for a
healthy candidate. This wrapper fixes the generation length at 512 tokens —
well below the natural stop, so every counted run completes the full request
(finish_reason=length) — and then delegates to decode_bench.main() unchanged.

Everything else (3 cold boots of strata-iq3_s.json on port 8101, production
strata.service stopped before measuring and restored after, median decode/prefill
TPS over the completed runs) is exactly decode_bench.py's protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decode_bench as db  # noqa: E402

db.GEN_TOKENS = 512  # request length; below the model's ~902-token natural stop

if __name__ == "__main__":
    sys.exit(db.main())
