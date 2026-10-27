# FlashHead draft head vs token subsets - Radeon 8060S (gfx1151), IQ3_S, 2026-10-27

`bench/flashhead_ab.py --samples 3 --tokens 1024 --variant en --variant cjk --variant fh --variant full`, one engine
binary for all four (this commit's build), the production config otherwise. Tables:
`tools/flashhead.py` (7,632 clusters x 32 + 4,096 static tokens from 1.7 M characters of the model's own pi agent
answers plus Chinese, Japanese and English text; `tables.log`), 256 probes.

- `summary.txt`: medians per language and variant (engine decode tok/s), drafts accepted of proposed and per token.
- `rows.jsonl`: every request (engine summary line, finish reason, whether the GPU read idle before it).
- `kernel-bench.txt`: one draft head evaluation, `flashhead_parity --bench` (synthetic data, the same shapes).
- `driver.log`: the run, in order.
- `en-2.stall.engine.log`: the first `en` sample-2 boot stopped itself mid-request ("no progress for 60 s ... verify
  window: waiting for the GPU", issue #29's watchdog) on the English-subset head, which runs none of the FlashHead
  code; the boot was run again (`--resume`).  One request (`en` sample 3, Chinese) started without ten idle GPU
  seconds; its rate (30.1) is within the other two (30.3, 31.4).
