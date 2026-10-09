"""bench/decode_bench.py — A/B GPU-exclusive decode/prefill TPS probe for strata-iq3_s.json.

One server shape per reading (kb: one_boot_per_stage): boots the candidate spec
`strata-iq3_s.json` on port 8101 three times (cold boots), takes one 1024-token
streaming generation per boot, and reports median decode_tps and prefill_tps.

GPU exclusivity: stops the production `strata.service` (port 8090) before
measuring and ALWAYS restarts it afterwards (finally block), verifying it
answers /v1/models again on 8090.

Protocol (EVO probe):
  - exit 0 + JSON = a reading (zeros included: a candidate that cannot boot is
    an honest bad reading)
  - exit 1 = the instrument/environment failed (service could not be stopped,
    production could not be restored); full key set is still printed
  - exit 75 = world failure (not used here)
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "strata-iq3_s.json"
PYTHON = REPO / ".venv" / "bin" / "python"
TMP = REPO / "bench" / "tmp"
PROD_PORT = 8090
MEAS_PORT = 8101
BOOTS = 3
GEN_TOKENS = 1024
READY_TIMEOUT_S = 900          # cold boot of the pack can take minutes
GEN_HARD_TIMEOUT_S = 600       # silence longer than this means the run is dead
MODEL_READY_TIMEOUT_S = 900    # production restart wait
GPU_SETTLE_S = 30

BASE = (
    "The scheduler maps each decoded block onto the resident expert set, and the "
    "cache layer reuses attention state across turns of the conversation. When the "
    "router changes shape, the pipeline waits for the outstanding kernels to drain "
    "before it swaps the packed weights. This note describes the observed timing "
    "behaviour of the decode path under a single idle GPU client, with the fixed "
    "prompt replayed verbatim so that every round of the benchmark sees the same "
    "token stream and the measured deltas stay attributable to the candidate "
    "configuration rather than to input noise.\n"
)
PROMPT = BASE * 36  # ~9.7 KB, roughly 2.2k tokens

KEYS = [
    "decode_tps",        # median decode tok/s (target of the run)
    "prefill_tps",       # median prefill tok/s (must not regress)
    "runs_complete",     # fraction of boots that completed all GEN_TOKENS
    "decode_tps_min",    # worst boot (spread check)
    "prefill_tps_min",
    "prompt_tokens",     # tokens per request as reported (or estimated)
    "prompt_tokens_estimated",  # 1.0 if prompt_tokens is a chars/4 estimate
]


def emit(metrics: dict) -> None:
    out = {k: float(metrics.get(k, 0.0)) for k in KEYS}
    print(json.dumps(out))


def run_cmd(argv, timeout=60):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def service_ctl(verb: str) -> bool:
    """Stop/start strata.service, trying the available systemctl flavours in order."""
    for argv in (
        ["sudo", "-n", "systemctl", verb, "strata.service"],
        ["systemctl", "--user", verb, "strata.service"],
        ["systemctl", verb, "strata.service"],
    ):
        r = run_cmd(argv, timeout=120)
        if r is not None and r.returncode == 0:
            return True
    return False


def http_json(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        pass
    return None


def port_closed(port: int) -> bool:
    return http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5.0) is None


def wait_gpu_free(deadline_s: float = 180.0) -> None:
    """Wait until no strata engine process and (best effort) GPU util settles."""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        r = run_cmd(["pgrep", "-f", "engine/strata"], timeout=10)
        if (r is None or r.returncode != 0) and port_closed(PROD_PORT):
            break
        time.sleep(5)
    for smi in ("rocm-smi", "/opt/rocm/bin/rocm-smi"):
        r = run_cmd([smi, "--showuse"], timeout=20)
        if r is None or r.returncode != 0:
            continue
        t1 = time.time()
        while time.time() - t1 < 120:
            r = run_cmd([smi, "--showuse"], timeout=20)
            txt = (r.stdout or "") if r else ""
            busy = False
            for tok in txt.replace("%", " ").split():
                try:
                    if float(tok) > 5.0:
                        busy = True
                        break
                except ValueError:
                    continue
            if not busy:
                break
            time.sleep(5)
        break
    time.sleep(GPU_SETTLE_S)


def make_meas_spec(idx: int) -> Path:
    cfg = json.loads(SPEC_PATH.read_text(encoding="utf-8-sig"))
    cfg["port"] = MEAS_PORT
    cfg["log"] = str(TMP / f"engine-boot{idx}.log")
    cfg["open_browser"] = False
    p = TMP / f"spec-boot{idx}.json"
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return p


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        if http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5.0) is not None:
            return True
        time.sleep(5)
    return False


def generate_once(port: int, model: str) -> dict | None:
    """One streaming 1024-token generation; returns timing metrics or None on failure."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": GEN_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.perf_counter()
    t_first = None
    t_last = None
    chunks = 0
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=GEN_HARD_TIMEOUT_S) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        chunks += 1
                        if t_first is None:
                            t_first = now
                        t_last = now
                    if ch.get("finish_reason"):
                        t_last = now
    except Exception:
        return None
    if t_first is None or t_last is None or t_last <= t_first:
        return None
    completion = int((usage or {}).get("completion_tokens") or chunks)
    prompt_tokens = (usage or {}).get("prompt_tokens")
    estimated = 0.0
    if not prompt_tokens:
        prompt_tokens = len(PROMPT) // 4
        estimated = 1.0
    ttft = t_first - t_start
    if ttft <= 0:
        return None
    return {
        "decode_tps": max(completion - 1, 1) / (t_last - t_first),
        "prefill_tps": float(prompt_tokens) / ttft,
        "completion_tokens": float(completion),
        "prompt_tokens": float(prompt_tokens),
        "estimated": estimated,
    }


def boot_and_measure(idx: int, model: str) -> dict | None:
    spec = make_meas_spec(idx)
    logf = open(TMP / f"server-boot{idx}.log", "w")
    proc = subprocess.Popen(
        [str(PYTHON), "serve/server.py", "--engine", "strata", "--config", str(spec),
         "--port", str(MEAS_PORT)],
        cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        if not wait_ready(MEAS_PORT, proc, READY_TIMEOUT_S):
            return None
        return generate_once(MEAS_PORT, model)
    finally:
        run_cmd(["pkill", "-g", str(proc.pid)], timeout=15)
        try:
            proc.terminate()
            proc.wait(timeout=60)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        logf.close()
        # let the engine release the GPU before the next boot
        time.sleep(20)


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    metrics = {k: 0.0 for k in KEYS}
    instrument_error = None

    if not SPEC_PATH.exists():
        print(f"candidate spec missing: {SPEC_PATH}", file=sys.stderr)
        emit(metrics)
        return 1
    try:
        model = json.loads(SPEC_PATH.read_text(encoding="utf-8-sig")).get("model_name") or "default"
    except Exception:
        model = "default"

    # GPU exclusivity: take production down.
    if not service_ctl("stop"):
        instrument_error = "could not stop strata.service (no working systemctl path)"
    if instrument_error is None:
        wait_gpu_free()

    results = []
    try:
        if instrument_error is None:
            for i in range(BOOTS):
                r = boot_and_measure(i, model)
                if r is not None and r["completion_tokens"] >= GEN_TOKENS:
                    results.append(r)
    finally:
        # Production restore ALWAYS runs, whatever happened above.
        restored = False
        if service_ctl("start"):
            t0 = time.time()
            while time.time() - t0 < MODEL_READY_TIMEOUT_S:
                if http_json(f"http://127.0.0.1:{PROD_PORT}/v1/models", timeout=10.0) is not None:
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
    metrics["runs_complete"] = len(results) / BOOTS

    emit(metrics)
    if instrument_error:
        print(instrument_error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
