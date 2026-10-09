"""bench/state_check.py — read-only diagnostic: production/measurement server and
leftover probe state after a possibly-interrupted preflight. Prints JSON, exits 0.
Changes nothing on the machine.
"""
import json
import subprocess
import urllib.request


def http_ok(url: str, timeout: float = 8.0) -> float:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 1.0 if resp.status == 200 else 0.0
    except Exception:
        return 0.0


def pgrep_count(pattern: str) -> float:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return 0.0
        return float(len([ln for ln in r.stdout.splitlines() if ln.strip()]))
    except Exception:
        return -1.0


def main() -> int:
    out = {
        "prod_8090_up": http_ok("http://127.0.0.1:8090/v1/models"),
        "meas_8101_up": http_ok("http://127.0.0.1:8101/v1/models"),
        "probe_procs": pgrep_count("bench/decode_bench.py"),
        "engine_procs": pgrep_count("engine/strata"),
        "server_py_procs": pgrep_count("serve/server.py"),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
