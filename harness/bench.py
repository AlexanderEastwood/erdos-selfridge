#!/usr/bin/env python3
"""Fixed benchmark slice for a sieve variant.

Usage: bench.py [--sieve path/to/gpu_sieve.py] [--tune JSON] [--ks 150,170,190] [--timeout 300]
Runs the given sieve file (default: the working tree's) as a subprocess on each k,
hard-gates on agreement with data/b003458.txt, and prints one JSON line:
  {"ok": bool, "wall": s, "checks": n, "rate": checks/s, "per_k": {...}}
A variant that gets ANY k wrong (or crashes/times out) scores ok=false regardless of speed.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")

ap = argparse.ArgumentParser()
ap.add_argument("--sieve", default=os.path.join(ROOT, "sieve", "gpu_sieve.py"))
ap.add_argument("--tune", default=None)
ap.add_argument("--ks", default="120,150,170,190")
ap.add_argument("--timeout", type=float, default=300)
a = ap.parse_args()

per_k, ok, wall, checks = {}, True, 0.0, 0
for k in [int(x) for x in a.ks.split(",")]:
    cmd = [PY, a.sieve, str(k), "--timeout", str(a.timeout)]
    if a.tune:
        cmd += ["--tune", a.tune]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout + 30)
        last = [l for l in p.stdout.splitlines() if l.startswith("{")]
        d = json.loads(last[-1]) if last else {"status": "NO_OUTPUT", "stderr": p.stderr[-500:]}
    except subprocess.TimeoutExpired:
        d = {"status": "TIMEOUT"}
    except Exception as e:  # noqa: BLE001
        d = {"status": "ERROR", "error": str(e)}
    dt = time.time() - t0
    good = d.get("status") == "FOUND" and d.get("matches_bfile") is True
    ok &= good
    wall += dt
    checks += int(d.get("residues_checked", 0) or 0)
    per_k[k] = {"status": d.get("status"), "match": d.get("matches_bfile"), "wall": round(dt, 2),
                "checks": d.get("residues_checked"), "g": str(d.get("g")), "stderr": d.get("stderr")}
print(json.dumps({"ok": ok, "wall": round(wall, 2), "checks": checks,
                  "rate": round(checks / wall) if wall else 0, "per_k": per_k}))
sys.exit(0 if ok else 1)
