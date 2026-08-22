#!/usr/bin/env python3
"""Referee checks: reference vs OEIS b-file (k<=60, fast) and GPU sieve vs b-file (k<=140)."""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref

b = {int(l.split()[0]): int(l.split()[1]) for l in open(os.path.join(ROOT, "data", "b003458.txt")) if l.strip()}
for k in range(1, 61):
    assert ref.g_crt(k) == b[k], f"ref mismatch at k={k}"
for k in range(1, 25):
    assert ref.g_bruteforce(k) == b[k], f"brute mismatch at k={k}"
print("reference OK (k<=60)")
if "--gpu" in sys.argv:
    py = os.path.join(ROOT, ".venv", "bin", "python")
    out = subprocess.run([py, os.path.join(ROOT, "sieve", "gpu_sieve.py"), *map(str, range(1, 141))],
                         capture_output=True, text=True, check=True).stdout
    res = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    bad = [r["k"] for r in res if not r.get("matches_bfile")]
    assert len(res) == 140 and not bad, f"GPU mismatches: {bad}"
    print("GPU sieve OK (k<=140)")
