#!/usr/bin/env python3
"""Hard correctness gate: a sieve variant must reproduce every b-file term k=1..KMAX."""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sieve = sys.argv[1]
KMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 140
b = {int(l.split()[0]): int(l.split()[1]) for l in open(f"{ROOT}/data/b003458.txt") if l.strip()}
py = f"{ROOT}/.venv/bin/python"
out = subprocess.run([py, sieve, *map(str, range(1, KMAX + 1))],
                     capture_output=True, text=True).stdout
res = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
bad = [r["k"] for r in res if not r.get("matches_bfile")]
dis = [r["k"] for r in res if r.get("status") == "DISAGREEMENT"]
print(f"{os.path.basename(sieve)}: {len(res)}/{KMAX} results, mismatches={bad}, DISAGREEMENTS={dis}")
ok = len(res) == KMAX and not bad and not dis
print("GATE PASS" if ok else "GATE FAIL")
sys.exit(0 if ok else 1)
