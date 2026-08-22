#!/usr/bin/env python3
"""Push MISSION rule 2(a) -- "CUDA sieve result == frozen reference result" -- above k=140.

The staging brief notes g_crt is unusable past k~140.  That is true at its DEFAULT
max_wheel_modulus=1e9; the parameter is part of the frozen function's public signature, and
raising it shrinks the residual search by orders of magnitude at the cost of RAM for the
residue list.  Nothing in ref/ is modified -- we only call it with a different argument.

A term confirmed here has the frozen referee's own independent, exhaustive answer, which is
strictly stronger evidence than b-file agreement plus a cross-wheel GPU re-run.
"""
import json, os, resource, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref  # frozen; called read-only

k = int(sys.argv[1])
cap = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 15
out = sys.argv[3] if len(sys.argv) > 3 else None
b = {int(l.split()[0]): int(l.split()[1]) for l in open(f"{ROOT}/data/b003458.txt") if l.strip()}
t0 = time.time()
g = ref.g_crt(k, max_wheel_modulus=cap)
wall = time.time() - t0
rec = {"k": k, "referee_g": str(g), "bfile": str(b.get(k)), "agrees": g == b.get(k),
       "max_wheel_modulus": cap, "wall": round(wall, 1),
       "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
       "method": "ref.g_crt (FROZEN referee, exhaustive)", "ts": round(time.time(), 1)}
print(json.dumps(rec), flush=True)
if out:
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
