#!/usr/bin/env python3
"""Correctness gate for the GPU MITM search: must reproduce every b-file term in range."""
import io, contextlib, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sieve")); sys.path.insert(0, os.path.join(ROOT, "ref"))
from gpu_mitm import mitm_gpu
lo, hi = int(sys.argv[1]), int(sys.argv[2])
mem = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
bf = {}
for l in open(os.path.join(ROOT, "data", "b003458.txt")):
    t = l.split()
    if len(t) == 2 and t[0].isdigit():
        bf[int(t[0])] = int(t[1])
bad, oor, n, t0 = [], [], 0, time.time()
for k in range(lo, hi + 1):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            r = mitm_gpu(k, mem_gb=mem, verbose=False)
        st = r.get("status")
        if st == "MITM_OUT_OF_RANGE":
            oor.append(k); continue
        if st != "FOUND":
            bad.append((k, st)); continue
        n += 1
        if r["g"] != bf[k]:
            bad.append((k, str(r["g"]), str(bf[k])))
    except Exception as e:
        bad.append((k, "ERR", repr(e)[:100]))
    if k % 20 == 0:
        print(f"  k={k} ok={n} bad={len(bad)} oor={len(oor)} {time.time()-t0:.0f}s", flush=True)
print(f"\nGPU MITM gate k={lo}..{hi}: solved={n} mismatches={bad[:4]} out-of-range={oor}")
print("GATE PASS" if not bad else "GATE FAIL")
sys.exit(0 if not bad else 1)
