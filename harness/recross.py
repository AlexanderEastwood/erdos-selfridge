#!/usr/bin/env python3
"""Re-cross terms whose cross-check happened to select the SAME wheel as the primary run.

A cross-check is only independent evidence if the second run decomposes the search over a
DIFFERENT set of CRT moduli.  When two planners agree on the wheel, the re-run proves
reproducibility but not independence.

Choosing the alternative wheel matters.  The obvious move -- drop the primary wheel's
largest prime -- is a trap: it shrinks M, and the block count is g/M, so the run drowns in
per-block host overhead (k=149 went from 0.4 s to >6 min that way, ~8e5 blocks).  Instead we
BAN one prime and let the planner re-optimise around it, giving a wheel of comparable
modulus and cost but a genuinely different composition.
"""
import json, math, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
SIEVE = os.path.join(ROOT, "sieve", "gpu_sieve_v4.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("sv4", SIEVE)
_sv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sv)


def alt_wheel(k, primary):
    """Best wheel the planner can build while BANNED from one of the primary's primes.

    Keeps the modulus -- and so the block count -- in the same range as the primary run, so
    the re-check costs about what the original did, while forcing a different decomposition.
    """
    plan = _sv.Plan(k)
    log2E = -math.log2(plan.density_all)
    for banned in sorted(primary, reverse=True):
        sub = {p: v for p, v in plan.info.items() if p != banned}
        w = _sv._truecost_wheel(sub, log2E)
        if w and sorted(w) != sorted(primary):
            return sorted(w)
    return None
path = sys.argv[1]
recs = [json.loads(l) for l in open(path) if l.strip()]
need = [r for r in recs if r.get("cross_agrees") is not None and r.get("wheel") == r.get("cross_wheel")]
print(f"# {len(need)} terms need an independent re-cross")
out = []
for r in need:
    k, w = r["k"], list(r["wheel"])
    forced = alt_wheel(k, w)
    if forced is None:
        print(f"k={k}: no distinct alternative wheel found, skipping")
        continue
    t0 = time.time()
    p = subprocess.run([PY, SIEVE, str(k), "--wheel", json.dumps(forced), "--timeout", "900"],
                       capture_output=True, text=True, timeout=1100)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    d = json.loads(lines[-1]) if lines else {"status": "NO_OUTPUT"}
    agree = d.get("status") == "FOUND" and str(d.get("g")) == r["g"]
    rec = {"k": k, "recross": True, "ts": round(time.time(), 1),
           "primary_wheel": w, "forced_wheel": forced,
           "forced_g": str(d.get("g")), "expected_g": r["g"],
           "recross_agrees": agree, "status": d.get("status"),
           "wall": round(time.time() - t0, 2)}
    out.append(rec)
    with open(path, "a") as fh:          # append as we go: crash-resumable
        fh.write(json.dumps(rec) + "\n")
    print(f"k={k} wheel {w} -> {forced}: {'AGREE' if agree else 'MISMATCH <<< RED ALERT'} ({rec['wall']}s)", flush=True)
bad = [r for r in out if not r["recross_agrees"]]
print(f"\n{len(out)-len(bad)}/{len(out)} independent re-crosses agree")
if bad:
    print("RED ALERT:", bad)
