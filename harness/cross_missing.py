#!/usr/bin/env python3
"""Cross-check the confirmed terms that the grind's --cross-max-wall threshold skipped.

Those terms carry the three primary evidence items but no second, independent search.
This runs one with a forced alternative wheel (same construction as recross.py: ban a
prime and let the planner re-optimise, so the modulus and block count stay comparable).
"""
import importlib.util, json, math, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
SIEVE = os.path.join(ROOT, "sieve", "gpu_sieve_v4.py")
_sp = importlib.util.spec_from_file_location("sv4", SIEVE)
_sv = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_sv)

path = sys.argv[1]
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 3600
# optional 3rd arg: comma-separated k to restrict to (cost control -- the alternative wheel
# for the most expensive terms can cost hours, which is rarely worth it for evidence that is
# already supplementary)
only = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else None
recs = [json.loads(l) for l in open(path) if l.strip()]
conf = [r for r in recs if r.get("verdict") == "CONFIRMED_PUBLISHED"]
have = {r["k"] for r in recs if r.get("cross_agrees") is True or r.get("recross_agrees") is True}
need = [r for r in conf if r["k"] not in have and (only is None or r["k"] in only)]
print(f"# {len(need)} confirmed terms lack any independent cross-check: {[r['k'] for r in need]}",
      flush=True)
for r in need:
    k, w = r["k"], list(r["wheel"])
    plan = _sv.Plan(k); log2E = -math.log2(plan.density_all)
    forced = None
    for banned in sorted(w, reverse=True):
        sub = {p: v for p, v in plan.info.items() if p != banned}
        cand = _sv._truecost_wheel(sub, log2E)
        if cand and sorted(cand) != sorted(w):
            forced = sorted(cand); break
    if forced is None:
        print(f"k={k}: no distinct wheel available, skipping", flush=True); continue
    t0 = time.time()
    try:
        p = subprocess.run([PY, SIEVE, str(k), "--wheel", json.dumps(forced),
                            "--timeout", str(timeout)], capture_output=True, text=True,
                           timeout=timeout + 120)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        d = json.loads(lines[-1]) if lines else {"status": "NO_OUTPUT"}
    except Exception as e:  # noqa: BLE001
        d = {"status": "ERROR", "error": repr(e)[:200]}
    agree = d.get("status") == "FOUND" and str(d.get("g")) == r["g"]
    rec = {"k": k, "recross": True, "ts": round(time.time(), 1), "primary_wheel": w,
           "forced_wheel": forced, "forced_g": str(d.get("g")), "expected_g": r["g"],
           "recross_agrees": agree, "status": d.get("status"),
           "wall": round(time.time() - t0, 2), "note": "cross-check skipped by --cross-max-wall"}
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"k={k} {w} -> {forced}: {'AGREE' if agree else 'MISMATCH <<< RED ALERT'} "
          f"({rec['wall']}s)", flush=True)
