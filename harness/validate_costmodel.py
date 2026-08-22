#!/usr/bin/env python3
"""Did the cost model that drove v3/v4's planner actually predict reality?

The planner minimises  R / (1 - e^{-M/E}),  whose second factor is the predicted number of
BLOCKS scanned, under H1 (g ~ Exp(E), E = 1/density_all).  Every grind record stores the
block index at which g was actually found, so the prediction is directly checkable against
the runs we did -- and blocks-scanned, not residue checks, is where the model could be wrong.

Reported: observed blocks (= block index + 1) vs predicted mean, and the distribution of
u = 1 - e^{-g/E}, which is Uniform(0,1) exactly when H1 holds.
"""
from __future__ import annotations
import importlib.util, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIEVE = os.path.join(ROOT, "sieve", "gpu_sieve_v4.py")
spec = importlib.util.spec_from_file_location("sv", SIEVE)
sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "overnight", "2026-08-21", "confirmations.jsonl")
recs = [json.loads(l) for l in open(path) if l.strip()]
recs = [r for r in recs if r.get("verdict") == "CONFIRMED_PUBLISHED" and r.get("block") is not None]

rows, us = [], []
for r in recs:
    k = r["k"]
    try:
        plan = sv.Plan(k, r.get("tune"), r.get("wheel"))
    except Exception:
        continue
    E = 1.0 / plan.density_all
    M = plan.M
    x = M / E
    pred_blocks = 1.0 / (-math.expm1(-x)) if x < 40 else 1.0
    obs_blocks = r["block"] + 1
    g = int(r["g"])
    us.append(1.0 - math.exp(-g / E))
    rows.append((k, obs_blocks, pred_blocks, obs_blocks / pred_blocks))

n = len(rows)
tot_obs = sum(r[1] for r in rows)
tot_pred = sum(r[2] for r in rows)
print(f"n = {n} confirmed terms with a recorded block index\n")
print(f"total blocks actually scanned : {tot_obs}")
print(f"total blocks predicted        : {tot_pred:.1f}")
print(f"ratio observed/predicted      : {tot_obs/tot_pred:.3f}   (1.000 = perfectly calibrated)")
ratios = sorted(r[3] for r in rows)
print(f"per-k ratio: median {ratios[n//2]:.3f}  q10 {ratios[int(0.1*(n-1))]:.3f}  "
      f"q90 {ratios[int(0.9*(n-1))]:.3f}")
# H1 => u uniform on (0,1); KS test
us.sort()
D = max(max(abs((i+1)/n - u), abs(i/n - u)) for i, u in enumerate(us))
print(f"\nu = 1 - e^(-g/E) should be Uniform(0,1) under H1:")
print(f"  mean {sum(us)/n:.4f} (0.5 expected), KS distance {D:.4f}, "
      f"5% critical {1.36/math.sqrt(n):.4f}")
print(f"  verdict: {'consistent with H1' if D < 1.36/math.sqrt(n) else 'DEVIATES from H1'}")
worst = sorted(rows, key=lambda r: -r[3])[:5]
print("\nworst over-runs (observed/predicted blocks):")
for k, o, p, ra in worst:
    print(f"  k={k}: {o} blocks vs {p:.1f} predicted ({ra:.1f}x)")
