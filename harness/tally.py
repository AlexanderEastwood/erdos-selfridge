#!/usr/bin/env python3
"""Summarise the confirmation log."""
import json, collections, sys
path = sys.argv[1] if len(sys.argv) > 1 else "overnight/2026-08-21/confirmations.jsonl"
all_recs = [json.loads(l) for l in open(path) if l.strip()]
recs = [r for r in all_recs if "verdict" in r]
recross = [r for r in all_recs if r.get("recross")]
print("verdicts:", dict(collections.Counter(r["verdict"] for r in recs)))
ks = sorted(r["k"] for r in recs if r["verdict"] == "CONFIRMED_PUBLISHED")
if ks:
    print(f"confirmed: {len(ks)} terms, k={min(ks)}..{max(ks)}")
    lo, hi = min(ks), max(ks)
    gaps = [k for k in range(lo, hi + 1) if k not in ks]
    print("gaps inside that range:", gaps or "none")
same = [r for r in recs if r.get("cross_agrees") is not None and r.get("wheel") == r.get("cross_wheel")]
diff = [r for r in recs if r.get("cross_agrees") is True and r.get("wheel") != r.get("cross_wheel")]
print("cross-wheel agreement (DIFFERENT wheel):", len(diff))
print("cross-run agreement (same wheel, reproducibility only):", len(same))
print("forced-alt-wheel re-crosses:", sum(1 for r in recross if r.get("recross_agrees")),
      "agree /", len(recross), "run")
# cross_agrees=False means EITHER the cross run disagreed (a real red alert) OR the cross
# run never produced a result (killed, timed out). Only the first is a disagreement; do not
# conflate them, and never raise a red alert for a run that simply did not finish.
real = [r for r in recs if r.get("cross_agrees") is False and r.get("cross_status") == "FOUND"]
failed = [r for r in recs if r.get("cross_agrees") is False and r.get("cross_status") != "FOUND"]
real += [r for r in recs if r.get("recross_agrees") is False and r.get("status") == "FOUND"]
print("cross-wheel DISagreement (real, RED ALERT):", len(real), [r["k"] for r in real] or "")
print("cross runs that failed to complete (not a disagreement):", len(failed),
      [r["k"] for r in failed] or "")
print("ref_is_good failures:", sum(1 for r in recs if r.get("ref_is_good") is False))
print("RED ALERTS:", sum(1 for r in recs if r.get("RED_ALERT")))
print("total sieve wall (s):", round(sum((r.get("wall") or 0) + (r.get("cross_wall") or 0) for r in recs), 1))
tot = sum(r.get("residues_checked") or 0 for r in recs)
print(f"total residue checks: {tot:.4e}")
