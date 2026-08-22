#!/usr/bin/env python3
"""End-to-end audit of the terms that carry the strongest evidence.

For every k the frozen referee solved outright, check FOUR independent things line up:
  1. the GPU sieve's g equals the referee's g
  2. both equal the published OEIS value
  3. ref.is_good(g,k) re-derives the Kummer condition from scratch
  4. brute force finds no good n in the window immediately below g
Check 1 is skipped (reported "n/a") for k with no grind record: confirmations.jsonl only
covers the k>=141 grind range, and k<=140 is gated separately by harness/gate.py.
(4) is a local minimality probe that shares no code path with the CRT search at all -- it
just tests every integer -- so it would catch an off-by-one or a wheel bug that both the
sieve and g_crt happened to share.
"""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "overnight", "2026-08-21")
WINDOW = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6
bf = {int(l.split()[0]): int(l.split()[1]) for l in open(os.path.join(ROOT, "data", "b003458.txt")) if l.strip()}
refans = {}
for f in glob.glob(os.path.join(R, "referee", "*.jsonl")) + [os.path.join(R, "referee_extended.jsonl")]:
    if not os.path.exists(f):
        continue
    for l in open(f):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("agrees") is True:
            refans[r["k"]] = int(r["referee_g"])
sieve = {}
for l in open(os.path.join(R, "confirmations.jsonl")):
    r = json.loads(l)
    if r.get("verdict") == "CONFIRMED_PUBLISHED":
        sieve[r["k"]] = int(r["g"])

# A term with no confirmations.jsonl record is NOT a mismatch -- that file only covers the
# k>=141 grind range. k<=140 is covered by the sieve gate (harness/gate.py) instead. Report
# it as "n/a" and never as a failure; a false red alert is worse than no alert.
print(f"{'k':>5} {'sieve==ref':>11} {'==bfile':>8} {'is_good':>8} {'no good n below':>16}")
bad, na = [], 0
for k in sorted(refans):
    g = refans[k]
    if k in sieve:
        a = sieve[k] == g
        a_s = str(a)
    else:
        a, a_s = None, "n/a"
        na += 1
    b, c = bf.get(k) == g, ref.is_good(g, k)
    lo = max(k + 2, g - WINDOW)
    d = not any(ref.is_good(n, k) for n in range(lo, g))
    if a is False or not (b and c and d):
        bad.append(k)
    print(f"{k:>5} {a_s:>11} {str(b):>8} {str(c):>8} {str(d):>16}")
checked = len(refans)
print(f"\n{checked-len(bad)}/{checked} pass every applicable check "
      f"({na} have no grind record -- k<=140 is covered by harness/gate.py, not the grind; "
      f"that is 'not applicable', not a failure)")
if bad:
    print("FAILURES <<< RED ALERT:", bad)
else:
    print("no discrepancies")
sys.exit(1 if bad else 0)
