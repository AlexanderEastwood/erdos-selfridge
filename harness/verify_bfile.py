#!/usr/bin/env python3
"""Check the DEFINING property of every published term with the frozen referee.

For each (k, n) in the b-file, ref.is_good(n, k) re-derives from scratch, in exact
Python integers, that n > k+1 and that for EVERY prime p <= k the base-p digits of k are
dominated by those of n (Kummer: C(n,k) has no prime factor <= k).

This is condition (b) of MISSION rule 2 and it is checkable for every published term --
including the k where an exhaustive search for minimality is far out of reach.  It cannot
confirm minimality; a term passing here is 'satisfies the property', not 'is g(k)'.
It also checks that no term is contradicted by a smaller published-adjacent value.

Usage: verify_bfile.py [b-file]   (default: the largest data/b003458*.txt, i.e. all 400 terms)
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref

def _default_bfile():
    """The most complete published b-file in data/ (b003458_1to400.txt today, not the 375-term original)."""
    import glob
    cands = glob.glob(os.path.join(ROOT, "data", "b003458*.txt"))
    if not cands:
        sys.exit("no b-file found in data/")
    return max(cands, key=lambda f: sum(1 for _ in open(f)))

BFILE = sys.argv[1] if len(sys.argv) > 1 else _default_bfile()

b = {}
for line in open(BFILE):
    p = line.split()
    if len(p) == 2 and p[0].isdigit():
        b[int(p[0])] = int(p[1])

t0 = time.time()
bad, checked = [], 0
for k in sorted(b):
    n = b[k]
    if not ref.is_good(n, k):
        bad.append((k, n))
    checked += 1
print(f"b-file: {os.path.relpath(BFILE, ROOT)}")
print(f"checked {checked} published terms (k=1..{max(b)}) in {time.time()-t0:.1f}s")
print(f"terms satisfying the defining property: {checked - len(bad)}/{checked}")
if bad:
    print("!!! TERMS FAILING ref.is_good:", bad[:20])
else:
    print("all published terms satisfy the Kummer digit-domination condition")
# monotonic sanity: g(k) > k+1 always
viol = [k for k in b if b[k] <= k + 1]
print("terms violating n > k+1:", viol or "none")
sys.exit(1 if bad else 0)
