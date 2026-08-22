#!/usr/bin/env python3
"""Exact CPU check of the v7 incremental-filter identity.

v7 carries the filter residue in 32-bit arithmetic across the inner-ring sweep instead of
recomputing a 64-bit modulo per residue. This test proves the substitution is exact:

    n mod q_j == (BQ_j + T0Q[j][d0] + (c ? q_j - (M mod q_j) : 0)) mod q_j

    BQ_j       = (b mod q_j + (t*M mod q_j)) mod q_j      -- once per outer iteration
    T0Q[j][d0] = term0[d0] mod q_j                        -- host-precomputed table
    s          = b + term0[d0];  c = 1 iff s >= M;  s -= c*M
    n          = t*M + s

It also asserts the range assumption the kernel relies on (the sum is < 3*q_j, so two
conditional subtracts always suffice). Pure CPU -- run it before spending any GPU on v7.
"""
import argparse, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sieve"))
import gpu_sieve_v7 as v7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ks", default="30,60,100,200,280,377,378")
ap.add_argument("--trials", type=int, default=40)
ap.add_argument("--seed", type=int, default=20260822)
a = ap.parse_args()
random.seed(a.seed)

total = bad = 0
for k in [int(x) for x in a.ks.split(",")]:
    P = v7.Plan(k)
    if not P.radix:
        continue
    r0 = P.radix[0]
    offs, terms = [], []
    for j, (q, res) in enumerate(P.rings):
        offs.append(len(terms))
        terms.extend((x * P.basis[j]) % P.M for x in res)
    inner = terms[offs[0]:offs[0] + r0]
    nf = min(int(v7.TUNABLES.get("fast_filters", 4)), len(P.rest))
    for _ in range(a.trials):
        b = 0
        for j in range(1, len(P.rings)):
            b = (b + terms[offs[j] + random.randrange(P.radix[j])]) % P.M
        t = random.randrange(0, 5)
        for d0 in random.sample(range(r0), min(r0, 6)):
            s = b + inner[d0]
            c = 1 if s >= P.M else 0
            s -= c * P.M
            n = t * P.M + s
            for j in range(nf):
                q = P.info[P.rest[j]][0]
                BQ = (b % q + (t * P.M) % q) % q
                m = BQ + inner[d0] % q + ((q - (P.M % q)) % q if c else 0)
                assert m < 3 * q, f"k={k}: range assumption 'sum < 3q' violated"
                while m >= q:
                    m -= q
                total += 1
                if m != n % q:
                    bad += 1
                    print(f"  MISMATCH k={k} j={j} q={q} got={m} want={n % q}")
print(f"checked {total} (k, outer index, d0, filter) combinations")
print("INCREMENTAL FILTER TEST PASS" if not bad else f"INCREMENTAL FILTER TEST FAIL ({bad})")
sys.exit(0 if not bad else 1)
