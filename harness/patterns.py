#!/usr/bin/env python3
"""Where does g(k) actually land, versus the natural random model?

For each prime p <= k, the fraction of n that survive p is
    density_p = prod over base-p digits d of k of (p - d)/p        (closed form: |A_p|/q_p)
If the events were independent and n were "random", the number of n below x that are good
would be ~ x * density_all, so g(k) would be roughly Exponential with mean E(k) = 1/density_all.
This script tests that model against all 375 published terms.  Pure CPU, no GPU.
"""
from __future__ import annotations
import json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref


def log10_density_all(k: int) -> float:
    s = 0.0
    for p in ref.primes_upto(k):
        x = k
        while x:
            s += math.log10((p - x % p) / p)
            x //= p
    return s


def main():
    b, src = {}, {}
    # b-file first, then any known extras (S-W addendum k=376,377 and terms this project verifies)
    for path in (os.path.join(ROOT, "data", "b003458.txt"),
                 os.path.join(ROOT, "data", "known_extra.txt")):
        if not os.path.exists(path):
            continue
        for line in open(path):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                b.setdefault(int(parts[0]), int(parts[1]))
                src.setdefault(int(parts[0]), os.path.basename(path))
    rows = []
    for k in sorted(b):
        if k < 2:
            continue
        ld = log10_density_all(k)          # log10 density_all  (negative)
        lE = -ld                           # log10 E(k) = expected magnitude of g(k)
        lg = math.log10(b[k])
        rows.append({"k": k, "log10_g": lg, "log10_E": lE, "log10_ratio": lg - lE,
                     "ratio": 10 ** (lg - lE), "source": src.get(k)})
    import time
    out = os.path.join(ROOT, "overnight", time.strftime("%Y-%m-%d", time.gmtime()), "patterns_gk.json")
    json.dump(rows, open(out, "w"))
    rs = [r["ratio"] for r in rows]
    lrs = [r["log10_ratio"] for r in rows]
    n = len(rs)
    mean = sum(rs) / n
    srt = sorted(rs)
    med = srt[n // 2]
    print(f"n = {n} known terms (k=2..{max(b)})")
    print(f"ratio r(k) = g(k) / E(k),  E(k) = 1/density_all(k)")
    print(f"  mean   {mean:.4f}")
    print(f"  median {med:.4f}      Exp(1) median would be {math.log(2):.4f}")
    print(f"  min    {srt[0]:.5f}   max {srt[-1]:.4f}")
    for qq in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  q{int(qq*100):02d}    {srt[int(qq*(n-1))]:.4f}", end="")
    print()
    frac_lt = sum(1 for r in rs if r < 1) / n
    print(f"  P(r < 1) = {frac_lt:.3f}   (Exp(1) would give {1-math.exp(-1):.3f})")
    # Kolmogorov-Smirnov against Exp(1)
    D = max(max(abs((i + 1) / n - (1 - math.exp(-x))), abs(i / n - (1 - math.exp(-x))))
            for i, x in enumerate(srt))
    print(f"  KS distance to Exp(1): {D:.4f}   (5% critical ~ {1.36/math.sqrt(n):.4f})")
    # drift: is the model biased with k?
    half = n // 2
    print(f"  mean r, k<=median-k: {sum(rs[:half])/half:.4f}   k>median-k: {sum(rs[half:])/(n-half):.4f}")
    # how good is E(k) as a magnitude predictor?
    sd = (sum((x - sum(lrs)/n) ** 2 for x in lrs) / n) ** 0.5
    print(f"  log10 r: mean {sum(lrs)/n:+.4f}  sd {sd:.4f}  "
          f"=> E(k) predicts log10 g(k) to about +/-{sd:.2f} decades")
    print(f"  (for scale, log10 g ranges {rows[0]['log10_g']:.2f} .. {rows[-1]['log10_g']:.2f})")
    # the tail: where do the largest ratios sit? (relevant to timeout policy: P(r>c)=e^-c)
    top = sorted(rows, key=lambda r: -r["ratio"])[:6]
    print("  largest r(k): " + ", ".join(f"k={r['k']} r={r['ratio']:.2f}" for r in top))
    for c in (3, 5, 8):
        print(f"  observed P(r > {c}) = {sum(1 for r in rs if r > c)/n:.4f}  "
              f"(Exp(1): {math.exp(-c):.4f})", end="")
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
