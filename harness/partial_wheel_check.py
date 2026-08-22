#!/usr/bin/env python3
"""Independent check of the partial-prime-power wheel claim (Sorenson-Webster sec 5).

Read-only analysis. Does NOT touch sieve/gpu_sieve_v5.py -- the point is to verify that
claim with separate code rather than trust it.

Model. For prime p let t_p = #base-p digits of k and a_0..a_{t_p-1} those digits (LSD first).
Taking the low T digits of p into the wheel costs modulus p^T and admits prod_{i<T}(p-a_i)
residues. So with a choice T_p per prime:

    M = prod_p p^{T_p},   R = prod_p prod_{i<T_p} (p - a_i)
    candidates below B  =  B * R / M
    E = 1/density_all,  density_all = prod_p prod_{i<t_p}(p-a_i)/p^{t_p}

Minimising cost = E*R/M means MAXIMISING  V = sum_p log2( p^{T_p} / prod_{i<T_p}(p-a_i) )
subject to  sum_p T_p*log2 p <= log2(m_cap).  Whole prime powers are the special case
T_p in {0, t_p}; allowing every T is a multiple-choice knapsack over strictly more options,
so it can only do better -- the question is by how much.
"""
from __future__ import annotations
import math, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref

SCALE = 96


def digits(x, p):
    d = []
    while x:
        x, r = divmod(x, p); d.append(r)
    return d


def items_for(k):
    """(prime, [(weight_bits, value_bits) for T=1..t_p]) and log2 E."""
    out, log2E = [], 0.0
    for p in ref.primes_upto(k):
        ds = digits(k, p)
        opts, cum = [], 0.0
        for T in range(1, len(ds) + 1):
            cum += math.log2(p - ds[T - 1])
            opts.append((T * math.log2(p), T * math.log2(p) - cum))
        out.append((p, opts))
        log2E += opts[-1][1]          # full prime: log2(q_p/|A_p|)
    return out, log2E


def best_V(items, cap_bits, whole_only=False):
    """Multiple-choice knapsack: max total value under a bit budget."""
    cap = int(cap_bits * SCALE)
    NEG = float("-inf")
    dp = [NEG] * (cap + 1); dp[0] = 0.0
    for p, opts in items:
        nd = dp[:]
        chosen = [opts[-1]] if whole_only else opts
        for wb, vb in chosen:
            w = int(round(wb * SCALE))
            if w > cap:
                continue
            for c in range(cap, w - 1, -1):
                prev = dp[c - w]
                if prev != NEG and prev + vb > nd[c]:
                    nd[c] = prev + vb
        dp = nd
    return max(dp)


def main():
    ks = [int(x) for x in sys.argv[1:]] or [255, 300, 376, 377, 378]
    RATE = 1.05e11
    print(f"measured sieve rate {RATE:.2e} residue checks/s\n")
    print(f"{'k':>4} {'log2 E':>8} {'whole-power':>13} {'partial-power':>14} "
          f"{'gain':>10} {'partial time':>14}")
    for k in ks:
        items, log2E = items_for(k)
        cap = log2E                      # block sieve: M <~ E
        Vw = best_V(items, cap, whole_only=True)
        Vp = best_V(items, cap, whole_only=False)
        cw = 2 ** (log2E - Vw)
        cp_ = 2 ** (log2E - Vp)
        secs = cp_ / RATE
        t = (f"{secs:.0f} s" if secs < 120 else
             f"{secs/60:.0f} min" if secs < 7200 else
             f"{secs/3600:.1f} h" if secs < 172800 else f"{secs/86400:.1f} d")
        print(f"{k:>4} {log2E:8.1f} {cw:13.3e} {cp_:14.3e} {cw/cp_:9.1f}x {t:>14}")


if __name__ == "__main__":
    main()
