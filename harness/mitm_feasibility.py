#!/usr/bin/env python3
"""Feasibility study: can meet-in-the-middle break the M <~ E ceiling?

TODAY. A block [tM,(t+1)M) is scanned in full, because the mixed-radix index order has no
relation to the order of n. So M must stay <~ E = 1/density_all, or the final partial block
is mostly wasted. That cap is what limits how many primes the wheel can absorb, and it is
the term that makes k=376 cost ~1e20 residue checks.

THE IDEA. Wheel-compatible residues are r = sum_p a_p*c_p mod M (c_p the CRT basis,
a_p in A_p). Split the wheel primes into two halves; enumerate the R1 partial sums of one
half and the R2 of the other (R1*R2 = R). Sort the second list. For each s1, the residues
with (s1+s2) mod M < B form a CONTIGUOUS interval of s2 values, so binary search yields
exactly the r < B without touching the other R - R*B/M combinations.

    cost ~ R1 + R2*log(R2) + (matches)  ~  sqrt(R) + B*density_wheel      [B = search bound]
    memory ~ sqrt(R) machine words

M no longer needs to be <~ E, so the wheel can absorb far more primes. This script asks
what that is actually worth, by minimising max(sqrt(R), E*density_wheel) over the same
DP frontier the v3 planner already computes.
"""
from __future__ import annotations
import math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref

SCALE = 64          # DP cells per bit of log2(M)
MAXBITS = 4096      # M may now be astronomically larger than E


def frontier(k):
    """min achievable log2(R) for each log2(M) bucket, over primes <= k."""
    ps = ref.primes_upto(k)
    info = {p: ref.allowed_residues(k, p) for p in ps}
    logE = -sum(math.log2(len(v[1]) / v[0]) for v in info.values())
    w = [max(1, round(math.log2(info[p][0]) * SCALE)) for p in ps]
    a = [math.log2(len(info[p][1])) for p in ps]
    cap = min(MAXBITS * SCALE, sum(w))
    INF = float("inf")
    dp = [INF] * (cap + 1); dp[0] = 0.0
    for i in range(len(ps)):
        wi, ai = w[i], a[i]
        for c in range(cap, wi - 1, -1):
            prev = dp[c - wi]
            if prev < INF and prev + ai < dp[c]:
                dp[c] = prev + ai
    return dp, cap, logE


def main():
    mem_gb = float(os.environ.get("MITM_MEM_GB", "0")) or None
    ks = [int(x) for x in sys.argv[1:]] or [250, 300, 375, 376]
    if mem_gb:
        cap_logR = 2 * math.log2(mem_gb * 1e9 / 8)
        print(f"# memory cap {mem_gb:g} GB  =>  log2(R) <= {cap_logR:.1f}\n")
    print(f"{'k':>5} {'log2 E':>8} {'today log2':>11} {'MITM log2':>10} {'gain':>10} "
          f"{'log2 M*':>8} {'mem GB':>9}")
    for k in ks:
        dp, cap, logE = frontier(k)
        today = mitm = float("inf")
        bestM = bestR = 0
        for c in range(cap + 1):
            logR = dp[c]
            if logR == float("inf"):
                continue
            logM = c / SCALE
            # today: must keep M <~ E; cost = R/(1-e^{-M/E}) with the exponential model
            x = 2.0 ** min(logM - logE, 60.0)
            denom = -math.expm1(-x) if x < 40 else 1.0
            today = min(today, logR - math.log2(denom))
            # MITM: cost ~ max(sqrt(R), E*density_wheel) = max(R/2, E + R - M) in log2
            if mem_gb and logR > cap_logR:
                continue
            c_mitm = max(logR / 2.0, logE + logR - logM)
            if c_mitm < mitm:
                mitm, bestM, bestR = c_mitm, logM, logR
        mem = 2.0 ** (bestR / 2.0) * 8 / 1e9
        print(f"{k:>5} {logE:8.1f} {today:11.1f} {mitm:10.1f} {2**(today-mitm):10.2e} "
              f"{bestM:8.1f} {mem:9.2e}")
    print("\ncolumns are log2(residue-checks); 'gain' is today/MITM as a plain ratio")
    print("'mem GB' is the meet-in-the-middle table at the cost-optimal split -- the")
    print("binding practical constraint, and the reason this is a study and not a patch.")


if __name__ == "__main__":
    main()
