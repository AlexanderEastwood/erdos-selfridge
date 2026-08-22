#!/usr/bin/env python3
"""Meet-in-the-middle search for g(k): break the M <~ E ceiling.

WHY. The block sieve must scan a whole block, so the wheel modulus M has to stay near
E = 1/density_all or the final partial block is wasted. That cap is what limits how many
primes the wheel can absorb, and the wheel's density is what sets the candidate count.

HOW. Wheel-compatible residues are r = sum_p a_p*c_p mod M (c_p the CRT basis, a_p in A_p).
Split the wheel primes into two halves and form
    S1 = { sum_{p in W1} a_p c_p mod M },   S2 = { sum_{p in W2} a_p c_p mod M }
with |S1|*|S2| = R. Sort S2. For each s1, the s2 giving (s1+s2) mod M < B form a CONTIGUOUS
(circular) interval of S2, found by binary search. So we enumerate only the ~R*B/M residues
below the bound instead of all R of them, and M is free to exceed E.

    cost ~ |S1| binary searches + (R*B/M) candidate tests,   memory ~ |S2| * 16 bytes

The candidate-test term dominates in every regime we care about, and a candidate test is the
same operation the block sieve already does -- so the comparison against it is apples to
apples.

Exactness: 128-bit residues are held as a numpy structured dtype (hi, lo) of uint64, which
sorts and searchsorts lexicographically, i.e. as true 128-bit unsigned integers. Every
survivor is re-verified with the FROZEN referee before it is accepted.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from math import prod

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref  # frozen referee, read-only

U128 = np.dtype([("hi", "<u8"), ("lo", "<u8")])
SCALE = 64


def to128(vals):
    """Python ints -> structured (hi, lo) array. Values must be < 2^128."""
    a = np.empty(len(vals), dtype=U128)
    mask = (1 << 64) - 1
    a["hi"] = np.fromiter((v >> 64 for v in vals), dtype=np.uint64, count=len(vals))
    a["lo"] = np.fromiter((v & mask for v in vals), dtype=np.uint64, count=len(vals))
    return a


def from128(a):
    return (int(a["hi"]) << 64) | int(a["lo"])


def choose_wheel_mitm(k, info, log2E, mem_bytes, min_m_over_e=4.0):
    """DP for min log2(R) at each log2(M) bucket, then minimise max(sqrt(R), E*R/M).

    MITM only searches n < M, so the modulus MUST be comfortably larger than E or g(k) can
    simply lie above it (P(g >= M) = e^{-M/E}).  We therefore require M >= min_m_over_e * E
    where the available primes allow it -- without this the optimiser happily picks a small
    M, because the cost model alone does not know the search would then miss the answer."""
    ps = list(info)
    w = [max(1, round(math.log2(info[p][0]) * SCALE)) for p in ps]
    a = [math.log2(len(info[p][1])) for p in ps]
    cap = min(4096 * SCALE, sum(w))
    INF = float("inf")
    dp = [INF] * (cap + 1); dp[0] = 0.0
    take = [bytearray(cap + 1) for _ in ps]
    for i in range(len(ps)):
        wi, ai, ti = w[i], a[i], take[i]
        for c in range(cap, wi - 1, -1):
            prev = dp[c - wi]
            if prev < INF and prev + ai < dp[c]:
                dp[c] = prev + ai; ti[c] = 1
    cap_logR = 2 * math.log2(mem_bytes / 16)     # optimistic: assumes a perfect 50/50 split
    total_logM = sum(w) / SCALE
    need_logM = min(log2E + math.log2(min_m_over_e), total_logM)

    def rebuild(c):
        wh, cc = [], c
        for i in range(len(ps) - 1, -1, -1):
            if take[i][cc]:
                wh.append(ps[i]); cc -= w[i]
        wh.sort()
        return wh

    # Rank buckets by predicted cost, then accept the first whose ACTUAL split fits memory.
    # The DP bounds sqrt(R), but |A_p| are lumpy, so the best achievable |S2| can be far
    # above sqrt(R) -- checking the real split is what keeps the memory budget honest.
    cands = []
    for c in range(cap + 1):
        logR = dp[c]
        if logR == INF or logR > cap_logR:
            continue
        logM = c / SCALE
        if logM < need_logM:
            continue
        cands.append((max(logR / 2.0, log2E + logR - logM), c))
    cands.sort()
    budget = mem_bytes / 16
    for _, c in cands:
        wh = rebuild(c)
        _, _, n1, n2 = split_wheel(wh, info)
        if min(n1, n2) <= budget:          # the smaller half is the one we sort and hold
            return wh
    return None


def split_wheel(wheel, info):
    """Partition into two halves with |S1| and |S2| as balanced as possible (greedy)."""
    order = sorted(wheel, key=lambda p: -len(info[p][1]))
    w1, w2, n1, n2 = [], [], 1, 1
    for p in order:
        if n1 <= n2:
            w1.append(p); n1 *= len(info[p][1])
        else:
            w2.append(p); n2 *= len(info[p][1])
    return sorted(w1), sorted(w2), n1, n2


def build_side(primes, info, basis_of, M):
    """All CRT partial sums for one half, as Python ints mod M."""
    acc = [0]
    for p in primes:
        c = basis_of[p]
        add = [(a * c) % M for a in info[p][1]]
        acc = [(x + y) % M for x in acc for y in add]
    return acc


def mitm_g(k, mem_gb=8.0, bound_mult=4.0, verbose=True):
    t0 = time.time()
    ps = ref.primes_upto(k)
    info = {p: ref.allowed_residues(k, p) for p in ps}
    density_all = prod(len(v[1]) / v[0] for v in info.values())
    E = 1.0 / density_all
    log2E = -math.log2(density_all)

    wheel = choose_wheel_mitm(k, info, log2E, mem_gb * 1e9)
    if not wheel:
        raise RuntimeError("no wheel fits the memory budget")
    M = prod(info[p][0] for p in wheel)
    basis_of = {}
    for p in wheel:
        q = info[p][0]; Mi = M // q
        basis_of[p] = Mi * pow(Mi, -1, q)
    rest = sorted([p for p in ps if p not in wheel],
                  key=lambda p: len(info[p][1]) / info[p][0])

    w1, w2, n1, n2 = split_wheel(wheel, info)
    if verbose:
        print(f"# k={k} wheel={wheel}", file=sys.stderr)
        print(f"# M=2^{M.bit_length()}  E=2^{log2E:.1f}  R=2^{math.log2(n1*n2):.1f} "
              f"|S1|=2^{math.log2(n1):.1f} |S2|=2^{math.log2(n2):.1f} rest={len(rest)}",
              file=sys.stderr)

    S1 = build_side(w1, info, basis_of, M)
    S2 = build_side(w2, info, basis_of, M)
    A2 = np.sort(to128(S2), order=("hi", "lo"))
    if verbose:
        print(f"# built S1={len(S1)} S2={len(S2)} in {time.time()-t0:.1f}s "
              f"(S2 table {A2.nbytes/1e9:.2f} GB)", file=sys.stderr)

    allowed = {p: set(info[p][1]) for p in rest}

    def scan(B):
        """All wheel-compatible n < B, as Python ints. B <= M."""
        out = []
        key = np.empty(1, dtype=U128)
        for s1 in S1:
            lo = (-s1) % M
            hi = (lo + B) % M
            if lo < hi:
                ranges = ((lo, hi),)
            else:
                ranges = ((lo, M), (0, hi))
            for a, b in ranges:
                key["hi"] = a >> 64; key["lo"] = a & ((1 << 64) - 1)
                i = np.searchsorted(A2, key[0], side="left")
                key["hi"] = b >> 64; key["lo"] = b & ((1 << 64) - 1)
                j = np.searchsorted(A2, key[0], side="left")
                for t in range(int(i), int(j)):
                    out.append((s1 + from128(A2[t])) % M)
        return out

    B = int(bound_mult * E)
    while True:
        if B > M:
            B = M
        t1 = time.time()
        cands = scan(B)
        tested = 0
        good = []
        for n in sorted(cands):
            if n <= k + 1:
                continue
            tested += 1
            ok = True
            for p in rest:
                if (n % info[p][0]) not in allowed[p]:
                    ok = False; break
            if ok and ref.is_good(n, k):
                good.append(n); break
        if verbose:
            print(f"# B={B:.3e} candidates={len(cands)} tested={tested} "
                  f"scan={time.time()-t1:.1f}s", file=sys.stderr)
        if good:
            return {"k": k, "g": good[0], "wheel": wheel, "M_bits": M.bit_length(),
                    "S1": len(S1), "S2": len(S2), "candidates": len(cands),
                    "bound": B, "wall": round(time.time() - t0, 2)}
        if B >= M:
            raise RuntimeError("exhausted the full modulus without a hit")
        B = min(B * 4, M)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int, nargs="+")
    ap.add_argument("--mem-gb", type=float, default=8.0)
    ap.add_argument("--bound-mult", type=float, default=4.0)
    a = ap.parse_args()
    bf = {}
    for line in open(os.path.join(ROOT, "data", "b003458.txt")):
        t = line.split()
        if len(t) == 2 and t[0].isdigit():
            bf[int(t[0])] = int(t[1])
    for k in a.k:
        r = mitm_g(k, a.mem_gb, a.bound_mult)
        r["bfile"] = str(bf.get(k))
        r["matches_bfile"] = (bf.get(k) == r["g"])
        r["g"] = str(r["g"]); r["bound"] = str(r["bound"])
        print(json.dumps(r), flush=True)
