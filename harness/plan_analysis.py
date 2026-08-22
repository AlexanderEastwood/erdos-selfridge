#!/usr/bin/env python3
"""Compare wheel-planner heuristics by EXPECTED residue checks (pure CPU, no GPU).

expected_residue_checks = density_wheel / density_all = 1 / density_rest
                        = exp2( V_all - V_wheel ),   V_p = log2(q_p / |A_p|)
so minimizing work == MAXIMIZING total wheel value V_wheel subject to
sum_p log2(q_p) <= log2(m_cap).  That is a 0/1 knapsack.
"""
from __future__ import annotations
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
import erdos_ref as ref
from math import prod

SCALE = 512  # log2 units per DP cell


def prime_info(k):
    ps = ref.primes_upto(k)
    return {p: ref.allowed_residues(k, p) for p in ps}


def m_cap_for(k, info, m_cap_bits=120, m_cap_factor=0.05):
    density_all = prod(len(v[1]) / v[0] for v in info.values()) if info else 1.0
    cap = min(1 << m_cap_bits, int(m_cap_factor / density_all) or 1)
    return min(cap, 1 << 126), density_all


def greedy(info, m_cap):
    order = sorted(info, key=lambda p: -(math.log(info[p][0] / len(info[p][1])) / math.log(info[p][0])))
    wheel, M = [], 1
    for p in order:
        q = info[p][0]
        if M * q <= m_cap:
            wheel.append(p); M *= q
    return wheel


def knapsack(info, m_cap):
    """0/1 knapsack on log2 weights, exact-product-verified."""
    ps = list(info)
    cap = int(math.log2(m_cap) * SCALE)
    w = [max(1, math.ceil(math.log2(info[p][0]) * SCALE)) for p in ps]
    v = [math.log2(info[p][0] / len(info[p][1])) for p in ps]
    NEG = -1e18
    dp = [0.0] + [NEG] * cap
    take = [bytearray(cap + 1) for _ in ps]
    for i, p in enumerate(ps):
        wi, vi, ti = w[i], v[i], take[i]
        if wi > cap:
            continue
        for c in range(cap, wi - 1, -1):
            cand = dp[c - wi] + vi
            if cand > dp[c]:
                dp[c] = cand; ti[c] = 1
    best_c = max(range(cap + 1), key=lambda c: dp[c])
    wheel, c = [], best_c
    for i in range(len(ps) - 1, -1, -1):
        if take[i][c]:
            wheel.append(ps[i]); c -= w[i]
    wheel.reverse()
    # safety: the log-rounding could in principle overshoot; drop worst items until it fits
    while prod(info[p][0] for p in wheel) > m_cap:
        worst = min(wheel, key=lambda p: math.log2(info[p][0] / len(info[p][1])) / math.log2(info[p][0]))
        wheel.remove(worst)
    return wheel


def checks_for(info, wheel, density_all):
    d_rest = prod(len(info[p][1]) / info[p][0] for p in info if p not in wheel)
    M = prod(info[p][0] for p in wheel)
    d_wheel = prod(len(info[p][1]) / info[p][0] for p in wheel)
    return 1.0 / d_rest, M, d_wheel


if __name__ == "__main__":
    ks = [int(x) for x in sys.argv[1:]] or [150, 190, 200, 250, 300, 375]
    print(f"{'k':>5} {'greedy_checks':>14} {'knap_checks':>14} {'ratio':>8} {'Mg_bits':>8} {'Mk_bits':>8} {'|Wg|':>5} {'|Wk|':>5}")
    for k in ks:
        info = prime_info(k)
        m_cap, d_all = m_cap_for(k, info)
        wg = greedy(info, m_cap); wk = knapsack(info, m_cap)
        cg, Mg, _ = checks_for(info, wg, d_all)
        ck, Mk, _ = checks_for(info, wk, d_all)
        print(f"{k:>5} {cg:14.4e} {ck:14.4e} {cg/ck:8.2f} {Mg.bit_length():8d} {Mk.bit_length():8d} {len(wg):5d} {len(wk):5d}")
