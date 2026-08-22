#!/usr/bin/env python3
"""Vectorised meet-in-the-middle search for g(k). Exact 128-bit throughout.

See harness/mitm_search.py for the derivation. This version replaces the Python loops with
numpy so it can run at frontier scale:

  * S1/S2 built by vectorised 128-bit modular outer-sums (harness/u128.py)
  * the per-s1 binary searches are one batched np.searchsorted over a structured array
  * the variable-length gathers use the standard ragged-index idiom, no Python loop
  * rest-prime filtering is vectorised: n mod q via (hi%q)*(2^64%q) + lo%q, then a table
    lookup, applied as a shrinking mask so later primes only see survivors

Every survivor is still re-verified by the FROZEN referee before acceptance.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from math import prod
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import erdos_ref as ref
import u128
from mitm_search import choose_wheel_mitm, split_wheel

U128 = np.dtype([("hi", "<u8"), ("lo", "<u8")])
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mod_small(hi, lo, q, p64modq):
    """(hi*2^64 + lo) mod q for q < 2^32, elementwise, exact in uint64."""
    q64 = np.uint64(q)
    return ((hi % q64) * np.uint64(p64modq) + (lo % q64)) % q64


def mitm_g(k, mem_gb=8.0, bound_mult=1.0, verbose=True, max_rounds=12):
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
    if n2 > n1:                 # hold the SMALLER half in the sorted table
        w1, w2, n1, n2 = w2, w1, n2, n1

    mhi, mlo = u128.split(M)
    h1, l1 = u128.build_side(w1, info, basis_of, M)
    h2, l2 = u128.build_side(w2, info, basis_of, M)
    A2 = u128.to_struct(h2, l2)
    A2.sort(order=("hi", "lo"))
    if verbose:
        print(f"# k={k} wheel={wheel} M=2^{M.bit_length()} E=2^{log2E:.1f}", file=sys.stderr)
        print(f"# |S1|={h1.size:.3e} |S2|={h2.size:.3e} ({A2.nbytes/1e9:.2f} GB) "
              f"rest={len(rest)} built in {time.time()-t0:.1f}s", file=sys.stderr)

    # rest-prime lookup tables
    tabs = []
    for p in rest:
        q, res = info[p]
        t = np.zeros(q, dtype=bool); t[list(res)] = True
        tabs.append((q, (1 << 64) % q, t))

    def keys_of(vhi, vlo):
        a = np.empty(vhi.size, dtype=U128); a["hi"] = vhi; a["lo"] = vlo
        return a

    def scan(B):
        """Return all wheel-compatible n in [0,B), as (hi,lo) arrays. B <= M."""
        bhi, blo = u128.split(B)
        zero = np.zeros(h1.size, dtype=np.uint64)
        # lo_bound = (-s1) mod M  == (M - s1) mod M
        negh, negl = u128.add_mod(np.full(h1.size, mhi, np.uint64) - h1 -
                                  ((np.full(h1.size, mlo, np.uint64) < l1).astype(np.uint64)),
                                  np.full(h1.size, mlo, np.uint64) - l1,
                                  zero, zero, mhi, mlo)
        hih, hil = u128.add_scalar_mod(negh, negl, bhi, blo, mhi, mlo)
        i = np.searchsorted(A2, keys_of(negh, negl), side="left")
        j = np.searchsorted(A2, keys_of(hih, hil), side="left")
        wrap = (negh > hih) | ((negh == hih) & (negl > hil))
        outs = []
        for lo_i, hi_i, s1h, s1l in ((i, np.where(wrap, A2.size, j), h1, l1),
                                     (np.zeros_like(i), np.where(wrap, j, 0), h1, l1)):
            cnt = (hi_i - lo_i).astype(np.int64)
            cnt[cnt < 0] = 0
            tot = int(cnt.sum())
            if tot == 0:
                continue
            off = np.concatenate(([0], np.cumsum(cnt)[:-1]))
            src = np.repeat(lo_i, cnt) + (np.arange(tot, dtype=np.int64) - np.repeat(off, cnt))
            sel = A2[src]
            rh = np.repeat(s1h, cnt); rl = np.repeat(s1l, cnt)
            nh, nl = u128.add_mod(rh, rl, sel["hi"], sel["lo"], mhi, mlo)
            outs.append((nh, nl))
        if not outs:
            return np.empty(0, np.uint64), np.empty(0, np.uint64)
        return (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]))

    B = max(int(bound_mult * E), 1)
    checked_total = 0
    for _ in range(max_rounds):
        B = min(B, M)
        t1 = time.time()
        nh, nl = scan(B)
        n_cand = nh.size
        keep = np.ones(n_cand, dtype=bool)
        for q, p64, tab in tabs:
            if not keep.any():
                break
            idx = np.flatnonzero(keep)
            r = _mod_small(nh[idx], nl[idx], q, p64)
            keep[idx] = tab[r]
        checked_total += n_cand
        surv = np.flatnonzero(keep)
        best = None
        for s in surv:
            n = (int(nh[s]) << 64) | int(nl[s])
            if n > k + 1 and ref.is_good(n, k) and (best is None or n < best):
                best = n
        if verbose:
            print(f"# B={B:.3e} candidates={n_cand:.3e} survivors={surv.size} "
                  f"scan={time.time()-t1:.1f}s", file=sys.stderr)
        if best is not None:
            return {"k": k, "g": best, "wheel": wheel, "M_bits": M.bit_length(),
                    "S1": int(h1.size), "S2": int(h2.size), "candidates": checked_total,
                    "bound": B, "wall": round(time.time() - t0, 2)}
        if B >= M:
            # g(k) lies at or above the wheel modulus. MITM searches n < M only, so it
            # cannot answer here; say so plainly rather than returning something wrong.
            # The block sieve (sieve/gpu_sieve_v4.py) has no such limit and is the fallback.
            return {"k": k, "status": "MITM_OUT_OF_RANGE", "M_bits": M.bit_length(),
                    "reason": "g(k) >= M; use the block sieve for this k",
                    "candidates": checked_total, "wall": round(time.time() - t0, 2)}
        B = min(B * 4, M)
    raise RuntimeError("round limit reached")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int, nargs="+")
    ap.add_argument("--mem-gb", type=float, default=8.0)
    ap.add_argument("--bound-mult", type=float, default=1.0)
    a = ap.parse_args()
    bf = {}
    for line in open(os.path.join(ROOT, "data", "b003458.txt")):
        t = line.split()
        if len(t) == 2 and t[0].isdigit():
            bf[int(t[0])] = int(t[1])
    for k in a.k:
        r = mitm_g(k, a.mem_gb, a.bound_mult)
        if r.get("status") == "MITM_OUT_OF_RANGE":
            print(json.dumps(r), flush=True); continue
        r["bfile"] = str(bf.get(k)); r["matches_bfile"] = (bf.get(k) == r["g"])
        r["g"] = str(r["g"]); r["bound"] = str(r["bound"])
        print(json.dumps(r), flush=True)
