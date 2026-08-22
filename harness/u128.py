#!/usr/bin/env python3
"""Vectorised 128-bit modular arithmetic on numpy uint64 (hi, lo) pairs.

The MITM builder has to form millions-to-billions of CRT partial sums mod M, with M up to
~2^110. Python ints are exact but far too slow at that scale, and numpy has no uint128, so
carry and comparison are done by hand. Everything here is exact -- no floats anywhere.
"""
from __future__ import annotations
import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def split(v: int):
    return np.uint64(v >> 64), np.uint64(v & 0xFFFFFFFFFFFFFFFF)


def add_mod(ahi, alo, bhi, blo, mhi, mlo):
    """(a + b) mod m for a,b < m < 2^128, elementwise. Exact."""
    lo = alo + blo
    carry = (lo < alo).astype(np.uint64)          # unsigned wraparound detects carry
    hi = ahi + bhi + carry
    # conditional subtract of m while (hi,lo) >= (mhi,mlo)
    ge = (hi > mhi) | ((hi == mhi) & (lo >= mlo))
    borrow = ((lo < mlo) & ge).astype(np.uint64)
    lo = np.where(ge, lo - mlo, lo)
    hi = np.where(ge, hi - mhi - borrow, hi)
    return hi, lo


def add_scalar_mod(ahi, alo, bhi, blo, mhi, mlo):
    """(a + scalar b) mod m, elementwise over a."""
    return add_mod(ahi, alo,
                   np.full(ahi.shape, bhi, dtype=np.uint64),
                   np.full(alo.shape, blo, dtype=np.uint64), mhi, mlo)


def build_side(primes, info, basis_of, M, dtype_pair=True):
    """All CRT partial sums for one half of the wheel, as (hi, lo) uint64 arrays.

    Builds by repeated outer-sum: acc = { acc + a*c_p mod M } for each wheel prime.
    """
    mhi, mlo = split(M)
    hi = np.zeros(1, dtype=np.uint64)
    lo = np.zeros(1, dtype=np.uint64)
    for p in primes:
        c = basis_of[p]
        terms = [(a * c) % M for a in info[p][1]]
        thi = np.fromiter((t >> 64 for t in terms), dtype=np.uint64, count=len(terms))
        tlo = np.fromiter((t & 0xFFFFFFFFFFFFFFFF for t in terms), dtype=np.uint64,
                          count=len(terms))
        n, m = hi.size, len(terms)
        HI = np.repeat(hi, m); LO = np.repeat(lo, m)
        THI = np.tile(thi, n); TLO = np.tile(tlo, n)
        hi, lo = add_mod(HI, LO, THI, TLO, mhi, mlo)
    return hi, lo


def to_struct(hi, lo):
    """Pack into a structured array that sorts as a true 128-bit unsigned integer."""
    a = np.empty(hi.size, dtype=np.dtype([("hi", "<u8"), ("lo", "<u8")]))
    a["hi"] = hi; a["lo"] = lo
    return a


def check():
    """Exactness self-test against Python ints."""
    import random
    rng = random.Random(12345)
    for _ in range(200):
        M = rng.randrange(3, 1 << 110) | 1
        a = [rng.randrange(M) for _ in range(64)]
        b = [rng.randrange(M) for _ in range(64)]
        mhi, mlo = split(M)
        ahi = np.array([x >> 64 for x in a], dtype=np.uint64)
        alo = np.array([x & 0xFFFFFFFFFFFFFFFF for x in a], dtype=np.uint64)
        bhi = np.array([x >> 64 for x in b], dtype=np.uint64)
        blo = np.array([x & 0xFFFFFFFFFFFFFFFF for x in b], dtype=np.uint64)
        rhi, rlo = add_mod(ahi, alo, bhi, blo, mhi, mlo)
        for i in range(64):
            want = (a[i] + b[i]) % M
            got = (int(rhi[i]) << 64) | int(rlo[i])
            if want != got:
                return False, (M, a[i], b[i], want, got)
    return True, None


if __name__ == "__main__":
    ok, detail = check()
    print("u128 add_mod exactness vs Python ints:", "PASS" if ok else f"FAIL {detail}")
