#!/usr/bin/env python3
"""FROZEN reference implementation of the Erdős–Selfridge function g(k) (OEIS A003458).

g(k) = least n > k+1 such that C(n, k) has no prime factor <= k.
By Kummer's theorem: for every prime p <= k, every base-p digit of k is <= the
corresponding base-p digit of n (i.e. subtracting k from n in base p needs no borrow).

This file is the REFEREE. Do not edit it. Pure Python, arbitrary precision, no tricks.
Two independent search strategies are provided:
  * g_bruteforce(k)  -- scans n = k+2, k+3, ...   (tiny k only)
  * g_crt(k)         -- exhaustive CRT-wheel search in increasing blocks (medium k)
Both are exact and return the minimal n.
"""
from __future__ import annotations
import sys
from itertools import product
from math import prod


def primes_upto(k: int) -> list[int]:
    if k < 2:
        return []
    s = bytearray([1]) * (k + 1)
    s[0] = s[1] = 0
    for i in range(2, int(k ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(k + 1) if s[i]]


def digits(x: int, p: int) -> list[int]:
    """Base-p digits of x, least significant first. digits(0,p) == []."""
    d = []
    while x:
        x, r = divmod(x, p)
        d.append(r)
    return d


def dominates(n: int, k: int, p: int) -> bool:
    """True iff every base-p digit of k is <= the corresponding digit of n
    (equivalently p does not divide C(n,k), Kummer)."""
    while k:
        if k % p > n % p:
            return False
        k //= p
        n //= p
    return True


def is_good(n: int, k: int) -> bool:
    """True iff n > k+1 and C(n,k) has no prime factor <= k."""
    if n <= k + 1:
        return False
    return all(dominates(n, k, p) for p in primes_upto(k))


def g_bruteforce(k: int, limit: int | None = None) -> int | None:
    n = k + 2
    ps = primes_upto(k)
    ps.sort(key=lambda p: -p)  # large primes reject fastest for 2-digit k
    while limit is None or n <= limit:
        if all(dominates(n, k, p) for p in ps):
            return n
        n += 1
    return None


def allowed_residues(k: int, p: int) -> tuple[int, list[int]]:
    """(q, sorted list of r in [0,q) whose base-p digits dominate k's), q = p**len(digits(k,p))."""
    kd = digits(k, p)
    q = p ** len(kd)
    res = []
    # mixed radix: digit i ranges over [kd[i], p-1]
    for combo in product(*[range(d, p) for d in kd]):
        r = 0
        for i, c in enumerate(combo):
            r += c * p ** i
        res.append(r)
    res.sort()
    return q, res


def _crt_basis(moduli: list[int]) -> tuple[int, list[int]]:
    M = prod(moduli)
    basis = []
    for q in moduli:
        Mi = M // q
        basis.append(Mi * pow(Mi, -1, q))
    return M, basis


def g_crt(k: int, wheel: list[int] | None = None, max_wheel_modulus: int = 10 ** 9,
          verbose: bool = False) -> int:
    """Exhaustive, exact search. Picks wheel primes (largest density gain per modulus
    size) with product <= max_wheel_modulus, enumerates every residue class mod M that
    satisfies the wheel primes, and scans blocks [tM, (t+1)M) in increasing t, checking
    the remaining primes with plain Python integers. Returns the minimal good n."""
    ps = primes_upto(k)
    info = {p: allowed_residues(k, p) for p in ps}
    if wheel is None:
        import math
        scored = sorted(ps, key=lambda p: -math.log(info[p][0] / len(info[p][1])) / math.log(info[p][0]))
        wheel, M = [], 1
        for p in scored:
            q = info[p][0]
            if M * q <= max_wheel_modulus:
                wheel.append(p); M *= q
    rest = [p for p in ps if p not in wheel]
    rest.sort(key=lambda p: len(info[p][1]) / info[p][0])  # most restrictive first
    moduli = [info[p][0] for p in wheel]
    M, basis = _crt_basis(moduli)
    sets = [info[p][1] for p in wheel]
    # all residues r mod M compatible with wheel primes
    residues = []
    for combo in product(*sets):
        residues.append(sum(a * b for a, b in zip(combo, basis)) % M)
    residues.sort()
    if verbose:
        print(f"k={k} wheel={wheel} M={M} residues={len(residues)} rest={len(rest)}", file=sys.stderr)
    t = 0
    while True:
        base = t * M
        for r in residues:
            n = base + r
            if n <= k + 1:
                continue
            if all(dominates(n, k, p) for p in rest):
                return n  # residues sorted & blocks increasing => first hit is minimal
        t += 1


def verify_term(k: int, n: int, recheck_minimality: bool = True, **crt_kwargs) -> dict:
    """Independent check of a claimed term. Returns dict with 'good' (digit domination
    for all primes <= k) and, if requested, 'minimal' (no smaller good n exists)."""
    out = {"k": k, "n": n, "good": is_good(n, k)}
    if recheck_minimality and out["good"]:
        out["minimal"] = g_crt(k, **crt_kwargs) == n
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Erdős–Selfridge g(k) reference (frozen)")
    ap.add_argument("k", type=int, nargs="+")
    ap.add_argument("--method", choices=["crt", "brute"], default="crt")
    ap.add_argument("--check", type=int, default=None, help="verify this n instead of searching")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    for k in a.k:
        if a.check is not None:
            print(verify_term(k, a.check, recheck_minimality=False))
        elif a.method == "brute":
            print(k, g_bruteforce(k))
        else:
            print(k, g_crt(k, verbose=a.v))
