#!/usr/bin/env python3
"""How fast does g(k) grow?  Fit log g(k) against candidate laws over the 375 terms.

Erdos-Selfridge-type questions are usually about the GROWTH RATE, so this fits
    log g(k) ~ C * k^alpha        (equivalently log log g ~ log C + alpha * log k)
by least squares on log log g vs log k, and compares against the deterministic quantity
    log E(k) = -log density_all(k) = sum_p sum_{base-p digits d of k} log(p/(p-d))
which H1 says is the right centring.  Reported with the caveat that a power-law fit over one
decade of k is weak evidence for any particular exponent.
"""
from __future__ import annotations
import math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref


def log_E(k):
    s = 0.0
    for p in ref.primes_upto(k):
        x = k
        while x:
            s += math.log(p / (p - x % p))
            x //= p
    return s


def fit(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    a = sxy/sxx; b = my - a*mx
    ss_res = sum((y - (a*x+b))**2 for x, y in zip(xs, ys))
    ss_tot = sum((y-my)**2 for y in ys)
    return a, b, 1 - ss_res/ss_tot


b = {}
for line in open(os.path.join(ROOT, "data", "b003458.txt")):
    p = line.split()
    if len(p) == 2 and p[0].isdigit():
        b[int(p[0])] = int(p[1])
ks = [k for k in sorted(b) if k >= 20]
lg = [math.log(b[k]) for k in ks]
lE = [log_E(k) for k in ks]

print(f"n = {len(ks)} terms, k = {ks[0]}..{ks[-1]}\n")
a, c, r2 = fit([math.log(k) for k in ks], [math.log(x) for x in lg])
print(f"log g(k) ~ C * k^alpha       : alpha = {a:.4f}, C = {math.exp(c):.4f}, R^2 = {r2:.5f}")
a2, c2, r22 = fit([math.log(k) for k in ks], [math.log(x) for x in lE])
print(f"log E(k) ~ C * k^alpha       : alpha = {a2:.4f}, C = {math.exp(c2):.4f}, R^2 = {r22:.5f}")
a3, c3, r23 = fit(ks, lg)
print(f"log g(k) ~ a*k + b (linear)  : a = {a3:.5f}, R^2 = {r23:.5f}")
a4, c4, r24 = fit([k/math.log(k) for k in ks], lg)
print(f"log g(k) ~ a*k/log k         : a = {a4:.5f}, R^2 = {r24:.5f}")
print(f"\nlog E(k) explains log g(k) with R^2 = {fit(lE, lg)[2]:.5f}"
      f"  (slope {fit(lE, lg)[0]:.4f}, 1.0 expected under H1)")
print("\nCaveat: k spans barely one decade, so competing laws are hard to separate -- the")
print("R^2 values above differ by little.  The useful statement is the relative one: the")
print("deterministic log E(k) tracks log g(k) far better than any smooth function of k.")
