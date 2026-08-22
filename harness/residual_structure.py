#!/usr/bin/env python3
"""Is r(k) = g(k)/E(k) structureless, as H1's i.i.d. Exp(1) model implies?

H1 says the normalised location of g(k) is Exp(1). If that is the whole story, r(k) should
correlate with NOTHING about k -- not its factorisation, not its digits, not its neighbours.
Any surviving correlation would be a genuinely new handle on where g(k) lands.

Method: Spearman rank correlation of log r(k) against a battery of features of k, with a
Bonferroni correction (we test many features, so an uncorrected p<0.05 proves nothing).
"""
from __future__ import annotations
import json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref


def spearman(x, y):
    n = len(x)
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    rho = num / den if den else 0.0
    z = rho * math.sqrt(n - 1)
    p = math.erfc(abs(z) / math.sqrt(2))
    return rho, p


def factorise(k):
    f, d, x = {}, 2, k
    while d * d <= x:
        while x % d == 0:
            f[d] = f.get(d, 0) + 1; x //= d
        d += 1
    if x > 1:
        f[x] = f.get(x, 0) + 1
    return f


def main():
    rows = json.load(open(os.path.join(ROOT, "overnight", "2026-08-21", "patterns_gk.json")))
    rows = [r for r in rows if r["k"] >= 10]           # tiny k are degenerate
    ks = [r["k"] for r in rows]
    lr = [r["log10_ratio"] for r in rows]
    feats = {}
    feats["k"] = [float(k) for k in ks]
    feats["popcount(k)"] = [float(bin(k).count("1")) for k in ks]
    feats["k mod 2"] = [float(k % 2) for k in ks]
    feats["is prime"] = [1.0 if len(factorise(k)) == 1 and list(factorise(k).values())[0] == 1 else 0.0 for k in ks]
    feats["omega(k) distinct pf"] = [float(len(factorise(k))) for k in ks]
    feats["largest prime factor"] = [float(max(factorise(k))) for k in ks]
    feats["smallest prime factor"] = [float(min(factorise(k))) for k in ks]
    feats["#divisors"] = [float(math.prod(e + 1 for e in factorise(k).values())) for k in ks]
    feats["frac to next pow2"] = [math.log2(k) - int(math.log2(k)) for k in ks]
    feats["#base-2 digits"] = [float(k.bit_length()) for k in ks]
    feats["sum base-2 digits/len"] = [bin(k).count("1") / k.bit_length() for k in ks]
    feats["#primes <= k"] = [float(len(ref.primes_upto(k))) for k in ks]
    # neighbour: does r(k) know about r(k-1)?
    lag = [lr[i - 1] for i in range(1, len(lr))]
    cur = lr[1:]

    n = len(rows)
    tests = len(feats) + 1
    alpha = 0.05 / tests
    print(f"n = {n} terms (k=10..375), {tests} tests, Bonferroni alpha = {alpha:.5f}\n")
    print(f"{'feature':<26} {'spearman rho':>13} {'p':>10}   {'verdict':<10}")
    print("-" * 66)
    out = []
    for name, v in feats.items():
        rho, p = spearman(v, lr)
        sig = "SIGNIFICANT" if p < alpha else ("(nominal)" if p < 0.05 else "")
        print(f"{name:<26} {rho:>13.4f} {p:>10.4f}   {sig}")
        out.append({"feature": name, "rho": rho, "p": p, "significant": p < alpha})
    rho, p = spearman(lag, cur)
    sig = "SIGNIFICANT" if p < alpha else ("(nominal)" if p < 0.05 else "")
    print(f"{'r(k-1) [serial corr]':<26} {rho:>13.4f} {p:>10.4f}   {sig}")
    out.append({"feature": "serial r(k-1)", "rho": rho, "p": p, "significant": p < alpha})
    nsig = sum(1 for o in out if o["significant"])
    print(f"\n{nsig} of {tests} features significant after Bonferroni correction.")
    if nsig == 0:
        print("=> No detectable structure. Consistent with r(k) being i.i.d. Exp(1):")
        print("   H1 appears to capture essentially all the predictable signal in g(k).")
    json.dump(out, open(os.path.join(ROOT, "overnight", "2026-08-21", "residual_structure.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
