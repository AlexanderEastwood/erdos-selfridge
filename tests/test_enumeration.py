#!/usr/bin/env python3
"""CPU emulation of the sieve kernels' block enumeration.

Validates, without a GPU: the wheel Plan, the mixed-radix index -> n mapping, that a block's
emitted indices cover range(R) exactly once, and that the minimal good n found by walking a
block equals the published term. Run this whenever a kernel's index scheme changes -- it isolates
"the enumeration is wrong" from "the CUDA translation is wrong".

Usage: test_enumeration.py [--kmax 40] [--engines gpu_sieve_v5,gpu_sieve_v6]
"""
import argparse, importlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sieve"))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kmax", type=int, default=40)
ap.add_argument("--extra", default="45,50,55,60,70,80,90,100")
ap.add_argument("--engines", default="gpu_sieve_v5,gpu_sieve_v6")
ap.add_argument("--max-blocks", type=int, default=40)
a = ap.parse_args()

b = {int(l.split()[0]): int(l.split()[1])
     for l in open(os.path.join(ROOT, "data", "b003458.txt")) if l.strip()}
mods = [importlib.import_module(m) for m in a.engines.split(",")]


def emulate(mod, k, max_blocks):
    P = mod.Plan(k)
    r0 = P.radix[0]
    R_outer = P.R // r0
    assert R_outer * r0 == P.R, f"{mod.__name__} k={k}: R not divisible by radix0"
    for t in range(max_blocks):
        seen, good = set(), []
        for oidx in range(R_outer):
            for d0 in range(r0):
                idx = oidx * r0 + d0
                seen.add(idx)
                n = P.idx_to_n(idx, t)
                if n > k + 1 and all(ref.dominates(n, k, p) for p in P.rest):
                    good.append(n)
        assert seen == set(range(P.R)), f"{mod.__name__} k={k}: index coverage broken"
        if good:
            return min(good), t, P
    return None, None, P


ks = list(range(2, a.kmax + 1)) + [int(x) for x in a.extra.split(",") if x.strip()]
bad = []
for k in ks:
    res = [emulate(m, k, a.max_blocks) for m in mods]
    ns = [r[0] for r in res]
    if len(set(ns)) != 1 or ns[0] != b[k]:
        bad.append((k, ns, b[k]))
    P0 = res[0][2]
    for m, (_, _, P) in zip(mods[1:], res[1:]):
        assert sorted(P0.wheel) == sorted(P.wheel), f"k={k}: {m.__name__} changed the wheel"
        assert P0.M == P.M and P0.R == P.R, f"k={k}: {m.__name__} changed N or R"
print(f"engines: {[m.__name__ for m in mods]}")
print(f"{len(ks)} k values tested; index coverage + idx_to_n + wheel/N/R equivalence verified")
print("ENUMERATION TEST PASS" if not bad else f"ENUMERATION TEST FAIL: {bad}")
sys.exit(0 if not bad else 1)
