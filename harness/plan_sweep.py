#!/usr/bin/env python3
"""CPU-only planner sweep for gpu_sieve_v5.

Two jobs:
  * sweep TUNABLES (planner/n_mult/cost_scale/m_cap_bits) for a given k and report the plan
    minimising expected_checks_true = R / (1 - e^{-N/E});
  * build an alternate wheel with one or more primes BANNED from the wheel (the confirmation-run
    recipe: ban, then let the planner re-optimise -- never just drop a prime, which shrinks N).

A banned prime is still a FILTER prime (Plan puts every p with wheelT[p] < t_p into .rest), so a
banned-wheel scan remains exhaustive and minimal. Emits --wheel JSON ready for the sieve CLI.

Usage:
  plan_sweep.py 378 --sweep
  plan_sweep.py 378 --ban 2,3 [--tune '{"planner":"truecost"}']
"""
from __future__ import annotations
import argparse, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sieve"))
import gpu_sieve_v5 as v5  # noqa: E402
import erdos_ref as ref    # noqa: E402  (v5 already put ref/ on the path)


def describe(plan) -> dict:
    s = plan.summary()
    return {"wheel": s["wheel"], "N_bits": s["N_bits"], "R": s["residues_per_block"],
            "N_over_E": round(s["N_over_E"], 4), "n_filters": s["n_filters"],
            "checks_true": s["expected_checks_true"], "E_g": s["E_g"]}


def sweep(k: int, grid_scales, grid_caps, grid_mults):
    out = []
    for planner in ("truecost", "knapsack"):
        mults = grid_mults if planner == "knapsack" else [1.0]
        for cs in grid_scales:
            for cap in grid_caps:
                for nm in mults:
                    tun = {"planner": planner, "cost_scale": cs, "m_cap_bits": cap, "n_mult": nm}
                    try:
                        p = v5.Plan(k, tun)
                    except AssertionError as e:
                        out.append({"tun": tun, "error": str(e)}); continue
                    d = describe(p); d["tun"] = tun
                    out.append(d)
    ok = [d for d in out if "error" not in d]
    ok.sort(key=lambda d: d["checks_true"])
    return ok


def banned_wheel(k: int, ban: list[int], tun: dict):
    ps = [p for p in ref.primes_upto(k) if p not in ban]
    full = v5.Plan(k, tun)                      # for E[g]; density uses ALL primes
    log2E = -math.log2(full.density_all)
    wheel = v5.plan_wheel(k, ps, log2E, dict(v5.TUNABLES, **tun))
    plan = v5.Plan(k, tun, wheel=[[p, T] for p, T in wheel])
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ban", default=None, help="comma-separated primes to exclude from the wheel")
    ap.add_argument("--tune", type=json.loads, default={})
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    if a.sweep:
        res = sweep(a.k, [64, 128, 256, 512, 1024], [120, 110, 100, 90, 80], [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        base = v5.Plan(a.k).summary()
        print(f"# k={a.k}  default plan: R={base['residues_per_block']:.4e} "
              f"N_bits={base['N_bits']} checks_true={base['expected_checks_true']:.4e}")
        for d in res[:a.top]:
            print(f"checks_true={d['checks_true']:.4e} R={d['R']:.4e} N_bits={d['N_bits']:3d} "
                  f"N/E={d['N_over_E']:.3f} filters={d['n_filters']:3d} tun={json.dumps(d['tun'])}")
        best = res[0]
        print("\n# BEST")
        print(json.dumps({"k": a.k, "tun": best["tun"], "wheel": best["wheel"],
                          "R": best["R"], "N_bits": best["N_bits"],
                          "checks_true": best["checks_true"],
                          "speedup_vs_default": base["expected_checks_true"] / best["checks_true"]}))
    if a.ban:
        ban = [int(x) for x in a.ban.split(",")]
        plan = banned_wheel(a.k, ban, a.tune)
        d = describe(plan)
        print(json.dumps({"k": a.k, "banned": ban, **d,
                          "wheel_json": json.dumps([[p, T] for p, T in plan.wheel])}, default=str))


if __name__ == "__main__":
    main()
