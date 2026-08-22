#!/usr/bin/env python3
"""Extend MISSION rule 2(a) -- sieve result == FROZEN REFERENCE result -- as far as it goes.

The staging brief says g_crt is unusable past k~140.  That holds at its DEFAULT
max_wheel_modulus=1e9.  The parameter is part of the frozen function's signature, so raising
it is a call, not an edit: a bigger wheel shrinks the residual scan by orders of magnitude,
at the cost of RAM for the residue list (g_crt materialises all R of them).

For each k we pick the LARGEST cap from a ladder whose residue list still fits R_MAX, run
the referee, and compare with both the b-file and the GPU sieve's answer.  A term that
passes here has the referee's own independent exhaustive answer -- strictly stronger than
b-file agreement plus a cross-wheel GPU re-run.
"""
from __future__ import annotations
import argparse, json, math, os, resource, subprocess, sys, time
from math import prod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref  # frozen; called read-only
PY = os.path.join(ROOT, ".venv", "bin", "python")
LADDER = [10 ** 19, 10 ** 18, 10 ** 17, 10 ** 16, 10 ** 15, 10 ** 14, 10 ** 13,
          10 ** 12, 10 ** 11, 10 ** 10, 10 ** 9]


def plan_for(k, cap):
    """Replicate g_crt's own wheel choice so we can predict R and the residual work."""
    ps = ref.primes_upto(k)
    info = {p: ref.allowed_residues(k, p) for p in ps}
    scored = sorted(ps, key=lambda p: -math.log(info[p][0] / len(info[p][1])) / math.log(info[p][0]))
    wheel, M = [], 1
    for p in scored:
        q = info[p][0]
        if M * q <= cap:
            wheel.append(p); M *= q
    R = prod(len(info[p][1]) for p in wheel)
    d_rest = prod(len(info[p][1]) / info[p][0] for p in ps if p not in wheel)
    return R, M, (1.0 / d_rest if d_rest else float("inf"))


def choose_cap(k, r_max, work_max):
    best = None
    for cap in LADDER:
        R, M, work = plan_for(k, cap)
        if R <= r_max:
            best = (cap, R, M, work)
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r-max", type=float, default=2e7, help="max residues held in RAM (~32B each)")
    ap.add_argument("--work-max", type=float, default=6e9, help="skip k whose residual scan exceeds this")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--order", choices=["k", "cost"], default="k",
                    help="cost = cheapest first, so the most terms land soonest")
    a = ap.parse_args()

    ks = []
    for part in a.ks.split(","):
        if "-" in part:
            x, y = part.split("-"); ks.extend(range(int(x), int(y) + 1))
        elif part.strip():
            ks.append(int(part))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["k"])
            except Exception:
                pass
    bfile = {int(l.split()[0]): int(l.split()[1]) for l in open(f"{ROOT}/data/b003458.txt") if l.strip()}

    ks = [k for k in ks if k not in done]
    if a.order == "cost":
        est = {}
        for k in ks:
            ch = choose_cap(k, a.r_max, a.work_max)
            est[k] = ch[3] if ch else float("inf")
        ks = [k for k in sorted(ks, key=lambda k: est[k]) if est[k] <= a.work_max]
        print(f"# cost-ordered, {len(ks)} feasible: "
              f"{[(k, f'{est[k]:.1e}') for k in ks[:6]]} ...", flush=True)

    for k in ks:
        ch = choose_cap(k, a.r_max, a.work_max)
        if ch is None:
            print(f"k={k} SKIP (no cap keeps the residue list under {a.r_max:.0e})", flush=True)
            continue
        cap, R, M, work = ch
        if work > a.work_max:
            print(f"k={k} SKIP (residual scan ~{work:.2e} > {a.work_max:.0e})", flush=True)
            continue
        t0 = time.time()
        try:
            p = subprocess.run([PY, os.path.join(ROOT, "harness", "referee_extend.py"),
                                str(k), str(cap)], capture_output=True, text=True,
                               timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(line[-1]) if line else {"k": k, "status": "NO_OUTPUT",
                                                     "stderr": p.stderr[-300:]}
        except subprocess.TimeoutExpired:
            rec = {"k": k, "status": "TIMEOUT", "wall": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            rec = {"k": k, "status": "ERROR", "error": repr(e)[:200]}
        rec.update({"predicted_R": R, "predicted_work": work, "cap_used": cap})
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        ok = rec.get("agrees")
        flag = "REFEREE-CONFIRMED" if ok else ("MISMATCH <<< RED ALERT" if ok is False else rec.get("status"))
        print(f"k={k} {flag} wall={rec.get('wall')}s R={R:.1e} cap=1e{len(str(cap))-1}", flush=True)


if __name__ == "__main__":
    main()
