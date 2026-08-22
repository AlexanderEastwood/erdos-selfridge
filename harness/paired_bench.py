#!/usr/bin/env python3
"""Paired A/B bench: interleave incumbent and variant so shared-GPU load cancels out.

The grind may be running on the same GPU, so an unpaired timing is meaningless. Each
rep runs control then variant back to back; we compare medians of the per-rep ratio.
Correctness is a hard gate: any k that disagrees with the b-file scores the variant out.
"""
from __future__ import annotations
import argparse, json, os, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
BASE = os.path.join(ROOT, "sieve", "gpu_sieve_v3.py")


ENV = dict(os.environ)
# sandbox copies live outside sieve/, so the sieve's relative ".."/ref lookup misses;
# make the frozen referee importable by absolute path for every child process.
ENV["PYTHONPATH"] = os.path.join(ROOT, "ref") + os.pathsep + ENV.get("PYTHONPATH", "")


def run(sieve, tune, ks, timeout):
    cmd = [PY, os.path.join(ROOT, "harness", "bench.py"), "--sieve", sieve, "--ks", ks,
           "--timeout", str(timeout)]
    if tune:
        cmd += ["--tune", json.dumps(tune)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120, env=ENV)
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        return json.loads(line[-1]) if line else {"ok": False}
    except Exception:
        return {"ok": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="JSON list of {name,sieve?,tune?}")
    ap.add_argument("--ks", default="190,195")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    variants = json.loads(open(a.variants).read()) if os.path.exists(a.variants) else json.loads(a.variants)

    results = []
    for v in variants:
        sieve = v.get("sieve") or BASE
        ratios, ok = [], True
        for _ in range(a.reps):
            c = run(BASE, None, a.ks, a.timeout)
            t = run(sieve, v.get("tune"), a.ks, a.timeout)
            if not (c.get("ok") and t.get("ok")):
                ok = False
                break
            ratios.append(c["wall"] / t["wall"])
        rec = {"name": v["name"], "tune": v.get("tune"), "sieve": os.path.basename(sieve),
               "correct": ok, "speedup_vs_incumbent": round(statistics.median(ratios), 4) if ratios else None,
               "reps": ratios}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    results.sort(key=lambda r: -(r["speedup_vs_incumbent"] or 0))
    if a.out:
        with open(a.out, "a") as fh:
            for r in results:
                fh.write(json.dumps({**r, "ts": round(time.time(), 1), "source": "paired_bench"}) + "\n")
    print("\n# ranked:")
    for r in results:
        print(f"  {r['speedup_vs_incumbent']}x  {r['name']}  correct={r['correct']}")


if __name__ == "__main__":
    main()
