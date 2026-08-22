#!/usr/bin/env python3
"""Reconstruct the per-block audit trail for a run that predates run_k.py's audit recording.

This does NOT scrape survivor counts out of progress lines (those are sampled, so they could
miss a survivor found between samples). It derives them:

  A kernel survivor is exactly an n in [tN, (t+1)N) that dominates k in every base p <= k.
  Such an n is either
     * good            -> the run reports FOUND, or
     * n <= k+1        -> is_good rejects it, or
     * neither         -> impossible: that is precisely a DISAGREEMENT, which the run reports.
  So for any block the run completed WITHOUT reporting FOUND or DISAGREEMENT, the survivor set
  is exactly { n in [tN,(t+1)N) : n <= k+1 and n dominates k in every base }.

  For t >= 1 that set is EMPTY, because tN >= N >> k+1.  A proof, not an observation.
  For t = 0 it is computable by brute force over n = 0..k+1.

Blocks in which the run reported FOUND cannot be fully reconstructed this way (the good survivors
are not derivable without re-sieving), and are emitted with reconstructible=false.

Usage: reconstruct_audit.py --k 378 --log <driver log> [--engine gpu_sieve_v6] [--ban 3] [-o out.json]
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sieve"))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref  # noqa: E402

SURV_HASH_RECIPE = ("sha256 of the UTF-8 bytes of '\\n'.join(decimal(n) for n in sorted(survivors)), "
                    "no trailing newline; the empty list hashes to sha256(b'')")


def survivor_hash(ns) -> str:
    return hashlib.sha256("\n".join(str(int(n)) for n in sorted(ns)).encode()).hexdigest()


def small_survivors(k: int) -> list[int]:
    """Every n <= k+1 that dominates k in every base p <= k (i.e. a kernel survivor is_good rejects)."""
    ps = ref.primes_upto(k)
    return [n for n in range(0, k + 2) if all(ref.dominates(n, k, p) for p in ps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--engine", default="gpu_sieve_v6")
    ap.add_argument("--ban", default=None)
    ap.add_argument("--run-json", default=None,
                    help="the run's result JSON/log, used to fill the FOUND block's tiers")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    eng = importlib.import_module(a.engine)
    wheel = None
    if a.ban:
        ban = [int(x) for x in a.ban.split(",")]
        ps = [p for p in ref.primes_upto(a.k) if p not in ban]
        full = eng.Plan(a.k)
        wheel = [[p, T] for p, T in eng.plan_wheel(a.k, ps, -math.log2(full.density_all),
                                                   dict(eng.TUNABLES))]
    plan = eng.Plan(a.k, None, wheel)

    txt = open(a.log).read()
    completed = sorted(int(m) for m in re.findall(r"block t=(\d+) complete", txt))
    found = re.search(r'"status": "FOUND".*?"block": (\d+)', txt)
    found_block = int(found.group(1)) if found else None
    if '"status": "DISAGREEMENT"' in txt:
        raise SystemExit("log contains a DISAGREEMENT -- refusing to reconstruct; investigate first")

    smalls = small_survivors(a.k)
    rows = []
    for t in completed:
        surv = [n for n in smalls if t * plan.M <= n < (t + 1) * plan.M]
        rows.append({
            "t": t,
            "residues_enumerated": plan.R,
            "kernel_survivors": len(surv),
            "after_n_gt_k_plus_1": sum(1 for n in surv if n > a.k + 1),
            "passed_ref_is_good": 0,
            "survivors_n": [str(n) for n in surv],
            "sha256_survivors": survivor_hash(surv),
            "reconstructible": True,
            "basis": ("proof: t>=1 so the block starts above k+1, and any survivor that were "
                      "neither good nor <=k+1 would have been reported as a DISAGREEMENT")
            if t >= 1 else "brute force over n=0..k+1 (the only survivors a completed block can hold)",
        })
    if found_block is not None:
        row = {"t": found_block, "residues_enumerated": plan.R, "reconstructible": False,
               "basis": "block reported FOUND; good survivors are not derivable without re-sieving"}
        # fill what the run itself recorded, plus the derivable n<=k+1 count for this block
        rj = None
        if a.run_json and os.path.exists(a.run_json):
            t_ = open(a.run_json).read()
            try:
                rj = json.loads(t_)
            except json.JSONDecodeError:
                ls = [l for l in t_.splitlines() if l.startswith("{")]
                rj = json.loads(ls[-1]) if ls else None
        n_small = sum(1 for n in smalls if found_block * plan.M <= n < (found_block + 1) * plan.M)
        if rj and rj.get("survivors_in_block") is not None:
            ks = int(rj["survivors_in_block"])
            row["kernel_survivors"] = ks
            row["after_n_gt_k_plus_1"] = ks - n_small
            allgood = rj.get("all_good_in_block")
            m = re.search(r"survivors=(\d+) good=(\d+)", txt)
            if allgood is not None:
                row["passed_ref_is_good"] = len(allgood)
                # the complete survivor list is known exactly: the good ones plus the small ones
                full = sorted([int(x) for x in allgood] +
                              [n for n in smalls
                               if found_block * plan.M <= n < (found_block + 1) * plan.M])
                row["survivors_n"] = [str(n) for n in full]
                row["sha256_survivors"] = survivor_hash(full)
                row["reconstructible"] = True
                row["basis"] = ("run recorded the complete good-survivor list; the remaining "
                                "survivors are exactly the derivable n <= k+1")
            elif m:
                row["passed_ref_is_good"] = int(m.group(2))
                row["basis"] += (f"; tier counts taken from the run (survivors={m.group(1)}, "
                                 f"good={m.group(2)}), of which {n_small} are the derivable n<=k+1. "
                                 "No hash: this run did not record the full good-survivor list.")
        rows.append(row)
    out = {"k": a.k, "engine": a.engine, "wheel": [[p, T] for p, T in plan.wheel],
           "N": str(plan.M), "N_bits": plan.M.bit_length(), "R": plan.R,
           "n_filters": len(plan.rest), "filter_primes": plan.rest,
           "surv_hash_recipe": SURV_HASH_RECIPE,
           "small_survivors_all_blocks": [str(n) for n in smalls],
           "reconstructed": True, "source_log": a.log, "block_audit": rows}
    js = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(js)
        print(f"wrote {a.out}")
    print(f"k={a.k} blocks completed: {completed}  found_block={found_block}")
    print(f"n<=k+1 kernel survivors (all in block 0): {smalls}")
    for r in rows:
        if r.get("reconstructible"):
            print(f"  t={r['t']}: residues={r['residues_enumerated']} survivors={r['kernel_survivors']} "
                  f"sha256={r['sha256_survivors'][:16]}...")


if __name__ == "__main__":
    main()
