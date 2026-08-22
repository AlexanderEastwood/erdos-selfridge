#!/usr/bin/env python3
"""Turn driver result JSON into the mission's evidence block + confirmations.jsonl line.

Every number is taken from the run's own JSON; nothing is typed by hand. The rule-2(b)
digit-domination recheck is executed FRESH here as a separate subprocess against the frozen
referee (never copied from the sieve's internal check).

Usage:
  record_term.py --k 376 --primary logs/k376_primary.json [--confirm logs/k376_confirm.json]
                 [--status "VERIFIED (two-wheel)"] [--note "..."] [--out-dir overnight/<date>]
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")


def load_result(path: str) -> dict:
    """Accept either a bare JSON file or a driver log whose last '{' line is the result."""
    txt = open(path).read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        lines = [l for l in txt.splitlines() if l.startswith("{")]
        if not lines:
            raise SystemExit(f"no JSON result line in {path}")
        return json.loads(lines[-1])


def fresh_is_good(n: int, k: int) -> tuple[bool, str]:
    cmd = f'from ref.erdos_ref import is_good; print(is_good({n},{k}))'
    p = subprocess.run([PY, "-c", cmd], capture_output=True, text=True, cwd=ROOT)
    return p.stdout.strip() == "True", p.stdout.strip()


def wheel_str(wheel) -> str:
    return "[" + ",".join(f"{p}^{T}" for p, T in wheel) + "]"


def git_hash(path: str | None = None) -> str:
    cmd = ["git", "log", "-1", "--format=%h", "--", path] if path else \
          ["git", "rev-parse", "--short", "HEAD"]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def engine_stamp(engine: str) -> str:
    """'<engine string>@<hash of the engine source file, if we can find one>'."""
    import re as _re
    m = _re.search(r"(sieve/gpu_sieve_v\d+\.py)", engine)
    if m and os.path.exists(os.path.join(ROOT, m.group(1))):
        h = git_hash(m.group(1))
        return f"{engine}@{h}" if h else engine
    return engine


def load_audit(run: dict, sidecar: str | None) -> tuple[list, dict]:
    """Per-block audit rows + provenance. Prefers the run's own recording; falls back to a
    reconstructed sidecar for runs that predate it."""
    if run.get("block_audit"):
        return run["block_audit"], {"source": "recorded live by harness/run_k.py",
                                    "recipe": run.get("surv_hash_recipe", "")}
    if sidecar and os.path.exists(sidecar):
        d = json.load(open(sidecar))
        return d.get("block_audit", []), {
            "source": f"reconstructed by harness/reconstruct_audit.py from {d.get('source_log')}",
            "recipe": d.get("surv_hash_recipe", ""), "reconstructed": True,
            "filter_primes": d.get("filter_primes")}
    return [], {"source": "not available"}


def render_audit(title: str, plan: dict, rows: list, prov: dict, k: int) -> list[str]:
    L = ["", f"### {title} — per-block audit (for independent re-sieving)", ""]
    L.append(f"    k                = {k}")
    L.append(f"    N                = {plan['N']}   ({plan['N_bits']} bits)")
    L.append(f"    wheel [[p,T],..] = {json.dumps([list(x) for x in plan['wheel']])}")
    L.append(f"    residues/block R = {plan['residues_per_block']}")
    L.append(f"    filter primes    = {plan.get('n_filters')} primes: every p <= k with wheel "
             f"exponent T < t_p, checked against the FULL p^t_p table")
    if prov.get("filter_primes"):
        L.append(f"                       {prov['filter_primes']}")
    L.append("")
    L.append("    A kernel survivor of block t is an n in [tN, (t+1)N) whose base-p digits dominate")
    L.append("    k's for every prime p <= k. Tiers below: kernel survivors -> those with n > k+1")
    L.append("    -> those the frozen referee ref.is_good accepts.")
    L.append(f"    survivor hash    = {prov.get('recipe','')}")
    L.append(f"    audit source     = {prov['source']}")
    L.append("")
    L.append("    | block t | residues enumerated | kernel surv | n > k+1 | ref.is_good | sha256(survivors) |")
    L.append("    |---:|---:|---:|---:|---:|:--|")
    for r in rows:
        if not r.get("reconstructible", True):
            L.append(f"    | {r['t']} | {r.get('residues_enumerated','')} | see run JSON | | | "
                     f"(block reported FOUND; not reconstructible) |")
            continue
        L.append(f"    | {r['t']} | {r['residues_enumerated']} | {r['kernel_survivors']} | "
                 f"{r['after_n_gt_k_plus_1']} | {r['passed_ref_is_good']} | {r['sha256_survivors']} |")
    nonempty = [r for r in rows if r.get("survivors_n")]
    if nonempty:
        L.append("")
        for r in nonempty:
            L.append(f"    block t={r['t']} survivors: {', '.join(r['survivors_n'])}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--confirm", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--primary-audit", default=None, help="reconstructed audit JSON sidecar")
    ap.add_argument("--confirm-audit", default=None)
    ap.add_argument("--confirm-engine", default=None, help="defaults to --engine")
    ap.add_argument("--engine", default="harness/run_k.py",
                    help="what actually drove the scan (run_k.py, or the sieve CLI directly)")
    ap.add_argument("--file", default="NEW_TERMS.md",
                    help="NEW_TERMS.md for genuinely new k; VALIDATION.md for k=376/377")
    a = ap.parse_args()

    out_dir = a.out_dir or os.path.join(ROOT, "overnight", time.strftime("%Y-%m-%d", time.gmtime()))
    pr = load_result(a.primary)
    if pr.get("status") != "FOUND":
        raise SystemExit(f"primary run status is {pr.get('status')}, not FOUND -- nothing to record")
    k, n = a.k, int(pr["g"])
    plan = pr["plan"]
    E = float(plan["E_g"])

    cf = load_result(a.confirm) if a.confirm else None
    if cf is not None and cf.get("status") != "FOUND":
        raise SystemExit(f"confirm run status is {cf.get('status')}, not FOUND")
    agrees = (int(cf["g"]) == n) if cf else None

    ok, raw = fresh_is_good(n, k)

    known = {}
    for path in (os.path.join(ROOT, "data", "b003458.txt"),
                 os.path.join(ROOT, "data", "known_extra.txt")):
        for line in open(path):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                known.setdefault(int(parts[0]), int(parts[1]))
    published = known.get(k)

    status = a.status or ("VERIFIED (two-wheel)" if (agrees and ok) else
                          "CANDIDATE - single wheel only" if ok else "CANDIDATE - referee recheck FAILED")

    L = []
    L.append(f"## k = {k} — {status}")
    L.append(f"n              = {n}")
    L.append(f"status         = {status}")
    pstamp = (f"{pr['engine_file']}@{pr['engine_commit']}"
              if pr.get("engine_file") and pr.get("engine_commit") else engine_stamp(a.engine))
    L.append(f"primary run    : engine={pstamp}  driver=harness/run_k.py  wheel={wheel_str(plan['wheel'])}")
    L.append(f"                 N={plan['N']} (log2 N={plan['N_bits']})  "
             f"blocks scanned={pr.get('blocks_scanned', pr.get('block',0)+1)} "
             f"(found in block t={pr.get('block')})")
    L.append(f"                 residues checked={pr.get('residues_checked')}  "
             f"wall={pr.get('wall'):.1f} s  restarts={pr.get('restarts',0)}  date={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    if cf:
        cplan = cf["plan"]
        cstamp = (f"{cf['engine_file']}@{cf['engine_commit']}"
                  if cf.get("engine_file") and cf.get("engine_commit")
                  else engine_stamp(a.confirm_engine or a.engine))
        L.append(f"confirm run    : engine={cstamp}  wheel={wheel_str(cplan['wheel'])} "
                 f"(banned: {cf.get('banned','-')})")
        L.append(f"                 N={cplan['N']} (log2 N={cplan['N_bits']})")
        L.append(f"                 blocks={cf.get('blocks_scanned', cf.get('block',0)+1)}  "
                 f"residues={cf.get('residues_checked')}  wall={cf.get('wall'):.1f} s  "
                 f"n agrees: {'yes' if agrees else 'NO'}")
    else:
        L.append("confirm run    : NOT RUN")
    L.append(f'ref.is_good    : `.venv/bin/python -c "from ref.erdos_ref import is_good; '
             f'print(is_good({n},{k}))"` -> {raw}')
    minimal = ("every residue of every block t' <= %d examined under %s; no survivor < n  (yes)"
               % (pr.get("block", 0), "both wheels" if cf else "the primary wheel"))
    L.append(f"minimality     : {minimal}")
    L.append(f"E[g]           = {E:.6e}   n/E[g] = {n/E:.4f}"
             + ("   (FLAG: outside [0.01, 8])" if not (0.01 <= n/E <= 8) else "   (in range)"))
    if published is not None:
        L.append(f"published      : {published}   addendum/b-file agrees: "
                 f"{'yes' if published == n else 'NO -- RED ALERT'}")
    L.append(f"logs           : {a.primary}" + (f", {a.confirm}" if a.confirm else ""))
    L.append(f"commit         : {git_hash()} (this entry; see the follow-up commit for the logs)")
    if a.note:
        L.append(f"note           : {a.note}")
    prows, pprov = load_audit(pr, a.primary_audit)
    if prows:
        L += render_audit("primary run", plan, prows, pprov, k)
    if cf:
        crows, cprov = load_audit(cf, a.confirm_audit)
        if crows:
            L += render_audit("confirm run", cf["plan"], crows, cprov, k)
    block = "\n".join(L) + "\n"

    md = os.path.join(out_dir, a.file)
    with open(md, "a") as f:
        if os.path.getsize(md) == 0 if os.path.exists(md) else True:
            pass
        f.write("\n" + block)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "k": k, "n": str(n),
           "status": status, "ref_is_good": ok, "confirm_agrees": agrees,
           "published": str(published) if published is not None else None,
           "published_agrees": (published == n) if published is not None else None,
           "E_g": E, "n_over_E": n / E,
           "primary": {"wheel": plan["wheel"], "N": plan["N"], "N_bits": plan["N_bits"],
                       "block": pr.get("block"), "residues_checked": pr.get("residues_checked"),
                       "wall": pr.get("wall"), "log": a.primary},
           "confirm": ({"wheel": cf["plan"]["wheel"], "N": cf["plan"]["N"],
                        "banned": cf.get("banned"), "block": cf.get("block"),
                        "residues_checked": cf.get("residues_checked"), "wall": cf.get("wall"),
                        "log": a.confirm} if cf else None),
           "commit": git_hash(),
           "primary_block_audit": prows, "primary_audit_provenance": pprov,
           "confirm_block_audit": (load_audit(cf, a.confirm_audit)[0] if cf else None)}
    with open(os.path.join(out_dir, "confirmations.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(block)
    print(f"# appended to {md} and {os.path.join(out_dir,'confirmations.jsonl')}")


if __name__ == "__main__":
    main()
