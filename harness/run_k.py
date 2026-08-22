#!/usr/bin/env python3
"""Crash-resumable, progress-logging driver for the gated sieve/gpu_sieve_v5.py.

Why this exists: v5's own `search()` only reports between blocks and can only resume at a block
boundary. At k>=378 a single block is ~5 h, so a crash (or a powerguard SIGKILL) costs hours and
the operator is blind while it runs. This driver imports v5 as a LIBRARY -- same Plan, same
planner, same NVRTC kernel, byte for byte -- and re-implements only the block loop, adding:

  * intra-block checkpointing (block index + residues scanned + survivors found so far),
  * resume from that checkpoint instead of restarting the block,
  * periodic progress to stderr and to the checkpoint file.

v5 itself is NOT edited, so its gate results carry over unchanged. This driver is separately
gated against the b-file (harness/gate_driver.py).

Correctness, unchanged from v5:
  * a block is only ever concluded after ALL R of its residues have been examined;
  * survivor indices are converted to n, sorted, and the smallest ref.is_good one wins;
  * blocks run t = 0, 1, 2, ... so the first block with a survivor holds the minimum;
  * any survivor with n > k+1 that ref.is_good rejects is a DISAGREEMENT (red alert), never
    silently dropped.
Resume is safe: the checkpoint stores the survivor list as of `idx_done`, so re-scanning the
tail after a crash reproduces exactly the same survivor set for the block.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, signal, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sieve"))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import importlib           # noqa: E402

# engine module first: gpu_sieve_v5 / gpu_sieve_v6 (they also put ref/ on sys.path themselves)
v5 = importlib.import_module(os.environ.get("ES_ENGINE", "gpu_sieve_v5"))
import erdos_ref as ref    # noqa: E402  (frozen referee, read-only)


SURV_HASH_RECIPE = ("sha256 of the UTF-8 bytes of '\\n'.join(decimal(n) for n in sorted(survivors)), "
                    "no trailing newline; the empty list hashes to sha256(b'')")


def survivor_hash(ns) -> str:
    """Hash of a block's survivor list, in a form a third party can reproduce exactly."""
    return hashlib.sha256("\n".join(str(int(n)) for n in sorted(ns)).encode()).hexdigest()


def atomic_write(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def run(k: int, ckpt_path: str, tun: dict, wheel, max_blocks, timeout, log_every: float,
        ckpt_every: float, min_free_mib: int, resume: bool) -> dict:
    """Engine-agnostic. A "unit" is one _launch index: 1 residue for the v5 kernel, r0
    residues for the v6 strip-mined kernel (r0 = size of the innermost ring)."""
    t_start = time.time()
    plan = v5.Plan(k, tun, wheel)
    wheel_json = [[p, T] for p, T in plan.wheel]

    engine = v5.__name__
    state = {"k": k, "engine": engine, "wheel": wheel_json, "M": str(plan.M), "R": plan.R,
             "t": 0, "unit_done": 0, "idx_done": 0, "survivor_idx": [], "residues_checked": 0,
             "wall_accum": 0.0, "restarts": 0, "block_audit": []}
    if resume and os.path.exists(ckpt_path):
        old = json.load(open(ckpt_path))
        if (old.get("k") == k and old.get("wheel") == wheel_json and old.get("R") == plan.R
                and old.get("engine", engine) == engine):
            state = old
            state.setdefault("block_audit", [])
            state["restarts"] = state.get("restarts", 0) + 1
            print(f"# RESUME k={k} engine={engine} block t={state['t']} "
                  f"residues_done={state['idx_done']}/{plan.R} "
                  f"({100.0*state['idx_done']/plan.R:.2f}%) survivors={len(state['survivor_idx'])} "
                  f"restarts={state['restarts']}", file=sys.stderr, flush=True)
        else:
            print(f"# checkpoint at {ckpt_path} is for a different plan -- starting fresh",
                  file=sys.stderr, flush=True)

    free = v5.free_vram_mib()
    if free is not None and free < min_free_mib:
        return {"k": k, "status": "SKIPPED_LOW_VRAM", "free_mib": free}

    sieve = v5.GpuSieve(plan)
    sieve.residues_checked = state["residues_checked"]
    checked_at_start = sieve.residues_checked   # so the reported rate is this process's, not cumulative
    r0 = getattr(sieve, "r0", 1)                       # residues per launch-unit
    units_total = getattr(sieve, "R_outer", plan.R)    # launch-units per block
    unit_chunk = getattr(sieve, "ochunk", sieve.chunk)
    assert units_total * r0 == plan.R, "unit accounting does not cover the block exactly"
    last_log = last_ckpt = time.time()
    t0 = time.time()

    def save():
        state["wall_accum"] = state.get("wall_accum", 0.0)
        atomic_write(ckpt_path, state)

    while max_blocks is None or state["t"] < max_blocks:
        t = state["t"]
        # per-block constants: tconst[j] = (t * M) mod q_j
        for j, q in enumerate(sieve.rest_q):
            sieve.tconst_h[j] = ((t % q) * sieve.rest_Mmodq[j]) % q
        sieve.tconst.set(sieve.tconst_h)

        while state["unit_done"] < units_total:
            if timeout and time.time() - t_start > timeout:
                state["residues_checked"] = sieve.residues_checked
                state["wall_accum"] = state.get("wall_accum", 0.0) + (time.time() - t0)
                save()
                return {"k": k, "status": "TIMEOUT", "block": t, "idx_done": state["idx_done"],
                        "block_audit": state["block_audit"], "surv_hash_recipe": SURV_HASH_RECIPE,
                        "unit_done": state["unit_done"], "units_total": units_total,
                        "R": plan.R, "residues_checked": sieve.residues_checked,
                        "wall": time.time() - t0, "checkpoint": ckpt_path}
            ustart = state["unit_done"]
            ucount = min(unit_chunk, units_total - ustart)
            sieve.out_n.fill(0)
            sieve._launch(ustart, ucount)
            nsv = int(sieve.out_n.get()[0])          # forces sync: chunk is done
            if nsv > sieve.out_cap:
                raise RuntimeError(f"survivor buffer overflow ({nsv} > {sieve.out_cap})")
            if nsv:
                state["survivor_idx"].extend(int(x) for x in sieve.out[:nsv].get())
            sieve.residues_checked += ucount * r0
            state["unit_done"] = ustart + ucount
            state["idx_done"] = state["unit_done"] * r0
            state["residues_checked"] = sieve.residues_checked

            now = time.time()
            if now - last_log >= log_every:
                el = now - t0
                frac = state["unit_done"] / units_total
                rate = (sieve.residues_checked - checked_at_start) / el if el else 0
                eta = (plan.R - state["idx_done"]) / rate if rate else float("inf")
                print(f"k={k} t={t} {100*frac:6.2f}%  idx={state['idx_done']:.6e}/{plan.R:.6e} "
                      f"surv={len(state['survivor_idx'])} checked={sieve.residues_checked:.4e} "
                      f"rate={rate:.3e}/s elapsed={el/3600:.2f}h block_eta={eta/3600:.2f}h",
                      file=sys.stderr, flush=True)
                last_log = now
            if now - last_ckpt >= ckpt_every:
                save()
                last_ckpt = now

        # block complete: every residue examined
        cands = sorted(plan.idx_to_n(i, t) for i in state["survivor_idx"])
        good = [n for n in cands if ref.is_good(n, k)]
        bad = [n for n in cands if n > k + 1 and n not in good]
        # per-block audit trail, so a third party can re-sieve this one block and compare counts
        state["block_audit"].append({
            "t": t,
            "residues_enumerated": plan.R,
            "kernel_survivors": len(cands),
            "after_n_gt_k_plus_1": sum(1 for n in cands if n > k + 1),
            "passed_ref_is_good": len(good),
            "survivors_n": [str(n) for n in cands] if len(cands) <= 64 else None,
            "sha256_survivors": survivor_hash(cands),
        })
        if bad:
            state["residues_checked"] = sieve.residues_checked
            save()
            return {"k": k, "status": "DISAGREEMENT", "block": t,
                    "block_audit": state["block_audit"], "surv_hash_recipe": SURV_HASH_RECIPE,
                    "kernel_survivors": [str(x) for x in cands[:20]],
                    "ref_rejected": [str(x) for x in bad[:20]],
                    "residues_checked": sieve.residues_checked, "wall": time.time() - t0}
        if good:
            state["residues_checked"] = sieve.residues_checked
            save()
            return {"k": k, "status": "FOUND", "g": str(good[0]), "block": t, "engine": engine,
                    "block_audit": state["block_audit"], "surv_hash_recipe": SURV_HASH_RECIPE,
                    "blocks_scanned": t + 1, "survivors_in_block": len(state["survivor_idx"]),
                    "all_good_in_block": [str(x) for x in good],
                    "residues_checked": sieve.residues_checked,
                    "wall": time.time() - t0, "restarts": state.get("restarts", 0)}
        print(f"k={k} block t={t} complete: no survivor passed the referee; advancing",
              file=sys.stderr, flush=True)
        state["t"] = t + 1
        state["unit_done"] = 0
        state["idx_done"] = 0
        state["survivor_idx"] = []
        save()

    return {"k": k, "status": "EXHAUSTED_BLOCKS", "blocks": state["t"],
            "block_audit": state["block_audit"], "surv_hash_recipe": SURV_HASH_RECIPE,
            "residues_checked": sieve.residues_checked, "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description="resumable driver for gpu_sieve_v5")
    ap.add_argument("k", type=int)
    ap.add_argument("--tag", default="primary", help="checkpoint/label tag, e.g. primary|confirm")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--tune", type=json.loads, default=None)
    ap.add_argument("--wheel", type=json.loads, default=None)
    ap.add_argument("--ban", default=None, help="comma-separated primes banned from the wheel")
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--log-every", type=float, default=60.0)
    ap.add_argument("--ckpt-every", type=float, default=120.0)
    ap.add_argument("--min-free-mib", type=int, default=2048)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--bfile", default=os.path.join(ROOT, "data", "b003458.txt"))
    ap.add_argument("--known", default=os.path.join(ROOT, "data", "known_extra.txt"))
    a = ap.parse_args()

    wheel = a.wheel
    if a.ban:
        ban = [int(x) for x in a.ban.split(",")]
        ps = [p for p in ref.primes_upto(a.k) if p not in ban]
        full = v5.Plan(a.k, a.tune)
        log2E = -math.log2(full.density_all)
        w = v5.plan_wheel(a.k, ps, log2E, dict(v5.TUNABLES, **(a.tune or {})))
        wheel = [[p, T] for p, T in w]
        print(f"# banned {ban} -> wheel {wheel}", file=sys.stderr, flush=True)

    ckpt_dir = a.ckpt_dir or os.path.join(ROOT, "overnight",
                                          time.strftime("%Y-%m-%d", time.gmtime()), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"k{a.k}_{a.tag}.json")

    res = run(a.k, ckpt, a.tune, wheel, a.max_blocks, a.timeout, a.log_every,
              a.ckpt_every, a.min_free_mib, not a.no_resume)
    plan = v5.Plan(a.k, a.tune, wheel)
    res["plan"] = plan.summary()
    res["engine_module"] = v5.__name__
    res["engine_file"] = f"sieve/{v5.__name__}.py"
    try:
        import subprocess as _sp
        res["engine_commit"] = _sp.run(["git", "log", "-1", "--format=%h", "--",
                                        f"sieve/{v5.__name__}.py"], capture_output=True,
                                       text=True, cwd=ROOT).stdout.strip()
    except Exception:
        res["engine_commit"] = None
    res["tag"] = a.tag
    if a.ban:
        res["banned"] = a.ban
    known = v5.load_known(a.bfile, a.known)
    if res.get("status") == "FOUND" and a.k in known:
        g_known, src = known[a.k]
        res["matches_bfile"] = (int(res["g"]) == g_known)
        res["bfile"] = str(g_known)
        res["known_source"] = src
    print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
