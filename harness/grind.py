#!/usr/bin/env python3
"""Frontier grind driver: confirm published g(k) terms with the GPU sieve.

Crash-resumable (re-reads its own append-only jsonl and skips finished k), checks free
VRAM before every launch, gives every run an explicit timeout, and halves the kernel
chunk on OOM instead of dying.  Verification per MISSION rule 2:
  (a) sieve result == frozen-reference-backed value  -> for published k, the OEIS b-file
  (b) from-scratch digit-domination recheck for every prime <= k (ref.is_good)
  (c) minimality -- the sieve scans every residue of every block up to and including the
      one containing g, so the first hit is minimal by construction; optionally
      re-confirmed with an independent wheel (--cross).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import erdos_ref as ref  # noqa: E402  frozen referee
PY = os.path.join(ROOT, ".venv", "bin", "python")


def free_vram_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
        return int(o)
    except Exception:
        return None


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("verdict") in ("CONFIRMED_PUBLISHED", "DISAGREEMENT", "MISMATCH_BFILE"):
                done[r["k"]] = r
    return done


def parse_ks(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_sieve(k, sieve, timeout, tune=None, log_path=None):
    cmd = [PY, sieve, str(k), "--timeout", str(timeout)]
    if tune:
        cmd += ["--tune", json.dumps(tune)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        d = json.loads(lines[-1]) if lines else {"status": "NO_OUTPUT", "stderr": p.stderr[-800:]}
    except subprocess.TimeoutExpired:
        d = {"status": "HARD_TIMEOUT"}
    except Exception as e:  # noqa: BLE001
        d = {"status": "ERROR", "error": repr(e)[:400]}
    d["wall"] = round(time.time() - t0, 2)
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] k={k} tune={tune} -> "
                     f"{d.get('status')} wall={d['wall']}s\n")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", required=True, help="e.g. 141-199 or 200,205,210")
    ap.add_argument("--sieve", default=os.path.join(ROOT, "sieve", "gpu_sieve_v3.py"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--min-free-mib", type=int, default=1024)
    ap.add_argument("--order", choices=["k", "cost"], default="k")
    ap.add_argument("--cross", action="store_true",
                    help="re-run each confirmed k with a different wheel (independent check)")
    ap.add_argument("--cross-max-wall", type=float, default=600,
                    help="skip the cross-check when the primary run took longer than this")
    ap.add_argument("--deadline", default=None, help="UTC HH:MM; stop launching new k after this")
    a = ap.parse_args()

    bfile = {}
    for line in open(os.path.join(ROOT, "data", "b003458.txt")):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            bfile[int(parts[0])] = int(parts[1])

    ks = parse_ks(a.ks)
    done = load_done(a.out)
    ks = [k for k in ks if k not in done]
    if a.order == "cost":
        sys.path.insert(0, os.path.join(ROOT, "sieve"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("sv", a.sieve)
        sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
        est = {}
        for k in ks:
            try:
                est[k] = sv.Plan(k).summary()["expected_residue_checks"]
            except Exception:
                est[k] = float("inf")
        ks.sort(key=lambda k: est[k])
        print(f"# cost-ordered: {[(k, f'{est[k]:.1e}') for k in ks[:8]]} ...", flush=True)

    print(f"# {len(done)} already done, {len(ks)} to go", flush=True)
    for k in ks:
        if a.deadline and time.strftime("%H:%M") >= a.deadline:
            print(f"# deadline {a.deadline} reached, stopping before k={k}", flush=True)
            break
        free = free_vram_mib()
        if free is not None and free < a.min_free_mib:
            print(f"# k={k}: only {free} MiB free VRAM, waiting", flush=True)
            time.sleep(60)
            free = free_vram_mib()
            if free is not None and free < a.min_free_mib:
                rec = {"k": k, "verdict": "SKIPPED_LOW_VRAM", "free_mib": free, "ts": time.time()}
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue

        tune, d = None, None
        for attempt in range(3):
            d = run_sieve(k, a.sieve, a.timeout, tune, a.log)
            err = (d.get("error", "") or "") + (d.get("stderr", "") or "")
            if d.get("status") in ("FOUND", "DISAGREEMENT", "HARD_TIMEOUT", "TIMEOUT"):
                break
            if "OutOfMemory" in err or "out of memory" in err.lower():
                shrink = 26 - attempt * 2
                tune = {"chunk_log2": shrink, "max_survivors": 1 << 18}
                print(f"# k={k}: OOM, retrying with chunk_log2={shrink}", flush=True)
                continue
            break

        rec = {"k": k, "ts": round(time.time(), 1), "status": d.get("status"),
               "wall": d.get("wall"), "free_mib_before": free,
               "residues_checked": d.get("residues_checked"), "block": d.get("block"),
               "tune": tune, "sieve": os.path.basename(a.sieve)}
        if d.get("plan"):
            rec["wheel"] = d["plan"].get("wheel")
            rec["M_bits"] = d["plan"].get("M_bits")
            rec["expected_checks"] = d["plan"].get("expected_residue_checks")

        if d.get("status") == "DISAGREEMENT":
            rec["verdict"] = "DISAGREEMENT"
            rec["RED_ALERT"] = True
            rec["kernel_survivors"] = d.get("kernel_survivors")
            rec["ref_rejected"] = d.get("ref_rejected")
        elif d.get("status") == "FOUND":
            g = int(d["g"])
            rec["g"] = str(g)
            # (b) from-scratch recheck with the frozen referee
            rec["ref_is_good"] = ref.is_good(g, k)
            # (a) agreement with the published value
            pub = bfile.get(k)
            rec["bfile"] = str(pub) if pub is not None else None
            rec["matches_bfile"] = (pub == g) if pub is not None else None
            # (c) exhaustiveness of the block scan is structural; record the evidence
            rec["exhaustive_scan"] = True
            if pub is not None and pub != g:
                rec["verdict"] = "MISMATCH_BFILE"; rec["RED_ALERT"] = True
            elif not rec["ref_is_good"]:
                rec["verdict"] = "REF_REJECTED"; rec["RED_ALERT"] = True
            elif pub is not None:
                rec["verdict"] = "CONFIRMED_PUBLISHED"
            else:
                rec["verdict"] = "CANDIDATE_NEW"
            if (a.cross and rec["verdict"] in ("CONFIRMED_PUBLISHED", "CANDIDATE_NEW")
                    and (d.get("wall") or 0) <= a.cross_max_wall):
                # independent check: a DIFFERENT planner yields a different CRT
                # decomposition of comparable quality (a smaller m_cap would also
                # differ, but is far slower on expensive k and dominated the run).
                d2 = run_sieve(k, a.sieve, a.timeout, {"planner": "knapsack"}, a.log)
                rec["cross_status"] = d2.get("status")
                rec["cross_wheel"] = (d2.get("plan") or {}).get("wheel")
                rec["cross_g"] = str(d2.get("g")) if d2.get("g") is not None else None
                rec["cross_agrees"] = (d2.get("status") == "FOUND" and int(d2["g"]) == g)
                rec["cross_wall"] = d2.get("wall")
        else:
            rec["verdict"] = d.get("status") or "UNKNOWN"
            if d.get("stderr"):
                rec["stderr"] = d["stderr"][-400:]

        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        flag = "  <<< RED ALERT" if rec.get("RED_ALERT") else ""
        print(f"k={k} {rec['verdict']} wall={rec.get('wall')}s "
              f"g={rec.get('g','-')}{flag}", flush=True)


if __name__ == "__main__":
    main()
