#!/usr/bin/env python3
"""Correctness gate for harness/run_k.py (the resumable driver around gpu_sieve_v5).

Runs the driver on every k in a range and requires exact agreement with data/b003458.txt.
Also exercises the RESUME path: for a few k it kills the run mid-block via a short --timeout,
then resumes from the checkpoint and requires the same answer as the uninterrupted run.

Usage: gate_driver.py [KMIN] [KMAX] [--resume-test k1,k2,...]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
DRV = os.path.join(ROOT, "harness", "run_k.py")

ap = argparse.ArgumentParser()
ap.add_argument("kmin", type=int, nargs="?", default=1)
ap.add_argument("kmax", type=int, nargs="?", default=200)
ap.add_argument("--resume-test", default="")
ap.add_argument("--ckpt-dir", default=None)
a = ap.parse_args()

bfile = {int(l.split()[0]): int(l.split()[1])
         for l in open(os.path.join(ROOT, "data", "b003458.txt")) if l.strip()}
ckpt_dir = a.ckpt_dir or tempfile.mkdtemp(prefix="gate_ckpt_")

def drive(k, extra=(), tag="gate"):
    cmd = [PY, DRV, str(k), "--tag", tag, "--ckpt-dir", ckpt_dir,
           "--log-every", "1e9", "--ckpt-every", "1e9", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"status": "NO_OUTPUT", "stderr": p.stderr[-400:]}

bad, dis, missing = [], [], []
n_ok = 0
for k in range(a.kmin, a.kmax + 1):
    r = drive(k, ["--no-resume"])
    if r.get("status") == "DISAGREEMENT":
        dis.append(k); continue
    if r.get("status") != "FOUND":
        missing.append((k, r.get("status"))); continue
    if int(r["g"]) != bfile[k]:
        bad.append((k, r["g"], bfile[k]))
    else:
        n_ok += 1
print(f"run_k.py driver: {n_ok}/{a.kmax - a.kmin + 1} match b-file, "
      f"mismatches={bad}, DISAGREEMENTS={dis}, missing={missing}")

# --- resume path -------------------------------------------------------------
resume_ok, resume_bad = [], []
for k in [int(x) for x in a.resume_test.split(",") if x.strip()]:
    tag = f"resume{k}"
    for f in (os.path.join(ckpt_dir, f"k{k}_{tag}.json"),):
        if os.path.exists(f):
            os.remove(f)
    # first pass: stop deterministically after 2 blocks, leaving a mid-search checkpoint
    r1 = drive(k, ["--no-resume", "--max-blocks", "2", "--ckpt-every", "0"], tag=tag)
    ck = os.path.join(ckpt_dir, f"k{k}_{tag}.json")
    partial = json.load(open(ck)) if os.path.exists(ck) else None
    t_at_ckpt = (partial or {}).get("t")
    # second pass: resume from that checkpoint and finish the search
    r2 = drive(k, [], tag=tag)
    got = int(r2["g"]) if r2.get("status") == "FOUND" else None
    # the test is only meaningful if the first pass really stopped mid-search
    exercised = r1.get("status") == "EXHAUSTED_BLOCKS" and (t_at_ckpt or 0) > 0
    if got == bfile[k] and exercised and r2.get("restarts", 0) > 0 and r2.get("block", 0) >= (t_at_ckpt or 0):
        resume_ok.append({"k": k, "first_pass": r1.get("status"), "ckpt_block": t_at_ckpt,
                          "resumed_and_found_in_block": r2.get("block"), "restarts": r2.get("restarts")})
    else:
        resume_bad.append({"k": k, "first_pass": r1.get("status"), "ckpt_block": t_at_ckpt,
                           "second_pass": r2.get("status"), "got": got, "want": bfile[k],
                           "exercised_resume": exercised, "restarts": r2.get("restarts")})
if a.resume_test:
    print(f"resume path: ok={resume_ok} bad={resume_bad}")

shutil.rmtree(ckpt_dir, ignore_errors=True)
ok = not bad and not dis and not missing and not resume_bad
print("DRIVER GATE PASS" if ok else "DRIVER GATE FAIL")
sys.exit(0 if ok else 1)
