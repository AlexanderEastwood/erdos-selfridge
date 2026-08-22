#!/usr/bin/env python3
"""Measure real system power and hold this session's GPU work under a ceiling.

Power sources, best first:
  * Corsair PSU hwmon "power total" -- true wall draw, the number the UPS actually sees
  * nvidia-smi power.draw          -- GPU only
  * RAPL package energy            -- CPU package only (may be unreadable without root)

Enforcement: when total draw exceeds `--ceiling`, SIGSTOP this session's sieve processes;
when it falls under `--resume-at`, SIGCONT them. That duty-cycles our own GPU work without
touching anyone else's, and needs no root and no system configuration change.

It ONLY ever signals processes matching --pattern (our sieve). vLLM, llama-server and
ComfyUI are never signalled. On exit every stopped process is resumed, so the guard dying
cannot leave work wedged.
"""
from __future__ import annotations
import argparse, atexit, os, signal, subprocess, sys, time, glob

PSU_GLOB = "/sys/class/hwmon/hwmon*/name"


def find_psu():
    for f in glob.glob(PSU_GLOB):
        try:
            if open(f).read().strip() == "corsairpsu":
                d = os.path.dirname(f)
                for p in glob.glob(os.path.join(d, "power*_label")):
                    if "total" in open(p).read().lower():
                        return p.replace("_label", "_input")
        except Exception:
            pass
    return None


def read_psu(path):
    try:
        return int(open(path).read().strip()) / 1e6      # uW -> W
    except Exception:
        return None


def read_gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return sum(float(x) for x in o)
    except Exception:
        return None


class Rapl:
    def __init__(self):
        self.f = None
        for c in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
            if os.access(c, os.R_OK):
                self.f = c
                break
        self.prev = self.read_raw()
        self.t = time.time()

    def read_raw(self):
        try:
            return int(open(self.f).read().strip())
        except Exception:
            return None

    def watts(self):
        if not self.f:
            return None
        cur, now = self.read_raw(), time.time()
        if cur is None or self.prev is None or now <= self.t:
            self.prev, self.t = cur, now
            return None
        d = cur - self.prev
        if d < 0:
            d = None
        w = (d / 1e6) / (now - self.t) if d is not None else None
        self.prev, self.t = cur, now
        return w


def pids(pattern):
    """PIDs matching the pattern, EXCLUDING ourselves and our own process group.

    Our argv contains the pattern string, so a naive pgrep -f matches this guard and we
    would SIGSTOP ourselves -- leaving the sieve stopped forever with nothing to resume it.
    """
    me, mypg = os.getpid(), os.getpgrp()
    out = []
    try:
        o = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10).stdout
        for x in o.split():
            pid = int(x)
            if pid == me or pid == os.getppid():
                continue
            try:
                if os.getpgid(pid) == mypg:      # never signal our own process group
                    continue
            except Exception:
                continue
            out.append(pid)
    except Exception:
        return []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", type=float, required=True, help="watts, total system")
    ap.add_argument("--resume-at", type=float, default=None, help="watts; default ceiling-60")
    ap.add_argument("--pattern", default=r"gpu_sieve_v[0-9]\.py [0-9]",
                    help="matches only a running sieve child, not the grind driver")
    ap.add_argument("--action", choices=["stop", "kill"], default="kill",
                    help="kill is the only thing that reliably drops GPU draw: CUDA "
                         "launches are async, so SIGSTOP leaves queued work running")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--sustain", type=int, default=3,
                    help="consecutive over-ceiling samples required before acting; a single "
                         "2 s transient (NVRTC compile spiking all cores while the GPU is "
                         "still hot) must not destroy minutes of GPU work")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    resume_at = a.resume_at if a.resume_at is not None else a.ceiling - 60

    psu = find_psu()
    rapl = Rapl()
    stopped: set[int] = set()

    def resume_all(*_):
        for p in list(stopped):
            try:
                os.kill(p, signal.SIGCONT)
            except Exception:
                pass
        stopped.clear()
    atexit.register(resume_all)
    signal.signal(signal.SIGTERM, lambda *_: (resume_all(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (resume_all(), sys.exit(0)))

    if a.log:
        with open(a.log, "a") as fh:
            fh.write("ts,psu_w,gpu_w,cpu_w,throttled\n")
    print(f"# psu={psu or 'NOT FOUND'} ceiling={a.ceiling}W resume_at={resume_at}W "
          f"pattern={a.pattern!r}", flush=True)

    hi = 0.0
    over = 0
    while True:
        p = read_psu(psu) if psu else None
        g = read_gpu()
        c = rapl.watts()
        total = p if p is not None else ((g or 0) + (c or 0))
        hi = max(hi, total or 0)
        over = over + 1 if (total is not None and total > a.ceiling) else 0
        if over >= a.sustain:
            victims = pids(a.pattern)
            if victims:
                sig = signal.SIGKILL if a.action == "kill" else signal.SIGSTOP
                for pid in victims:
                    try:
                        os.kill(pid, sig)
                        if a.action == "stop":
                            stopped.add(pid)
                    except Exception:
                        pass
                print(f"[{time.strftime('%H:%M:%S')}] !! {total:.0f}W > {a.ceiling}W for "
                      f"{over} samples -> {a.action.upper()} {sorted(victims)}", flush=True)
                over = 0
        elif total is not None and total < resume_at and stopped:
            print(f"[{time.strftime('%H:%M:%S')}] {total:.0f}W < {resume_at}W -> resuming", flush=True)
            resume_all()
        if a.log:
            with open(a.log, "a") as fh:
                fh.write(f"{time.time():.0f},{p if p is not None else ''},"
                         f"{g if g is not None else ''},{c if c is not None else ''},"
                         f"{1 if stopped else 0}\n")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
