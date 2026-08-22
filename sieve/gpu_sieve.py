#!/usr/bin/env python3
"""CUDA (CuPy/NVRTC) CRT-wheel search for the Erdős–Selfridge function g(k).

Method. For each prime p <= k let q = p^e (e = #base-p digits of k) and A_p = the set of
residues mod q whose base-p digits dominate k's digits. A "wheel" W of primes with
M = prod q_p is chosen; every residue r mod M compatible with W is enumerated by a
mixed-radix index, and blocks [tM, (t+1)M) are scanned for t = 0, 1, 2, ...  Each GPU
thread reconstructs r (128-bit, exact hi/lo arithmetic), then checks the remaining
primes via r mod q lookups into per-prime "allowed" byte tables. Survivors are sent to
the host, which rebuilds n with Python ints and re-verifies with the FROZEN reference
(ref/erdos_ref.is_good) before accepting. Within the first block that has a survivor,
all residues are scanned, so the minimum is exact.

No system nvcc required: CuPy compiles the kernel with NVRTC at runtime.

Everything between the TUNABLE markers is fair game for automated mutation; everything
else should be considered load-bearing.
"""
from __future__ import annotations
import argparse, json, math, os, signal, subprocess, sys, time
from math import prod

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
import erdos_ref as ref  # noqa: E402  (frozen referee, used read-only)

import numpy as np

try:
    import cupy as cp
except ImportError:  # allow import for planning/dry-run on machines without CuPy
    cp = None

MAX_M_BITS = 126  # partial sums stay < 2M < 2^127

# ============================ TUNABLE BEGIN ============================
# Knobs below may be mutated by the search harness. Keep them pure functions of their
# inputs; the correctness gate (agreement with the frozen reference) is enforced outside.

TUNABLES = {
    "m_cap_bits": 120,      # hard cap on log2(M) for the wheel modulus
    "m_cap_factor": 0.05,   # M <= m_cap_factor / density_all  (density_all ~ 1/E[g(k)])
    "threads": 256,         # CUDA block size
    "chunk_log2": 28,       # residues per kernel launch = 2**chunk_log2
    "max_survivors": 1 << 20,
}


def wheel_score(p: int, q: int, n_allowed: int) -> float:
    """Higher = better wheel prime. Default: density gain per bit of modulus."""
    return math.log(q / n_allowed) / math.log(q)


def choose_wheel(k: int, info: dict[int, tuple[int, list[int]]], m_cap: int) -> list[int]:
    """Greedy: take primes by wheel_score while product of moduli stays <= m_cap."""
    order = sorted(info, key=lambda p: -wheel_score(p, info[p][0], len(info[p][1])))
    wheel, M = [], 1
    for p in order:
        q = info[p][0]
        if M * q <= m_cap:
            wheel.append(p); M *= q
    return wheel


def order_rest(rest: list[int], info: dict) -> list[int]:
    """Order of the non-wheel primes inside the kernel: most restrictive first."""
    return sorted(rest, key=lambda p: len(info[p][1]) / info[p][0])


KERNEL_SRC = r'''
extern "C" __global__
void es_sieve(const unsigned long long idx_start, const unsigned long long count,
              const unsigned long long t,
              const int w, const unsigned int* __restrict__ radix,
              const unsigned int* __restrict__ res_off,
              const unsigned long long* __restrict__ term_lo, const unsigned long long* __restrict__ term_hi,
              const unsigned long long M_lo, const unsigned long long M_hi,
              const int nrest, const unsigned int* __restrict__ q_arr,
              const unsigned int* __restrict__ Mmodq, const unsigned int* __restrict__ p64modq,
              const unsigned int* __restrict__ tab_off, const unsigned char* __restrict__ allowed,
              unsigned long long* __restrict__ out, unsigned int* __restrict__ out_n,
              const unsigned int out_cap)
{
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long i = gid; i < count; i += stride) {
        unsigned long long idx = idx_start + i;
        // decode mixed radix, accumulate S = sum (a_i*c_i mod M) with a conditional
        // subtraction after each add, so S < M always (exact 128-bit hi/lo arithmetic)
        unsigned long long s_lo = 0, s_hi = 0;
        unsigned long long rem = idx;
        for (int j = 0; j < w; ++j) {
            unsigned int r = radix[j];
            unsigned int d = (unsigned int)(rem % r);
            rem /= r;
            unsigned int o = res_off[j] + d;
            unsigned long long lo = term_lo[o], hi = term_hi[o];
            unsigned long long nlo = s_lo + lo;
            s_hi += hi + (nlo < s_lo ? 1ULL : 0ULL);
            s_lo = nlo;
            if (s_hi > M_hi || (s_hi == M_hi && s_lo >= M_lo)) {
                unsigned long long mlo = s_lo - M_lo;
                s_hi -= M_hi + (s_lo < M_lo ? 1ULL : 0ULL);
                s_lo = mlo;
            }
        }
        bool ok = true;
        for (int j = 0; j < nrest; ++j) {
            unsigned int q = q_arr[j];
            unsigned long long m = ((s_hi % q) * (unsigned long long)p64modq[j] + (s_lo % q)) % q;
            m = (m + (t % q) * (unsigned long long)Mmodq[j]) % q;
            if (!allowed[tab_off[j] + m]) { ok = false; break; }
        }
        if (ok) {
            unsigned int slot = atomicAdd(out_n, 1u);
            if (slot < out_cap) out[slot] = idx;
        }
    }
}
'''
# ============================= TUNABLE END =============================


class Plan:
    def __init__(self, k: int, tun: dict | None = None, wheel: list[int] | None = None):
        self.k = k
        self.tun = dict(TUNABLES, **(tun or {}))
        ps = ref.primes_upto(k)
        self.info = {p: ref.allowed_residues(k, p) for p in ps}
        self.density_all = prod(len(v[1]) / v[0] for v in self.info.values()) if ps else 1.0
        m_cap = min(1 << self.tun["m_cap_bits"], int(self.tun["m_cap_factor"] / self.density_all) or 1)
        m_cap = min(m_cap, 1 << MAX_M_BITS)
        self.wheel = wheel if wheel is not None else choose_wheel(k, self.info, m_cap)
        self.rest = order_rest([p for p in ps if p not in self.wheel], self.info)
        self.moduli = [self.info[p][0] for p in self.wheel]
        self.M = prod(self.moduli)
        assert self.M < (1 << MAX_M_BITS), "wheel modulus too large for 128-bit path"
        self.basis = []
        for q in self.moduli:
            Mi = self.M // q
            self.basis.append(Mi * pow(Mi, -1, q))
        self.radix = [len(self.info[p][1]) for p in self.wheel]
        self.R = prod(self.radix)  # residues per block
        self.density_wheel = self.R / self.M if self.M else 1.0
        self.density_rest = prod(len(self.info[p][1]) / self.info[p][0] for p in self.rest)

    def idx_to_n(self, idx: int, t: int) -> int:
        r = 0
        for j, (p, rad) in enumerate(zip(self.wheel, self.radix)):
            idx, d = divmod(idx, rad)
            r += self.info[p][1][d] * self.basis[j]
        return t * self.M + (r % self.M)

    def summary(self) -> dict:
        return {"k": self.k, "wheel": self.wheel, "M_bits": self.M.bit_length(), "M": str(self.M),
                "residues_per_block": self.R, "density_wheel": self.density_wheel,
                "density_rest": self.density_rest, "expected_g": 1 / self.density_all,
                "expected_blocks": (1 / self.density_all) / self.M,
                "expected_residue_checks": self.density_wheel / self.density_all}


def free_vram_mib() -> int | None:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return int(o)
    except Exception:
        return None


class GpuSieve:
    def __init__(self, plan: Plan):
        if cp is None:
            raise RuntimeError("cupy not available")
        self.plan = plan
        P = plan
        self.kernel = cp.RawKernel(KERNEL_SRC, "es_sieve", options=("-std=c++17",))
        u32 = lambda a: cp.asarray(np.asarray(a, dtype=np.uint32))
        u64 = lambda a: cp.asarray(np.asarray(a, dtype=np.uint64))
        self.w = len(P.wheel)
        self.radix = u32(P.radix)
        offs, terms = [], []
        for j, p in enumerate(P.wheel):
            offs.append(len(terms)); terms.extend((a * P.basis[j]) % P.M for a in P.info[p][1])
        self.res_off = u32(offs)
        self.term_lo = u64([x & ((1 << 64) - 1) for x in terms]); self.term_hi = u64([x >> 64 for x in terms])
        self.M_lo = np.uint64(P.M & ((1 << 64) - 1)); self.M_hi = np.uint64(P.M >> 64)
        qs, mm, p64, toff, tab = [], [], [], [], bytearray()
        for p in P.rest:
            q, res = P.info[p]
            qs.append(q); mm.append(P.M % q); p64.append((1 << 64) % q); toff.append(len(tab))
            t = bytearray(q)
            for r in res: t[r] = 1
            tab += t
        self.nrest = len(P.rest)
        self.q_arr = u32(qs or [0]); self.Mmodq = u32(mm or [0]); self.p64modq = u32(p64 or [0])
        self.tab_off = u32(toff or [0]); self.allowed = cp.asarray(np.frombuffer(bytes(tab) or b"\0", dtype=np.uint8))
        self.out_cap = int(P.tun["max_survivors"])
        self.out = cp.zeros(self.out_cap, dtype=cp.uint64)
        self.out_n = cp.zeros(1, dtype=cp.uint32)
        self.threads = int(P.tun["threads"])
        self.chunk = 1 << int(P.tun["chunk_log2"])
        self.launches = 0
        self.residues_checked = 0

    def scan_block(self, t: int) -> list[int]:
        """Scan all residues of block t. Returns survivor indices (unverified)."""
        P = self.plan
        survivors = []
        start = 0
        while start < P.R:
            count = min(self.chunk, P.R - start)
            self.out_n.fill(0)
            blocks = min(65535 * 16, (count + self.threads - 1) // self.threads)
            self.kernel((blocks,), (self.threads,), (
                np.uint64(start), np.uint64(count), np.uint64(t), np.int32(self.w), self.radix,
                self.res_off, self.term_lo, self.term_hi, self.M_lo, self.M_hi,
                np.int32(self.nrest), self.q_arr, self.Mmodq, self.p64modq, self.tab_off, self.allowed,
                self.out, self.out_n, np.uint32(self.out_cap)))
            n = int(self.out_n.get()[0])
            if n > self.out_cap:
                raise RuntimeError(f"survivor buffer overflow ({n} > {self.out_cap}); raise max_survivors")
            if n:
                survivors.extend(int(x) for x in self.out[:n].get())
            self.launches += 1
            self.residues_checked += count
            start += count
        return survivors

    def search(self, max_blocks: int | None = None, log=None) -> dict:
        P = self.plan
        t0 = time.time()
        t = 0
        while max_blocks is None or t < max_blocks:
            surv = self.scan_block(t)
            cands = sorted(P.idx_to_n(i, t) for i in surv)
            good = [n for n in cands if ref.is_good(n, P.k)]
            bad = [n for n in cands if n > P.k + 1 and n not in good]
            if bad:
                # kernel said yes, referee said no -> RED ALERT, never accept silently
                return {"k": P.k, "status": "DISAGREEMENT", "kernel_survivors": cands[:20],
                        "ref_rejected": bad[:20], "block": t, "wall": time.time() - t0}
            if log and (good or time.time() - getattr(self, "_last_log", 0) > 10):
                self._last_log = time.time()
                log(f"k={P.k} block t={t} survivors={len(surv)} good={len(good)} "
                    f"checked={self.residues_checked:.3e} wall={time.time()-t0:.1f}s")
            if good:
                return {"k": P.k, "status": "FOUND", "g": good[0], "block": t,
                        "survivors_in_block": len(surv), "residues_checked": self.residues_checked,
                        "launches": self.launches, "wall": time.time() - t0}
            t += 1
        return {"k": P.k, "status": "EXHAUSTED_BLOCKS", "blocks": t,
                "residues_checked": self.residues_checked, "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description="GPU CRT-wheel search for g(k)")
    ap.add_argument("k", type=int, nargs="+")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--tune", type=json.loads, default=None, help='JSON dict overriding TUNABLES')
    ap.add_argument("--wheel", type=json.loads, default=None, help="explicit wheel primes (JSON list)")
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None, help="seconds; process exits 124 on expiry")
    ap.add_argument("--min-free-mib", type=int, default=2048)
    ap.add_argument("--bfile", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "b003458.txt"))
    a = ap.parse_args()
    if a.timeout:
        signal.signal(signal.SIGALRM, lambda *_: (print(json.dumps({"status": "TIMEOUT"}), flush=True), os._exit(124)))
        signal.setitimer(signal.ITIMER_REAL, a.timeout)
    known = {}
    if os.path.exists(a.bfile):
        for line in open(a.bfile):
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                known[int(parts[0])] = int(parts[1])
    for k in a.k:
        plan = Plan(k, a.tune, a.wheel)
        if a.plan_only:
            print(json.dumps(plan.summary())); continue
        free = free_vram_mib()
        if free is not None and free < a.min_free_mib:
            print(json.dumps({"k": k, "status": "SKIPPED_LOW_VRAM", "free_mib": free})); continue
        sieve = GpuSieve(plan)
        res = sieve.search(a.max_blocks, log=lambda s: print(s, file=sys.stderr, flush=True))
        if res.get("status") == "FOUND" and k in known:
            res["matches_bfile"] = (res["g"] == known[k])
            res["bfile"] = known[k]
        res["plan"] = plan.summary()
        print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
