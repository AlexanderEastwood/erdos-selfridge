#!/usr/bin/env python3
"""v4. CUDA (CuPy/NVRTC) CRT-wheel search for the Erdős–Selfridge function g(k).

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
# v2 mutations vs the baseline:
#   (1) choose_wheel: exact 0/1 knapsack DP on log2 weights instead of ratio-greedy.
#       expected_residue_checks = exp2(V_all - V_wheel) with V_p = log2(q_p/|A_p|),
#       so minimizing work == maximizing wheel value under a log2(m_cap) budget.
#   (2) KERNEL: generated per-plan with every divisor as a literal, so NVRTC emits
#       magic-multiply sequences instead of runtime 64-bit divisions.
#   (3) the t-dependent term of each rest-prime check is hoisted to the host (it is
#       loop-invariant across residues), removing one 64-bit mod per prime per residue.

TUNABLES = {
    "m_cap_bits": 120,      # hard cap on log2(M) for the wheel modulus
    "m_cap_factor": 0.05,   # M <= m_cap_factor / density_all  (density_all ~ 1/E[g(k)])
    "threads": 256,         # CUDA block size
    "chunk_log2": 30,       # residues per kernel launch = 2**chunk_log2
    "max_survivors": 1 << 20,
    "knapsack_scale": 512,  # DP cells per bit of modulus budget
    "planner": "truecost",  # "truecost" | "knapsack" | "greedy"
    "cost_scale": 256,      # DP cells per bit of log2(M) for the truecost planner
    "max_R_bits": 62,       # residues per block must index in uint64
}


def wheel_score(p: int, q: int, n_allowed: int) -> float:
    """Higher = better wheel prime. Default: density gain per bit of modulus."""
    return math.log(q / n_allowed) / math.log(q)


def _greedy_wheel(info: dict, m_cap: int) -> list[int]:
    order = sorted(info, key=lambda p: -wheel_score(p, info[p][0], len(info[p][1])))
    wheel, M = [], 1
    for p in order:
        q = info[p][0]
        if M * q <= m_cap:
            wheel.append(p); M *= q
    return wheel


def _knapsack_wheel(info: dict, m_cap: int, scale: int = 512) -> list[int]:
    """Exact 0/1 knapsack: maximize sum log2(q/|A|) s.t. sum log2(q) <= log2(m_cap)."""
    ps = list(info)
    if m_cap < 2:
        return []
    cap = int(math.log2(m_cap) * scale)
    if cap <= 0:
        return []
    w = [max(1, math.ceil(math.log2(info[p][0]) * scale)) for p in ps]
    v = [math.log2(info[p][0] / len(info[p][1])) for p in ps]
    NEG = float("-inf")
    dp = [0.0] + [NEG] * cap
    take = [bytearray(cap + 1) for _ in ps]
    for i in range(len(ps)):
        wi, vi, ti = w[i], v[i], take[i]
        if wi > cap or vi <= 0:
            continue
        for c in range(cap, wi - 1, -1):
            prev = dp[c - wi]
            if prev != NEG:
                cand = prev + vi
                if cand > dp[c]:
                    dp[c] = cand; ti[c] = 1
    best_c = max(range(cap + 1), key=lambda c: dp[c])
    wheel, c = [], best_c
    for i in range(len(ps) - 1, -1, -1):
        if take[i][c]:
            wheel.append(ps[i]); c -= w[i]
    wheel.reverse()
    # log-rounding can in principle overshoot: shed the least valuable until the exact
    # integer product fits.  (Exactness of M <= m_cap is load-bearing for the 128-bit path.)
    while wheel and prod(info[p][0] for p in wheel) > m_cap:
        worst = min(wheel, key=lambda p: wheel_score(p, info[p][0], len(info[p][1])))
        wheel.remove(worst)
    return wheel


def _truecost_wheel(info: dict, log2E: float, scale: int = 256,
                    max_M_bits: int = MAX_M_BITS, max_R_bits: int = 62) -> list[int]:
    """Choose the wheel by directly minimising the EXPECTED cost, with no m_cap knob.

    Under the empirical model g(k) ~ Exponential(mean E = 1/density_all) -- which holds
    to KS 0.046 over all 374 published terms, see harness/patterns.py -- the number of
    blocks scanned is  sum_{t>=0} P(g > tM) = 1/(1 - e^{-M/E}),  so

        expected residue checks = R / (1 - e^{-M/E}),   R = prod_{p in W} |A_p|.

    The old objective (maximise wheel value under M <= m_cap) is the M << E limit of this
    and silently ignores the cost of overshooting into a final partial block, so it needs
    an arbitrary safety factor.  Here: DP for the minimum achievable log2(R) at each
    log2(M) bucket, then evaluate the real cost at every bucket and take the best.
    """
    ps = list(info)
    cap = int(max_M_bits * scale)
    w = [max(1, round(math.log2(info[p][0]) * scale)) for p in ps]
    a = [math.log2(len(info[p][1])) for p in ps]
    INF = float("inf")
    dp = [INF] * (cap + 1); dp[0] = 0.0
    take = [bytearray(cap + 1) for _ in ps]
    for i in range(len(ps)):
        wi, ai, ti = w[i], a[i], take[i]
        if wi > cap:
            continue
        for c in range(cap, wi - 1, -1):
            prev = dp[c - wi]
            if prev < INF and prev + ai < dp[c]:
                dp[c] = prev + ai; ti[c] = 1
    best, best_c = INF, 0
    for c in range(cap + 1):
        v = dp[c]
        if v >= INF or v > max_R_bits:
            continue
        x = 2.0 ** min(c / scale - log2E, 60.0)
        denom = -math.expm1(-x) if x < 40 else 1.0
        cost = v - math.log2(denom)
        if cost < best:
            best, best_c = cost, c
    wheel, c = [], best_c
    for i in range(len(ps) - 1, -1, -1):
        if take[i][c]:
            wheel.append(ps[i]); c -= w[i]
    wheel.sort()
    # log-rounding safety: the exact integer product must fit the 128-bit path
    while wheel and prod(info[p][0] for p in wheel) >= (1 << MAX_M_BITS):
        worst = min(wheel, key=lambda p: wheel_score(p, info[p][0], len(info[p][1])))
        wheel.remove(worst)
    return wheel


def choose_wheel(k: int, info: dict[int, tuple[int, list[int]]], m_cap: int,
                 planner: str = "knapsack", scale: int = 512) -> list[int]:
    if planner == "greedy":
        return _greedy_wheel(info, m_cap)
    kn = _knapsack_wheel(info, m_cap, scale)
    gr = _greedy_wheel(info, m_cap)
    # never regress: keep whichever wheel has the smaller expected residue-check count
    def val(wh):
        return sum(math.log2(info[p][0] / len(info[p][1])) for p in wh)
    return kn if (kn and val(kn) >= val(gr)) else gr


def order_rest(rest: list[int], info: dict) -> list[int]:
    """Order of the non-wheel primes inside the kernel: most restrictive first."""
    return sorted(rest, key=lambda p: len(info[p][1]) / info[p][0])


def gen_kernel_src(radix: list[int], res_off: list[int], M: int,
                   rest_q: list[int], rest_taboff: list[int]) -> str:
    """Emit a kernel specialised to one plan: all divisors are compile-time literals,
    both loops fully unrolled.  NVRTC turns `x % <literal>` into multiply+shift."""
    M_lo, M_hi = M & ((1 << 64) - 1), M >> 64
    L = []
    L.append('extern "C" __global__')
    L.append('void es_sieve(const unsigned long long idx_start, const unsigned long long count,')
    L.append('              const unsigned long long* __restrict__ term_lo,')
    L.append('              const unsigned long long* __restrict__ term_hi,')
    L.append('              const unsigned int* __restrict__ tconst,')
    L.append('              const unsigned char* __restrict__ allowed,')
    L.append('              unsigned long long* __restrict__ out, unsigned int* __restrict__ out_n,')
    L.append('              const unsigned int out_cap)')
    L.append('{')
    L.append('    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;')
    L.append('    unsigned long long stride = gridDim.x * (unsigned long long)blockDim.x;')
    L.append(f'    const unsigned long long M_LO = {M_lo}ULL, M_HI = {M_hi}ULL;')
    L.append('    for (unsigned long long i = gid; i < count; i += stride) {')
    L.append('        unsigned long long idx = idx_start + i;')
    L.append('        unsigned long long s_lo = 0, s_hi = 0;')
    L.append('        unsigned long long rem = idx;')
    for j, (r, off) in enumerate(zip(radix, res_off)):
        if j == len(radix) - 1:
            L.append(f'        unsigned int d{j} = (unsigned int)rem;')
        else:
            L.append(f'        unsigned int d{j} = (unsigned int)(rem % {r}ULL); rem /= {r}ULL;')
        L.append(f'        {{ unsigned int o = {off}u + d{j};')
        L.append('          unsigned long long lo = term_lo[o], hi = term_hi[o];')
        L.append('          unsigned long long nlo = s_lo + lo;')
        L.append('          s_hi += hi + (nlo < s_lo ? 1ULL : 0ULL);')
        L.append('          s_lo = nlo;')
        L.append('          if (s_hi > M_HI || (s_hi == M_HI && s_lo >= M_LO)) {')
        L.append('              unsigned long long mlo = s_lo - M_LO;')
        L.append('              s_hi -= M_HI + (s_lo < M_LO ? 1ULL : 0ULL);')
        L.append('              s_lo = mlo; } }')
    L.append('        bool ok = true;')
    for j, (q, toff) in enumerate(zip(rest_q, rest_taboff)):
        p64 = (1 << 64) % q
        L.append(f'        {{ unsigned long long m = ((s_hi % {q}ULL) * {p64}ULL + (s_lo % {q}ULL)) % {q}ULL;')
        L.append(f'          m += tconst[{j}]; if (m >= {q}ULL) m -= {q}ULL;')
        L.append(f'          if (!allowed[{toff}u + m]) ok = false; }}')
        L.append('        if (!ok) continue;')
    L.append('        {')
    L.append('            unsigned int slot = atomicAdd(out_n, 1u);')
    L.append('            if (slot < out_cap) out[slot] = idx;')
    L.append('        }')
    L.append('    }')
    L.append('}')
    return "\n".join(L)


KERNEL_SRC = None  # v2 generates the kernel per plan; see gen_kernel_src
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
        if wheel is not None:
            self.wheel = wheel
        elif self.tun.get("planner", "truecost") == "truecost":
            log2E = -math.log2(self.density_all) if self.density_all > 0 else 0.0
            self.wheel = _truecost_wheel(self.info, log2E,
                                         int(self.tun.get("cost_scale", 256)),
                                         min(int(self.tun.get("m_cap_bits", 120)), MAX_M_BITS),
                                         int(self.tun.get("max_R_bits", 62)))
        else:
            self.wheel = choose_wheel(k, self.info, m_cap, self.tun["planner"],
                                      int(self.tun.get("knapsack_scale", 512)))
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
                "expected_residue_checks": self.density_wheel / self.density_all,
                "expected_checks_true": self._true_cost()}

    def _true_cost(self) -> float:
        """R / (1 - e^{-M/E}) -- expected residue checks including block overshoot."""
        try:
            E = 1.0 / self.density_all
            x = self.M / E
            denom = -math.expm1(-x) if x < 40 else 1.0
            return self.R / denom
        except (ZeroDivisionError, OverflowError):
            return float("inf")


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
        u32 = lambda a: cp.asarray(np.asarray(a, dtype=np.uint32))
        u64 = lambda a: cp.asarray(np.asarray(a, dtype=np.uint64))
        self.w = len(P.wheel)
        offs, terms = [], []
        for j, p in enumerate(P.wheel):
            offs.append(len(terms)); terms.extend((a * P.basis[j]) % P.M for a in P.info[p][1])
        self.term_lo = u64([x & ((1 << 64) - 1) for x in terms])
        self.term_hi = u64([x >> 64 for x in terms])
        # rest-prime tables
        self.rest_q, self.rest_Mmodq, toff, tab = [], [], [], bytearray()
        for p in P.rest:
            q, res = P.info[p]
            self.rest_q.append(q); self.rest_Mmodq.append(P.M % q); toff.append(len(tab))
            t = bytearray(q)
            for r in res:
                t[r] = 1
            tab += t
        self.nrest = len(P.rest)
        self.allowed = cp.asarray(np.frombuffer(bytes(tab) or b"\0", dtype=np.uint8))
        self.tconst_h = np.zeros(max(1, self.nrest), dtype=np.uint32)
        self.tconst = cp.zeros(max(1, self.nrest), dtype=cp.uint32)
        # specialise the kernel to this plan (all divisors become literals)
        self.src = gen_kernel_src(P.radix, offs, P.M, self.rest_q, toff)
        t_compile = time.time()
        self.kernel = cp.RawKernel(self.src, "es_sieve", options=("-std=c++17",))
        self.kernel.compile()
        self.compile_s = time.time() - t_compile
        self.out_cap = int(P.tun["max_survivors"])
        self.out = cp.zeros(self.out_cap, dtype=cp.uint64)
        self.out_n = cp.zeros(1, dtype=cp.uint32)
        self.threads = int(P.tun["threads"])
        self.chunk = 1 << int(P.tun["chunk_log2"])
        self.launches = 0
        self.residues_checked = 0

    def scan_block(self, t: int) -> list[int]:
        """Scan all residues of block t. Returns survivor indices (unverified).

        v4: the survivor counter is reset once per BLOCK, not once per launch, and read
        back once at the end.  The old code called out_n.get() after every launch, and
        each of those is a device-to-host sync that drains the pipeline -- with R/chunk
        launches per block that was the dominant per-launch cost, not the kernel.
        Survivor indices are absolute within the block (the kernel stores idx_start + i),
        so they stay meaningful when accumulated across launches.  Blocks essentially
        always hold 0-2 survivors; if one ever overflows the buffer we fall back to the
        old per-launch path for that block rather than losing a survivor.
        """
        P = self.plan
        # hoist the t-dependent term of every rest-prime check to the host
        for j, q in enumerate(self.rest_q):
            self.tconst_h[j] = ((t % q) * self.rest_Mmodq[j]) % q
        self.tconst.set(self.tconst_h)
        self.out_n.fill(0)
        start = 0
        while start < P.R:
            count = min(self.chunk, P.R - start)
            blocks = min(65535 * 16, (count + self.threads - 1) // self.threads)
            self.kernel((blocks,), (self.threads,), (
                np.uint64(start), np.uint64(count), self.term_lo, self.term_hi,
                self.tconst, self.allowed, self.out, self.out_n, np.uint32(self.out_cap)))
            self.launches += 1
            self.residues_checked += count
            start += count
        n = int(self.out_n.get()[0])          # one sync for the whole block
        if n > self.out_cap:
            return self._scan_block_chunked(t)
        return [int(x) for x in self.out[:n].get()] if n else []

    def _scan_block_chunked(self, t: int) -> list[int]:
        """Per-launch-drain fallback, used only when a block overflows the buffer."""
        P = self.plan
        survivors = []
        start = 0
        while start < P.R:
            count = min(self.chunk, P.R - start)
            self.out_n.fill(0)
            blocks = min(65535 * 16, (count + self.threads - 1) // self.threads)
            self.kernel((blocks,), (self.threads,), (
                np.uint64(start), np.uint64(count), self.term_lo, self.term_hi,
                self.tconst, self.allowed, self.out, self.out_n, np.uint32(self.out_cap)))
            n = int(self.out_n.get()[0])
            if n > self.out_cap:
                raise RuntimeError(f"survivor buffer overflow ({n} > {self.out_cap}); "
                                   "raise max_survivors")
            if n:
                survivors.extend(int(x) for x in self.out[:n].get())
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
