#!/usr/bin/env python3
"""v5. CUDA (CuPy/NVRTC) CRT-wheel search for g(k) with PARTIAL prime-power wheels
(Sorenson-Webster, ANTS 2020, section 5 "prime splitting and knapsack").

Method. For each prime p <= k let t_p = #base-p digits of k and q_p = p^t_p.  A residue
r mod p^T (1 <= T <= t_p) is admissible if its T low base-p digits dominate k's digits;
there are R_T = prod_{i<T}(p - a_ip) of them.  The wheel is a set of items (p, T), one
power per prime, with modulus N = prod p^T; every residue mod N compatible with the wheel
rings is enumerated by a mixed-radix index and blocks [tN, (t+1)N) are scanned for
t = 0, 1, 2, ...  Each GPU thread reconstructs r (128-bit, exact hi/lo arithmetic), then
checks every FILTER prime -- every prime whose full q_p is not entirely in the wheel, i.e.
non-wheel primes and partially-used wheel primes -- via r mod q_p lookups into per-prime
"allowed" byte tables (the full q_p table, so the high digits of a split prime are checked).
Survivors are sent to the host, which rebuilds n with Python ints and re-verifies with the
FROZEN reference (ref/erdos_ref.is_good) before accepting.  Within the first block that has
a survivor all residues are scanned, so the minimum is exact.

Why partial powers: a ring mod p^T has filter rate R_T / p^T, and for small p the first
few digits carry almost all of the filtering per bit of modulus.  Taking the full p^t_p
(v2-v4) spends many bits of N on the high digits of 2, 3, 5 ... which filter poorly;
splitting lets the knapsack spend those bits on extra primes instead.  Cost model (see
research/MORNING_REPORT.md sec 2): k=376 expected residue checks 1.9e19 -> ~2.7e14.

Planner.  Group knapsack DP (one T per prime) over log2(N) buckets, minimising log2(R)
per bucket; then either
  * planner="truecost" (default): evaluate R/(1 - e^{-N/E}) at every bucket and take the
    minimum (v3/v4 objective; E = 1/density_all is the expected g(k)); or
  * planner="knapsack": capacity log2(n_mult * E), maximise wheel value (S-W rule).

Everything between the TUNABLE markers is fair game for automated mutation; everything
else should be considered load-bearing.  CLI/JSON contract is the same as v4 plus
--known (extra known terms, default data/known_extra.txt, covers the addendum k=376,377).
"""
from __future__ import annotations
import argparse, json, math, os, signal, subprocess, sys, time
from itertools import product
from math import prod

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
import erdos_ref as ref  # noqa: E402  (frozen referee, used read-only)

import numpy as np

try:
    import cupy as cp
except ImportError:  # allow import for planning/dry-run on machines without CuPy
    cp = None

MAX_M_BITS = 126  # partial sums stay < 2N < 2^127

# ============================ TUNABLE BEGIN ============================
TUNABLES = {
    "planner": "truecost",  # "truecost" | "knapsack"
    "n_mult": 1.0,          # knapsack planner: capacity = log2(n_mult * E[g])
    "cost_scale": 256,      # DP cells per bit of log2(N)
    "m_cap_bits": 120,      # hard cap on log2(N)
    "max_R_bits": 62,       # residues per block must index in uint64
    "threads": 256,         # CUDA block size
    "chunk_log2": 30,       # residues per kernel launch = 2**chunk_log2
    "max_survivors": 1 << 20,
}


def partial_residues(k: int, p: int, T: int) -> tuple[int, list[int]]:
    """(p^T, sorted residues r in [0, p^T) whose T low base-p digits dominate k's)."""
    kd = ref.digits(k, p)
    assert 1 <= T <= len(kd)
    res = []
    for combo in product(*[range(d, p) for d in kd[:T]]):
        r = 0
        for i, c in enumerate(combo):
            r += c * p ** i
        res.append(r)
    res.sort()
    return p ** T, res


def partial_count(k: int, p: int, T: int) -> int:
    kd = ref.digits(k, p)
    return prod(p - d for d in kd[:T])


def _group_knapsack(k: int, ps: list[int], scale: int, max_M_bits: int):
    """DP: dp[c] = min achievable log2(R) with sum of rounded item weights == c cells.
    Returns (dp, choice, weights) where choice[i][c] is the T picked for prime ps[i] on
    the optimal path to cell c (0 = prime not in wheel) and weights[i][T] is its cells."""
    cap = int(max_M_bits * scale)
    INF = np.inf
    dp = np.full(cap + 1, INF)
    dp[0] = 0.0
    choice, weights = [], []
    for p in ps:
        tp = len(ref.digits(k, p))
        new = dp.copy()
        ch = np.zeros(cap + 1, dtype=np.int8)
        wT = {}
        for T in range(1, tp + 1):
            w = max(1, round(T * math.log2(p) * scale))
            a = math.log2(partial_count(k, p, T))
            wT[T] = w
            if w > cap:
                continue
            cand = dp[:cap + 1 - w] + a
            better = cand < new[w:]
            new[w:][better] = cand[better]
            ch[w:][better] = T
        dp = new
        choice.append(ch)
        weights.append(wT)
    return dp, choice, weights


def _backtrack(ps, choice, weights, c) -> list[tuple[int, int]]:
    wheel = []
    for i in range(len(ps) - 1, -1, -1):
        T = int(choice[i][c])
        if T:
            wheel.append((ps[i], T)); c -= weights[i][T]
    assert c == 0
    wheel.sort()
    return wheel


def plan_wheel(k: int, ps: list[int], log2E: float, tun: dict) -> list[tuple[int, int]]:
    scale = int(tun.get("cost_scale", 256))
    max_M_bits = min(int(tun.get("m_cap_bits", 120)), MAX_M_BITS)
    max_R_bits = float(tun.get("max_R_bits", 62))
    if not ps:
        return []
    dp, choice, weights = _group_knapsack(k, ps, scale, max_M_bits)
    cap = len(dp) - 1
    c_arr = np.arange(cap + 1)
    feasible = np.isfinite(dp) & (dp <= max_R_bits)
    if tun.get("planner", "truecost") == "knapsack":
        c_target = min(cap, int(math.floor((log2E + math.log2(float(tun.get("n_mult", 1.0)))) * scale)))
        feasible &= c_arr <= max(c_target, 0)
        value = np.where(feasible, c_arr / scale - dp, -np.inf)
        best_c = int(np.argmax(value))
        if not feasible[best_c]:
            best_c = 0
    else:
        x = np.minimum(2.0 ** np.clip(c_arr / scale - log2E, -200.0, 60.0), 1e300)
        denom = -np.expm1(-x)
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = np.where(feasible & (denom > 0), dp - np.log2(denom), np.inf)
        best_c = int(np.argmin(cost))
        if not np.isfinite(cost[best_c]):
            best_c = 0
    wheel = _backtrack(ps, choice, weights, best_c)
    # log-rounding safety: exact integer N must fit the 128-bit path and R must fit 62 bits
    def shrink(wh):
        # drop one digit from the item whose top digit contributes least value per bit
        def top_ratio(it):
            p, T = it
            RT, RTm = partial_count(k, p, T), partial_count(k, p, T - 1) if T > 1 else 1
            return (math.log2(p) - math.log2(RT / RTm)) / math.log2(p)
        it = min(wh, key=top_ratio)
        wh.remove(it)
        if it[1] > 1:
            wh.append((it[0], it[1] - 1)); wh.sort()
    while wheel and (prod(p ** T for p, T in wheel) >= (1 << MAX_M_BITS)
                     or prod(partial_count(k, p, T) for p, T in wheel) >= (1 << 62)):
        shrink(wheel)
    return wheel


def order_filters(k: int, filt: list[int], info: dict, wheelT: dict[int, int]) -> list[int]:
    """Kernel order of the filter primes: smallest conditional pass rate first (for a split
    prime the low T digits already passed the wheel ring, so condition on them)."""
    def pass_rate(p):
        q, res = info[p]
        T = wheelT.get(p, 0)
        return (len(res) / q) / (partial_count(k, p, T) / p ** T if T else 1.0)
    return sorted(filt, key=pass_rate)


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
    L.append('        unsigned long long rem = idx; (void)rem;')
    for j, (r, off) in enumerate(zip(radix, res_off)):
        if r == 1:
            L.append(f'        unsigned int d{j} = 0u;')
        elif j == len(radix) - 1:
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
# ============================= TUNABLE END =============================


def _norm_wheel(wheel) -> list[tuple[int, int]]:
    out = []
    for it in wheel:
        if isinstance(it, (list, tuple)):
            out.append((int(it[0]), int(it[1])))
        else:
            out.append((int(it), None))  # None = full power, resolved in Plan
    return out


class Plan:
    def __init__(self, k: int, tun: dict | None = None, wheel=None):
        self.k = k
        self.tun = dict(TUNABLES, **(tun or {}))
        ps = ref.primes_upto(k)
        self.info = {p: ref.allowed_residues(k, p) for p in ps}          # full q_p tables
        self.tp = {p: len(ref.digits(k, p)) for p in ps}
        self.density_all = prod(len(v[1]) / v[0] for v in self.info.values()) if ps else 1.0
        log2E = -math.log2(self.density_all) if self.density_all > 0 else 0.0
        if wheel is not None:
            self.wheel = [(p, self.tp[p] if T is None else T) for p, T in _norm_wheel(wheel)]
            self.wheel.sort()
        else:
            self.wheel = plan_wheel(k, ps, log2E, self.tun)
        for p, T in self.wheel:
            assert 1 <= T <= self.tp[p], f"bad wheel item {(p, T)}"
        assert len({p for p, _ in self.wheel}) == len(self.wheel), "prime used twice in wheel"
        self.wheelT = dict(self.wheel)
        self.rings = [partial_residues(k, p, T) for p, T in self.wheel]   # (p^T, residues)
        self.moduli = [q for q, _ in self.rings]
        self.M = prod(self.moduli)
        assert self.M < (1 << MAX_M_BITS), "wheel modulus too large for 128-bit path"
        # filters: every prime whose full q_p is not entirely inside the wheel
        filt = [p for p in ps if self.wheelT.get(p, 0) < self.tp[p]]
        self.rest = order_filters(k, filt, self.info, self.wheelT)
        self.basis = []
        for q in self.moduli:
            Mi = self.M // q
            self.basis.append(Mi * pow(Mi, -1, q))
        self.radix = [len(res) for _, res in self.rings]
        self.R = prod(self.radix)  # residues per block
        assert self.R < (1 << 62), "too many residues per block"
        self.density_wheel = self.R / self.M if self.M else 1.0
        # fraction of wheel residues that survive all filters (conditional on the rings)
        self.density_rest = self.density_all / self.density_wheel

    def idx_to_n(self, idx: int, t: int) -> int:
        r = 0
        for j, ((q, res), rad) in enumerate(zip(self.rings, self.radix)):
            idx, d = divmod(idx, rad)
            r += res[d] * self.basis[j]
        return t * self.M + (r % self.M)

    def summary(self) -> dict:
        E = 1 / self.density_all
        return {"k": self.k, "wheel": [[p, T] for p, T in self.wheel],
                "wheel_primes": [p for p, _ in self.wheel],
                "split_primes": [[p, T, self.tp[p]] for p, T in self.wheel if T < self.tp[p]],
                "N_bits": self.M.bit_length(), "M_bits": self.M.bit_length(), "M": str(self.M),
                "N": str(self.M), "n_filters": len(self.rest),
                "residues_per_block": self.R, "density_wheel": self.density_wheel,
                "density_rest": self.density_rest, "expected_g": E, "E_g": E,
                "expected_blocks": E / self.M, "N_over_E": self.M / E,
                "expected_residue_checks": self.density_wheel / self.density_all,
                "expected_checks_true": self._true_cost()}

    def _true_cost(self) -> float:
        """R / (1 - e^{-N/E}) -- expected residue checks including block overshoot."""
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
        u64 = lambda a: cp.asarray(np.asarray(a, dtype=np.uint64))
        self.w = len(P.wheel)
        offs, terms = [], []
        for j, (q, res) in enumerate(P.rings):
            offs.append(len(terms)); terms.extend((a * P.basis[j]) % P.M for a in res)
        self.term_lo = u64([x & ((1 << 64) - 1) for x in terms] or [0])
        self.term_hi = u64([x >> 64 for x in terms] or [0])
        # filter tables over the FULL q_p (high digits of split primes are checked here)
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

    def _launch(self, start: int, count: int):
        blocks = min(65535 * 16, (count + self.threads - 1) // self.threads)
        self.kernel((blocks,), (self.threads,), (
            np.uint64(start), np.uint64(count), self.term_lo, self.term_hi,
            self.tconst, self.allowed, self.out, self.out_n, np.uint32(self.out_cap)))
        self.launches += 1

    def scan_block(self, t: int) -> list[int]:
        """Scan all residues of block t; one device sync per block (v4 trick).  Survivor
        indices are absolute within the block, so accumulating across launches is safe."""
        P = self.plan
        for j, q in enumerate(self.rest_q):
            self.tconst_h[j] = ((t % q) * self.rest_Mmodq[j]) % q
        self.tconst.set(self.tconst_h)
        self.out_n.fill(0)
        start = 0
        while start < P.R:
            count = min(self.chunk, P.R - start)
            self._launch(start, count)
            self.residues_checked += count
            start += count
        n = int(self.out_n.get()[0])
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
            self._launch(start, count)
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


def load_known(*paths: str) -> dict[int, tuple[int, str]]:
    """k -> (g, source) from whitespace 'k g' files; later files do not override earlier."""
    known: dict[int, tuple[int, str]] = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for line in open(path):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                known.setdefault(int(parts[0]), (int(parts[1]), os.path.basename(path)))
    return known


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="GPU CRT-wheel search for g(k), partial-power wheel")
    ap.add_argument("k", type=int, nargs="+")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--tune", type=json.loads, default=None, help='JSON dict overriding TUNABLES')
    ap.add_argument("--wheel", type=json.loads, default=None,
                    help="explicit wheel: JSON list of [p,T] (or bare p = full power)")
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None, help="seconds; process exits 124 on expiry")
    ap.add_argument("--min-free-mib", type=int, default=2048)
    ap.add_argument("--bfile", default=os.path.join(here, "..", "data", "b003458.txt"))
    ap.add_argument("--known", default=os.path.join(here, "..", "data", "known_extra.txt"),
                    help="extra 'k g' file of known terms not in the b-file (addendum k=376,377)")
    a = ap.parse_args()
    if a.timeout:
        signal.signal(signal.SIGALRM, lambda *_: (print(json.dumps({"status": "TIMEOUT"}), flush=True), os._exit(124)))
        signal.setitimer(signal.ITIMER_REAL, a.timeout)
    known = load_known(a.bfile, a.known)
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
            g_known, src = known[k]
            res["matches_bfile"] = (res["g"] == g_known)
            res["bfile"] = g_known
            res["known_source"] = src
        res["plan"] = plan.summary()
        print(json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
