#!/usr/bin/env python3
"""GPU meet-in-the-middle search for g(k). Breaks the M <~ E ceiling of the block sieve.

Structure (see harness/mitm_search.py for the derivation):
  * the SMALLER half of the wheel is materialised as a sorted 128-bit table S2 on the GPU
  * the LARGER half is generated on the fly by mixed-radix decode inside the kernel, exactly
    as gpu_sieve_v4 does -- so only the small table costs memory, and S1 may be enormous
  * each thread forms one s1, binary-searches S2 for the (circular) interval of s2 with
    (s1+s2) mod M < B, walks that contiguous run, and tests the non-wheel primes with early
    exit; survivors are appended atomically

Only n < B are ever formed, so M is free to exceed E and the wheel can absorb far more
primes than the block sieve permits. Every survivor is re-verified by the FROZEN referee.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from math import prod
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import erdos_ref as ref
import u128
from mitm_search import choose_wheel_mitm, split_wheel

try:
    import cupy as cp
except ImportError:
    cp = None

KERNEL = r'''
extern "C" __global__
void mitm(const unsigned long long i_start, const unsigned long long count,
          const int w1, const unsigned int* __restrict__ radix,
          const unsigned int* __restrict__ res_off,
          const unsigned long long* __restrict__ term_hi,
          const unsigned long long* __restrict__ term_lo,
          const unsigned long long M_hi, const unsigned long long M_lo,
          const unsigned long long B_hi, const unsigned long long B_lo,
          const unsigned long long n2,
          const unsigned long long* __restrict__ s2_hi,
          const unsigned long long* __restrict__ s2_lo,
          const int nrest, const unsigned int* __restrict__ q_arr,
          const unsigned int* __restrict__ p64modq,
          const unsigned int* __restrict__ tab_off,
          const unsigned char* __restrict__ allowed,
          unsigned long long* __restrict__ out_hi,
          unsigned long long* __restrict__ out_lo,
          unsigned int* __restrict__ out_n, const unsigned int out_cap)
{
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long t = gid; t < count; t += stride) {
        unsigned long long idx = i_start + t;
        // ---- build s1 by mixed-radix decode (no S1 table in memory) ----
        unsigned long long s_hi = 0, s_lo = 0, rem = idx;
        for (int j = 0; j < w1; ++j) {
            unsigned int r = radix[j];
            unsigned int d = (j == w1 - 1) ? (unsigned int)rem : (unsigned int)(rem % r);
            if (j != w1 - 1) rem /= r;
            unsigned int o = res_off[j] + d;
            unsigned long long ah = term_hi[o], al = term_lo[o];
            unsigned long long nl = s_lo + al;
            s_hi += ah + (nl < s_lo ? 1ULL : 0ULL);
            s_lo = nl;
            if (s_hi > M_hi || (s_hi == M_hi && s_lo >= M_lo)) {
                unsigned long long t2 = s_lo - M_lo;
                s_hi -= M_hi + (s_lo < M_lo ? 1ULL : 0ULL);
                s_lo = t2;
            }
        }
        // ---- target interval: s2 in [lo, lo+B) mod M, where lo = (M - s1) mod M ----
        unsigned long long lo_hi, lo_lo;
        if (s_hi == 0 && s_lo == 0) { lo_hi = 0; lo_lo = 0; }
        else {
            lo_lo = M_lo - s_lo;
            lo_hi = M_hi - s_hi - (M_lo < s_lo ? 1ULL : 0ULL);
        }
        unsigned long long hi_lo = lo_lo + B_lo;
        unsigned long long hi_hi = lo_hi + B_hi + (hi_lo < lo_lo ? 1ULL : 0ULL);
        int wrap = 0;
        if (hi_hi > M_hi || (hi_hi == M_hi && hi_lo >= M_lo)) {
            unsigned long long t3 = hi_lo - M_lo;
            hi_hi -= M_hi + (hi_lo < M_lo ? 1ULL : 0ULL);
            hi_lo = t3; wrap = 1;
        }
        // ---- two lower_bound searches over the sorted 128-bit table ----
        unsigned long long a = 0, b = n2;
        while (a < b) { unsigned long long m = (a + b) >> 1;
            if (s2_hi[m] < lo_hi || (s2_hi[m] == lo_hi && s2_lo[m] < lo_lo)) a = m + 1; else b = m; }
        unsigned long long i0 = a;
        a = 0; b = n2;
        while (a < b) { unsigned long long m = (a + b) >> 1;
            if (s2_hi[m] < hi_hi || (s2_hi[m] == hi_hi && s2_lo[m] < hi_lo)) a = m + 1; else b = m; }
        unsigned long long j0 = a;
        unsigned long long r0s, r0e, r1s, r1e;
        if (!wrap) { r0s = i0; r0e = j0; r1s = 0; r1e = 0; }
        else       { r0s = i0; r0e = n2; r1s = 0;  r1e = j0; }
        for (int part = 0; part < 2; ++part) {
            unsigned long long st = part ? r1s : r0s, en = part ? r1e : r0e;
            for (unsigned long long u = st; u < en; ++u) {
                unsigned long long nh = s_hi + s2_hi[u];
                unsigned long long nl = s_lo + s2_lo[u];
                if (nl < s_lo) nh += 1ULL;
                if (nh > M_hi || (nh == M_hi && nl >= M_lo)) {
                    unsigned long long t4 = nl - M_lo;
                    nh -= M_hi + (nl < M_lo ? 1ULL : 0ULL);
                    nl = t4;
                }
                bool ok = true;
                for (int j = 0; j < nrest; ++j) {
                    unsigned int q = q_arr[j];
                    unsigned long long m2 = ((nh % q) * (unsigned long long)p64modq[j] + (nl % q)) % q;
                    if (!allowed[tab_off[j] + m2]) { ok = false; break; }
                }
                if (ok) {
                    unsigned int slot = atomicAdd(out_n, 1u);
                    if (slot < out_cap) { out_hi[slot] = nh; out_lo[slot] = nl; }
                }
            }
        }
    }
}
'''


def mitm_gpu(k, mem_gb=8.0, bound_mult=3.0, threads=256, chunk_log2=24,
             out_cap=1 << 20, verbose=True, max_rounds=8):
    if cp is None:
        raise RuntimeError("cupy not available")
    t0 = time.time()
    ps = ref.primes_upto(k)
    info = {p: ref.allowed_residues(k, p) for p in ps}
    density_all = prod(len(v[1]) / v[0] for v in info.values())
    E = 1.0 / density_all
    log2E = -math.log2(density_all)
    wheel = choose_wheel_mitm(k, info, log2E, mem_gb * 1e9)
    if not wheel:
        return {"k": k, "status": "NO_WHEEL_FITS_MEMORY"}
    M = prod(info[p][0] for p in wheel)
    basis_of = {}
    for p in wheel:
        q = info[p][0]; Mi = M // q
        basis_of[p] = Mi * pow(Mi, -1, q)
    rest = sorted([p for p in ps if p not in wheel],
                  key=lambda p: len(info[p][1]) / info[p][0])
    w1, w2, n1, n2 = split_wheel(wheel, info)
    if n2 > n1:                      # materialise the SMALLER half
        w1, w2, n1, n2 = w2, w1, n2, n1

    h2, l2 = u128.build_side(w2, info, basis_of, M)
    order = np.lexsort((l2, h2))
    h2 = np.ascontiguousarray(h2[order]); l2 = np.ascontiguousarray(l2[order])
    if verbose:
        print(f"# k={k} wheel={wheel} M=2^{M.bit_length()} E=2^{log2E:.1f} "
              f"|S1|={n1:.3e} |S2|={n2:.3e} ({h2.nbytes*2/1e9:.2f} GB) rest={len(rest)} "
              f"built {time.time()-t0:.1f}s", file=sys.stderr)

    d_s2h = cp.asarray(h2); d_s2l = cp.asarray(l2)
    radix, offs, terms = [], [], []
    for p in w1:
        radix.append(len(info[p][1])); offs.append(len(terms))
        terms.extend((a * basis_of[p]) % M for a in info[p][1])
    d_radix = cp.asarray(np.array(radix, np.uint32))
    d_off = cp.asarray(np.array(offs, np.uint32))
    d_th = cp.asarray(np.array([t >> 64 for t in terms], np.uint64))
    d_tl = cp.asarray(np.array([t & ((1 << 64) - 1) for t in terms], np.uint64))
    qs, p64, toff, tab = [], [], [], bytearray()
    for p in rest:
        q, res = info[p]
        qs.append(q); p64.append((1 << 64) % q); toff.append(len(tab))
        t = bytearray(q)
        for r in res:
            t[r] = 1
        tab += t
    d_q = cp.asarray(np.array(qs or [0], np.uint32))
    d_p64 = cp.asarray(np.array(p64 or [0], np.uint32))
    d_toff = cp.asarray(np.array(toff or [0], np.uint32))
    d_tab = cp.asarray(np.frombuffer(bytes(tab) or b"\0", np.uint8))
    d_oh = cp.zeros(out_cap, cp.uint64); d_ol = cp.zeros(out_cap, cp.uint64)
    d_on = cp.zeros(1, cp.uint32)
    kern = cp.RawKernel(KERNEL, "mitm", options=("-std=c++17",))

    Mh, Ml = np.uint64(M >> 64), np.uint64(M & ((1 << 64) - 1))
    B = max(int(bound_mult * E), 1)
    for _ in range(max_rounds):
        B = min(B, M)
        Bh, Bl = np.uint64(B >> 64), np.uint64(B & ((1 << 64) - 1))
        d_on.fill(0)
        t1 = time.time(); start = 0; chunk = 1 << chunk_log2
        while start < n1:
            cnt = min(chunk, n1 - start)
            blocks = min(65535 * 16, (cnt + threads - 1) // threads)
            kern((blocks,), (threads,), (
                np.uint64(start), np.uint64(cnt), np.int32(len(w1)), d_radix, d_off,
                d_th, d_tl, Mh, Ml, Bh, Bl, np.uint64(n2), d_s2h, d_s2l,
                np.int32(len(rest)), d_q, d_p64, d_toff, d_tab,
                d_oh, d_ol, d_on, np.uint32(out_cap)))
            start += cnt
        nsurv = int(d_on.get()[0])
        if nsurv > out_cap:
            return {"k": k, "status": "SURVIVOR_OVERFLOW", "n": nsurv}
        hs = d_oh[:nsurv].get(); ls = d_ol[:nsurv].get()
        cands = sorted(((int(a) << 64) | int(b)) for a, b in zip(hs, ls))
        good = [n for n in cands if n > k + 1 and ref.is_good(n, k)]
        if verbose:
            print(f"# B={B:.3e} survivors={nsurv} good={len(good)} "
                  f"scan={time.time()-t1:.1f}s", file=sys.stderr)
        if good:
            return {"k": k, "status": "FOUND", "g": min(good), "wheel": wheel,
                    "M_bits": M.bit_length(), "S1": int(n1), "S2": int(n2),
                    "bound": B, "wall": round(time.time() - t0, 2)}
        if B >= M:
            return {"k": k, "status": "MITM_OUT_OF_RANGE",
                    "reason": "g(k) >= M; use the block sieve", "M_bits": M.bit_length()}
        B = min(B * 4, M)
    return {"k": k, "status": "ROUND_LIMIT"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int, nargs="+")
    ap.add_argument("--mem-gb", type=float, default=8.0)
    ap.add_argument("--bound-mult", type=float, default=3.0)
    ap.add_argument("--chunk-log2", type=int, default=24)
    a = ap.parse_args()
    bf = {}
    for line in open(os.path.join(ROOT, "data", "b003458.txt")):
        t = line.split()
        if len(t) == 2 and t[0].isdigit():
            bf[int(t[0])] = int(t[1])
    for k in a.k:
        r = mitm_gpu(k, a.mem_gb, a.bound_mult, chunk_log2=a.chunk_log2)
        if r.get("status") == "FOUND":
            r["bfile"] = str(bf.get(k)) if k in bf else None
            r["matches_bfile"] = (bf.get(k) == r["g"]) if k in bf else None
            r["g"] = str(r["g"]); r["bound"] = str(r["bound"])
        print(json.dumps(r, default=str), flush=True)
