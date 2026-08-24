# Erdős–Selfridge function g(k) — GPU tabulation

Computes and verifies terms of the **Erdős–Selfridge function** g(k) (OEIS
[A003458](https://oeis.org/A003458)): the least integer n > k+1 such that the binomial
coefficient C(n, k) has no prime factor ≤ k. By Kummer's theorem this holds iff, for every
prime p ≤ k, each base-p digit of k is ≤ the corresponding base-p digit of n.

This repository accompanies the note *Further values of the Erdős–Selfridge function and a GPU
algorithm for computing them* (`paper/`). It reproduces the published table and reports **seven new terms, g(378) through g(384)**,
beyond the previously published frontier of k = 377.

## New values (each independently verified: two wheels + frozen referee)

```
g(378) = 11243132307156301763663607287294
g(379) = 161870983573549868804425756301179
g(380) = 10462825184429793014317942235516
g(381) = 12870452058086999925869938534781
g(382) = 33595253716498387794413412981758
g(383) = 540148968489634107903617360993663
g(384) = 347602760349418009297709548536219
```

Verify any of them in seconds with the reference implementation (pure Python, no GPU), e.g.:

```bash
python ref/erdos_ref.py 379 --check 161870983573549868804425756301179
```

## Layout

| path | what it is |
|---|---|
| `ref/erdos_ref.py` | **frozen reference** — brute force + exact CRT search + `is_good`. The referee. |
| `sieve/gpu_sieve*.py` | GPU CRT-wheel sieves (CuPy/NVRTC). `gpu_sieve_v6.py` is the current engine; earlier versions kept for provenance. |
| `harness/` | benchmark, gating, driver (`run_k.py`), evidence generation (`record_term.py`), analysis. |
| `tests/` | agreement tests (referee & sieve vs OEIS b-file), enumeration-equivalence tests. |
| `data/b003458.txt` | OEIS A003458 b-file (375 terms, frozen). `known_extra.txt` adds g(376)-g(381); `b003458_1to381.txt` is the full merged table submitted to OEIS. |
| `evidence/NEW_TERMS.md` | per-block certification ledger for g(378)-g(384): wheel, modulus, survivor counts per tier, SHA-256 per block. |
| `paper/` | the accompanying note (LaTeX + HTML). |

## Method (brief)

For each prime p ≤ k, admissible residues mod p^t form a set; a **wheel** modulus N = ∏ p^T is
chosen by a knapsack over *partial* prime powers (Sorenson–Webster, ANTS XIV 2020). The GPU
enumerates every wheel-admissible residue by mixed-radix index using exact 128-bit arithmetic,
filters the remaining primes via byte lookup tables, and scans blocks [tN, (t+1)N) in increasing t;
the least survivor of the first non-empty block is g(k). Every candidate is re-checked by the
frozen referee, and each accepted term is confirmed by a second run with a materially different
wheel. See the paper for details.

## Certification

A value is accepted only if (i) two runs with different wheels agree, (ii) the frozen referee
confirms Kummer's condition, and (iii) the block scan below the answer is exhaustive. The per-block
ledger in `evidence/` lets a third party re-sieve any single block on a CPU and match the counts.

## Requirements

Reference/tests: Python 3.10+. GPU sieve: an NVIDIA GPU with CuPy (CUDA 12/13) — `cupy-cuda13x[ctk]`.

## License

MIT (see `LICENSE`).
