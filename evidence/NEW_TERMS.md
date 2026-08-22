# NEW TERMS — beyond the published frontier

Terms here are **not** in `data/b003458.txt` (375 terms) nor in the Sorenson–Webster 2021
addendum (k=376, 377). **Nothing here has been submitted anywhere**, per the mission ground rules.

Status labels:
* **VERIFIED (two-wheel)** — two runs with materially different wheels agree on n, a fresh
  `ref.is_good(n,k)` passes, and both scans were exhaustive and minimal by construction.
* **CANDIDATE** — anything less. The reason is stated.

Each entry carries a **per-block audit** so a third party can re-sieve any single block on a CPU
from the published parameters and compare counts.

## k = 378 — CANDIDATE — single wheel only; the two-wheel confirmation run has not been done yet
n              = 11243132307156301763663607287294
status         = CANDIDATE — single wheel only; the two-wheel confirmation run has not been done yet
primary run    : engine=sieve/gpu_sieve_v6.py (via harness/run_k.py)@2d62616  driver=harness/run_k.py  wheel=[3^6,101^1,199^1,79^1,197^1,131^1,7^2,97^1,43^1,2^9,193^1,191^1,127^1,5^1,19^1]
                 N=1390243345488918710642394017280 (log2 N=101)  blocks scanned=9 (found in block t=8)
                 residues checked=7982180204544000  wall=36095.4 s  restarts=0  date=2026-08-22T06:23:48Z
confirm run    : NOT RUN
ref.is_good    : `.venv/bin/python -c "from ref.erdos_ref import is_good; print(is_good(11243132307156301763663607287294,378))"` -> True
minimality     : every residue of every block t' <= 8 examined under the primary wheel; no survivor < n  (yes)
E[g]           = 2.442861e+30   n/E[g] = 4.6024   (in range)
logs           : overnight/2026-08-21/logs/k378_primary.log
commit         : 63a1141 (this entry; see the follow-up commit for the logs)
note           : Primary scan complete and exhaustive: blocks t=0..8 ALL fully scanned (7,982,180,204,544,000 residues), answer found in block t=8, 10.03 h at 2.211e11 checks/s, restarts=0, zero DISAGREEMENTs. g/E[g]=4.60, a 1-in-100 upper-tail draw (largest ratio among the 376 known terms is 7.10), which is why this k cost 10 h against an expected 2.6 h. Confirmation run with a materially different wheel (ban=[3], N2=2107560498989903673324279534080) is queued.

### primary run — per-block audit (for independent re-sieving)

    k                = 378
    N                = 1390243345488918710642394017280   (101 bits)
    wheel [[p,T],..] = [[3, 6], [101, 1], [199, 1], [79, 1], [197, 1], [131, 1], [7, 2], [97, 1], [43, 1], [2, 9], [193, 1], [191, 1], [127, 1], [5, 1], [19, 1]]
    residues/block R = 886908911616000
    filter primes    = 72 primes: every p <= k with wheel exponent T < t_p, checked against the FULL p^t_p table
                       [23, 211, 137, 139, 223, 103, 67, 227, 229, 233, 5, 239, 11, 83, 241, 107, 149, 151, 251, 31, 17, 109, 257, 29, 59, 263, 37, 157, 269, 13, 271, 41, 71, 277, 113, 281, 283, 163, 293, 89, 61, 167, 53, 73, 307, 311, 313, 47, 317, 173, 43, 331, 7, 337, 179, 181, 347, 349, 353, 359, 19, 79, 367, 97, 101, 373, 127, 131, 191, 193, 197, 199]

    A kernel survivor of block t is an n in [tN, (t+1)N) whose base-p digits dominate
    k's for every prime p <= k. Tiers below: kernel survivors -> those with n > k+1
    -> those the frozen referee ref.is_good accepts.
    survivor hash    = sha256 of the UTF-8 bytes of '\n'.join(decimal(n) for n in sorted(survivors)), no trailing newline; the empty list hashes to sha256(b'')
    audit source     = reconstructed by harness/reconstruct_audit.py from overnight/2026-08-21/logs/k378_primary.log

    | block t | residues enumerated | kernel surv | n > k+1 | ref.is_good | sha256(survivors) |
    |---:|---:|---:|---:|---:|:--|
    | 0 | 886908911616000 | 2 | 0 | 0 | cf4c98846d083a163a00168221063550e48c3cc3d91ec79dc10cdf15dd63f10c |
    | 1 | 886908911616000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 2 | 886908911616000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 3 | 886908911616000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 4 | 886908911616000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 5 | 886908911616000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 6 | 886908911616000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 7 | 886908911616000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 8 | 886908911616000 | 1 | 1 | 1 | 6bc2d169a52f5ab3e75dca1661f97f2cc6ac924091119002ffca0f56d988cea0 |

    block t=0 survivors: 378, 379
    block t=8 survivors: 11243132307156301763663607287294
