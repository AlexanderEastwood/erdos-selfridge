# NEW TERMS — beyond the published frontier

Terms here are **not** in `data/b003458.txt` (375 terms) nor in the Sorenson–Webster 2021
addendum (k=376, 377). **Nothing here has been submitted anywhere**, per the mission ground rules.

Status labels:
* **VERIFIED (two-wheel)** — two runs with materially different wheels agree on n, a fresh
  `ref.is_good(n,k)` passes, and both scans were exhaustive and minimal by construction.
* **CANDIDATE** — anything less. The reason is stated.

Each entry carries a **per-block audit** so a third party can re-sieve any single block on a CPU
from the published parameters and compare counts.

## k = 378 — VERIFIED (two-wheel)
n              = 11243132307156301763663607287294
status         = VERIFIED (two-wheel)
primary run    : engine=sieve/gpu_sieve_v6.py (via harness/run_k.py)@2d62616  driver=harness/run_k.py  wheel=[3^6,101^1,199^1,79^1,197^1,131^1,7^2,97^1,43^1,2^9,193^1,191^1,127^1,5^1,19^1]
                 N=1390243345488918710642394017280 (log2 N=101)  blocks scanned=9 (found in block t=8)
                 residues checked=7982180204544000  wall=36095.4 s  restarts=0  date=2026-08-22T11:49:11Z
confirm run    : engine=sieve/gpu_sieve_v7.py@f064db3  wheel=[23^2,211^1,199^1,79^1,197^1,131^1,7^2,97^1,43^1,2^9,193^1,191^1,127^1,5^1,19^1] (banned: 3)
                 N=2107560498989903673324279534080 (log2 N=101)
                 blocks=6  residues=7587998466048000  wall=19254.0 s  n agrees: yes
ref.is_good    : `.venv/bin/python -c "from ref.erdos_ref import is_good; print(is_good(11243132307156301763663607287294,378))"` -> True
minimality     : primary examined every n < 12512190109400268395781546155520 (= 9 x N1) exhaustively; confirm examined every n < 12645362993939422039945677204480 (= 6 x N2) exhaustively; n = 11243132307156301763663607287294 lies below both bound, and no good n smaller than it was found by either run  (yes)
E[g]           = 2.442861e+30   n/E[g] = 4.6024   (in range)
logs           : overnight/2026-08-21/logs/k378_primary.log, overnight/2026-08-22/logs/k378_confirm.log
commit         : 8b3d7a2 (this entry; see the follow-up commit for the logs)
note           : Independent on THREE axes: different wheel (confirm bans 3 entirely, using 23^2 and 211 where the primary uses 3^6 and 101), different modulus (N2 != N1), and different CUDA kernel (v7 vs v6, separately gated against the b-file). They found the term in different blocks (t=8 vs t=5), and g/N1=8.0872 and g/N2=5.3347 each independently predict their own block. The per-block survivor SHA-256s agree across the two runs wherever the survivor sets must coincide: block 0 (both cf4c9884..., survivors {378,379}) and the found block (both 6bc2d169..., survivor {g}). Zero DISAGREEMENTs, zero restarts in either run.

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

### confirm run — per-block audit (for independent re-sieving)

    k                = 378
    N                = 2107560498989903673324279534080   (101 bits)
    wheel [[p,T],..] = [[23, 2], [211, 1], [199, 1], [79, 1], [197, 1], [131, 1], [7, 2], [97, 1], [43, 1], [2, 9], [193, 1], [191, 1], [127, 1], [5, 1], [19, 1]]
    residues/block R = 1264666411008000
    filter primes    = 72 primes: every p <= k with wheel exponent T < t_p, checked against the FULL p^t_p table

    A kernel survivor of block t is an n in [tN, (t+1)N) whose base-p digits dominate
    k's for every prime p <= k. Tiers below: kernel survivors -> those with n > k+1
    -> those the frozen referee ref.is_good accepts.
    survivor hash    = sha256 of the UTF-8 bytes of '\n'.join(decimal(n) for n in sorted(survivors)), no trailing newline; the empty list hashes to sha256(b'')
    audit source     = recorded live by harness/run_k.py

    | block t | residues enumerated | kernel surv | n > k+1 | ref.is_good | sha256(survivors) |
    |---:|---:|---:|---:|---:|:--|
    | 0 | 1264666411008000 | 2 | 0 | 0 | cf4c98846d083a163a00168221063550e48c3cc3d91ec79dc10cdf15dd63f10c |
    | 1 | 1264666411008000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 2 | 1264666411008000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 3 | 1264666411008000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 4 | 1264666411008000 | 0 | 0 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
    | 5 | 1264666411008000 | 1 | 1 | 1 | 6bc2d169a52f5ab3e75dca1661f97f2cc6ac924091119002ffca0f56d988cea0 |

    block t=0 survivors: 378, 379
    block t=5 survivors: 11243132307156301763663607287294
