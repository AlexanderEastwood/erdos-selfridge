# Erdős–Selfridge function g(k): values through 400

The Erdős–Selfridge function g(k), [OEIS A003458](https://oeis.org/A003458), is the least integer n > k+1 such that C(n,k) has no prime factor ≤ k. Kummer's theorem turns this into simultaneous base-p digit constraints for every prime p ≤ k.

**The complete table now contains g(1) through g(400).** This repository reports 23 computed terms, g(378)–g(400), beyond the Sorenson–Sorenson–Webster frontier of 377. On September 14, 2026, the existing OEIS b-file contained 1–389; this release preserves those values and supplies 390–400 for submission. Repository availability is separate from OEIS editorial approval.

- [Complete 400-row b-file](data/b003458_1to400.txt), with decimal integers and consecutive indices.
- [Research note](paper/main.html), [LaTeX source](paper/main.tex), and [updated figure](paper/terms_growth.png).
- [Evidence and reproduction guide](evidence/through400/certificate.html), [paired run records](evidence/through400/runs.json), and [validation receipt](evidence/through400/verification.json).

## Check the release without a GPU

```sh
python3 harness/verify_release.py
python3 -m unittest discover -s tests -p test_release.py
python3 tests/test_agree.py
```

Python 3.11+; no third-party packages are needed for these checks. The release validator checks all 400 witnesses using both the frozen Kummer reference and exact Python binomial arithmetic (17,391 prime-divisor tests), preserves the old 389-row OEIS prefix, and audits 108 unique completed block records across 46 runs.

**Admissibility and minimality are different checks.** The quick CPU checks establish that each value satisfies the defining condition. Minimality is supported by the recorded exhaustive block scans and agreement of two different wheels for each of 378–400. The wheels share implementation components; this is not a formally verified proof or a fresh independent exhaustive replay. Survivor hashes check the published lists, not whether a faulty kernel could have omitted a candidate. Full reproduction requires enumerating the recorded wheels again.

## Computed frontier

| k | g(k) | Completed blocks: primary / confirmation |
|---:|---:|---:|
| 378 | 11243132307156301763663607287294 | 9 / 6 |
| 379 | 161870983573549868804425756301179 | 1 / 1 |
| 380 | 10462825184429793014317942235516 | 1 / 1 |
| 381 | 12870452058086999925869938534781 | 1 / 1 |
| 382 | 33595253716498387794413412981758 | 14 / 4 |
| 383 | 540148968489634107903617360993663 | 2 / 1 |
| 384 | 347602760349418009297709548536219 | 1 / 2 |
| 385 | 327295190388354179623724094491095 | 1 / 1 |
| 386 | 13244365243698813468350652166046 | 1 / 1 |
| 387 | 17088927744279024569389598842319 | 1 / 1 |
| 388 | 1379978640683593021393172825519 | 1 / 1 |
| 389 | 4171959526360182919803787001674199 | 2 / 1 |
| 390 | 1967924476778959819324408694516718 | 1 / 1 |
| 391 | 1407678907549820296859900034849791 | 2 / 3 |
| 392 | 2149910863290282683111595748554173 | 4 / 1 |
| 393 | 8221905832123087594889003885519 | 1 / 1 |
| 394 | 26940132483154282770968897574794 | 1 / 1 |
| 395 | 70422314661266035686410669061023 | 2 / 1 |
| 396 | 7350984493848874781629926017999 | 1 / 2 |
| 397 | 7282852096749166908452220090055597 | 5 / 5 |
| 398 | 358644756463853013270803607289823 | 2 / 1 |
| 399 | 1463367633345557449307696345739199 | 7 / 5 |
| 400 | 58305842808280308870124770403739 | 2 / 3 |

## Method and reproduction

The implementation builds on the partial-prime-power wheel and filter method of Brianna Sorenson, Jonathan P. Sorenson, and Jonathan Webster. It uses CRT to enumerate wheel-admissible residues, strip-mined CUDA loops to amortize residue construction, and residual prime-power filters. Blocks [tN,(t+1)N) are scanned in increasing order and the minimum admissible survivor of the first nonempty block is returned. The final 397 and 399 confirmations used a qualified word-mask filter kernel, retaining original candidate IDs and the existing checkpoint coverage.

`evidence/through400/runs.json` records both wheels, exact moduli, residue counts, all completed blocks, survivor lists and hashes. Repeated identical checkpoint rows have been collapsed; reconstructed historical audit rows remain explicitly labeled. The 397 confirmation includes 26,233,006,904,538,000 previously completed residue positions; they are reused coverage, not additional novel work.

To print a fresh exhaustive GPU reproduction command for either recorded wheel:

```sh
python3 harness/replay_release.py 399 confirmation
# Add --execute to start the scan on an available CUDA GPU.
```

The replay helper uses archived v9 for the recorded mathematical scope. It imports no checkpoint and does not reproduce historical timing. These full searches can take days. GPU execution requires a compatible NVIDIA driver, CUDA, NumPy and CuPy; the CPU release validator does not. Archived engines are supplied unchanged for provenance, not declared portable to every CUDA stack.

## Files and provenance

| Path | Purpose |
|---|---|
| `data/b003458_1to400.txt` | Current complete table. |
| `data/terms.json` | Same values as decimal strings, avoiding floating-point truncation. |
| `data/b003458.txt` | Frozen historical 375-row source; preserved for old tests. |
| `data/b003458_1to387.txt` | Previous published repository snapshot; retained. |
| `data/known_extra.txt` | Supplemental values 376–400 for existing harnesses. |
| `ref/erdos_ref.py` | Frozen exact reference; SHA-256 checked by the validator. |
| `sieve/` | Original engines plus archived v8, v9, v11 and the completed word-mask confirmation engine. |
| `evidence/through400/source-hashes.json` | Hashes of the included source snapshots. Only the final word-mask records contain a complete matching per-run source manifest. |
| `paper/make_terms_growth.py` | Figure generator; install pinned `paper/requirements.txt` to run it. |

The earlier [Zenodo archive](https://doi.org/10.5281/zenodo.22057210) predates this update; no claim is made that it contains the 400-term release. Older manuscript versions remain in Git history. Computational work and drafting used AI assistance under the author's direction; the scope and limitations of the checks are stated above.

MIT license for the repository's code. Historical OEIS material retains its source attribution and [OEIS licensing](https://oeis.org/wiki/The_OEIS_Contributor%27s_License_Agreement).
