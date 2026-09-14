#!/usr/bin/env python3
"""Check the 400-term publication and captured search records; no GPU replay.

Exact binomial checks establish admissibility, not minimality. The block checks
validate the internal consistency of the recorded exhaustive scans. A full
independent minimality reproduction must enumerate the recorded wheels again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import comb, prod
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref.erdos_ref import digits, dominates, is_good, primes_upto

FROZEN_SHA = 'a2374d4ea0cecaed6f30b1a9f07d382eb62c3e31e8a99a70fdf4cd57f9b9c645'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path, count: int) -> dict[int, int]:
    raw = path.read_bytes()
    require(raw.endswith(b'\n') and b'\r' not in raw, f'{path}: expected LF-terminated text')
    rows: dict[int, int] = {}
    for line in raw.decode('ascii').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        require(re.fullmatch(r'[1-9][0-9]* [1-9][0-9]*', line) is not None, f'{path}: invalid row')
        k, n = map(int, line.split())
        require(k == len(rows) + 1, f'{path}: missing, repeated, or unordered index {k}')
        rows[k] = n
    require(len(rows) == count, f'{path}: expected {count} rows, got {len(rows)}')
    return rows


def check_run(k: int, n: int, run: dict[str, Any]) -> int:
    ps = primes_upto(k)
    wheel: list[list[int]] = run['wheel']
    require(len({p for p, _ in wheel}) == len(wheel), f'k={k}: repeated wheel prime')
    require(all(p in ps and 1 <= exponent <= len(digits(k, p)) for p, exponent in wheel), f'k={k}: invalid wheel')
    modulus = prod(p ** exponent for p, exponent in wheel)
    residues = prod(prod(p - d for d in digits(k, p)[:exponent]) for p, exponent in wheel)
    require(modulus == int(run['N']), f'k={k}: wheel modulus mismatch')
    require(residues == int(run['residues_per_block']), f'k={k}: wheel residue count mismatch')
    require(n // modulus == run['block'], f'k={k}: answer outside terminal block')
    if 'reported_g' in run:
        require(n == int(run['reported_g']), f'k={k}: terminal answer mismatch')
    blocks: list[dict[str, Any]] = run['block_audit']
    require([b['t'] for b in blocks] == list(range(run['block'] + 1)), f'k={k}: missing or repeated block')
    for block in blocks:
        index = block['t']
        require(block['residues_enumerated'] == residues, f'k={k}, t={index}: incomplete residue count')
        survivors = sorted(map(int, block['survivors_n']))
        require(len(set(survivors)) == len(survivors), f'k={k}: repeated survivor')
        require(all(index * modulus <= v < (index + 1) * modulus for v in survivors), f'k={k}: survivor outside block')
        digest = hashlib.sha256('\n'.join(map(str, survivors)).encode()).hexdigest()
        require(digest == block['sha256_survivors'], f'k={k}: survivor hash mismatch')
        require(len(survivors) == block['kernel_survivors'], f'k={k}: survivor count mismatch')
        require(all(all(dominates(v, k, p) for p in ps) for v in survivors), f'k={k}: inadmissible recorded survivor')
        valid = [v for v in survivors if v > k + 1]
        require(block['after_n_gt_k_plus_1'] == block['passed_ref_is_good'] == len(valid), f'k={k}: acceptance count mismatch')
        if index < run['block']:
            require(not valid, f'k={k}: earlier valid survivor')
        else:
            require(bool(valid) and min(valid) == n, f'k={k}: terminal minimum mismatch')
    for path, digest in run.get('engine_revision', {}).items():
        require(sha(ROOT / path) == digest, f'k={k}: source revision mismatch: {path}')
    provenance = run['provenance']
    if provenance.get('record_kind') == 'completed block checkpoint':
        require(provenance['idx_done'] == provenance['R'] == residues, f'k={k}: checkpoint incomplete')
    # Wall times and cumulative counters can include resumed legs. Coverage is
    # calculated from unique complete block receipts, never from elapsed time.
    return len(blocks)


def verify() -> dict[str, Any]:
    require(sha(ROOT / 'ref/erdos_ref.py') == FROZEN_SHA, 'Frozen reference changed')
    evidence = ROOT / 'evidence/through400'
    values = read_table(ROOT / 'data/b003458_1to400.txt', 400)
    before = read_table(evidence / 'oeis-before-b003458.txt', 389)
    require(all(values[k] == n for k, n in before.items()), 'Existing OEIS entry changed')
    historical = read_table(ROOT / 'data/b003458.txt', 375)
    require(all(values[k] == n for k, n in historical.items()), 'Historical table changed')
    json_values = json.loads((ROOT / 'data/terms.json').read_text())
    require(json_values == {str(k): str(n) for k, n in values.items()}, 'JSON table differs')
    prime_checks = 0
    for k, n in values.items():
        require(is_good(n, k), f'k={k}: reference rejects value')
        coefficient = comb(n, k)
        ps = primes_upto(k)
        require(all(coefficient % p != 0 for p in ps), f'k={k}: exact binomial check failed')
        prime_checks += len(ps)
    terms = json.loads((evidence / 'runs.json').read_text())['terms']
    require([term['k'] for term in terms] == list(range(378, 401)), 'Incomplete frontier records')
    block_count = 0
    for term in terms:
        k, n = term['k'], int(term['n'])
        require(n == values[k], f'k={k}: evidence/table mismatch')
        a, b = term['primary'], term['confirmation']
        require(sorted(a['wheel']) != sorted(b['wheel']) and a['N'] != b['N'], f'k={k}: wheels not different')
        for run in (a, b):
            block_count += check_run(k, n, run)
    source_hashes = json.loads((evidence / 'source-hashes.json').read_text())
    for path, digest in source_hashes.items():
        require(sha(ROOT / path) == digest, f'Archived source changed: {path}')
    return dict(status='PASS', checked_utc=datetime.now(timezone.utc).isoformat(),
                terms_checked=400, exact_binomial_prime_divisor_checks=prime_checks,
                existing_oeis_terms_unchanged=389, oeis_terms_added=list(range(390, 401)),
                paired_wheel_terms=23, complete_unique_block_records=block_count,
                bfile_sha256=sha(ROOT / 'data/b003458_1to400.txt'),
                runs_sha256=sha(evidence / 'runs.json'), frozen_reference_sha256=FROZEN_SHA,
                scope='Fresh exact admissibility checks for all 400 values and consistency checks of captured completed block records for 378..400. No fresh exhaustive GPU or CPU minimality replay. Different wheels share implementation components; hashes authenticate lists, not omitted-candidate exclusions.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON receipt path')
    args = parser.parse_args()
    result = verify()
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
