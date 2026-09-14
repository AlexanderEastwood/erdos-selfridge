#!/usr/bin/env python3
"""Print or execute a fresh exhaustive GPU scan of a published wheel.

Uses the archived v9 enumerator for every recorded wheel; this reproduces the
mathematical scope, not the historical timing or exact historical engine.
No checkpoint is imported. Large terms can require days of GPU time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def command(k: int, role: str) -> list[str]:
    terms: list[dict[str, Any]] = json.loads((ROOT / 'evidence/through400/runs.json').read_text())['terms']
    term = next(term for term in terms if term['k'] == k)
    run = term[role]
    return [sys.executable, str(ROOT / 'sieve/gpu_sieve_v9.py'), str(k),
            '--wheel', json.dumps(run['wheel'], separators=(',', ':')),
            '--max-blocks', str(run['block'] + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('k', type=int, choices=range(378, 401))
    parser.add_argument('role', choices=('primary', 'confirmation'))
    parser.add_argument('--execute', action='store_true', help='Actually start the exhaustive GPU scan')
    args = parser.parse_args()
    argv = command(args.k, args.role)
    print(shlex.join(argv), flush=True)
    if args.execute:
        subprocess.run(argv, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
