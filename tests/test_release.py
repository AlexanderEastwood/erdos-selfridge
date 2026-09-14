"""Reject damaged coverage evidence and malformed publication tables."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.verify_release import check_run, read_table


class ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        terms = json.loads((ROOT / 'evidence/through400/runs.json').read_text())['terms']
        self.term: dict[str, Any] = next(term for term in terms if term['k'] == 397)

    def test_missing_completed_block_is_rejected(self) -> None:
        run = deepcopy(self.term['confirmation'])
        del run['block_audit'][1]
        with self.assertRaisesRegex(ValueError, 'missing or repeated block'):
            check_run(397, int(self.term['n']), run)

    def test_partial_residue_count_is_rejected(self) -> None:
        run = deepcopy(self.term['confirmation'])
        run['block_audit'][0]['residues_enumerated'] -= 1
        with self.assertRaisesRegex(ValueError, 'incomplete residue count'):
            check_run(397, int(self.term['n']), run)

    def test_changed_survivor_is_rejected(self) -> None:
        run = deepcopy(self.term['confirmation'])
        run['block_audit'][-1]['survivors_n'][0] = str(int(self.term['n']) + 1)
        with self.assertRaisesRegex(ValueError, 'survivor hash mismatch'):
            check_run(397, int(self.term['n']), run)

    def test_duplicate_bfile_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'b.txt'
            path.write_text('1 3\n1 5\n')
            with self.assertRaisesRegex(ValueError, 'missing, repeated, or unordered'):
                read_table(path, 2)


if __name__ == '__main__':
    unittest.main()
