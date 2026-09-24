# Run logs (378–400)

Raw operational output from the searches that produced the values in `data/b003458_1to400.txt`,
one directory per run date, mirroring the layout the harness wrote.

| what | files |
|---|---|
| sieve run logs (`k<k>_primary.log`, `k<k>_confirm*.log`, driver and utility logs) | 129 |
| per-run checkpoints (`checkpoints/*.json`) — block index, residues done, resume state | 42 |
| `confirmations.jsonl` — the per-term confirmation ledger as it was written live | 17 |
| `power.csv.gz` — PSU telemetry for the 2026-08-21 run (supports the power figures in §Running time) | 1 |

These are the *operational* record: progress lines, rates, timeouts, restarts, and resume points.
The authoritative per-block evidence for each published value is `evidence/through400/runs.json`
(wheel, modulus, per-block residues enumerated, kernel survivors, survivor digests), summarised in
`evidence/through400/certificate.html` with the validation receipt in `verification.json`.

## What is deliberately not here

* **AI terminal session captures** (`claude_session*.log`, ~34 MB). They are ANSI-escape recordings
  of working sessions, not evidence anyone would re-check, and they carry contact and network
  details unrelated to the computation.
* **Working notes** (`*.md`) and **`experiments.jsonl`** — internal narrative and operational
  commentary rather than search evidence.

Logs are lightly structured and were written for operators, not readers: a term's story is often
split across several files as runs timed out at the 12-hour wall and resumed. Where a log and the
per-block ledger disagree on presentation, `runs.json` is authoritative.
