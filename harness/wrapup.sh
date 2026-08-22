#!/usr/bin/env bash
# Final wrap-up: close verification gaps, re-audit, re-sync the report, commit.
# Safe to run more than once. Does NOT start any new frontier work.
set -uo pipefail
cd "$(dirname "$0")/.."
R=overnight/2026-08-21
PY=.venv/bin/python

echo "== 1. stop launching new work (leave anything already running to finish) =="
echo "   grind:   $(ps -eo cmd --no-headers | grep -c 'harness/grin[d].py')"
echo "   referee: $(ps -eo cmd --no-headers | grep -c 'referee_swee[p].py')"

echo "== 2. cross-check any confirmed term that still lacks an independent wheel =="
$PY harness/cross_missing.py $R/confirmations.jsonl 3600 2>&1 | tail -20

echo "== 3. four-way audit of every referee-solved term =="
$PY harness/audit.py $R 100000 2>&1 | tail -4

echo "== 4. verify the frozen files are still untouched =="
git diff --quiet cdf3ca0..HEAD -- ref/ sieve/gpu_sieve.py data/ tests/ \
  && echo "   OK: ref/, baseline sieve, data/ and tests/ unchanged since staging" \
  || echo "   !! FROZEN FILES CHANGED -- investigate before trusting anything"

echo "== 5. re-sync the report and tally =="
$PY harness/refresh_report.py
$PY harness/tally.py

echo "== 6. red-alert sweep across every result file =="
for pat in '"RED_ALERT"' '"ref_is_good": false' '"matches_bfile": false' '"agrees": false' 'DISAGREEMENT'; do
  n=$(grep -rl -- "$pat" $R/*.jsonl $R/referee/*.jsonl 2>/dev/null | wc -l)
  echo "   $pat -> $n files"
done

echo "== 7. commit =="
git add -A && git commit -q -m "wrap-up: close cross-check gaps, re-audit, re-sync report" 2>/dev/null \
  && echo "   committed" || echo "   nothing to commit"
echo "== done. Read $R/MORNING_REPORT.md =="
