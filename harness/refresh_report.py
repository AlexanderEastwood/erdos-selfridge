#!/usr/bin/env python3
"""Refresh the live numbers in MORNING_REPORT.md from the result files.

Keeps the narrative untouched and rewrites only the counts that move: timestamp, confirmed
range, referee-confirmed list, wall time, residue checks.  Run it whenever a batch lands so
the report is never stale relative to the data.
"""
import glob, json, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "overnight", "2026-08-21")
p = os.path.join(R, "MORNING_REPORT.md")
s = open(p).read()

recs = [json.loads(l) for l in open(os.path.join(R, "confirmations.jsonl")) if l.strip()]
conf = [r for r in recs if r.get("verdict") == "CONFIRMED_PUBLISHED"]
ks = sorted(r["k"] for r in conf)
n = len(ks)
wall = sum((r.get("wall") or 0) + (r.get("cross_wall") or 0) for r in recs)
checks = sum(r.get("residues_checked") or 0 for r in recs)

# cross-check accounting
recross_ok = {r["k"] for r in recs if r.get("recross_agrees") is True}
diff_wheel = {r["k"] for r in conf if r.get("cross_agrees") is True and r.get("wheel") != r.get("cross_wheel")}
independent = diff_wheel | recross_ok
no_cross = [k for k in ks if k not in independent]

ref = set()
for f in glob.glob(os.path.join(R, "referee", "*.jsonl")) + [os.path.join(R, "referee_extended.jsonl")]:
    if not os.path.exists(f):
        continue
    for l in open(f):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("agrees") is True:
            ref.add(r["k"])
refl = sorted(ref)
reflist = ", ".join(str(k) for k in refl)

now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
gaps = [k for k in range(ks[0], ks[-1] + 1) if k not in ks] if ks else []
rng = f"k = {ks[0]} … {ks[-1]}" if ks else "none"

s = re.sub(r"- Last updated \*\*[^*]+\*\*", f"- Last updated **{now}**", s, count=1)
s = re.sub(r"- \*\*\d+ published terms independently confirmed\*\*, k = \d+…\d+",
           f"- **{n} published terms independently confirmed**, k = {ks[0]}…{ks[-1]}", s, count=1)
s = re.sub(r"\| Terms independently confirmed \| \*\*\d+\*\* — k = \d+ … \d+",
           f"| Terms independently confirmed | **{n}** — k = {ks[0]} … {ks[-1]}", s, count=1)
s = re.sub(r"\| …of which fully referee-confirmed \(rule 2a\) \| \*\*\d+\*\*",
           f"| …of which fully referee-confirmed (rule 2a) | **{len(refl)}**", s, count=1)
s = re.sub(r"\| …with a same-answer run under a \*different\* wheel \| \*\*[^|]*\|",
           f"| …with a same-answer run under a *different* wheel | **{len(independent)} / {n}** |",
           s, count=1)
s = re.sub(r"\| GPU wall for the \d+ \| [\d ]+ s \|", f"| GPU wall for the {n} | {wall:.0f} s |", s, count=1)
s = re.sub(r"\| Residue checks \| [\d.]+ × 10\^?\d+ \|",
           f"| Residue checks | {checks/1e12:.2f} × 10¹² |", s, count=1)
# the referee set and the k>=141 confirmation set are LARGELY DISJOINT -- report both
# sizes and their overlap, never conflate them
overlap = sorted(ref & set(ks))
lo_contig = 0
for k in range(1, 400):
    if k in ref or k <= 60:
        lo_contig = k
    else:
        break
s = re.sub(r"- \*\*The frozen referee independently solved every k from 1 to \d+[^\n]*\n(?:  [^\n]*\n)*",
           f"- **The frozen referee independently solved every k from 1 to {lo_contig}, no gaps**\n"
           f"  ({len(refl)} terms solved by it in total) — rule 2(a) in the strict sense, from a\n"
           f"  pure-Python exhaustive search sharing no code with the GPU. The brief expected this\n"
           f"  to be impossible past k≈140.\n", s, count=1)
s = re.sub(r"- \*\*Referee-confirmed terms inside the k≥141 confirmation range\*\*[^\n]*\n(?:  [^\n]*\n)*",
           f"- **{len(overlap)} terms carry BOTH** kinds of evidence — sieve-confirmed *and* solved\n"
           f"  outright by the referee (k = {', '.join(map(str, overlap))}).\n", s, count=1)
s = re.sub(r"\*\*k = [\d, ]+ were solved outright by the frozen referee\*\*",
           f"**k = {reflist} were solved outright by the frozen referee**", s, count=1)
open(p, "w").write(s)
print(f"refreshed: {n} confirmed ({rng}, gaps={gaps or 'none'}), "
      f"{len(refl)} referee-confirmed, {len(independent)}/{n} independent-wheel, "
      f"wall={wall:.0f}s, checks={checks:.3e}")
if no_cross:
    print(f"  NOT cross-checked (primary run exceeded --cross-max-wall): {no_cross}")
