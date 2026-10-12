"""Per-prompt paired comparison of accepted tokens/step between two runs on the same prompts.

Prompts where either run loops (see summarize.looping) are excluded. The 95% interval is a
bootstrap over prompts (4,000 resamples, seed 0).

Usage: python paired.py --out results/paired_stats.csv "name=before.csv,after.csv" ...
"""
import argparse
import csv
import os
import random
import statistics as st

from summarize import looping

p = argparse.ArgumentParser()
p.add_argument("pairs", nargs="+", help="label=before.csv,after.csv")
p.add_argument("--out", required=True)
args = p.parse_args()

rows = []
for pair in args.pairs:
    label, paths = pair.split("=", 1)
    before, after = ({r["id"]: r for r in csv.DictReader(open(path))} for path in paths.split(","))
    ids = [k for k in before if not looping(before[k]["ea_text"]) and not looping(after[k]["ea_text"])]
    diffs = [float(after[k]["accept_len"]) - float(before[k]["accept_len"]) for k in ids]
    random.seed(0)
    boot = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(4000))
    base = st.mean(float(before[k]["accept_len"]) for k in ids)
    rows.append({
        "comparison": label, "prompts": len(ids),
        "before_mean": round(base, 3), "after_mean": round(base + st.mean(diffs), 3),
        "mean_change": round(st.mean(diffs), 3), "pct_change": round(100 * st.mean(diffs) / base, 1),
        "ci95_low": round(boot[100], 3), "ci95_high": round(boot[3899], 3),
        "after_higher": sum(d > 0 for d in diffs), "after_lower": sum(d < 0 for d in diffs),
    })
    print(rows[-1])

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
