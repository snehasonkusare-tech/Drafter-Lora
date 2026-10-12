"""Print and save a summary table from bench CSVs.

Usage: python summarize.py --out results/summary.csv "baseline (general)=results/stage1_baseline.csv" ...
"""
import argparse
import csv
import os


def looping(text, n=8, threshold=0.3):
    """True if at least 30% of the output's 8-word chunks are repeats (the model is stuck in a loop)."""
    w = text.split()
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return bool(grams) and 1 - len(set(grams)) / len(grams) >= threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label=path.csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    for run in args.runs:
        label, path = run.split("=", 1)
        data = list(csv.DictReader(open(path)))
        num = lambda k: [float(r[k]) for r in data]
        ea_tok, ea_t = sum(num("ea_new_tokens")), sum(num("ea_time_s"))
        base_tok, base_t = sum(num("base_new_tokens")), sum(num("base_time_s"))
        ok = [r for r in data if not looping(r["ea_text"])]
        rows.append({
            "run": label,
            "prompts": len(data),
            "accept_len": round(sum(num("ea_new_tokens")) / sum(num("ea_steps")), 3),
            "ea_tok_per_s": round(ea_tok / ea_t, 2),
            "base_tok_per_s": round(base_tok / base_t, 2),
            "speedup": round((ea_tok / ea_t) / (base_tok / base_t), 3),
            "looping_pct": round(100 * (len(data) - len(ok)) / len(data), 1),
            "accept_len_no_loops": round(sum(float(r["ea_new_tokens"]) for r in ok) / sum(float(r["ea_steps"]) for r in ok), 3) if ok else "",
            "same_output_pct": round(100 * sum(r["same_output"] == "True" for r in data) / len(data), 1),
        })

    cols = list(rows[0])
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
