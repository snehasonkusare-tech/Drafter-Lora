"""Stage 4 chart: speedup over plain decoding for the three setups, on medical and general prompts.

Reads two summary CSVs written by summarize.py, each from a single Kaggle session
(so speeds within a group are directly comparable).
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--medical", default="results/summary.csv")
p.add_argument("--general", default="results/summary_stage3b.csv")
p.add_argument("--out", default="results/chart.png")
args = p.parse_args()

SERIES = [  # fixed order and colours (validated categorical palette, slots 1-3)
    ("Original model, original head", "#2a78d6"),
    ("Strong LoRA, original head", "#eb6834"),
    ("Strong LoRA, head + LoRA adapter", "#1baf7a"),
]
groups = []
for label, path in [("Medical prompts (MedMCQA)", args.medical), ("General prompts (Alpaca)", args.general)]:
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 3, f"{path}: expected the 3 runs in SERIES order"
    groups.append((label, [float(r["speedup"]) for r in rows], [float(r["accept_len"]) for r in rows]))

fig, ax = plt.subplots(figsize=(9, 4.8))
width, gap = 0.26, 0.02
for g, (glabel, speedups, accepts) in enumerate(groups):
    for i, ((name, color), s, a) in enumerate(zip(SERIES, speedups, accepts)):
        x = g + (i - 1) * (width + gap)
        ax.bar(x, s, width, color=color, label=name if g == 0 else None)
        ax.text(x, s + 0.03, f"{s:.2f}x", ha="center", va="bottom", fontsize=10, color="#0b0b0b")
        ax.text(x, 0.06, f"{a:.2f}\ntok/step", ha="center", va="bottom", fontsize=8, color="#0b0b0b")
ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=10, color="#52514e")
ax.set_ylabel("Speedup over plain decoding", fontsize=10, color="#52514e")
ax.set_ylim(0, 2.4)
ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", colors="#52514e", length=0)
ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
ax.set_title("A 1.7 MB adapter on the EAGLE-3 head restores the speedup lost to a LoRA fine-tune",
             fontsize=11, loc="left", color="#0b0b0b")
fig.text(0.01, 0.005, "Qwen3-1.7B + AngelSlim EAGLE-3 head, Kaggle T4, greedy, max 256 new tokens, 100 prompts per bar",
         fontsize=8, color="#52514e")
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(args.out, dpi=160)
print("saved", args.out)
