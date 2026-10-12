"""How far did a fine-tune move the target? Score each run's outputs with the ORIGINAL model.

top1_agreement: share of output tokens the original model would also have picked greedily
                (about 100% for the original model's own outputs; lower = bigger shift).
mean_nll:       average negative log-likelihood of the output tokens under the original model.

Usage: python shift.py --out results/shift.csv "mild=results/stage2_med_broken.csv" ...
"""
import argparse
import csv
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE, chat_ids, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", help="label=bench.csv")
p.add_argument("--prompts", default="prompts/med_eval.jsonl")
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=100)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map={"": 0}).eval()
prompts = {ex["id"]: ex["prompt"] for ex in read_jsonl(args.prompts)}

rows = []
for run in args.runs:
    label, path = run.split("=", 1)
    agree = total = 0
    nll = 0.0
    for r in list(csv.DictReader(open(path)))[: args.n]:
        prompt_ids = chat_ids(tok, prompts[r["id"]])
        out_ids = tok(r["ea_text"], add_special_tokens=False).input_ids
        if not out_ids:
            continue
        ids = torch.tensor([prompt_ids + out_ids], device="cuda")
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1:-1].float()
        target = ids[0, len(prompt_ids):]
        agree += (logits.argmax(-1) == target).sum().item()
        nll += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
        total += len(out_ids)
    rows.append({"run": label, "tokens": total, "top1_agreement": round(agree / total, 4),
                 "mean_nll": round(nll / total, 4)})
    print(rows[-1])

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
