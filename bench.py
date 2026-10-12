"""Benchmark speculative decoding vs plain decoding on a prompt set; one CSV row per prompt.

accept_len = new tokens / draft-verify steps (includes the target's bonus token, as in the EAGLE papers).
Resumable: prompts already in --out are skipped.
"""
import argparse
import csv
import os
import time

import torch

from common import BASE, HEAD, chat_ids, load_ea_model, read_jsonl, set_seed
from head_lora import apply_head_adapter

p = argparse.ArgumentParser()
p.add_argument("--base", default=BASE, help="target model (HF id or merged LoRA folder)")
p.add_argument("--head", default=HEAD)
p.add_argument("--adapter", default=None, help="folder with a draft-head LoRA adapter")
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=100)
p.add_argument("--max_new_tokens", type=int, default=256)
args = p.parse_args()
set_seed()

model = load_ea_model(args.base, args.head)
if args.adapter:
    apply_head_adapter(model.ea_layer, args.adapter)
tok = model.get_tokenizer()


def timed(fn, ids):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out, new_tokens, idx = fn(ids, temperature=0.0, max_new_tokens=args.max_new_tokens, log=True)
    torch.cuda.synchronize()
    return out[0, ids.shape[1]:].tolist(), int(new_tokens), int(idx) + 1, time.perf_counter() - t


prompts = read_jsonl(args.prompts)[: args.n]
warm = torch.tensor([chat_ids(tok, prompts[0]["prompt"])], device="cuda")
timed(model.eagenerate, warm)
timed(model.naivegenerate, warm)

fields = ["id", "input_len", "ea_new_tokens", "ea_steps", "accept_len", "ea_time_s", "ea_tok_per_s",
          "base_new_tokens", "base_time_s", "base_tok_per_s", "speedup", "same_output", "ea_text"]
done = set()
if os.path.exists(args.out):
    done = {r["id"] for r in csv.DictReader(open(args.out))}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
f = open(args.out, "a", newline="")
w = csv.DictWriter(f, fieldnames=fields)
if not done:
    w.writeheader()

for i, ex in enumerate(prompts):
    if ex["id"] in done:
        continue
    ids = torch.tensor([chat_ids(tok, ex["prompt"])], device="cuda")
    ea_out, ea_new, ea_steps, ea_t = timed(model.eagenerate, ids)
    base_out, base_new, _, base_t = timed(model.naivegenerate, ids)
    n = min(len(ea_out), len(base_out))
    row = {
        "id": ex["id"], "input_len": ids.shape[1],
        "ea_new_tokens": ea_new, "ea_steps": ea_steps, "accept_len": round(ea_new / ea_steps, 4),
        "ea_time_s": round(ea_t, 4), "ea_tok_per_s": round(ea_new / ea_t, 2),
        "base_new_tokens": base_new, "base_time_s": round(base_t, 4), "base_tok_per_s": round(base_new / base_t, 2),
        "speedup": round((ea_new / ea_t) / (base_new / base_t), 4),
        "same_output": ea_out[:n] == base_out[:n],
        "ea_text": tok.decode(ea_out, skip_special_tokens=True),
    }
    w.writerow(row)
    f.flush()
    print(f"[{i + 1}/{len(prompts)}] accept_len={row['accept_len']:.2f} "
          f"ea={row['ea_tok_per_s']:.1f} tok/s base={row['base_tok_per_s']:.1f} tok/s")
f.close()
