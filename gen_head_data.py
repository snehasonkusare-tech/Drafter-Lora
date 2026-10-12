"""Stage 3 data: run the LoRA-tuned target (greedy) on in-domain prompts and save prompt+output token ids.

Generation uses EAGLE with the original head only for speed: greedy speculative decoding is
verified by the target, so the text is the target's own greedy output. Resumable.
"""
import argparse
import json
import os

import torch

from common import HEAD, chat_ids, load_ea_model, read_jsonl, set_seed

p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="merged LoRA-tuned target")
p.add_argument("--prompts", default="prompts/med_headtrain.jsonl")
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=600)
p.add_argument("--max_new_tokens", type=int, default=256)
args = p.parse_args()
set_seed()

model = load_ea_model(args.base, HEAD)
tok = model.get_tokenizer()

done = set()
if os.path.exists(args.out):
    done = {json.loads(line)["id"] for line in open(args.out)}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "a") as f:
    prompts = read_jsonl(args.prompts)[: args.n]
    for i, ex in enumerate(prompts):
        if ex["id"] in done:
            continue
        prompt_ids = chat_ids(tok, ex["prompt"])
        out = model.eagenerate(torch.tensor([prompt_ids], device="cuda"), temperature=0.0,
                               max_new_tokens=args.max_new_tokens)
        f.write(json.dumps({"id": ex["id"], "prompt_len": len(prompt_ids), "input_ids": out[0].tolist()}) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(prompts)}")
print("saved", args.out)
