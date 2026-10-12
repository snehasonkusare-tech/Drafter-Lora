"""Stage 2: LoRA fine-tune the target (Qwen3-1.7B) on MedMCQA, then merge and save it.

The merged model is saved in 2 GB shards so it has a model.safetensors.index.json,
which the EAGLE head needs to find the target's embeddings.
"""
import argparse
import csv
import os
import random

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE, chat_ids, read_jsonl, set_seed

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/med_train.jsonl")
p.add_argument("--adapter_out", required=True)
p.add_argument("--merged_out", required=True)
p.add_argument("--log", required=True)
p.add_argument("--max_examples", type=int, default=1000)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--grad_accum", type=int, default=8)
p.add_argument("--max_len", type=int, default=768)
args = p.parse_args()
set_seed()

tok = AutoTokenizer.from_pretrained(BASE)
end_ids = tok("<|im_end|>\n", add_special_tokens=False).input_ids

examples = []
for ex in read_jsonl(args.data)[: args.max_examples]:
    prompt = chat_ids(tok, ex["prompt"])
    answer = tok(ex["answer"], add_special_tokens=False).input_ids + end_ids
    ids = (prompt + answer)[: args.max_len]
    labels = ([-100] * len(prompt) + answer)[: args.max_len]
    examples.append((ids, labels))
schedule = []  # each epoch is a fresh shuffle of the examples
for _ in range(args.epochs):
    order = list(examples)
    random.shuffle(order)
    schedule += order
examples = schedule

if not os.path.exists(os.path.join(args.adapter_out, "adapter_config.json")):
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for param in model.parameters():  # fp16 base, fp32 adapter weights for stable updates
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log = csv.writer(open(args.log, "w", newline=""))
    log.writerow(["step", "loss"])
    running = []
    for i, (ids, labels) in enumerate(examples):
        ids_t = torch.tensor([ids], device="cuda")
        labels_t = torch.tensor([labels], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids_t, labels=labels_t).loss / args.grad_accum
        scaler.scale(loss).backward()
        running.append(loss.item() * args.grad_accum)
        if (i + 1) % args.grad_accum == 0 or i == len(examples) - 1:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            step = (i + 1 + args.grad_accum - 1) // args.grad_accum
            avg = sum(running) / len(running)
            log.writerow([step, round(avg, 4)])
            running = []
            if step % 10 == 0 or i == len(examples) - 1:
                print(f"step {step} loss {avg:.4f}")
    model.save_pretrained(args.adapter_out)
    del model, opt
    torch.cuda.empty_cache()
else:
    print("adapter exists, skipping training:", args.adapter_out)

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map={"": 0})
merged = PeftModel.from_pretrained(base, args.adapter_out).merge_and_unload()
merged.save_pretrained(args.merged_out, max_shard_size="2GB", safe_serialization=True)
tok.save_pretrained(args.merged_out)
assert os.path.exists(os.path.join(args.merged_out, "model.safetensors.index.json")), "missing shard index"
print("merged target saved to", args.merged_out)
