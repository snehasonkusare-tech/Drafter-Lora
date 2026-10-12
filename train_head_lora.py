"""Stage 3: train a LoRA adapter on the EAGLE-3 draft head against the LoRA-tuned target.

The head's base weights stay frozen. For each sequence the target's hidden states (3 taps)
and logits are computed on the fly; the head is trained to match the target's next-token
distribution over the draft vocabulary, with the same loss as EAGLE-3's training code.

Simplification vs. the official EAGLE-3 recipe: only the first drafting step is trained
(teacher-forced on target features). The official recipe also unrolls later draft steps
("training-time test"). Noted in the README.

Writes the adapter, a per-epoch log (loss + top-1 agreement on held-out sequences), and
results/adapter_size.csv comparing adapter size with the full head.
"""
import argparse
import csv
import os
import random

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from common import HEAD, load_ea_model, read_jsonl, set_seed
from head_lora import add_head_lora, save_head_adapter

p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="merged LoRA-tuned target")
p.add_argument("--data", required=True, help="output of gen_head_data.py")
p.add_argument("--out", required=True, help="adapter folder")
p.add_argument("--log", required=True)
p.add_argument("--size_csv", default=None)
p.add_argument("--max_seqs", type=int, default=100000)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--grad_accum", type=int, default=8)
p.add_argument("--val_frac", type=float, default=0.1)
args = p.parse_args()
set_seed()

model = load_ea_model(args.base, HEAD)
head = model.ea_layer
head.float()  # train in fp32; the target stays fp16
head.reset()  # no tree mask: plain causal attention during training
meta = add_head_lora(head, args.rank)
trainable = [q for q in head.parameters() if q.requires_grad]
print(f"trainable adapter params: {sum(q.numel() for q in trainable):,}")

seqs = read_jsonl(args.data)[: args.max_seqs]
random.shuffle(seqs)
n_val = max(1, int(len(seqs) * args.val_frac))
val, train = seqs[:n_val], seqs[n_val:]
print(f"train seqs={len(train)} val seqs={len(val)}")


def loss_and_acc(ex):
    ids = torch.tensor([ex["input_ids"]], device="cuda")
    with torch.no_grad():
        out = model.base_model.model(input_ids=ids)
        logits = model.base_model.lm_head(out[0])
        feats = torch.cat(out["hidden_states"], dim=-1)
    # Same alignment as EAGLE drafting: target features at position t + token t+1
    # -> predict the target's distribution for token t+2.
    hs_in = feats[:, :-1].float()
    ids_in = ids[:, 1:]
    target = logits[:, 1:].float()
    hidden = head(hs_in, input_ids=ids_in)
    draft_logp = F.log_softmax(head.lm_head(head.norm(hidden)).float(), dim=-1)
    with torch.no_grad():
        t2d = head.t2d
        target_p = F.softmax(target[..., t2d], dim=-1)
        mask = t2d[target.argmax(-1)].float()  # target's top token is in the draft vocab
        start = max(ex["prompt_len"] - 2, 0)  # only score the response tokens
        mask[:, :start] = 0
    denom = mask.sum().clamp(min=1)
    loss = -(mask * (target_p * draft_logp).sum(-1)).sum() / denom
    acc = ((draft_logp.argmax(-1) == target_p.argmax(-1)).float() * mask).sum() / denom
    return loss, acc.item()


def evaluate():
    head.eval()
    with torch.no_grad():
        res = [loss_and_acc(ex) for ex in val]
    head.train()
    return sum(r[0].item() for r in res) / len(res), sum(r[1] for r in res) / len(res)


opt = torch.optim.AdamW(trainable, lr=args.lr)
os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
log = csv.writer(open(args.log, "w", newline=""))
log.writerow(["epoch", "train_loss", "val_loss", "val_top1_agreement"])
vl, va = evaluate()
log.writerow([0, "", round(vl, 4), round(va, 4)])
print(f"before training: val_loss={vl:.4f} val_top1={va:.4f}")

head.train()
for epoch in range(1, args.epochs + 1):
    random.shuffle(train)
    total = 0.0
    for i, ex in enumerate(train):
        loss, _ = loss_and_acc(ex)
        (loss / args.grad_accum).backward()
        total += loss.item()
        if (i + 1) % args.grad_accum == 0 or i == len(train) - 1:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()
    vl, va = evaluate()
    log.writerow([epoch, round(total / len(train), 4), round(vl, 4), round(va, 4)])
    print(f"epoch {epoch}: train_loss={total / len(train):.4f} val_loss={vl:.4f} val_top1={va:.4f}")

sd = save_head_adapter(head, meta, args.out)
print("adapter saved to", args.out)

if args.size_csv:
    head_file = hf_hub_download(HEAD, "pytorch_model.bin")
    head_sd = torch.load(head_file, map_location="cpu")
    head_params = sum(v.numel() for k, v in head_sd.items() if k not in ("d2t", "t2d"))
    adapter_file = os.path.join(args.out, "head_lora.pt")
    row = {
        "adapter_params": sum(v.numel() for v in sd.values()),
        "adapter_file_mb": round(os.path.getsize(adapter_file) / 1e6, 2),
        "head_params": head_params,
        "head_file_mb": round(os.path.getsize(head_file) / 1e6, 2),
    }
    row["param_ratio_pct"] = round(100 * row["adapter_params"] / row["head_params"], 2)
    os.makedirs(os.path.dirname(args.size_csv) or ".", exist_ok=True)
    with open(args.size_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print(row)
