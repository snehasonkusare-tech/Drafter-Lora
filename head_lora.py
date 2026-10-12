"""Add, save and load a LoRA adapter on the EAGLE-3 draft head (model.ea_layer)."""
import json
import os

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer

# Every linear layer in the head except lm_head: the attention + MLP of the single
# decoder layer, and fc (the projection of the target's 3 hidden-state taps).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc"]


def add_head_lora(head, rank):
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=TARGET_MODULES)
    inject_adapter_in_model(cfg, head)
    for name, param in head.named_parameters():
        param.requires_grad = "lora_" in name
    return {"rank": rank, "alpha": 2 * rank, "target_modules": TARGET_MODULES}


def save_head_adapter(head, meta, folder):
    os.makedirs(folder, exist_ok=True)
    sd = {n: p.detach().to(torch.float16).cpu() for n, p in head.named_parameters() if "lora_" in n}
    torch.save(sd, os.path.join(folder, "head_lora.pt"))
    with open(os.path.join(folder, "adapter_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return sd


def apply_head_adapter(head, folder):
    """Load the adapter, merge it into the head's weights, and return the head in fp16."""
    meta = json.load(open(os.path.join(folder, "adapter_config.json")))
    cfg = LoraConfig(r=meta["rank"], lora_alpha=meta["alpha"], lora_dropout=0.0,
                     target_modules=meta["target_modules"])
    inject_adapter_in_model(cfg, head)
    sd = torch.load(os.path.join(folder, "head_lora.pt"), map_location="cpu")
    lora_keys = {n for n, _ in head.named_parameters() if "lora_" in n}
    if set(sd) != lora_keys:
        raise ValueError(f"adapter keys do not match the head: {sorted(set(sd) ^ lora_keys)[:5]}")
    head.load_state_dict(sd, strict=False)
    for m in head.modules():
        if isinstance(m, LoraLayer):
            m.merge()
    head.to(torch.float16)
    print(f"merged head adapter from {folder} ({sum(v.numel() for v in sd.values()):,} params)")
