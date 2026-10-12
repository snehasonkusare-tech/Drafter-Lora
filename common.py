"""Shared settings and helpers for the drafter-lora scripts."""
import json
import os
import random

import numpy as np
import torch

BASE = "Qwen/Qwen3-1.7B"
HEAD = "AngelSlim/Qwen3-1.7B_eagle3"
WORK = os.environ.get("WORK", "/kaggle/working")
SEED = 42

# EAGLE-3 drafting settings, fixed for every run so results are comparable.
TOTAL_TOKEN = 60
DEPTH = 7
TOP_K = 10


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chat_ids(tokenizer, prompt):
    """Token ids for a single user turn, ready for the assistant to answer (thinking off)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def medmcqa_prompt(ex):
    opts = "\n".join(f"{l}. {ex['op' + l.lower()]}" for l in "ABCD")
    return f"{ex['question']}\n\n{opts}\n\nAnswer with the correct option and a short explanation."


def medmcqa_answer(ex):
    l = "ABCD"[ex["cop"]]
    return f"Answer: {l}. {ex['op' + l.lower()]}\n\nExplanation: {ex['exp'].strip()}"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_ea_model(base_path, head_path=HEAD):
    from eagle.model.ea_model import EaModel

    model = EaModel.from_pretrained(
        base_model_path=base_path,
        ea_model_path=head_path,
        use_eagle3=True,
        total_token=TOTAL_TOKEN,
        depth=DEPTH,
        top_k=TOP_K,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    return model
