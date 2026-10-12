"""Build the fixed prompt sets used by every stage.

prompts/general_eval.jsonl   100 Alpaca instructions (Stage 1 baseline)
prompts/med_eval.jsonl       100 MedMCQA validation questions (Stage 2/3 eval)
data/med_train.jsonl         MedMCQA train questions + answers (Stage 2 target LoRA)
prompts/med_headtrain.jsonl  MedMCQA train questions, disjoint from the above (Stage 3 head data)
"""
import argparse
import os
import re

from datasets import load_dataset

from common import SEED, medmcqa_answer, medmcqa_prompt, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--out", default=".")
p.add_argument("--n_eval", type=int, default=100)
p.add_argument("--n_target_train", type=int, default=1000)
p.add_argument("--n_head_train", type=int, default=600)
args = p.parse_args()

alpaca = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=SEED)
alpaca = alpaca.filter(lambda ex: ex["input"] == "").select(range(args.n_eval))
write_jsonl(os.path.join(args.out, "prompts/general_eval.jsonl"),
            [{"id": f"alpaca-{i}", "prompt": ex["instruction"]} for i, ex in enumerate(alpaca)])


def usable(ex):
    return ex["exp"] is not None and 50 <= len(ex["exp"]) <= 1500 and ex["choice_type"] == "single"


def clean_for_training(ex):
    """Training answers only: drop explanations with answer-key prefixes ("Ans. is 'a' i.e. ..."),
    textbook citations or run-together sentences. The first LoRA trained on these looped badly."""
    if not usable(ex):
        return False
    e = ex["exp"].strip()
    return (100 <= len(e) <= 800
            and not re.match(r"(?i)^\s*ans", e)
            and not re.search(r"(?i)\bref\b|reference|textbook|\bpg\b|edition", e)
            and not re.search(r"[a-z]\.[A-Z]", e))


val = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=SEED).filter(usable)
write_jsonl(os.path.join(args.out, "prompts/med_eval.jsonl"),
            [{"id": ex["id"], "prompt": medmcqa_prompt(ex)} for ex in val.select(range(args.n_eval))])

train = load_dataset("openlifescienceai/medmcqa", split="train").shuffle(seed=SEED).filter(clean_for_training)
n1, n2 = args.n_target_train, args.n_target_train + args.n_head_train
write_jsonl(os.path.join(args.out, "data/med_train.jsonl"),
            [{"id": ex["id"], "prompt": medmcqa_prompt(ex), "answer": medmcqa_answer(ex)}
             for ex in train.select(range(n1))])
write_jsonl(os.path.join(args.out, "prompts/med_headtrain.jsonl"),
            [{"id": ex["id"], "prompt": medmcqa_prompt(ex)} for ex in train.select(range(n1, n2))])

print(f"general_eval={len(alpaca)} med_eval={args.n_eval} med_train={n1} med_headtrain={n2 - n1}")
