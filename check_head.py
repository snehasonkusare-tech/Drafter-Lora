"""Stage 1 smoke test: does the EAGLE-3 head load cleanly and produce the same text as plain decoding?"""
import argparse

import torch
from huggingface_hub import hf_hub_download

from common import BASE, HEAD, chat_ids, load_ea_model, set_seed

p = argparse.ArgumentParser()
p.add_argument("--base", default=BASE)
p.add_argument("--head", default=HEAD)
args = p.parse_args()
set_seed()

model = load_ea_model(args.base, args.head)
tok = model.get_tokenizer()

# EaModel loads the head with strict=False, so check the keys ourselves.
sd = torch.load(hf_hub_download(args.head, "pytorch_model.bin"), map_location="cpu")
expected = set(model.ea_layer.state_dict()) - {"embed_tokens.weight"}  # embeddings come from the target
missing = sorted(expected - set(sd))
unexpected = sorted(set(sd) - expected)
print("head tensors in file:", len(sd))
print("missing keys:", missing or "none")
print("unexpected keys:", unexpected or "none")

prompt = "Explain in three sentences why the sky is blue."
ids = torch.tensor([chat_ids(tok, prompt)], device="cuda")
out_ea, new_ea, idx = model.eagenerate(ids, temperature=0.0, max_new_tokens=128, log=True)
out_base, new_base, _ = model.naivegenerate(ids, temperature=0.0, max_new_tokens=128, log=True)
text_ea = tok.decode(out_ea[0, ids.shape[1]:], skip_special_tokens=True)
text_base = tok.decode(out_base[0, ids.shape[1]:], skip_special_tokens=True)
print("\n--- EAGLE output ---\n" + text_ea)
print("\n--- plain output ---\n" + text_base)
print(f"\naccepted tokens/step on this prompt: {int(new_ea) / (int(idx) + 1):.2f}")
print("outputs identical:", text_ea == text_base)
if missing or unexpected:
    raise SystemExit("Head keys do not match the model. Stop and investigate before benchmarking.")
