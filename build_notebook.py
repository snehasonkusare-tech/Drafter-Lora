"""Build drafter_lora_kaggle.ipynb: the scripts as %%writefile cells plus the run cells for each stage.

Run locally after editing any script:  python build_notebook.py
One stage only (for an unattended Kaggle run):  python build_notebook.py --stages 1 --out kaggle_run/stage1.ipynb
"""
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("--stages", default="1,2,2b,3,3b,4", help="e.g. 1 for a Stage-1-only notebook")
p.add_argument("--out", default="drafter_lora_kaggle.ipynb")
args = p.parse_args()
keep = {"0"} | set(args.stages.split(","))

SCRIPTS = ["common.py", "head_lora.py", "make_prompts.py", "check_head.py", "bench.py", "summarize.py", "paired.py", "shift.py",
           "train_target_lora.py", "gen_head_data.py", "train_head_lora.py", "plot.py"]
EAGLE_COMMIT = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"
SMOKE = "/kaggle/working/smoke"

cells = []
stage = "0"  # "0" = intro, setup and scripts (always included)


def md(text):
    if stage in keep:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})


def code(text):
    if stage in keep:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": text.strip()})


md("""
# drafter-lora

Does a small LoRA adapter on an EAGLE-3 draft head restore speculative decoding speed after the
target model is LoRA fine-tuned?

**Before running:** Settings → Accelerator → **GPU T4 x2** (only one GPU is used), and
Settings → **Internet on** (needs a phone-verified Kaggle account).

Run one stage at a time. Each stage runs a small smoke test first, then the real run.
At the end of each stage, copy the printed tables back to Claude and download
`/kaggle/working/drafter-lora-results.zip` (Kaggle deletes files when the session ends).
""")

md("## Setup")
code(f"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORK"] = "/kaggle/working"
os.environ["PYTHONPATH"] = "/kaggle/working/EAGLE:/kaggle/working/drafter-lora"
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q "transformers==4.53.3" "peft==0.16.0"
!test -d /kaggle/working/EAGLE || git clone -q https://github.com/SafeAILab/EAGLE.git /kaggle/working/EAGLE
!cd /kaggle/working/EAGLE && git checkout -q {EAGLE_COMMIT} && git log -1 --format="EAGLE commit %h (%cd)"
!python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '|', torch.cuda.get_device_name(0))"
os.makedirs("/kaggle/working/drafter-lora", exist_ok=True)
%cd /kaggle/working/drafter-lora
""")
code("""
# When earlier stages ran as separate Kaggle notebooks attached as inputs, bring their outputs in.
import glob
for src in glob.glob("/kaggle/input/**/drafter-lora", recursive=True):
    for sub in ["results", "prompts", "data", "adapters"]:
        if os.path.isdir(f"{src}/{sub}"):
            !cp -rn {src}/{sub} .
for found in glob.glob("/kaggle/input/**/target_lora*", recursive=True):  # saved target LoRA adapters
    name = os.path.basename(found)
    if os.path.isdir(found) and not os.path.exists(f"/kaggle/working/{name}"):
        os.symlink(found, f"/kaggle/working/{name}")
!ls results prompts 2>/dev/null; ls -l /kaggle/working | grep target || true
""")

md("## Scripts\nEach cell writes one script into `/kaggle/working/drafter-lora`.")
for name in SCRIPTS:
    code(f"%%writefile {name}\n" + open(name).read())

stage = "1"
md("""
## Stage 1 - Baseline
Original Qwen3-1.7B + original EAGLE-3 head on 100 general (Alpaca) prompts.
""")
code("!python make_prompts.py")
code("# Smoke test: head keys match, and EAGLE output equals plain greedy output\n!python check_head.py")
code("""# Small run first: 5 prompts, 128 tokens
!python bench.py --prompts prompts/general_eval.jsonl --n 5 --max_new_tokens 128 --out results/smoke/stage1.csv
!python summarize.py "smoke=results/smoke/stage1.csv"
""")
code("""# Real run: 100 prompts, 256 tokens
!python bench.py --prompts prompts/general_eval.jsonl --out results/stage1_baseline.csv
!python summarize.py "baseline (general)=results/stage1_baseline.csv" --out results/summary_stage1.csv
""")
code("!cd /kaggle/working && zip -qr drafter-lora-results.zip drafter-lora/results drafter-lora/prompts && echo zipped")
md("**Stop here.** Send Claude the outputs above before starting Stage 2.")

stage = "2"
md("""
## Stage 2 - Break it
LoRA fine-tune the target on MedMCQA, then run the *original* head against it on 100 held-out
medical prompts. The original target is also run on the same medical prompts, so the drop isn't
confused with a change of domain.
""")
code("""# Rebuild the training data with the cleaned-explanation filter. The eval prompts must not change:
# compare the two checksums printed below.
!md5sum prompts/med_eval.jsonl prompts/general_eval.jsonl 2>/dev/null
!python make_prompts.py
!md5sum prompts/med_eval.jsonl prompts/general_eval.jsonl
!head -c 600 data/med_train.jsonl
""")
code(f"""# Small run first: 40 examples, then a 3-prompt benchmark
!python train_target_lora.py --max_examples 40 --grad_accum 4 --adapter_out {SMOKE}/target_lora --merged_out {SMOKE}/target_merged --log results/smoke/target_lora_loss.csv
!python bench.py --base {SMOKE}/target_merged --prompts prompts/med_eval.jsonl --n 3 --max_new_tokens 64 --out results/smoke/stage2.csv
!python summarize.py "smoke=results/smoke/stage2.csv"
!rm -rf {SMOKE}/target_lora {SMOKE}/target_merged
""")
code("""# Real run: 1000 examples, 1 epoch
!python train_target_lora.py --adapter_out /kaggle/working/target_lora --merged_out /kaggle/working/target_merged --log results/target_lora_loss.csv
""")
code("""!python bench.py --prompts prompts/med_eval.jsonl --out results/stage2_med_original.csv
!python bench.py --base /kaggle/working/target_merged --prompts prompts/med_eval.jsonl --out results/stage2_med_broken.csv
!python summarize.py "baseline (general)=results/stage1_baseline.csv" "original target (medical)=results/stage2_med_original.csv" "LoRA target, original head (medical)=results/stage2_med_broken.csv" --out results/summary_stage2.csv
""")
code("!cd /kaggle/working && zip -qr drafter-lora-results.zip drafter-lora/results drafter-lora/prompts target_lora && echo zipped")
md("""
**Stop here.** Send Claude the table. If accepted tokens per step barely dropped, that is a real
finding and Stage 3 should change.
""")

stage = "2b"
md("""
## Stage 2b - How hard do you have to fine-tune before the head falls behind?
The mild fine-tune (1,000 examples, 1 epoch, rank 16) did not lower acceptance. Two stronger
fine-tunes on the same medical data, each benchmarked on the same 100 eval prompts.
`shift.py` measures how far each fine-tune moved the model: the share of its output tokens the
original model would also have picked (the original model's own outputs should be near 100%).
""")
code("""# 2,000 training examples (eval prompts must be unchanged: compare the checksum with Stage 2's)
!python make_prompts.py --n_target_train 2000
!md5sum prompts/med_eval.jsonl prompts/general_eval.jsonl
""")
code("""# Small run first: 16 examples x 2 epochs, 3-prompt benchmark, shift on 3 outputs
!python train_target_lora.py --max_examples 16 --epochs 2 --rank 64 --grad_accum 4 --adapter_out /kaggle/working/smoke/l2_lora --merged_out /kaggle/working/smoke/l2_merged --log results/smoke/l2_loss.csv
!python bench.py --base /kaggle/working/smoke/l2_merged --prompts prompts/med_eval.jsonl --n 3 --max_new_tokens 64 --out results/smoke/stage2b.csv
!python shift.py --n 3 --out results/smoke/shift.csv "original=results/stage2_med_original.csv" "smoke=results/smoke/stage2b.csv"
!rm -rf /kaggle/working/smoke
""")
code("""# Medium: 2,000 examples x 3 epochs, rank 64, lr 1e-4
!python train_target_lora.py --max_examples 2000 --epochs 3 --rank 64 --lr 1e-4 --adapter_out /kaggle/working/target_lora_medium --merged_out /kaggle/working/target_merged_medium --log results/target_lora_loss_medium.csv
!python bench.py --base /kaggle/working/target_merged_medium --prompts prompts/med_eval.jsonl --out results/stage2b_med_medium.csv
!rm -rf /kaggle/working/target_merged_medium
""")
code("""# Strong: same, lr 3e-4
!python train_target_lora.py --max_examples 2000 --epochs 3 --rank 64 --lr 3e-4 --adapter_out /kaggle/working/target_lora_strong --merged_out /kaggle/working/target_merged_strong --log results/target_lora_loss_strong.csv
!python bench.py --base /kaggle/working/target_merged_strong --prompts prompts/med_eval.jsonl --out results/stage2b_med_strong.csv
!rm -rf /kaggle/working/target_merged_strong
""")
code("""!python shift.py --out results/shift.csv "original=results/stage2_med_original.csv" "mild=results/stage2_med_broken.csv" "medium=results/stage2b_med_medium.csv" "strong=results/stage2b_med_strong.csv"
!python summarize.py "original target (medical)=results/stage2_med_original.csv" "mild LoRA=results/stage2_med_broken.csv" "medium LoRA=results/stage2b_med_medium.csv" "strong LoRA=results/stage2b_med_strong.csv" --out results/summary_stage2b.csv
""")
code("!cd /kaggle/working && zip -qr drafter-lora-results.zip drafter-lora/results drafter-lora/prompts target_lora_medium target_lora_strong && echo zipped")
md("**Stop here.** Send Claude the two tables.")

stage = "3"
md("""
## Stage 3 - Fix it
Target: the **strong** fine-tune from Stage 2b (its LoRA adapter is re-merged here, no retraining).
Generate data with it, train a LoRA adapter on the draft head (base weights frozen), and benchmark
on the same 100 medical prompts. The original model and the strong model with the original head are
re-run in this same session, so all speeds in the final table come from one machine.
""")
code("""# Head-training prompts must not overlap the strong model's 2,000 training examples:
# regenerate them (rows 2000-2600 of the shuffled cleaned train split). Eval checksum must match Stage 2.
!python make_prompts.py --n_target_train 2000
!md5sum prompts/med_eval.jsonl prompts/general_eval.jsonl
!python -c "import json; a={json.loads(l)['id'] for l in open('data/med_train.jsonl')}; b={json.loads(l)['id'] for l in open('prompts/med_headtrain.jsonl')}; print('train', len(a), 'headtrain', len(b), 'overlap', len(a & b))"
# Re-merge the strong target from its saved adapter (training is skipped because the adapter exists)
!python train_target_lora.py --max_examples 2000 --epochs 3 --rank 64 --lr 3e-4 --adapter_out /kaggle/working/target_lora_strong --merged_out /kaggle/working/target_merged_strong --log results/smoke/remerge_log.csv
""")
code(f"""# Small run first: 10 sequences, 1 epoch, 3-prompt benchmark
!python gen_head_data.py --base /kaggle/working/target_merged_strong --n 10 --out {SMOKE}/head_data.jsonl
!python train_head_lora.py --base /kaggle/working/target_merged_strong --data {SMOKE}/head_data.jsonl --epochs 1 --grad_accum 2 --out {SMOKE}/head_lora --log results/smoke/head_lora_log.csv
!python bench.py --base /kaggle/working/target_merged_strong --adapter {SMOKE}/head_lora --prompts prompts/med_eval.jsonl --n 3 --max_new_tokens 64 --out results/smoke/stage3.csv
!python summarize.py "smoke=results/smoke/stage3.csv"
""")
code("""# Real run: 600 sequences, 2 epochs
!python gen_head_data.py --base /kaggle/working/target_merged_strong --out data/head_data.jsonl
!python train_head_lora.py --base /kaggle/working/target_merged_strong --data data/head_data.jsonl --out adapters/head_lora --log results/head_lora_log.csv --size_csv results/adapter_size.csv
!cat results/head_lora_log.csv
""")
code("""# All three medical runs in this one session
!python bench.py --prompts prompts/med_eval.jsonl --out results/stage3_med_original.csv
!python bench.py --base /kaggle/working/target_merged_strong --prompts prompts/med_eval.jsonl --out results/stage3_med_strong_orighead.csv
!python bench.py --base /kaggle/working/target_merged_strong --adapter adapters/head_lora --prompts prompts/med_eval.jsonl --out results/stage3_med_strong_fixed.csv
!python summarize.py "original model=results/stage3_med_original.csv" "strong LoRA, original head=results/stage3_med_strong_orighead.csv" "strong LoRA, head + adapter=results/stage3_med_strong_fixed.csv" --out results/summary.csv
!cat results/adapter_size.csv
""")
code("!cd /kaggle/working && zip -qr drafter-lora-results.zip drafter-lora/results drafter-lora/prompts drafter-lora/adapters && echo zipped")
md("**Stop here.** Send Claude the table, the training log and the adapter size.")

stage = "3b"
md("""
## Stage 3b - Does the head adapter hurt outside the medical domain?
The adapter was trained on medical answers only. Re-run the 100 general (Alpaca) prompts from
Stage 1: original model, strong fine-tune with the original head, strong fine-tune with head + adapter.
""")
code("""!md5sum prompts/general_eval.jsonl
!ls adapters/head_lora
!python train_target_lora.py --max_examples 2000 --epochs 3 --rank 64 --lr 3e-4 --adapter_out /kaggle/working/target_lora_strong --merged_out /kaggle/working/target_merged_strong --log results/smoke/remerge_log.csv
""")
code("""!python bench.py --prompts prompts/general_eval.jsonl --out results/stage3b_gen_original.csv
!python bench.py --base /kaggle/working/target_merged_strong --prompts prompts/general_eval.jsonl --out results/stage3b_gen_strong_orighead.csv
!python bench.py --base /kaggle/working/target_merged_strong --adapter adapters/head_lora --prompts prompts/general_eval.jsonl --out results/stage3b_gen_strong_fixed.csv
!python summarize.py "original model (general)=results/stage3b_gen_original.csv" "strong LoRA, original head (general)=results/stage3b_gen_strong_orighead.csv" "strong LoRA, head + adapter (general)=results/stage3b_gen_strong_fixed.csv" --out results/summary_stage3b.csv
""")
md("**Stop here.** Send Claude the table.")

stage = "4"
md("## Stage 4 - Paired statistics and chart (no GPU needed; also runs locally on the results CSVs)")
code("""!python paired.py --out results/paired_stats.csv "medical: adapter vs no adapter=results/stage3_med_strong_orighead.csv,results/stage3_med_strong_fixed.csv" "general: adapter vs no adapter=results/stage3b_gen_strong_orighead.csv,results/stage3b_gen_strong_fixed.csv"
!python plot.py --medical results/summary.csv --general results/summary_stage3b.csv --out results/chart.png
from IPython.display import Image
Image("results/chart.png")
""")

for c in cells:  # notebook format stores source as a list of lines
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
with open(args.out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {args.out} ({len(cells)} cells)")
