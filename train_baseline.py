"""
STEP 3: Train Baseline (Full Fine-Tuning on GPT-2 Medium)
==========================================================
Fine-tunes GPT-2 Medium on the PubMed training set.
This is the "full fine-tuning" baseline — all weights updated.
We'll compare LoRA variants against this in the final project.

Estimated time: ~30–60 min on a single GPU (varies by VRAM)
GPU memory needed: ~6 GB (GPT-2 Medium is 355M params)

Usage:
    python train_baseline.py

Output (in ./models/full_ft/):
    pytorch_model.bin / model.safetensors  - model weights
    config.json                            - model config
    tokenizer files                        - for inference
    training_log.txt                       - loss per epoch
"""

import argparse
import json
import math
import os
import subprocess
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
MODEL_NAME   = "gpt2-medium"    # 355M parameter GPT-2 variant
DATA_PATH    = "./data/train.txt"
OUTPUT_ROOT  = "./models"

# Training hyperparameters
MAX_LENGTH   = 256      # token length per sample (shorter = faster training)
BATCH_SIZE   = 4        # reduce to 2 if you get OOM errors
GRAD_ACCUM   = 4        # effective batch size = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS       = 3        # 3 epochs is sufficient for a meaningful baseline
LR           = 2e-5     # standard LR for GPT-2 fine-tuning
WARMUP_RATIO = 0.1      # 10% of steps used for LR warmup
SEED         = 42
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 with full FT, LoRA, or Prefix tuning")
    parser.add_argument("--method", choices=["full_ft", "lora", "prefix"], default="full_ft")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--prefix-virtual-tokens", type=int, default=20)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()

def resolve_output_dir(args):
    if args.output_dir:
        return args.output_dir
    if args.method == "full_ft":
        return os.path.join(OUTPUT_ROOT, "full_ft")
    if args.method == "lora":
        return os.path.join(OUTPUT_ROOT, f"lora_r{args.lora_rank}")
    return os.path.join(OUTPUT_ROOT, f"prefix_v{args.prefix_virtual_tokens}")

args = parse_args()
output_dir = resolve_output_dir(args)
log_path = os.path.join(output_dir, "training_log.txt")
meta_path = os.path.join(output_dir, "run_metadata.json")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except Exception:
        return "unknown"

# ── Dataset class ─────────────────────────────────────────────
class TextDataset(Dataset):
    """Simple line-by-line text dataset with tokenization."""

    def __init__(self, path, tokenizer, max_length):
        print(f"Loading dataset from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        print(f"  Tokenizing {len(lines):,} samples (max_length={max_length})...")
        self.examples = []
        for line in tqdm(lines, desc="  Tokenizing"):
            enc = tokenizer(
                line,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze()
            # For causal LM, labels = input_ids (next-token prediction)
            self.examples.append({
                "input_ids": input_ids,
                "labels":    input_ids.clone(),
            })

        print(f"  Dataset size: {len(self.examples):,} samples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ── Load tokenizer & model ────────────────────────────────────
print("\n" + "=" * 50)
print(f" Loading {args.model_name}...")
print("=" * 50)
print("(First run will download ~1.5 GB from HuggingFace)")

tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
# GPT-2 has no pad token by default — use EOS as pad
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(args.model_name)
model.config.pad_token_id = tokenizer.eos_token_id

if args.method == "lora":
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
elif args.method == "prefix":
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.prefix_virtual_tokens,
    )
    model = get_peft_model(model, prefix_config)

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_pct = 100.0 * trainable_params / total_params
print(f"\nModel parameters   : {total_params / 1e6:.1f}M total")
print(f"Trainable params   : {trainable_params / 1e6:.3f}M ({trainable_pct:.3f}%)")
print(f"Method             : {args.method}")
if args.method == "lora":
    print(f"LoRA rank          : {args.lora_rank}")
if args.method == "prefix":
    print(f"Prefix tokens      : {args.prefix_virtual_tokens}")

# ── Build dataset & dataloader ────────────────────────────────
dataset    = TextDataset(args.data_path, tokenizer, args.max_length)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,      # keep 0 for WSL compatibility
    pin_memory=True if device.type == "cuda" else False,
)

# ── Optimizer & scheduler ─────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

updates_per_epoch = max(1, math.ceil(len(dataloader) / args.grad_accum))
total_steps   = updates_per_epoch * args.epochs
warmup_steps  = int(total_steps * args.warmup_ratio)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"\nTraining plan:")
print(f"  Epochs          : {args.epochs}")
print(f"  Batches/epoch   : {len(dataloader)}")
print(f"  Effective batch : {args.batch_size * args.grad_accum}")
print(f"  Total steps     : {total_steps}")
print(f"  Warmup steps    : {warmup_steps}")

# ── Training loop ─────────────────────────────────────────────
print("\n" + "=" * 50)
print(" Training...")
print("=" * 50)

log_lines = ["epoch,step,loss,elapsed_min\n"]
start_time = time.time()

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss  = 0.0
    step_count  = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss    = outputs.loss / args.grad_accum   # normalize for gradient accumulation
        loss.backward()

        epoch_loss += outputs.loss.item()
        step_count += 1

        # Gradient accumulation: step every grad_accum batches and on final remainder.
        if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update progress bar
        avg_loss = epoch_loss / step_count
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # End-of-epoch logging
    avg_epoch_loss = epoch_loss / step_count
    elapsed = (time.time() - start_time) / 60
    print(f"\n  Epoch {epoch} complete | avg loss: {avg_epoch_loss:.4f} | elapsed: {elapsed:.1f} min")
    log_lines.append(f"{epoch},{step_count},{avg_epoch_loss:.6f},{elapsed:.2f}\n")

# ── Save model ────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f" Saving model to {output_dir}/")
print("=" * 50)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

with open(log_path, "w", encoding="utf-8") as f:
    f.writelines(log_lines)

total_time = (time.time() - start_time) / 60
print(f"\nTraining complete in {total_time:.1f} minutes.")
print(f"Model saved to    : {output_dir}/")
print(f"Training log      : {log_path}")
print("\nNext: run run_mia.py")