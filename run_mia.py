"""
STEP 4: Membership Inference Attack (Loss-Threshold)
=====================================================
Runs the loss-threshold MIA against the fine-tuned model.

Core idea:
  - Fine-tuned models have LOWER loss on training samples (members)
    than on unseen samples (non-members).
  - The attack predicts "member" if loss(model, x) < threshold τ.
  - We sweep τ to build a full ROC curve.

Usage:
    python run_mia.py

Outputs (in ./results/):
    roc_curve.png          - ROC curve plot (ready for slides)
    loss_distribution.png  - loss histogram (members vs non-members)
    mia_results.txt        - AUC, TPR@1%FPR, TPR@5%FPR numbers
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works in WSL)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
MODEL_DIR       = "./models/full_ft"
MEMBER_PATH     = "./data/member_eval.txt"
NONMEMBER_PATH  = "./data/nonmember_eval.txt"
OUTPUT_DIR      = "./results"
MAX_LENGTH      = 256
BATCH_SIZE      = 8
# ─────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load model ────────────────────────────────────────────────
print("\n[1/4] Loading fine-tuned model...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()
model.to(device)
print("  Model loaded.")

# ── Utility Functions ─────────────────────────────────────────
def compute_losses(model, tokenizer, text_path, label, max_length, device):
    """
    For each text sample, compute the average per-token cross-entropy loss.
    Lower loss = model is more 'familiar' with the text = likely member.
    """
    with open(text_path, "r", encoding="utf-8") as f:
        texts = [l.strip() for l in f if l.strip()]

    losses = []
    with torch.no_grad():
        for text in tqdm(texts, desc=f"  Computing losses ({label})"):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)

            # Skip very short sequences (< 5 tokens)
            if input_ids.shape[1] < 5:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            # outputs.loss is the mean per-token cross-entropy loss
            losses.append(outputs.loss.item())

    return np.array(losses)


def tpr_at_fpr(fpr_arr, tpr_arr, target_fpr):
    """Find the TPR at the closest FPR ≤ target_fpr."""
    mask = fpr_arr <= target_fpr
    if not mask.any():
        return 0.0
    return tpr_arr[mask][-1]

# ── Compute per-sample loss ───────────────────────────────────
print("\n[2/4] Computing per-sample losses...")
member_losses     = compute_losses(model, tokenizer, MEMBER_PATH, "members", MAX_LENGTH, device)
nonmember_losses  = compute_losses(model, tokenizer, NONMEMBER_PATH, "non-members", MAX_LENGTH, device)

print(f"\n  Member losses     — mean: {member_losses.mean():.4f}, std: {member_losses.std():.4f}")
print(f"  Non-member losses — mean: {nonmember_losses.mean():.4f}, std: {nonmember_losses.std():.4f}")

# ── Compute MIA metrics ───────────────────────────────────────
print("\n[3/4] Computing MIA metrics...")

# Ground truth labels: 1 = member, 0 = non-member
y_true  = np.concatenate([
    np.ones(len(member_losses)),
    np.zeros(len(nonmember_losses))
])

# Attack score: NEGATIVE loss (higher score → more likely member)
# We negate because lower loss = more likely member
y_score = np.concatenate([
    -member_losses,
    -nonmember_losses
])

# ROC curve & AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# TPR at specific low FPR thresholds (key privacy metrics)
tpr_at_1  = tpr_at_fpr(fpr, tpr, 0.01)
tpr_at_5  = tpr_at_fpr(fpr, tpr, 0.05)
tpr_at_10 = tpr_at_fpr(fpr, tpr, 0.10)

print(f"\n  ┌─────────────────────────────────────┐")
print(f"  │   MIA Results (Full Fine-Tuning)    │")
print(f"  ├─────────────────────────────────────┤")
print(f"  │  AUC-ROC         : {roc_auc:.4f}          │")
print(f"  │  TPR @ 1%  FPR   : {tpr_at_1:.4f}          │")
print(f"  │  TPR @ 5%  FPR   : {tpr_at_5:.4f}          │")
print(f"  │  TPR @ 10% FPR   : {tpr_at_10:.4f}          │")
print(f"  │  Random baseline  : AUC = 0.5000         │")
print(f"  └─────────────────────────────────────┘")

# Save text results
results_text = f"""MIA Results — Full Fine-Tuning Baseline
========================================
Model            : GPT-2 Medium (Full Fine-Tuning)
Dataset          : PubMed Abstracts
Members eval     : {len(member_losses)} samples
Non-members eval : {len(nonmember_losses)} samples
Attack           : Loss-Threshold (Yeom et al. 2018)

Results:
  AUC-ROC          : {roc_auc:.4f}
  TPR @ 1%  FPR   : {tpr_at_1:.4f}
  TPR @ 5%  FPR   : {tpr_at_5:.4f}
  TPR @ 10% FPR   : {tpr_at_10:.4f}
  Random baseline  : AUC = 0.5000

Member loss stats:
  Mean : {member_losses.mean():.4f}
  Std  : {member_losses.std():.4f}
  Min  : {member_losses.min():.4f}
  Max  : {member_losses.max():.4f}

Non-member loss stats:
  Mean : {nonmember_losses.mean():.4f}
  Std  : {nonmember_losses.std():.4f}
  Min  : {nonmember_losses.min():.4f}
  Max  : {nonmember_losses.max():.4f}
"""
with open(os.path.join(OUTPUT_DIR, "mia_results.txt"), "w") as f:
    f.write(results_text)

# ── Generate plots ────────────────────────────────────────────
print("\n[4/4] Generating plots...")

plt.style.use("seaborn-v0_8-whitegrid")
FIG_DPI = 150

# ── Plot 1: ROC Curve ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(fpr, tpr,
        color="steelblue", lw=2.5,
        label=f"Full Fine-Tuning (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1],
        color="gray", lw=1.5, linestyle="--",
        label="Random Classifier (AUC = 0.500)")

# Mark TPR @ 1% FPR on the curve
ax.scatter([0.01], [tpr_at_1], color="red", zorder=5, s=80)
ax.annotate(
    f"TPR={tpr_at_1:.3f}\n@ FPR=1%",
    xy=(0.01, tpr_at_1),
    xytext=(0.08, tpr_at_1 - 0.08),
    fontsize=9,
    color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2)
)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
ax.set_title("ROC Curve — Loss-Threshold MIA\nGPT-2 Medium (Full Fine-Tuning) on PubMed", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.tick_params(labelsize=11)

plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=FIG_DPI)
plt.close()
print(f"  Saved: {roc_path}")

# ── Plot 2: Loss Distribution ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

bins = np.linspace(
    min(member_losses.min(), nonmember_losses.min()),
    max(member_losses.max(), nonmember_losses.max()),
    60
)

ax.hist(member_losses,    bins=bins, alpha=0.65, color="steelblue",
        label=f"Members (n={len(member_losses)})", density=True)
ax.hist(nonmember_losses, bins=bins, alpha=0.65, color="salmon",
        label=f"Non-members (n={len(nonmember_losses)})", density=True)

ax.axvline(member_losses.mean(),    color="steelblue", linestyle="--", lw=2,
           label=f"Member mean = {member_losses.mean():.3f}")
ax.axvline(nonmember_losses.mean(), color="salmon",    linestyle="--", lw=2,
           label=f"Non-member mean = {nonmember_losses.mean():.3f}")

ax.set_xlabel("Per-Sample Cross-Entropy Loss", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Loss Distribution: Members vs. Non-Members\nGPT-2 Medium (Full Fine-Tuning) on PubMed", fontsize=13)
ax.legend(fontsize=10)
ax.tick_params(labelsize=11)

plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "loss_distribution.png")
plt.savefig(dist_path, dpi=FIG_DPI)
plt.close()
print(f"  Saved: {dist_path}")

# ── Done ──────────────────────────────────────────────────────
print(f"""
================================================
 All done! Results saved to: {OUTPUT_DIR}/

   roc_curve.png         → use this in your slides
   loss_distribution.png → use this in your slides
   mia_results.txt       → your key numbers

 Key numbers to report in mid-term:
   AUC-ROC        : {roc_auc:.4f}
   TPR @ 1% FPR  : {tpr_at_1:.4f}
   (Random baseline AUC = 0.5000)
================================================
""")