"""
Runs the loss-threshold MIA against one or more trained models.

Usage:
    python run_mia.py
    python run_mia.py --model-dirs ./models/full_ft ./models/lora_r4 ./models/lora_r16

Outputs (in ./results/):
    roc_curve.png              - comparative ROC curve plot
    loss_distribution.png      - histogram for first evaluated model
    mia_results.txt            - legacy summary for first model
    mia_results_all.json       - per-model structured metrics
    mia_results_summary.csv    - per-model tabular metrics
"""

import argparse
import csv
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works in WSL)
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# ── Utility Functions ─────────────────────────────────────────
def compute_losses(model, tokenizer, text_path, label, max_length, device):
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

            if input_ids.shape[1] < 5:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            losses.append(outputs.loss.item())

    return np.array(losses)


def tpr_at_fpr(fpr_arr, tpr_arr, target_fpr):
    mask = fpr_arr <= target_fpr
    if not mask.any():
        return 0.0
    return tpr_arr[mask][-1]

# ── Core Evaluation Logic ─────────────────────────────────────
def evaluate_model(model_dir, member_path, nonmember_path, max_length, device):
    print(f"\nEvaluating model: {model_dir}")
    print("[1/3] Loading model...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    print("  Model loaded.")

    print("[2/3] Computing per-sample losses...")
    member_losses = compute_losses(model, tokenizer, member_path, "members", max_length, device)
    nonmember_losses = compute_losses(model, tokenizer, nonmember_path, "non-members", max_length, device)

    if len(member_losses) == 0 or len(nonmember_losses) == 0:
        raise ValueError(f"Empty loss arrays for {model_dir}. Check data files and tokenizer settings.")
    if len(member_losses) != len(nonmember_losses):
        raise ValueError(f"Sample count mismatch for {model_dir}. Use balanced eval splits for fair MIA comparison.")

    print(f"  Member losses     — mean: {member_losses.mean():.4f}, std: {member_losses.std():.4f}")
    print(f"  Non-member losses — mean: {nonmember_losses.mean():.4f}, std: {nonmember_losses.std():.4f}")

    print("[3/3] Computing MIA metrics...")
    y_true = np.concatenate([
        np.ones(len(member_losses)),
        np.zeros(len(nonmember_losses)),
    ])
    y_score = np.concatenate([
        -member_losses,
        -nonmember_losses,
    ])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    tpr_at_1 = tpr_at_fpr(fpr, tpr, 0.01)
    tpr_at_5 = tpr_at_fpr(fpr, tpr, 0.05)
    tpr_at_10 = tpr_at_fpr(fpr, tpr, 0.10)

    return {
        "model_dir": model_dir,
        "model_name": os.path.basename(model_dir.rstrip("/")),
        "metrics": {
            "auc_roc": float(roc_auc),
            "tpr_at_1_fpr": float(tpr_at_1),
            "tpr_at_5_fpr": float(tpr_at_5),
            "tpr_at_10_fpr": float(tpr_at_10),
            "member_count": int(len(member_losses)),
            "nonmember_count": int(len(nonmember_losses)),
            "member_loss_mean": float(member_losses.mean()),
            "member_loss_std": float(member_losses.std()),
            "member_loss_min": float(member_losses.min()),
            "member_loss_max": float(member_losses.max()),
            "nonmember_loss_mean": float(nonmember_losses.mean()),
            "nonmember_loss_std": float(nonmember_losses.std()),
            "nonmember_loss_min": float(nonmember_losses.min()),
            "nonmember_loss_max": float(nonmember_losses.max()),
            "loss_gap": float(nonmember_losses.mean() - member_losses.mean()),
        },
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "member_losses": member_losses.tolist(),
        "nonmember_losses": nonmember_losses.tolist(),
    }

# ── Reporting and Plotting ────────────────────────────────────
def save_csv_summary(output_path, results):
    rows = []
    for item in results:
        m = item["metrics"]
        rows.append({
            "model_name": item["model_name"],
            "model_dir": item["model_dir"],
            "auc_roc": m["auc_roc"],
            "tpr_at_1_fpr": m["tpr_at_1_fpr"],
            "tpr_at_5_fpr": m["tpr_at_5_fpr"],
            "tpr_at_10_fpr": m["tpr_at_10_fpr"],
            "member_count": m["member_count"],
            "nonmember_count": m["nonmember_count"],
            "member_loss_mean": m["member_loss_mean"],
            "nonmember_loss_mean": m["nonmember_loss_mean"],
            "loss_gap": m["loss_gap"],
        })
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_legacy_summary(output_path, first_result):
    m = first_result["metrics"]
    results_text = f"""MIA Results — {first_result['model_name']}
========================================
Model            : {first_result['model_name']}
Model directory  : {first_result['model_dir']}
Members eval     : {m['member_count']} samples
Non-members eval : {m['nonmember_count']} samples
Attack           : Loss-Threshold (Yeom et al. 2018)

Results:
  AUC-ROC          : {m['auc_roc']:.4f}
  TPR @ 1%  FPR   : {m['tpr_at_1_fpr']:.4f}
  TPR @ 5%  FPR   : {m['tpr_at_5_fpr']:.4f}
  TPR @ 10% FPR   : {m['tpr_at_10_fpr']:.4f}
  Random baseline  : AUC = 0.5000

Member loss stats:
  Mean : {m['member_loss_mean']:.4f}
  Std  : {m['member_loss_std']:.4f}
  Min  : {m['member_loss_min']:.4f}
  Max  : {m['member_loss_max']:.4f}

Non-member loss stats:
  Mean : {m['nonmember_loss_mean']:.4f}
  Std  : {m['nonmember_loss_std']:.4f}
  Min  : {m['nonmember_loss_min']:.4f}
  Max  : {m['nonmember_loss_max']:.4f}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(results_text)

def plot_comparative_roc(output_path, results):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))

    for item in results:
        fpr = np.array(item["roc"]["fpr"])
        tpr = np.array(item["roc"]["tpr"])
        ax.plot(fpr, tpr, lw=2.2,
                label=f"{item['model_name']} (AUC={item['metrics']['auc_roc']:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random (AUC=0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
    ax.set_title("Comparative ROC — Loss-Threshold MIA", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_first_loss_distribution(output_path, first_result):
    member_losses = np.array(first_result["member_losses"])
    nonmember_losses = np.array(first_result["nonmember_losses"])

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(member_losses.min(), nonmember_losses.min()),
        max(member_losses.max(), nonmember_losses.max()),
        60,
    )
    ax.hist(member_losses, bins=bins, alpha=0.65, color="steelblue",
            label=f"Members (n={len(member_losses)})", density=True)
    ax.hist(nonmember_losses, bins=bins, alpha=0.65, color="salmon",
            label=f"Non-members (n={len(nonmember_losses)})", density=True)
    ax.axvline(member_losses.mean(), color="steelblue", linestyle="--", lw=2,
               label=f"Member mean = {member_losses.mean():.3f}")
    ax.axvline(nonmember_losses.mean(), color="salmon", linestyle="--", lw=2,
               label=f"Non-member mean = {nonmember_losses.mean():.3f}")

    ax.set_xlabel("Per-Sample Cross-Entropy Loss", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Loss Distribution: Members vs. Non-Members\n{first_result['model_name']}", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def print_console_summary(results):
    print("\n┌──────────────────────────────────────────────────────────────┐")
    print("│                 MIA Comparative Summary                      │")
    print("├──────────────────────────────────────────────────────────────┤")
    for item in results:
        m = item["metrics"]
        print(f"│ {item['model_name']:<16} AUC={m['auc_roc']:.4f}  TPR@1%={m['tpr_at_1_fpr']:.4f}          │")
    print("└──────────────────────────────────────────────────────────────┘")

def parse_args():
    parser = argparse.ArgumentParser(description="Run loss-threshold MIA across one or more models")
    parser.add_argument("--model-dirs", nargs="+", default=["./models/full_ft"],
                        help="Model directories to evaluate.")
    parser.add_argument("--member-path", type=str, default="./data/member_eval.txt")
    parser.add_argument("--nonmember-path", type=str, default="./data/nonmember_eval.txt")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-model-progress", action="store_true",
                        help="Disable top-level progress bar over models")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dirs = args.model_dirs
    print(f"\nDiscovered {len(model_dirs)} model(s) for evaluation.")
    
    model_iter = model_dirs
    if not args.no_model_progress:
        model_iter = tqdm(model_dirs, desc="MIA models")
        
    results = [evaluate_model(m_dir, args.member_path, args.nonmember_path, args.max_length, device) for m_dir in model_iter]
    print_console_summary(results)

    # Save structured outputs
    json_path = os.path.join(args.output_dir, "mia_results_all.json")
    csv_path = os.path.join(args.output_dir, "mia_results_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    save_csv_summary(csv_path, results)

    # Backward compatible single-file summary and plots based on first model.
    first = results[0]
    legacy_txt_path = os.path.join(args.output_dir, "mia_results.txt")
    write_legacy_summary(legacy_txt_path, first)

    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    dist_path = os.path.join(args.output_dir, "loss_distribution.png")
    plot_comparative_roc(roc_path, results)
    plot_first_loss_distribution(dist_path, first)

    print(f"\nSaved structured results: {json_path}")
    print(f"Saved summary CSV       : {csv_path}")
    print(f"Saved legacy summary    : {legacy_txt_path}")
    print(f"Saved comparative ROC   : {roc_path}")
    print(f"Saved loss distribution : {dist_path}")

if __name__ == "__main__":
    main()