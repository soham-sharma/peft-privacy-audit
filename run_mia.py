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
from peft import PeftConfig, PeftModel
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
MEMBER_PATH     = "./data/member_eval.txt"
NONMEMBER_PATH  = "./data/nonmember_eval.txt"
OUTPUT_DIR      = "./results"
MAX_LENGTH      = 256
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run loss-threshold MIA across one or more models")
    parser.add_argument("--model-dirs", nargs="+", default=None,
                        help="Model directories to evaluate. If omitted, auto-discovers under ./models")
    parser.add_argument("--member-path", type=str, default=MEMBER_PATH)
    parser.add_argument("--nonmember-path", type=str, default=NONMEMBER_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--no-model-progress", action="store_true",
                        help="Disable top-level progress bar over models")
    parser.add_argument("--include-untrained-baseline", action="store_true",
                        help="Include an unfine-tuned baseline model in evaluation")
    parser.add_argument("--baseline-model-name", type=str, default="gpt2-medium",
                        help="HuggingFace model ID for the untrained baseline")
    return parser.parse_args()


def discover_model_dirs(root="./models"):
    dirs = []
    if not os.path.isdir(root):
        return dirs
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        has_full_model = os.path.exists(os.path.join(path, "config.json"))
        has_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
        if has_full_model or has_adapter:
            dirs.append(path)
    return dirs


def is_hf_reference(model_ref):
    return model_ref.startswith("hf://")


def model_ref_to_name(model_ref):
    if is_hf_reference(model_ref):
        base = model_ref.split("hf://", 1)[1]
        return f"untrained_{base.replace('-', '_').replace('/', '_')}"
    return os.path.basename(model_ref.rstrip("/"))


def load_model_and_tokenizer(model_ref, device):
    if is_hf_reference(model_ref):
        base_model_name = model_ref.split("hf://", 1)[1]
        tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(base_model_name)
        model.config.pad_token_id = tokenizer.eos_token_id
        model.eval()
        model.to(device)
        return model, tokenizer

    model_dir = model_ref
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        peft_config = PeftConfig.from_pretrained(model_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2LMHeadModel.from_pretrained(peft_config.base_model_name_or_path)
        base_model.config.pad_token_id = tokenizer.eos_token_id
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    model.to(device)
    return model, tokenizer

# ── Compute per-sample loss ───────────────────────────────────
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


def evaluate_model(model_ref, args, device):
    print(f"\nEvaluating model: {model_ref}")
    print("[1/3] Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_ref, device)
    print("  Model loaded.")

    print("[2/3] Computing per-sample losses...")
    member_losses = compute_losses(model, tokenizer, args.member_path, "members", args.max_length, device)
    nonmember_losses = compute_losses(model, tokenizer, args.nonmember_path, "non-members", args.max_length, device)

    if len(member_losses) == 0 or len(nonmember_losses) == 0:
        raise ValueError(
            f"Empty loss arrays for {model_ref}. "
            "Check data files and tokenizer settings."
        )
    if len(member_losses) != len(nonmember_losses):
        raise ValueError(
            f"Sample count mismatch for {model_ref}: "
            f"members={len(member_losses)}, non-members={len(nonmember_losses)}. "
            "Use balanced eval splits for fair MIA comparison."
        )

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
        "model_dir": model_ref,
        "model_name": model_ref_to_name(model_ref),
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
    ax.set_ylim(0.0, 1.05)
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dirs = args.model_dirs if args.model_dirs else discover_model_dirs("./models")
    model_dirs = [d for d in model_dirs if os.path.isdir(d) or is_hf_reference(d)]
    if args.include_untrained_baseline:
        baseline_ref = f"hf://{args.baseline_model_name}"
        if baseline_ref not in model_dirs:
            model_dirs.append(baseline_ref)
    if not model_dirs:
        raise ValueError("No model directories found. Train at least one model before running MIA.")
    if not os.path.exists(args.member_path):
        raise FileNotFoundError(f"Member file not found: {args.member_path}")
    if not os.path.exists(args.nonmember_path):
        raise FileNotFoundError(f"Non-member file not found: {args.nonmember_path}")

    print(f"\nDiscovered {len(model_dirs)} model(s) for evaluation.")
    model_iter = model_dirs
    if not args.no_model_progress:
        model_iter = tqdm(model_dirs, desc="MIA models")
    results = [evaluate_model(model_dir, args, device) for model_dir in model_iter]
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