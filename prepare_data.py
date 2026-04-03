"""
Downloads a subset of PubMed abstracts from HuggingFace datasets,
splits into train / member / non-member sets for MIA, and saves to disk.

Usage:
    python prepare_data.py

Output files (in ./data/):
    train.txt           - text used to fine-tune the model (the "members")
    member_eval.txt     - held-out sample of training texts  (MIA positives)
    nonmember_eval.txt  - texts the model never saw          (MIA negatives)
    data_stats.txt      - summary of dataset sizes
"""

import os
import random
from datasets import load_dataset
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
SEED             = 42
TRAIN_SIZE       = 2000   # samples used to fine-tune
MEMBER_EVAL      = 500    # subset of train set for MIA evaluation
NONMEMBER_EVAL   = 500    # held-out samples never seen by model
MIN_LENGTH       = 100    # minimum characters per abstract
MAX_LENGTH       = 1024   # truncate anything longer than this
OUTPUT_DIR       = "./data"
# ─────────────────────────────────────────────────────────────

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print(" Step 2: Preparing PubMed Dataset")
print("=" * 50)

# ── 1. Load PubMed abstracts ──────────────────────────────────
print("\n[1/4] Downloading PubMed abstracts from HuggingFace...")
print("      (This may take a few minutes on first run)")

# pubmed_qa has clean, real medical abstracts - perfect for our use case
dataset = load_dataset(
    "pubmed_qa",
    "pqa_unlabeled",   # the large unlabeled split - just abstracts, no Q&A labels
    split="train",
    trust_remote_code=True
)

print(f"      Raw dataset size: {len(dataset):,} samples")

# ── 2. Extract and clean abstracts ───────────────────────────
print("\n[2/4] Extracting and filtering abstracts...")

abstracts = []
for item in tqdm(dataset, desc="Processing"):
    # pubmed_qa stores context as a dict with sentences
    ctx = item.get("context", {})
    if isinstance(ctx, dict):
        sentences = ctx.get("contexts", [])
        text = " ".join(sentences) if sentences else ""
    else:
        text = str(ctx)

    text = text.strip()

    # Filter: must be substantive text
    if len(text) < MIN_LENGTH:
        continue

    # Truncate very long abstracts
    text = text[:MAX_LENGTH]
    abstracts.append(text)

print(f"      After filtering: {len(abstracts):,} abstracts")

# Shuffle before splitting
random.shuffle(abstracts)

# ── 3. Split into train / member_eval / nonmember_eval ───────
print("\n[3/4] Splitting dataset...")

total_needed = TRAIN_SIZE + NONMEMBER_EVAL
if len(abstracts) < total_needed:
    raise ValueError(
        f"Not enough abstracts ({len(abstracts)}) for requested sizes. "
        f"Lower TRAIN_SIZE or NONMEMBER_EVAL."
    )

# Training set (these become "members")
train_texts      = abstracts[:TRAIN_SIZE]

# Member eval: a random subset of the training texts
# These are texts the model HAS seen → MIA should label as "member"
member_indices   = random.sample(range(TRAIN_SIZE), MEMBER_EVAL)
member_texts     = [train_texts[i] for i in member_indices]

# Non-member eval: texts held out entirely
# These are texts the model has NOT seen → MIA should label as "non-member"
nonmember_texts  = abstracts[TRAIN_SIZE : TRAIN_SIZE + NONMEMBER_EVAL]

print(f"      Train set       : {len(train_texts):,} samples")
print(f"      Member eval     : {len(member_texts):,} samples")
print(f"      Non-member eval : {len(nonmember_texts):,} samples")

# ── 4. Save to disk ───────────────────────────────────────────
print("\n[4/4] Saving to disk...")

def save_texts(texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            # Use a clear separator between samples
            f.write(text.replace("\n", " ") + "\n")

save_texts(train_texts,     os.path.join(OUTPUT_DIR, "train.txt"))
save_texts(member_texts,    os.path.join(OUTPUT_DIR, "member_eval.txt"))
save_texts(nonmember_texts, os.path.join(OUTPUT_DIR, "nonmember_eval.txt"))

# Save a stats summary
stats = f"""Dataset Statistics
==================
Train (fine-tuning) : {len(train_texts):,} samples
Member eval (MIA+)  : {len(member_texts):,} samples
Non-member eval (MIA-): {len(nonmember_texts):,} samples

Avg abstract length (train): {sum(len(t) for t in train_texts) / len(train_texts):.0f} chars
Min length filter applied   : {MIN_LENGTH} chars
Max length cap applied      : {MAX_LENGTH} chars
Random seed                 : {SEED}
"""
with open(os.path.join(OUTPUT_DIR, "data_stats.txt"), "w") as f:
    f.write(stats)

print(stats)
print(f"Files saved to: {OUTPUT_DIR}/")