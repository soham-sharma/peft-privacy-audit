#!/bin/bash

set -e  # Exit immediately on error

echo "========================================"
echo " MIA Project - Environment Setup"
echo "========================================"

# --- 1. Create and activate a virtual environment ---
echo "[1/5] Creating virtual environment..."
python3 -m venv mia_env
source mia_env/bin/activate

# --- 2. Upgrade pip ---
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# --- 3. Install PyTorch with CUDA 12.1 support ---
echo "[3/5] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- 4. Install HuggingFace + PEFT + data libraries ---
echo "[4/5] Installing HuggingFace stack..."
pip install \
    transformers==4.40.0 \
    peft==0.10.0 \
    datasets==2.19.0 \
    accelerate==0.29.0 \
    evaluate==0.4.1 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4 \
    numpy==1.26.4 \
    tqdm \
    pandas

# --- 5. Verify GPU is accessible ---
echo "[5/5] Verifying GPU access..."
python3 -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not found. Check your PyTorch install.')
"

echo ""
echo "========================================"
echo " Setup complete"
echo "========================================"