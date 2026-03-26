# cs593-tml — Membership Inference Attack Experiments

Short description
- This repository contains code and data for membership inference attack (MIA) experiments used in the CS593 project.

Quick project overview
- Data: `data/` holds training/eval splits and evaluation logs.
- Models: `models/` stores trained model artifacts.
- Env: `mia_env/` is a Python virtual environment for reproducible runs.
- Results / logs: `results/`, `log.txt`, and `training_log.txt`.

Quick start
1. Activate the provided virtualenv: `source mia_env/bin/activate`
2. Prepare data: `python prepare_data.py`
3. Run baseline training: `python train_baseline.py`
4. Run MIA: `python run_mia.py`

Notes
- This README is an initial project comment. Update it with experiment details, requirements, and reproduction steps.
