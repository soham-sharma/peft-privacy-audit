# cs593-TML — Does Parameter-Efficient Fine-Tuning Protect Privacy?

- This repository contains code and data for membership inference attack (MIA) experiments used in the CS593 project.

Quick project overview
- Data: `data/` holds training/eval splits and evaluation logs.
- Models: `models/` stores trained model artifacts.
- Env: `mia_env/` is a Python virtual environment for reproducible runs.
- Results: `results/`.

Quick start
1. Setup the environment `bash setup.sh`
2. Activate the virtualenv: `source mia_env/bin/activate`
3. Prepare data: `python prepare_data.py`
4. Run baseline training: `python train_baseline.py`
5. Run MIA: `python run_mia.py`
