# CS593-TML — Does Parameter-Efficient Fine-Tuning Protect Privacy?

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
4. Run training (examples):
	- Full FT baseline: `python train_baseline.py --method full_ft`
	- LoRA: `python train_baseline.py --method lora --lora-rank` 
	- Prefix tuning: `python train_baseline.py --method prefix --prefix-virtual-tokens 20`
5. Run MIA across all discovered models: `python run_mia.py`
6. Or run MIA for specific models:
	- `python run_mia.py --model-dirs ./models/full_ft ./models/lora_r4 ./models/lora_r16 ./models/lora_r64 ./models/prefix_v20`
