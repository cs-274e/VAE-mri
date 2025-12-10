#!/bin/bash
#SBATCH --job-name=eval_age
#SBATCH --partition=arcus.p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=100:00:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=udelaqui@uci.edu

echo "=== Job started on $(hostname) at $(date)"

mkdir -p logs

PYTHON="/home/udelaqui/.conda/envs/denoise116/bin/python"   # <-- adjust if `which python` says different

echo "Using python: $PYTHON"

# ---- CUDA / env diagnostics ----
$PYTHON - << 'PY'
import sys, torch
print("sys.executable:", sys.executable)
print("torch:", torch.__version__, "torch.version.cuda:", torch.version.cuda)
print("cuda.is_available():", torch.cuda.is_available())
print("cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device idx:", torch.cuda.current_device())
    print("current device name:", torch.cuda.get_device_name(0))
PY

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# ---- run training ----
cd /baldig/bioprojects2/uxue/VAE-mri

# $PYTHON conditional_ddpm/train/train_conditional_ddpm_3_age_resume.py
$PYTHON conditional_ddpm/test/test_trajectories.py

echo "=== Job finished at $(date)"
