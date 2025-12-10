#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=arcus.p
#SBATCH --nodes=1                # ensure single node
#SBATCH --ntasks=1               # one process using multiple GPUs
#SBATCH --gres=gpu:2             # <<< request 4 GPUs on this node
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=8        # maybe bump a bit for dataloader (optional)
#SBATCH --mem=300G
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
    for i in range(torch.cuda.device_count()):
        print(f"device {i}: {torch.cuda.get_device_name(i)}")
PY

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# ---- run training ----
cd /baldig/bioprojects2/uxue/VAE-mri

$PYTHON conditional_ddpm/test/evaluation_for_corrected_3_age.py
