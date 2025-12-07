#!/usr/bin/env python3
"""
Evaluation script for conditional DDPM model on MRI T1->T2 translation.

This script:
1. Runs conditional sampling T2 | T1 on ALL slices in the test set
2. Computes SSIM, PSNR, MSE comparing generated T2 to ground truth T2
3. Saves metrics to a text file and optionally per-slice metrics

No argparse; configuration is at the top of the file.
"""

import os
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# =====================================================
# IMPORT YOUR TRAINING COMPONENTS
#   Change "ddpm_train" to the actual filename of your
#   training script (without .py) if it differs.
# =====================================================
THIS_DIR = Path(__file__).resolve().parent         # .../conditional_ddpm/test
PROJECT_ROOT = THIS_DIR.parent                     # .../conditional_ddpm
sys.path.append(str(PROJECT_ROOT))

from train.train_conditional_ddpm_3 import (
    PairedNiftiSliceDataset,
    DiffusionConfig,
    UNetConditional,
)

# =========================
# CONFIG (edit these)
# =========================

TEST_ROOT = "dataset/test"

# path to the trained checkpoint from your training script
CKPT_PATH = "conditional_ddpm/train/best_models/ddpm_cond_t1_to_t2_3_corrected_2.pth"

# must match training
IMAGE_SIZE = 192
NUM_STEPS = 400          # T in DiffusionConfig, must match training
BATCH_SIZE = 1           # keep 1 for simplest handling of slice indices

RESULTS_DIR = "conditional_ddpm/test/eval_outputs_corrected_2_192/test_split"

SAVE_INDIVIDUAL_METRICS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Metric helpers (same logic as pix2pix eval)
# =========================

def compute_metrics(img1, img2, resize_method='bicubic'):
    """Compute SSIM, PSNR, and MSE between two images.
    
    Args:
        img1: numpy array of shape (H, W) or (H, W, C), values in [0, 255]
              This is typically the ground truth image
        img2: numpy array of shape (H, W) or (H, W, C), values in [0, 255]
              This is typically the generated image
        resize_method: method to use for resizing if dimensions don't match
                      'bicubic', 'bilinear', 'nearest', or 'lanczos'
    
    Returns:
        dict with 'ssim', 'psnr', 'mse' keys
    """
    # Convert to grayscale if needed (take first channel or average)
    if len(img1.shape) == 3:
        img1 = img1[:, :, 0] if img1.shape[2] == 1 else np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = img2[:, :, 0] if img2.shape[2] == 1 else np.mean(img2, axis=2)
    
    # Ensure images are in [0, 255] range and are uint8
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    # Resize img2 (generated) to match img1 (ground truth) dimensions if they don't match
    if img1.shape != img2.shape:
        h1, w1 = img1.shape
        
        resize_map = {
            'bicubic': Image.BICUBIC,
            'bilinear': Image.BILINEAR,
            'nearest': Image.NEAREST,
            'lanczos': Image.LANCZOS
        }
        pil_method = resize_map.get(resize_method.lower(), Image.BICUBIC)
        
        img2_pil = Image.fromarray(img2)
        img2_pil = img2_pil.resize((w1, h1), pil_method)
        img2 = np.array(img2_pil)
    
    # Compute SSIM (data_range is 255 for uint8 images)
    ssim_value = ssim(img1, img2, data_range=255)
    
    # Compute PSNR
    psnr_value = psnr(img1, img2, data_range=255)
    
    # Compute MSE
    mse_value = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    return {
        'ssim': ssim_value,
        'psnr': psnr_value,
        'mse': mse_value
    }


def tensor_to_numpy(tensor):
    """Convert a tensor to numpy array in [0, 255] range.
    
    Args:
        tensor: torch.Tensor of shape (B, C, H, W) or (C, H, W) with values in [-1, 1]
    
    Returns:
        numpy array of shape (H, W) or (H, W, C) with values in [0, 255]
        (we always use the first item in the batch if batch dimension exists)
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]  # (C,H,W)
        image_numpy = tensor.detach().cpu().float().numpy()
    else:
        image_numpy = tensor
    
    # Handle grayscale (1 channel) or RGB (3 channels)
    if image_numpy.shape[0] == 1:
        # Grayscale: (1, H, W) -> (H, W)
        image_numpy = image_numpy[0]
    else:
        # RGB: (3, H, W) -> (H, W, 3)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    # Convert from [-1, 1] to [0, 255]
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
    
    return image_numpy


# =========================
# DDPM Sampling: T2 | T1
# =========================

@torch.no_grad()
def sample_t2_from_t1(model, diffusion, t1, device):
    """
    Run full reverse diffusion to sample T2 given T1.

    Args:
        model: UNetConditional
        diffusion: DiffusionConfig
        t1: (B,1,H,W) tensor in [-1,1]
        device: torch.device
    
    Returns:
        x0: (B,1,H,W) sampled T2 in [-1,1]
    """
    model.eval()
    B, _, H, W = t1.shape
    T = diffusion.T

    x = torch.randn(B, 1, H, W, device=device)  # x_T ~ N(0, I)

    betas = diffusion.betas
    alphas = diffusion.alphas
    alphas_cumprod = diffusion.alphas_cumprod

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        model_in = torch.cat([x, t1], dim=1)  # (B,2,H,W)
        eps_theta = model(model_in, t_batch)  # predicted noise

        beta_t = betas[t].view(1, 1, 1, 1)
        alpha_t = alphas[t].view(1, 1, 1, 1)
        alpha_bar_t = alphas_cumprod[t].view(1, 1, 1, 1)

        if t > 0:
            alpha_bar_prev = alphas_cumprod[t - 1].view(1, 1, 1, 1)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        # Posterior mean p(x_{t-1} | x_t, t1)
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coeff1 * (x - coeff2 * eps_theta)

        if t > 0:
            # Posterior variance
            var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

    x = torch.clamp(x, -1.0, 1.0)
    return x


# =========================
# Evaluation
# =========================

def evaluate():
    print(f">>> Using device: {DEVICE}")

    # Dataset & DataLoader
    print(f"[Stage] Building test dataset from {TEST_ROOT} ...")
    test_dataset = PairedNiftiSliceDataset(
        root_split_dir=TEST_ROOT,
        image_size=IMAGE_SIZE,
        min_nonzero_fraction=0.0,
        crop_to_brain=True,
    )
    print(f"[Stage] Test dataset ready with {len(test_dataset)} slices.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Diffusion config and model
    diffusion = DiffusionConfig(
        T=NUM_STEPS,
        beta_start=1e-4,
        beta_end=0.02,
        device=DEVICE,
    )

    # Infer H,W
    t1_sample, _, _ = test_dataset[0]
    _, H, W = t1_sample.shape
    print(f"Evaluating on slices of size {H}x{W}")

    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128).to(DEVICE)
    print(f"Loading checkpoint from {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Evaluation loop
    print(f"\nEvaluating on {len(test_dataset)} test slices...")
    all_metrics = []
    individual_metrics = []

    if BATCH_SIZE != 1 and SAVE_INDIVIDUAL_METRICS:
        print("WARNING: SAVE_INDIVIDUAL_METRICS assumes BATCH_SIZE == 1 for clean slice indexing.")

    with torch.no_grad():
        for i, (t1, t2, meta) in enumerate(test_loader):
            if i % 10 == 0:
                print(f"Processing slice batch {i+1}/{len(test_loader)}...")

            t1 = t1.to(DEVICE)  # (B,1,H,W)
            t2 = t2.to(DEVICE)  # (B,1,H,W)

            # Sample T2 | T1 via DDPM
            t2_pred = sample_t2_from_t1(model, diffusion, t1, DEVICE)  # (B,1,H,W)

            # Convert to numpy [0,255]
            t2_pred_np = tensor_to_numpy(t2_pred)   # generated
            t2_gt_np = tensor_to_numpy(t2)          # ground truth

            # Compute metrics (handles resizing if needed)
            metrics = compute_metrics(t2_gt_np, t2_pred_np)
            all_metrics.append(metrics)

            if SAVE_INDIVIDUAL_METRICS and BATCH_SIZE == 1:
                # meta is a tuple (subj_idx, z) for batch_size=1
                subj_idx, z = meta  # each is a scalar
                subj_idx_val = int(subj_idx)
                z_val = int(z)
                individual_metrics.append({
                    'subject_index': subj_idx_val,
                    'slice_index': z_val,
                    **metrics
                })

    # Aggregate metrics
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_mse  = np.mean([m['mse']  for m in all_metrics])

    std_ssim = np.std([m['ssim'] for m in all_metrics])
    std_psnr = np.std([m['psnr'] for m in all_metrics])
    std_mse  = np.std([m['mse']  for m in all_metrics])

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (DDPM T1->T2)")
    print("="*60)
    print(f"Number of test slices: {len(all_metrics)}")
    print(f"\nAverage Metrics:")
    print(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"  MSE:  {avg_mse:.4f} ± {std_mse:.4f}")
    print("="*60)

    # Save results
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = results_dir / 'evaluation_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS (DDPM T1->T2)\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"Test root: {TEST_ROOT}\n")
        f.write(f"Number of test slices: {len(all_metrics)}\n")
        f.write(f"\nAverage Metrics:\n")
        f.write(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"  PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB\n")
        f.write(f"  MSE:  {avg_mse:.4f} ± {std_mse:.4f}\n")
        f.write("="*60 + "\n")

    print(f"\nResults saved to: {metrics_file}")

    if SAVE_INDIVIDUAL_METRICS and BATCH_SIZE == 1:
        individual_file = results_dir / 'individual_metrics.json'
        with open(individual_file, 'w') as f:
            json.dump(individual_metrics, f, indent=2)
        print(f"Individual metrics saved to: {individual_file}")


if __name__ == "__main__":
    evaluate()
