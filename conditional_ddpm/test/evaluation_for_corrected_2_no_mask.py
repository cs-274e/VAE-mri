#!/usr/bin/env python3
"""
Evaluation script for conditional DDPM model on MRI T1->T2 translation (no-mask training).

Pipeline:
1. Use PairedNiftiSliceDataset on dataset/test (same preprocessing as training, full FOV, no masking).
2. Run conditional DDPM sampling to generate T2 | T1 at IMAGE_SIZE x IMAGE_SIZE.
3. For each slice:
   - Resize the prediction back to the full FOV (here bbox is full image, so it's essentially a resize).
   - Transpose to match the PNG orientation used by save_mri_images.py.
   - Compare to T2 PNG from mri_slice_images/test/<subject>/<prefix>_zXXX.png.
4. Compute PSNR and SSIM only.
5. Aggregate metrics:
   - Overall PSNR/SSIM across all test slices.
   - Per-subject PSNR/SSIM for each third of the volume along z.
   - Global PSNR/SSIM per third across the whole test set.
6. Save example prediction images per subject for z in {1, 11, 21, 31} (if available),
   each as its own PNG in a per-subject folder, with the same size and orientation
   as your colleague's PNG dataset.
"""

import os
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# =====================================================
# IMPORT TRAINING COMPONENTS (NO-MASK VERSION)
# =====================================================
THIS_DIR = Path(__file__).resolve().parent         # .../conditional_ddpm/test
PROJECT_ROOT = THIS_DIR.parent                     # .../conditional_ddpm
sys.path.append(str(PROJECT_ROOT))

# ⚠️ Adjust the module name here if your file has a different name
from train.train_conditional_ddpm_3_no_mask import (
    PairedNiftiSliceDataset,
    DiffusionConfig,
    UNetConditional,
)

# =========================
# CONFIG (edit these)
# =========================

# NIfTI-based test split (same style as training)
TEST_ROOT = "dataset/test"

# Path to trained DDPM checkpoint (no-mask)
CKPT_PATH = (
    "conditional_ddpm/train/best_models/"
    "ddpm_cond_t1_to_t2_3_corrected_2_no_mask.pth"
)

# Root of PNG slices prepared by your colleague
PNG_ROOT = "mri_slice_images/test"

# Prefix used when saving T2 PNGs via save_mri_images.py
# If they used: prefix='t2', files are t2_z000.png, t2_z001.png, ...
# Change to "slice" if PNGs are named slice_z000.png, etc.
PNG_PREFIX_T2 = "t2"

# Must match training
IMAGE_SIZE = 192
NUM_STEPS = 400          # T in DiffusionConfig
BATCH_SIZE = 1           # keep 1 for clean subject/z bookkeeping

RESULTS_DIR = "conditional_ddpm/test/eval_outputs_corrected_2_no_mask"

SAVE_INDIVIDUAL_METRICS = True

# z-indices (0-based) to save per-subject example predictions
EXAMPLE_ZS = [1, 11, 21, 31]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Metric helpers (PSNR + SSIM in PyTorch)
# =========================

def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure numpy image (H,W) in uint8 [0,255]."""
    if len(img.shape) == 3:
        img = img[:, :, 0] if img.shape[2] == 1 else np.mean(img, axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _torch_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute SSIM between two single-channel torch images in [0,255].

    Args:
        img1, img2: (1,1,H,W) float32 tensors
    Returns:
        scalar SSIM value
    """
    K1, K2 = 0.01, 0.03
    L = 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    def gaussian_window(size=11, sigma=1.5):
        coords = torch.arange(size) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma * sigma))
        g = g / g.sum()
        window = g[:, None] @ g[None, :]
        return window

    window = gaussian_window().unsqueeze(0).unsqueeze(0)  # (1,1,11,11)
    window = window.to(img1.device, dtype=img1.dtype)

    mu1 = torch.nn.functional.conv2d(img1, window, padding=5)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=5) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=5) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=5) - mu1_mu2

    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_n / (ssim_d + 1e-12)

    return ssim_map.mean().item()


def compute_metrics(img_gt: np.ndarray, img_pred: np.ndarray, resize_method: str = 'bicubic'):
    """
    Compute PSNR and SSIM between two images.

    Args:
        img_gt:   numpy array (H,W) or (H,W,C), values in [0,255] (ground truth)
        img_pred: numpy array (H,W) or (H,W,C), values in [0,255] (prediction)

    Returns:
        dict with 'psnr', 'ssim'
    """
    # Convert to grayscale uint8
    img1 = _to_gray_uint8(img_gt)
    img2 = _to_gray_uint8(img_pred)

    # Resize prediction to match GT if shapes differ
    if img1.shape != img2.shape:
        h1, w1 = img1.shape
        resize_map = {
            'bicubic': Image.BICUBIC,
            'bilinear': Image.BILINEAR,
            'nearest': Image.NEAREST,
            'lanczos': Image.LANCZOS,
        }
        pil_method = resize_map.get(resize_method.lower(), Image.BICUBIC)
        img2_pil = Image.fromarray(img2)
        img2_pil = img2_pil.resize((w1, h1), pil_method)
        img2 = np.array(img2_pil)

    # Convert to torch tensors (1,1,H,W)
    img1_t = torch.from_numpy(img1.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    img2_t = torch.from_numpy(img2.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # MSE
    mse = torch.nn.functional.mse_loss(img1_t, img2_t).item()

    # PSNR
    if mse == 0:
        psnr_val = 100.0
    else:
        psnr_val = 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)

    # SSIM
    ssim_val = _torch_ssim(img1_t, img2_t)

    return {"psnr": psnr_val, "ssim": ssim_val}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to numpy array in [0, 255] range.

    Args:
        tensor: torch.Tensor of shape (B,C,H,W) or (C,H,W) with values in [-1,1]

    Returns:
        numpy array of shape (H,W) or (H,W,C) in [0,255]
        (always uses the first item in the batch if batch dimension exists)
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]  # (C,H,W)
        image_numpy = tensor.detach().cpu().float().numpy()
    else:
        image_numpy = tensor

    if image_numpy.shape[0] == 1:
        image_numpy = image_numpy[0]
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
    return image_numpy


# =========================
# DDPM Sampling: T2 | T1
# =========================

@torch.no_grad()
def sample_t2_from_t1(
    model: UNetConditional,
    diffusion: DiffusionConfig,
    t1: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
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
        eps_theta = model(model_in, t_batch)

        beta_t = betas[t].view(1, 1, 1, 1)
        alpha_t = alphas[t].view(1, 1, 1, 1)
        alpha_bar_t = alphas_cumprod[t].view(1, 1, 1, 1)

        if t > 0:
            alpha_bar_prev = alphas_cumprod[t - 1].view(1, 1, 1, 1)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coeff1 * (x - coeff2 * eps_theta)

        if t > 0:
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
        crop_to_brain=False,  # full FOV (no masking) like training
    )
    print(f"[Stage] Test dataset ready with {len(test_dataset)} slices.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    diffusion = DiffusionConfig(
        T=NUM_STEPS,
        beta_start=1e-4,
        beta_end=0.02,
        device=DEVICE,
    )

    # For reconstructing original FOV:
    # shape: (H_vol, W_vol, Z), bbox: (y_min, y_max, x_min, x_max)
    volume_stats = test_dataset.volume_stats

    # Model input size
    t1_sample, _, _ = test_dataset[0]
    _, H_model, W_model = t1_sample.shape
    print(f"Evaluating with model input size {H_model}x{W_model}")

    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128).to(DEVICE)
    print(f"Loading checkpoint from {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Overall metrics
    all_psnr = []
    all_ssim = []

    # Per-slice metrics
    individual_metrics = []

    # Per-subject, per-third metrics
    per_subject_bins = defaultdict(
        lambda: {0: {'psnr': [], 'ssim': []},
                 1: {'psnr': [], 'ssim': []},
                 2: {'psnr': [], 'ssim': []}}
    )

    # Global bins
    global_bins = {
        0: {'psnr': [], 'ssim': []},
        1: {'psnr': [], 'ssim': []},
        2: {'psnr': [], 'ssim': []},
    }

    # Example prediction slices per subject
    # examples[subj_name][z] = np.ndarray (H_png, W_png)
    examples = defaultdict(dict)

    with torch.no_grad():
        for i, (t1, t2, meta) in enumerate(test_loader):
            if i % 10 == 0:
                print(f"Processing slice batch {i+1}/{len(test_loader)}...")

            t1 = t1.to(DEVICE)  # (B,1,H_model,W_model)
            # t2 is normalized NIfTI patch; not used directly for metrics
            t2 = t2.to(DEVICE)

            subj_idx_tensor, z_tensor = meta

            if isinstance(subj_idx_tensor, torch.Tensor):
                subj_idx = int(subj_idx_tensor[0].item() if subj_idx_tensor.ndim > 0 else subj_idx_tensor.item())
            else:
                subj_idx = int(subj_idx_tensor)

            if isinstance(z_tensor, torch.Tensor):
                z_val = int(z_tensor[0].item() if z_tensor.ndim > 0 else z_tensor.item())
            else:
                z_val = int(z_tensor)

            subj_path = test_dataset.subjects[subj_idx]
            subj_name = subj_path.name  # e.g. "s0711"

            # --- Load PNG ground truth T2 (full FOV, in PNG normalization) ---
            png_dir = Path(PNG_ROOT) / subj_name
            png_t2_path = png_dir / f"{PNG_PREFIX_T2}_z{z_val:03d}.png"
            if not png_t2_path.exists():
                print(f"  WARNING: PNG GT not found at {png_t2_path}, skipping slice.")
                continue

            t2_gt_img = Image.open(png_t2_path).convert("L")
            W_png, H_png = t2_gt_img.size  # (width,height)
            t2_gt_np = np.array(t2_gt_img)  # (H_png, W_png), uint8

            # --- Generate T2 patch at model resolution (IMAGE_SIZE x IMAGE_SIZE) ---
            t2_pred_patch = sample_t2_from_t1(model, diffusion, t1, DEVICE)  # (1,1,H_model,W_model)

            # --- Reconstruct full-FOV predicted slice in original volume space ---
            stats = volume_stats[subj_idx]
            H_vol, W_vol, Z_vol = stats['shape']
            y_min, y_max, x_min, x_max = stats['bbox']  # here full FOV: y_min=0,x_min=0

            patch_h = y_max - y_min + 1  # == H_vol
            patch_w = x_max - x_min + 1  # == W_vol

            # Resize prediction back to full FOV (since bbox==full FOV)
            t2_pred_full = torch.nn.functional.interpolate(
                t2_pred_patch,
                size=(patch_h, patch_w),
                mode="bilinear",
                align_corners=False,
            )  # (1,1,H_vol,W_vol)

            # Convert to numpy [0,255] in original NIfTI orientation
            full_pred_np = tensor_to_numpy(t2_pred_full)  # (H_vol, W_vol)

            # PNG was created as: data[:, :, z].transpose(1, 0)
            # -> transpose our prediction to align with PNG axes
            full_pred_png_np = full_pred_np.T  # (W_vol, H_vol) ~ (H_png, W_png)

            # --- Compute metrics (PSNR + SSIM) in full FOV PNG space ---
            metrics = compute_metrics(t2_gt_np, full_pred_png_np)
            psnr_val = float(metrics['psnr'])
            ssim_val = float(metrics['ssim'])

            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)

            if SAVE_INDIVIDUAL_METRICS:
                individual_metrics.append({
                    'subject_name': subj_name,
                    'subject_index': subj_idx,
                    'slice_z': z_val,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'png_path': str(png_t2_path),
                })

            # --- Assign to volume third along z ---
            Z_total = Z_vol
            third_size = Z_total / 3.0
            if z_val < third_size:
                bin_idx = 0
            elif z_val < 2 * third_size:
                bin_idx = 1
            else:
                bin_idx = 2

            per_subject_bins[subj_name][bin_idx]['psnr'].append(psnr_val)
            per_subject_bins[subj_name][bin_idx]['ssim'].append(ssim_val)
            global_bins[bin_idx]['psnr'].append(psnr_val)
            global_bins[bin_idx]['ssim'].append(ssim_val)

            # --- Store prediction examples for specified z-values (prediction only) ---
            if z_val in EXAMPLE_ZS and z_val not in examples[subj_name]:
                # full_pred_png_np is already full-FOV, PNG orientation, uint8-like
                examples[subj_name][z_val] = full_pred_png_np

    # =========================
    # Aggregate + Print metrics
    # =========================

    avg_psnr = float(np.mean(all_psnr)) if all_psnr else float("nan")
    avg_ssim = float(np.mean(all_ssim)) if all_ssim else float("nan")
    std_psnr = float(np.std(all_psnr)) if all_psnr else float("nan")
    std_ssim = float(np.std(all_ssim)) if all_ssim else float("nan")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (DDPM T1->T2, full-FOV PNG, NO MASK)")
    print("=" * 60)
    print(f"Number of evaluated slices: {len(all_psnr)}")
    print(f"\nOverall (all slices):")
    print(f"  PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

    # Per subject + per third
    print("\nPer-subject metrics by volume thirds (z-axis):")
    for subj_name, bins in per_subject_bins.items():
        print(f"\nSubject {subj_name}:")
        for bin_idx, label in zip([0, 1, 2], ["1st third", "2nd third", "3rd third"]):
            ps = bins[bin_idx]['psnr']
            ss = bins[bin_idx]['ssim']
            if len(ps) == 0:
                print(f"  {label}: no slices")
                continue
            ps_mean = float(np.mean(ps))
            ss_mean = float(np.mean(ss))
            print(f"  {label}: PSNR={ps_mean:.4f} dB, SSIM={ss_mean:.4f}")

    # Global per-third metrics
    print("\nGlobal metrics by volume thirds (across all subjects):")
    for bin_idx, label in zip([0, 1, 2], ["1st third", "2nd third", "3rd third"]):
        ps = global_bins[bin_idx]['psnr']
        ss = global_bins[bin_idx]['ssim']
        if len(ps) == 0:
            print(f"  {label}: no slices")
            continue
        ps_mean = float(np.mean(ps))
        ss_mean = float(np.mean(ss))
        print(f"  {label}: PSNR={ps_mean:.4f} dB, SSIM={ss_mean:.4f}")

    # =========================
    # Save metrics to disk
    # =========================

    metrics_file = results_dir / "evaluation_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS (DDPM T1->T2, full-FOV PNG, NO MASK)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"NIfTI test root: {TEST_ROOT}\n")
        f.write(f"PNG root: {PNG_ROOT}\n")
        f.write(f"PNG prefix (T2): {PNG_PREFIX_T2}\n")
        f.write(f"Number of evaluated slices: {len(all_psnr)}\n\n")

        f.write("Overall (all slices):\n")
        f.write(f"  PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB\n")
        f.write(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n\n")

        f.write("Per-subject metrics by volume thirds (z-axis):\n")
        for subj_name, bins in per_subject_bins.items():
            f.write(f"\nSubject {subj_name}:\n")
            for bin_idx, label in zip([0, 1, 2], ["1st third", "2nd third", "3rd third"]):
                ps = bins[bin_idx]['psnr']
                ss = bins[bin_idx]['ssim']
                if len(ps) == 0:
                    f.write(f"  {label}: no slices\n")
                    continue
                ps_mean = float(np.mean(ps))
                ss_mean = float(np.mean(ss))
                f.write(f"  {label}: PSNR={ps_mean:.4f} dB, SSIM={ss_mean:.4f}\n")

        f.write("\nGlobal metrics by volume thirds (across all subjects):\n")
        for bin_idx, label in zip([0, 1, 2], ["1st third", "2nd third", "3rd third"]):
            ps = global_bins[bin_idx]['psnr']
            ss = global_bins[bin_idx]['ssim']
            if len(ps) == 0:
                f.write(f"  {label}: no slices\n")
                continue
            ps_mean = float(np.mean(ps))
            ss_mean = float(np.mean(ss))
            f.write(f"  {label}: PSNR={ps_mean:.4f} dB, SSIM={ss_mean:.4f}\n")

    print(f"\nSummary metrics saved to: {metrics_file}")

    if SAVE_INDIVIDUAL_METRICS:
        individual_file = results_dir / "individual_metrics.json"
        with open(individual_file, 'w') as f:
            json.dump(individual_metrics, f, indent=2)
        print(f"Per-slice metrics saved to: {individual_file}")

    # =========================
    # Save individual prediction PNGs per subject & z
    # =========================

    print("\nSaving individual T2 prediction slices per subject...")
    for subj_name, slices_dict in examples.items():
        if not slices_dict:
            continue
        subj_dir = results_dir / subj_name
        subj_dir.mkdir(parents=True, exist_ok=True)

        for z_val, pred_img in slices_dict.items():
            # pred_img is (H_png, W_png), uint8-like
            img = Image.fromarray(pred_img.astype(np.uint8))
            out_path = subj_dir / f"t2_pred_z{z_val:03d}.png"
            img.save(out_path)
            print(f"  Saved prediction slice z={z_val} for {subj_name} to {out_path}")


if __name__ == "__main__":
    evaluate()
