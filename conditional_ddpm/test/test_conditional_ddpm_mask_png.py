import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

# 👇 adjust this import to match your training script filename
# e.g. if your training script is train_ddpm_png_masked.py
import sys

THIS_FILE = Path(__file__).resolve()
COND_DDPM_DIR = THIS_FILE.parents[1]      # .../conditional_ddpm
TRAIN_DIR = COND_DDPM_DIR / "train"
sys.path.insert(0, str(TRAIN_DIR))

from train_conditional_ddpm_mask_png import (
    PairedPNGSliceDataset,
    DiffusionConfig,
    UNetConditional,
)

# =========================
# Sampling (reverse diffusion)
# =========================

@torch.no_grad()
def sample_t2_given_t1(
    model,
    diffusion: DiffusionConfig,
    t1: torch.Tensor,          # (B,1,H,W), normalized to [-1,1]
    device: torch.device = None,
    num_steps: int = None,
):
    """
    DDPM sampling: start from noise for T2 and iteratively denoise,
    conditioned on T1. Uses the same betas/alphas as training.
    """
    model.eval()
    device = device or next(model.parameters()).device
    t1 = t1.to(device)

    B, _, H, W = t1.shape
    T = diffusion.T
    if num_steps is None or num_steps > T:
        num_steps = T

    # Start from pure noise
    x_t = torch.randn(B, 1, H, W, device=device)

    betas = diffusion.betas
    alphas = diffusion.alphas
    alpha_bars = diffusion.alphas_cumprod

    for step in reversed(range(num_steps)):
        t_idx = step  # 0..T-1
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

        alpha_t = alphas[t_idx]
        beta_t = betas[t_idx]
        alpha_bar_t = alpha_bars[t_idx]

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # conditional input [x_t, t1]
        model_in = torch.cat([x_t, t1], dim=1)
        eps_theta = model(model_in, t_batch)

        # DDPM mean (epsilon parameterization)
        coef1 = 1.0 / sqrt_alpha_t
        coef2 = beta_t / sqrt_one_minus_alpha_bar_t
        mean = coef1 * (x_t - coef2 * eps_theta)

        if t_idx > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean

    x0 = torch.clamp(x_t, -1.0, 1.0)
    return x0  # (B,1,H,W)


# =========================
# Utility: convert tensor [-1,1] -> uint8 image
# =========================

def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: (H,W) or (1,H,W) torch tensor in [-1,1]
    returns: uint8 numpy array [H,W] in [0,255]
    """
    if img_tensor.ndim == 3:
        img_tensor = img_tensor[0]  # (H,W)
    img = img_tensor.detach().cpu().clamp(-1.0, 1.0)
    img = (img + 1.0) * 0.5 * 255.0
    img = img.numpy().astype(np.uint8)
    return img


# =========================
# Main testing / sampling script
# =========================

def main():
    # --- paths ---
    test_root = "mri_slice_images/test"
    ckpt_path = "conditional_ddpm/train/best_models/ddpm_cond_t1_to_t2_3_masked_png.pth"
    out_root = Path("conditional_ddpm/test/eval_outputs_masked_png_192")
    out_root.mkdir(parents=True, exist_ok=True)

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)
    if device.type == "cuda":
        print(">>> GPU name:", torch.cuda.get_device_name(0))

    # --- dataset & dataloader ---
    print("[Stage] Building TEST dataset...")
    dataset = PairedPNGSliceDataset(
        test_root,
        image_size=192,           # must match training
        min_nonzero_fraction=0.0,
    )
    print(f"[Stage] Test dataset size: {len(dataset)} slices")

    dataloader = DataLoader(
        dataset,
        batch_size=1,             # easier to handle metadata
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- diffusion config & model (must match training) ---
    diffusion = DiffusionConfig(
        T=200,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    model = UNetConditional(
        in_ch=2,
        base_ch=32,
        time_emb_dim=128,
    ).to(device)

    print(f"[Stage] Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- sampling loop ---
    max_samples = None  # set to e.g. 100 to limit
    print("[Stage] Sampling on test set...")

    for i, batch in enumerate(dataloader):
        if max_samples is not None and i >= max_samples:
            break

        t1, t2_gt, mask, meta = batch  # meta is (subj_idx_batch, suffix_batch)

        # unpack metadata (because of DataLoader collate)
        subj_idx_batch, suffix_batch = meta
        subj_idx = int(subj_idx_batch.item())      # tensor -> int
        suffix = suffix_batch[0]                   # string, e.g. "z031.png"

        subj_dir = dataset.subjects[subj_idx]      # Path to mri_slice_images/test/sXXXX
        subj_name = subj_dir.name                  # "s0660"

        # move tensors to device
        t1 = t1.to(device)            # (1,1,H,W)
        t2_gt = t2_gt.to(device)      # (1,1,H,W)

        # sample T2 from T1
        t2_pred = sample_t2_given_t1(
            model,
            diffusion,
            t1,
            device=device,
            num_steps=None,          # full T steps (200)
        )                            # (1,1,H,W)

        # convert to uint8 images
        t1_img = tensor_to_uint8(t1[0, 0])
        t2_gt_img = tensor_to_uint8(t2_gt[0, 0])
        t2_pred_img = tensor_to_uint8(t2_pred[0, 0])

        # per-subject output directory
        subj_out_dir = out_root / subj_name
        subj_out_dir.mkdir(parents=True, exist_ok=True)

        # 1) save predicted T2 alone
        pred_name = f"pred_t2_{suffix}"
        pred_path = subj_out_dir / pred_name
        Image.fromarray(t2_pred_img).save(pred_path)

        # 2) save comparison panel: [T1 | GT T2 | Pred T2]
        panel = np.concatenate([t1_img, t2_gt_img, t2_pred_img], axis=1)
        cmp_name = f"compare_t1_gt_pred_{suffix}"
        cmp_path = subj_out_dir / cmp_name
        Image.fromarray(panel).save(cmp_path)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(dataloader)} slices...")

    print("Done. Outputs are in:", out_root)


if __name__ == "__main__":
    main()
