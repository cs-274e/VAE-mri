import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# CONFIG (edit these)
# =========================

TEST_ROOT = "dataset/test"

# NEW: point to the recon-trained checkpoint
CKPT_PATH = "conditional_ddpm/best_models/ddpm_cond_t1_to_t2_3_corrected.pth"

# must match training
IMAGE_SIZE = 192

BATCH_SIZE = 128

OUT_DIR = "conditional_ddpm/eval_outputs_corrected_192"

NUM_SAVE = 16


# =========================
# Dataset (same as training logic)
# =========================

class PairedNiftiSliceDataset(Dataset):
    """
    Expects:

      root_split_dir/
        s0001/
          t1.nii.gz
          t2.nii.gz
        s0002/
          t1.nii.gz
          t2.nii.gz
        ...

    Returns (t1, t2, (subj_idx, z)):
      t1: (1, H, W) in [-1, 1]
      t2: (1, H, W) in [-1, 1]
    """

    def __init__(
        self,
        root_split_dir: str,
        image_size: Optional[int] = None,
        min_nonzero_fraction: float = 0.0,
    ):
        self.root = Path(root_split_dir)
        self.image_size = image_size
        self.min_nonzero_fraction = min_nonzero_fraction

        self.subjects: List[Path] = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        if not self.subjects:
            raise RuntimeError(f"No subject dirs found in {self.root}")

        self.nifti_imgs: List[dict] = []
        self.index: List[Tuple[int, int]] = []  # (subj_idx, z)

        print(f"[Dataset] Scanning subjects in {self.root} ...")
        for subj_idx, subj_dir in enumerate(self.subjects):
            t1_path = subj_dir / "t1.nii.gz"
            t2_path = subj_dir / "t2.nii.gz"
            if not t1_path.exists() or not t2_path.exists():
                print(f"  Warning: missing T1 or T2 in {subj_dir}, skipping.")
                continue

            t1_img = nib.load(str(t1_path))
            t2_img = nib.load(str(t2_path))
            if t1_img.shape != t2_img.shape:
                raise RuntimeError(
                    f"T1/T2 shapes differ in {subj_dir}: {t1_img.shape} vs {t2_img.shape}"
                )
            if len(t1_img.shape) != 3:
                raise RuntimeError(f"Expected 3D volume in {subj_dir}, got {t1_img.shape}")

            self.nifti_imgs.append({"t1": t1_img, "t2": t2_img})

            dataobj = t2_img.dataobj
            H, W, Z = t2_img.shape
            for z in range(Z):
                slice_np = np.asarray(dataobj[:, :, z], dtype=np.float32)
                nonzero_fraction = np.count_nonzero(slice_np) / slice_np.size
                if nonzero_fraction >= self.min_nonzero_fraction:
                    self.index.append((subj_idx, z))

        if not self.index:
            raise RuntimeError("No valid slices found; check min_nonzero_fraction or data.")

        print(f"[Dataset] Subjects: {len(self.nifti_imgs)}, slices: {len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _normalize_slice(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        vmin = np.percentile(x, 1)
        vmax = np.percentile(x, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        x = np.clip(x, vmin, vmax)
        x = (x - vmin) / (vmax - vmin)  # [0,1]
        x = x * 2.0 - 1.0               # [-1,1]
        return x

    @staticmethod
    def _resize_tensor(x: torch.Tensor, size: Optional[int]) -> torch.Tensor:
        if size is None:
            return x
        if x.shape[-2] == size and x.shape[-1] == size:
            return x
        x = x.unsqueeze(0)  # (1,1,H,W)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)    # (1,H,W)
        return x

    def __getitem__(self, idx: int):
        subj_idx, z = self.index[idx]
        imgs = self.nifti_imgs[subj_idx]
        t1_img = imgs["t1"]
        t2_img = imgs["t2"]

        t1_slice = np.asarray(t1_img.dataobj[:, :, z], dtype=np.float32)
        t2_slice = np.asarray(t2_img.dataobj[:, :, z], dtype=np.float32)

        t1_slice = self._normalize_slice(t1_slice)
        t2_slice = self._normalize_slice(t2_slice)

        t1 = torch.from_numpy(t1_slice.copy()).unsqueeze(0)  # (1,H,W)
        t2 = torch.from_numpy(t2_slice.copy()).unsqueeze(0)

        t1 = self._resize_tensor(t1, self.image_size)
        t2 = self._resize_tensor(t2, self.image_size)

        return t1, t2, (subj_idx, z)


# =========================
# Diffusion config (same as training)
# =========================

class DiffusionConfig:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


def sinusoidal_time_embedding(t, dim: int):
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(0, half, dtype=torch.float32, device=t.device)
        * (math.log(10000.0) / (half - 1))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.ReLU(inplace=True),
            )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        if self.time_mlp is not None and t_emb is not None:
            te = self.time_mlp(t_emb)
            h = h + te[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        return h


class UNetConditional(nn.Module):
    """
    Same as training:
      input:  (B, 2, H, W)  [noisy_T2, T1]
      output: (B, 1, H, W)  predicted noise
    """

    def __init__(self, in_ch=2, base_ch=32, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Encoder
        self.down1 = DoubleConv(in_ch, base_ch, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_ch, base_ch * 2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_ch * 2, base_ch * 4, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8, time_emb_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, time_emb_dim)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        x3 = self.down3(self.pool2(x2), t_emb)

        xb = self.bottleneck(self.pool3(x3), t_emb)

        x_up = self.up3(xb)
        x_up = torch.cat([x3, x_up], dim=1)
        x_up = self.dec3(x_up, t_emb)

        x_up = self.up2(x_up)
        x_up = torch.cat([x2, x_up], dim=1)
        x_up = self.dec2(x_up, t_emb)

        x_up = self.up1(x_up)
        x_up = torch.cat([x1, x_up], dim=1)
        x_up = self.dec1(x_up, t_emb)

        out = self.out_conv(x_up)
        return out


# =========================
# Sampling: conditional DDPM T2 | T1
# =========================

@torch.no_grad()
def sample_t2_from_t1(
    model: UNetConditional,
    diffusion: DiffusionConfig,
    t1: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the reverse diffusion process to sample T2_hat given T1.
    t1: (B,1,H,W) in [-1,1]
    returns T2_hat: (B,1,H,W) in [-1,1]
    """
    model.eval()
    B, _, H, W = t1.shape
    x = torch.randn(B, 1, H, W, device=device)  # start from noise
    T = diffusion.T

    betas = diffusion.betas
    alphas = diffusion.alphas
    alphas_cumprod = diffusion.alphas_cumprod

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict noise
        model_in = torch.cat([x, t1], dim=1)  # (B,2,H,W)
        eps_theta = model(model_in, t_batch)  # (B,1,H,W)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # Predicted x0
        x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x)

            x = (1.0 / sqrt_alpha_t) * (
                x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta
            ) + sigma_t * noise
        else:
            x = x0_pred

    return x


# =========================
# Main eval
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Config] device      =", device)
    print("[Config] test_root   =", TEST_ROOT)
    print("[Config] ckpt        =", CKPT_PATH)
    print("[Config] image_size  =", IMAGE_SIZE)
    print("[Config] batch_size  =", BATCH_SIZE)
    print("[Config] out_dir     =", OUT_DIR)

    print("[Stage] Building test dataset...")
    dataset = PairedNiftiSliceDataset(
        TEST_ROOT,
        image_size=IMAGE_SIZE,
        min_nonzero_fraction=0.0,
    )
    print("[Stage] Test dataset ready.")

    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print("[Stage] Test dataloader ready.")

    diffusion = DiffusionConfig(T=200, beta_start=1e-4, beta_end=0.02, device=device)
    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128).to(device)

    print(f"[Stage] Loading checkpoint from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print("[Stage] Checkpoint loaded.")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    mae_total = 0.0
    mse_total = 0.0
    n_pixels = 0
    saved = 0

    print("[Stage] Starting evaluation sampling...")
    with torch.no_grad():
        for batch_idx, (t1, t2, idx_info) in enumerate(tqdm(test_loader, desc="Testing")):
            t1 = t1.to(device)
            t2 = t2.to(device)

            t2_hat = sample_t2_from_t1(model, diffusion, t1, device=device)

            diff = t2_hat - t2
            mae = torch.mean(torch.abs(diff)).item()
            mse = torch.mean(diff ** 2).item()

            B = t1.size(0)
            n_pix_batch = B * t1.size(-2) * t1.size(-1)

            mae_total += mae * n_pix_batch
            mse_total += mse * n_pix_batch
            n_pixels += n_pix_batch

            # Save some example PNGs
            for i in range(B):
                if saved >= NUM_SAVE:
                    break

                img_real = t2[i, 0].detach().cpu().numpy()
                img_fake = t2_hat[i, 0].detach().cpu().numpy()

                def to_uint8(x):
                    x = (x + 1.0) * 0.5
                    x = np.clip(x, 0.0, 1.0)
                    return (x * 255).astype(np.uint8)

                real_uint8 = to_uint8(img_real)
                fake_uint8 = to_uint8(img_fake)

                Image.fromarray(real_uint8).save(out_dir / f"real_{saved:04d}.png")
                Image.fromarray(fake_uint8).save(out_dir / f"fake_{saved:04d}.png")

                saved += 1

    mae_mean = mae_total / float(n_pixels)
    mse_mean = mse_total / float(n_pixels)

    print("\n=== Evaluation results ===")
    print(f"Mean MAE (per-pixel in [-1,1] space): {mae_mean:.6f}")
    print(f"Mean MSE (per-pixel in [-1,1] space): {mse_mean:.6f}")
    print(f"Saved {saved} example slice pairs to {out_dir}")


if __name__ == "__main__":
    main()
