import os
import glob
from typing import List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import nibabel as nib  # NEW: for NIfTI

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Tuple, Optional  

# =========================
# Dataset: Paired T1/T2 axial slices from NIfTI
# =========================

class PairedNiftiSliceDataset(Dataset):
    """
    CHANGED: instead of loading PNG mid-slices, we now load
    T1/T2 NIfTI volumes from:

        dataset/train/
          s0001/T1.nii.gz, T2.nii.gz
          s0002/T1.nii.gz, T2.nii.gz
          ...

    We then create 2D axial slice pairs (T1[z], T2[z]) with simple
    per-slice normalization and resize to a fixed size.
    """

    def __init__(self, root_split_dir: str, image_size: Optional[int] = None,
                 min_nonzero_fraction: float = 0.05):
        self.root = Path(root_split_dir)
        self.image_size = image_size
        self.min_nonzero_fraction = min_nonzero_fraction

        # subject folders s0001, s0002, ...
        self.subjects: List[Path] = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        if not self.subjects:
            raise RuntimeError(f"No subject dirs found in {self.root}")

        self.nifti_imgs: List[dict] = []
        self.index: List[Tuple[int, int]] = []  # (subj_idx, z)

        print(f"Scanning subjects in {self.root} ...")
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

            # Select slices with enough non-zero content in T2
            dataobj = t2_img.dataobj
            H, W, Z = t2_img.shape
            for z in range(Z):
                slice_np = np.asarray(dataobj[:, :, z], dtype=np.float32)
                nonzero_fraction = np.count_nonzero(slice_np) / slice_np.size
                if nonzero_fraction >= self.min_nonzero_fraction:
                    self.index.append((subj_idx, z))

        if not self.index:
            raise RuntimeError("No valid slices found; check min_nonzero_fraction or data.")

        print(f"  Found {len(self.nifti_imgs)} subjects, {len(self.index)} valid slices total.")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _normalize_slice(x: np.ndarray) -> np.ndarray:
        """
        Normalize slice to [-1,1] with percentile clipping.
        """
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
        """
        x: (1,H,W) -> (1,size,size) if size is not None,
           otherwise return as-is.
        """
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

        t1 = torch.from_numpy(t1_slice).unsqueeze(0)  # (1,H,W)
        t2 = torch.from_numpy(t2_slice).unsqueeze(0)  # (1,H,W)

        t1 = self._resize_tensor(t1, self.image_size)
        t2 = self._resize_tensor(t2, self.image_size)

        return t1, t2, (subj_idx, z)


# =========================
# Diffusion config (same idea)
# =========================

class DiffusionConfig:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)


# =========================
# UNet building blocks (same as friend)
# =========================

def sinusoidal_time_embedding(t, dim: int):
    """
    t: (B,) integer timesteps
    returns: (B, dim) embedding
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(0, half, dtype=torch.float32, device=t.device)
        * (np.log(10000.0) / (half - 1))
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
            te = self.time_mlp(t_emb)  # (B, out_ch)
            h = h + te[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        return h


# =========================
# CONDITIONAL UNet (key difference)
# =========================

class UNetConditional(nn.Module):
    """
    CHANGED: This is your friend’s UNetFull, but:

    - in_ch = 2   (noisy T2, T1 condition)
    - out_ch = 1  (predicted noise for T2)

    Everything else (time embedding, encoder/decoder) is the same idea.
    """

    def __init__(self, in_ch=2, base_ch=32, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding MLP
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

        self.out_conv = nn.Conv2d(base_ch, 1, 1)  # predict 1-channel noise for T2

    def forward(self, x, t):
        # x: (B, 2, H, W) = [noisy T2, T1]
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        x3 = self.down3(self.pool2(x2), t_emb)

        # Bottleneck
        xb = self.bottleneck(self.pool3(x3), t_emb)

        # Decoder
        x_up = self.up3(xb)
        x_up = torch.cat([x3, x_up], dim=1)
        x_up = self.dec3(x_up, t_emb)

        x_up = self.up2(x_up)
        x_up = torch.cat([x2, x_up], dim=1)
        x_up = self.dec2(x_up, t_emb)

        x_up = self.up1(x_up)
        x_up = torch.cat([x1, x_up], dim=1)
        x_up = self.dec1(x_up, t_emb)

        out = self.out_conv(x_up)  # (B,1,H,W)
        return out


# =========================
# Training (conditional DDPM T2 | T1)
# =========================

def train():
    # CHANGED: we now point to directory with subject folders
    train_root = "dataset/train"
    out_ckpt = "conditional_ddpm/best_models/ddpm_cond_t1_to_t2_3.pth"

    batch_size = 128
    image_size = 192
    num_epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)
    if device.type == "cuda":
        print(">>> GPU name:", torch.cuda.get_device_name(0))
    
    print("[Stage] Building dataset...")
    dataset = PairedNiftiSliceDataset(train_root, image_size=image_size)
    print("[Stage] Dataset ready.")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print("[Stage] Dataloader ready.")

    # Get H,W from a sample
    t1_sample, t2_sample, _ = dataset[0]
    _, H, W = t1_sample.shape
    print(f"Training conditional DDPM on slices of size {H}x{W}")

    diffusion = DiffusionConfig(T=200, beta_start=1e-4, beta_end=0.02, device=device)

    # CHANGED: use conditional UNet with 2 input channels
    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()

    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Starting...")

        for t1, t2, _ in dataloader:
            t1 = t1.to(device)  # (B,1,H,W)
            t2 = t2.to(device)  # (B,1,H,W)
            B = t1.size(0)

            # sample timesteps and noise
            t = diffusion.sample_timesteps(B)      # (B,)
            noise = torch.randn_like(t2)           # same shape as T2

            alpha_bar_t = diffusion.alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

            # FORWARD NOISE: q(x_t | x_0 = T2)
            x_t = sqrt_alpha_bar_t * t2 + sqrt_one_minus * noise

            # CONDITIONAL INPUT: concat noisy T2 and T1
            model_in = torch.cat([x_t, t1], dim=1)  # (B,2,H,W)

            # predict noise
            eps_theta = model(model_in, t)
            loss = mse_loss(eps_theta, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            global_step += 1

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - train MSE: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_ckpt)
            print(f"  Saved best model to {out_ckpt}")

    print("Done.")


if __name__ == "__main__":
    train()
