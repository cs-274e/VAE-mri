import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset: Paired T1/T2 PNG axial slices
# =========================

class PairedPNGSliceDataset(Dataset):
    """
    Loads T1/T2 PNG slices from:

        mri_slice_images/train/
          s0001/t1_z031.png, t2_z031.png, ...
          s0002/t1_z010.png, t2_z010.png, ...
          ...

    Produces:
      - T1 slice (1,H,W) normalized to [-1,1]
      - T2 slice (1,H,W) normalized to [-1,1]
      - mask (1,H,W) from raw T2 (>0)
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

        # subject folders s0001, s0002, ...
        self.subjects: List[Path] = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        if not self.subjects:
            raise RuntimeError(f"No subject dirs found in {self.root}")

        # For each subject: mapping slice_id -> (t1_path, t2_path)
        self.slice_map: List[Dict[str, Dict[str, Path]]] = []
        self.index: List[Tuple[int, str]] = []  # (subj_idx, slice_id)

        print(f"Scanning subjects in {self.root} ...")

        for subj_idx, subj_dir in enumerate(self.subjects):
            t1_files = sorted(subj_dir.glob("t1_*.png"))
            if not t1_files:
                print(f"  Warning: no t1_*.png in {subj_dir}, skipping.")
                continue

            t1_dict: Dict[str, Path] = {}
            t2_dict: Dict[str, Path] = {}

            # build dicts keyed by "z031.png", etc.
            for t1_path in t1_files:
                suffix = t1_path.name[len("t1_"):]  # "z031.png"
                t1_dict[suffix] = t1_path
                t2_path = subj_dir / f"t2_{suffix}"
                if t2_path.exists():
                    t2_dict[suffix] = t2_path

            # intersection of available T1 & T2 slices
            common_suffixes = sorted(set(t1_dict.keys()) & set(t2_dict.keys()))
            if not common_suffixes:
                print(f"  Warning: no matching T1/T2 pairs in {subj_dir}, skipping.")
                continue

            subj_entry = {"t1": t1_dict, "t2": t2_dict}
            self.slice_map.append(subj_entry)

            for suf in common_suffixes:
                # optional: filter by non-zero fraction on T2
                t2_path = t2_dict[suf]
                t2_img_raw = self._load_png_as_float(t2_path)
                nonzero_fraction = np.count_nonzero(t2_img_raw) / t2_img_raw.size
                if nonzero_fraction >= self.min_nonzero_fraction:
                    self.index.append((len(self.slice_map) - 1, suf))

        if not self.index:
            raise RuntimeError(
                "No valid T1/T2 slice pairs found; check data path or min_nonzero_fraction."
            )

        print(f"  Found {len(self.slice_map)} subjects, {len(self.index)} slice pairs total.")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load_png_as_float(path: Path) -> np.ndarray:
        """
        Load a PNG as float32 2D array, no normalization yet.
        Assumes grayscale or will convert to grayscale.
        """
        img = Image.open(path).convert("F")  # 32-bit floating grayscale
        arr = np.array(img, dtype=np.float32)
        return arr

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
        subj_idx, suffix = self.index[idx]
        subj_entry = self.slice_map[subj_idx]
        t1_path = subj_entry["t1"][suffix]
        t2_path = subj_entry["t2"][suffix]

        # RAW T2 slice for mask (before normalization)
        t2_raw = self._load_png_as_float(t2_path)

        # NORMALIZED slices
        t1_slice = self._load_png_as_float(t1_path)
        t2_slice = t2_raw.copy()

        t1_slice = self._normalize_slice(t1_slice)
        t2_slice = self._normalize_slice(t2_slice)

        # to tensors
        t1 = torch.from_numpy(t1_slice).unsqueeze(0)  # (1,H,W)
        t2 = torch.from_numpy(t2_slice).unsqueeze(0)  # (1,H,W)

        # simple brain mask from raw T2 intensities
        mask_np = (t2_raw > 0).astype(np.float32)      # assumes background is exactly 0
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # (1,H,W)

        # resize all to desired image_size
        t1 = self._resize_tensor(t1, self.image_size)
        t2 = self._resize_tensor(t2, self.image_size)
        mask = self._resize_tensor(mask, self.image_size)

        # re-binarize mask after interpolation
        mask = (mask > 0.5).float()

        return t1, t2, mask, (subj_idx, suffix)


# =========================
# Diffusion config
# =========================

class DiffusionConfig:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)


# =========================
# UNet building blocks
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
# CONDITIONAL UNet
# =========================

class UNetConditional(nn.Module):
    """
    - in_ch = 2   (noisy T2, T1 condition)
    - out_ch = 1  (predicted noise for T2)
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
# Masked losses
# =========================

def masked_mse(pred, target, mask):
    """
    pred, target, mask: (B,1,H,W), mask in {0,1}
    """
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


def masked_l1(pred, target, mask):
    diff = torch.abs(pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


# =========================
# Training loop
# =========================

def train():
    train_root = "mri_slice_images/train"
    out_ckpt = "conditional_ddpm_png/best_models/ddpm_cond_t1_to_t2_3_masked_png.pth"

    batch_size = 64
    image_size = 192
    num_epochs = 100
    lambda_recon = 1.0      # reconstruction weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)
    if device.type == "cuda":
        print(">>> GPU name:", torch.cuda.get_device_name(0))

    print("[Stage] Building dataset...")
    dataset = PairedPNGSliceDataset(
        train_root,
        image_size=image_size,
        min_nonzero_fraction=0.0,
    )
    print("[Stage] Dataset ready.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print("[Stage] Dataloader ready.")

    t1_sample, t2_sample, mask_sample, _ = dataset[0]
    _, H, W = t1_sample.shape
    print(f"Training conditional DDPM on slices of size {H}x{W}")

    diffusion = DiffusionConfig(T=200, beta_start=1e-4, beta_end=0.02, device=device)

    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    best_loss = float("inf")
    ckpt_dir = Path(out_ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Starting...")

        for t1, t2, mask, _ in dataloader:
            t1 = t1.to(device)      # (B,1,H,W)
            t2 = t2.to(device)      # (B,1,H,W)
            mask = mask.to(device)  # (B,1,H,W)
            B = t1.size(0)

            # sample timesteps and noise
            t = diffusion.sample_timesteps(B)      # (B,)
            noise = torch.randn_like(t2)           # (B,1,H,W)

            alpha_bar_t = diffusion.alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

            # forward diffusion: q(x_t | x_0 = T2)
            x_t = sqrt_alpha_bar_t * t2 + sqrt_one_minus * noise

            # conditional input [noisy T2, T1]
            model_in = torch.cat([x_t, t1], dim=1)
            eps_theta = model(model_in, t)

            # x0 prediction from eps_theta
            x0_pred = (x_t - sqrt_one_minus * eps_theta) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # masked losses
            loss_eps = masked_mse(eps_theta, noise, mask)
            loss_recon = masked_l1(x0_pred, t2, mask)
            loss = loss_eps + lambda_recon * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - train total loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_ckpt)
            print(f"  Saved best model to {out_ckpt} (loss={best_loss:.6f})")

    print("Done.")


if __name__ == "__main__":
    train()
