import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from PIL import Image


# =========================
# Dataset: Paired T1/T2 axial slices from NIfTI
# =========================

class PairedNiftiSliceDataset(Dataset):
    """
    Loads T1/T2 NIfTI volumes from:

        dataset/train/
          s0001/t1.nii.gz, t2.nii.gz
          s0002/t1.nii.gz, t2.nii.gz
          ...

    Produces 2D axial slice pairs (T1[z], T2[z]),
    per-slice normalized to [-1,1] and optionally resized.
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
# Diffusion config
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
# DDPM sampler: T2 | T1
# =========================

@torch.no_grad()
def sample_t2_given_t1(model, diffusion: DiffusionConfig, t1: torch.Tensor,
                       device: torch.device = None):
    """
    DDPM sampler: generate T2 from T1 using your conditional UNet.

    t1: (B,1,H,W), normalized to [-1,1]
    returns: x0 (B,1,H,W) in [-1,1]
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    t1 = t1.to(device)

    B, _, H, W = t1.shape
    T = diffusion.T

    # start from pure noise for T2
    x_t = torch.randn(B, 1, H, W, device=device)

    betas = diffusion.betas
    alphas = diffusion.alphas
    alpha_bars = diffusion.alphas_cumprod

    for t in reversed(range(T)):  # t = T-1 ... 0
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_t = alpha_bars[t]

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # conditional input [x_t, t1]
        model_in = torch.cat([x_t, t1], dim=1)
        eps_theta = model(model_in, t_batch)

        # DDPM mean (epsilon-prediction)
        coef1 = 1.0 / sqrt_alpha_t
        coef2 = beta_t / sqrt_one_minus_alpha_bar_t
        mean = coef1 * (x_t - coef2 * eps_theta)

        if t > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean

    x0 = torch.clamp(x_t, -1.0, 1.0)
    return x0


# =========================
# Training on full dataset
# =========================

def train():
    train_root = "dataset/train"
    out_ckpt = "conditional_ddpm/best_models/ddpm_cond_t1_to_t2_3_corrected.pth"

    batch_size = 64
    image_size = 192
    num_epochs = 100
    lambda_recon = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Using device:", device)
    if device.type == "cuda":
        print(">>> GPU name:", torch.cuda.get_device_name(0))

    print("[Stage] Building dataset...")
    dataset = PairedNiftiSliceDataset(
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

    # Get H,W from a sample
    t1_sample, t2_sample, _ = dataset[0]
    _, H, W = t1_sample.shape
    print(f"Training conditional DDPM on slices of size {H}x{W}")

    diffusion = DiffusionConfig(T=200, beta_start=1e-4, beta_end=0.02, device=device)

    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()

    global_step = 0
    best_loss = float("inf")

    ckpt_dir = Path(out_ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

            # predict x0 (T2) from predicted noise
            x0_pred = (x_t - sqrt_one_minus * eps_theta) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # original noise-prediction loss
            loss_eps = mse_loss(eps_theta, noise)

            # reconstruction loss on T2
            loss_recon = torch.mean(torch.abs(x0_pred - t2))

            # combined loss
            loss = loss_eps + lambda_recon * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            global_step += 1

        epoch_loss = running_loss / len(dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"train total loss: {epoch_loss:.6f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_ckpt)
            print(f"  Saved best model to {out_ckpt} (loss={best_loss:.6f})")

    print("Done full training.")


# =========================
# Overfit a single slice
# =========================

def overfit_single_slice(
    train_root="dataset/train",
    out_ckpt="conditional_ddpm/debug_overfit_single_slice.pth",
    image_size=192,
    slice_idx=0,          # which slice in the full dataset to use
    repeats=1024,         # how many times to repeat that slice
    batch_size=8,
    num_epochs=200,
    lambda_recon=1.0      # stronger recon weight for overfitting
):
    """
    Train the same conditional DDPM model on ONE fixed (T1, T2) slice.

    Goal: see if the model can nearly perfectly reconstruct that slice.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Overfitting a single slice")
    print(">>> Using device:", device)
    if device.type == "cuda":
        print(">>> GPU name:", torch.cuda.get_device_name(0))

    # 1) Load full dataset & grab one slice
    full_ds = PairedNiftiSliceDataset(
        train_root,
        image_size=image_size,
        min_nonzero_fraction=0.0,
    )

    t1_single, t2_single, idx_info = full_ds[slice_idx]
    print(f"Using slice index {slice_idx}, subj/z = {idx_info}, shape={t1_single.shape}")

    # 2) Tiny dataset that just repeats this slice
    class SingleSliceDataset(Dataset):
        def __len__(self):
            return repeats

        def __getitem__(self, i):
            return t1_single.clone(), t2_single.clone(), (0, 0)

    single_ds = SingleSliceDataset()
    dataloader = DataLoader(
        single_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 3) Set up model & diffusion
    diffusion = DiffusionConfig(T=200, beta_start=1e-4, beta_end=0.02, device=device)

    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()

    best_loss = float("inf")
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)

    # 4) Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_eps = 0.0
        running_recon = 0.0
        n_samples = 0

        for t1_batch, t2_batch, _ in dataloader:
            t1 = t1_batch.to(device)  # (B,1,H,W)
            t2 = t2_batch.to(device)  # (B,1,H,W)
            B = t1.size(0)

            t = diffusion.sample_timesteps(B)           # (B,)
            noise = torch.randn_like(t2)

            alpha_bar_t = diffusion.alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

            # forward diffusion
            x_t = sqrt_alpha_bar_t * t2 + sqrt_one_minus * noise

            # conditional input
            model_in = torch.cat([x_t, t1], dim=1)
            eps_theta = model(model_in, t)

            # x0 prediction
            x0_pred = (x_t - sqrt_one_minus * eps_theta) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            loss_eps = mse_loss(eps_theta, noise)
            loss_recon = torch.mean(torch.abs(x0_pred - t2))

            loss = loss_eps + lambda_recon * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            running_eps += loss_eps.item() * B
            running_recon += loss_recon.item() * B
            n_samples += B

        epoch_loss = running_loss / n_samples
        epoch_eps = running_eps / n_samples
        epoch_recon = running_recon / n_samples

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"total={epoch_loss:.6f}  eps={epoch_eps:.6f}  recon={epoch_recon:.6f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_ckpt)
            print(f"  Saved best overfit model to {out_ckpt} (loss={best_loss:.6f})")

    print("Overfitting single slice done.")

    # 5) After training, do a sample from T1 of that slice
    model.eval()
    with torch.no_grad():
        t1_input = t1_single.unsqueeze(0).to(device)  # (1,1,H,W)
        x0_sample = sample_t2_given_t1(model, diffusion, t1_input, device=device)
        x0_sample_np = x0_sample.squeeze(0).squeeze(0).cpu().numpy()
        t2_gt_np = t2_single.numpy()

    return x0_sample_np, t2_gt_np


# =========================
# Utility: convert [-1,1] → uint8
# =========================

def to_uint8(x: np.ndarray) -> np.ndarray:
    """Convert [-1,1] float image (2D or 3D with singleton channel) to uint8 0–255."""
    x = np.squeeze(x)      # handle (1,H,W) or (H,W,1)
    x = (x + 1.0) * 0.5    # [-1,1] -> [0,1]
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)

# =========================
# Overfit + visualize helper
# =========================

def run_overfit_and_visualize(
    train_root="dataset/train",
    image_size=192,
    slice_idx=0,
    repeats=1024,
    batch_size=8,
    num_epochs=200,
    lambda_recon=1.0,
    save_png: bool = True,
    out_dir: str = "conditional_ddpm/overfit_debug"
):
    """
    Runs overfit_single_slice() and immediately visualizes
    predicted vs ground truth T2 for that slice.

    Also optionally saves PNGs to disk.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Overfit one slice
    pred, gt = overfit_single_slice(
        train_root=train_root,
        out_ckpt=os.path.join(out_dir, "overfit_single_slice.pth"),
        image_size=image_size,
        slice_idx=slice_idx,
        repeats=repeats,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lambda_recon=lambda_recon,
    )

    # Ensure 2D (H, W) for plotting/saving
    pred = np.squeeze(pred)  # e.g. (1, 192, 192) -> (192, 192)
    gt   = np.squeeze(gt)

    # 2) Visualization
    pred_u8 = to_uint8(pred)
    gt_u8   = to_uint8(gt)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"GT T2 (slice {slice_idx})")
    plt.imshow(gt_u8, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted T2 (overfit)")
    plt.imshow(pred_u8, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 3) Optional saving
    if save_png:
        Image.fromarray(gt_u8).save(os.path.join(out_dir, "gt_t2.png"))
        Image.fromarray(pred_u8).save(os.path.join(out_dir, "pred_t2_overfit.png"))
        print(f"Saved PNGs to {out_dir}/gt_t2.png and {out_dir}/pred_t2_overfit.png")

    # Also save raw arrays if you want to inspect later
    np.save(os.path.join(out_dir, "pred_t2_overfit.npy"), pred)
    np.save(os.path.join(out_dir, "gt_t2.npy"), gt)
    print(f"Saved NPYs to {out_dir}/pred_t2_overfit.npy and {out_dir}/gt_t2.npy")

    return pred, gt


# =========================
# Main
# =========================

if __name__ == "__main__":
    # Choose mode: "train" for full training, "overfit" for one-slice debug
    mode = "overfit"   # or "train"

    if mode == "train":
        train()
    else:
        run_overfit_and_visualize(
            train_root="dataset/train",
            image_size=192,
            slice_idx=0,      # change this to test other slices
            repeats=1024,
            batch_size=8,
            num_epochs=200,
            lambda_recon=1.0,
        )
