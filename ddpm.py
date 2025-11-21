import os
import glob
import random
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# =========================
# Dataset: 833 mid-slice T1 images
# =========================

class MidSliceT1Dataset(Dataset):
    """
    Loads T1 mid-slice PNGs from a directory, applies simple augmentation,
    resizes to 64x64, and returns tensors in [-1, 1] with shape (1, H, W).
    """

    def __init__(self, img_dir: str, img_size: int = 64, augment: bool = True):
        self.paths: List[str] = sorted(
            glob.glob(os.path.join(img_dir, "*_mid_T1.png"))
        )
        if not self.paths:
            raise RuntimeError(f"No *_mid_T1.png found in {img_dir}")
        print(f"Found {len(self.paths)} T1 mid-slices in {img_dir}")
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("L")  # grayscale PIL image

        # Resize to 64x64
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)

            # Small random rotation [-10, 10] degrees
            angle = random.uniform(-10.0, 10.0)
            img = TF.rotate(img, angle, fill=0)

        arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]

        # Optional tiny brightness/contrast jitter
        # Uncomment if you want more aggression
        # arr = np.clip(
        #     arr * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05),
        #     0.0, 1.0
        # )

        # Map to [-1, 1]
        arr = arr * 2.0 - 1.0
        arr = np.expand_dims(arr, axis=0)  # (1,H,W)
        return torch.from_numpy(arr)


# =========================
# Diffusion config
# =========================

class DiffusionConfig:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)


# =========================
# UNet pieces
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


class UNetSmall(nn.Module):
    """
    Tiny UNet for 64x64 grayscale images:
    - base_ch=16
    - 2 downsamples (64->32->16)
    """

    def __init__(self, in_ch=1, base_ch=16, time_emb_dim=64):
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
        self.pool1 = nn.MaxPool2d(2)  # 64->32

        self.down2 = DoubleConv(base_ch, base_ch * 2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)  # 32->16

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4, time_emb_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)  # 16->32
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)  # 32->64
        self.dec1 = DoubleConv(base_ch * 2, base_ch, time_emb_dim)

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        # t: (B,) integer timesteps
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.down1(x, t_emb)         # (B,16,64,64)
        x2 = self.down2(self.pool1(x1), t_emb)  # (B,32,32,32)

        # Bottleneck
        xb = self.bottleneck(self.pool2(x2), t_emb)  # (B,64,16,16)

        # Decoder
        x = self.up2(xb)  # (B,32,32,32)
        x = torch.cat([x2, x], dim=1)  # (B,64,32,32)
        x = self.dec2(x, t_emb)        # (B,32,32,32)

        x = self.up1(x)                # (B,16,64,64)
        x = torch.cat([x1, x], dim=1)  # (B,32,64,64)
        x = self.dec1(x, t_emb)        # (B,16,64,64)

        out = self.out_conv(x)         # (B,1,64,64)
        return out  # predicted noise ε


# =========================
# Sampling
# =========================

def save_samples(model, diffusion, device, out_dir, step, H=64, W=64, num_samples=8):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    T = diffusion.T

    with torch.no_grad():
        x = torch.randn(num_samples, 1, H, W, device=device)

        for t in reversed(range(T)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            eps_theta = model(x, t_batch)  # predicted noise

            beta_t = diffusion.betas[t]
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alphas_cumprod[t]

            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)

            # Predict x0
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if t > 0:
                sigma_t = torch.sqrt(beta_t)
                noise = torch.randn_like(x)
                # DDPM reverse update
                x = (1.0 / sqrt_alpha_t) * (
                    x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta
                ) + sigma_t * noise
            else:
                x = x0_pred

        x = x.cpu().numpy()

    for i in range(num_samples):
        img = x[i, 0]
        img = (img + 1.0) * 0.5  # [-1,1] -> [0,1]
        img = np.clip(img, 0.0, 1.0)
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(
            os.path.join(out_dir, f"sample_step{step:06d}_{i:02d}.png")
        )


# =========================
# Training
# =========================

def train():
    img_dir = "mid_slices_T1"
    out_samples_dir = "ddpm_samples"
    ckpt_path = "ddpm_mid_slices_t1_small.pth"

    dataset = MidSliceT1Dataset(img_dir, img_size=64, augment=True)
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = DiffusionConfig(T=100, beta_start=1e-4, beta_end=0.02, device=device)

    model = UNetSmall(in_ch=1, base_ch=16, time_emb_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()

    num_epochs = 500
    global_step = 0
    best_val = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x in train_loader:
            x = x.to(device)  # (B,1,64,64)
            b = x.size(0)

            t = diffusion.sample_timesteps(b)  # (B,)
            noise = torch.randn_like(x)

            alpha_bar_t = diffusion.alphas_cumprod[t].view(b, 1, 1, 1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

            # forward noising: q(x_t | x_0)
            x_t = sqrt_alpha_bar_t * x + sqrt_one_minus * noise

            eps_theta = model(x_t, t)
            loss = mse_loss(eps_theta, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b
            global_step += 1

            if global_step % 500 == 0:
                save_samples(
                    model,
                    diffusion,
                    device,
                    out_samples_dir,
                    global_step,
                    H=64,
                    W=64,
                    num_samples=8,
                )

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                b = x.size(0)
                t = diffusion.sample_timesteps(b)
                noise = torch.randn_like(x)

                alpha_bar_t = diffusion.alphas_cumprod[t].view(b, 1, 1, 1)
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

                x_t = sqrt_alpha_bar_t * x + sqrt_one_minus * noise
                eps_theta = model(x_t, t)
                loss = mse_loss(eps_theta, noise)
                val_loss += loss.item() * b

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- train MSE: {train_loss:.6f} "
            f"- val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    train()
