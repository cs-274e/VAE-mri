import os
import glob
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset: T1 mid-slices
# =========================

class MidSliceT1Dataset(Dataset):
    """
    Loads grayscale PNGs from a directory and returns tensors of shape (1, H, W)
    with values in [0,1].
    """

    def __init__(self, img_dir: str):
        self.paths: List[str] = sorted(
            glob.glob(os.path.join(img_dir, "*_mid_T1.png"))
        )
        if not self.paths:
            raise RuntimeError(f"No *_mid_T1.png found in {img_dir}")
        print(f"Found {len(self.paths)} T1 mid-slices in {img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
        arr = np.expand_dims(arr, axis=0)              # (1,H,W)
        return torch.from_numpy(arr)


# =========================
# Conv VAE
# =========================

class ConvVAE(nn.Module):
    def __init__(self, in_ch: int = 1, latent_dim: int = 64, img_shape=None):
        super().__init__()
        self.in_ch = in_ch
        self.latent_dim = latent_dim

        # Encoder: 3 downsampling conv blocks
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # We don't know the spatial size after convs ahead of time,
        # so we'll infer it at runtime the first time forward() is called.
        self._feat_shape = None  # (C, H', W')
        self.fc_mu = None
        self.fc_logvar = None
        self.fc_dec = None

        # If img_shape is given, we can pre-build fc layers
        if img_shape is not None:
            self._build_fc_layers(img_shape)

    def _build_fc_layers(self, img_shape):
        # img_shape = (C_in, H, W)
        # Pass a dummy through encoder to infer flattened size.
        c, h, w = img_shape
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = self.enc_conv1(x)
            x = self.enc_conv2(x)
            x = self.enc_conv3(x)
        _, c_out, h_out, w_out = x.shape
        self._feat_shape = (c_out, h_out, w_out)
        flat_dim = c_out * h_out * w_out

        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, flat_dim)

        # Decoder convs (mirror of encoder)
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, self.in_ch, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc_conv1(x)
        h = self.enc_conv2(h)
        h = self.enc_conv3(h)
        if self._feat_shape is None:
            # build fc layers using this input shape
            _, c_out, h_out, w_out = h.shape
            flat_dim = c_out * h_out * w_out
            self._feat_shape = (c_out, h_out, w_out)
            self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
            self.fc_logvar = nn.Linear(flat_dim, self.latent_dim)
            self.fc_dec = nn.Linear(self.latent_dim, flat_dim)

            self.dec_conv1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.dec_conv2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.dec_conv3 = nn.Sequential(
                nn.ConvTranspose2d(32, self.in_ch, 4, stride=2, padding=1),
            )

        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # map latent back to feature map and run transposed convs
        h_flat = self.fc_dec(z)
        c, h, w = self._feat_shape
        h_feat = h_flat.view(-1, c, h, w)
        h = self.dec_conv1(h_feat)
        h = self.dec_conv2(h)
        h = self.dec_conv3(h)
        # bound to [0,1]
        x_recon = torch.sigmoid(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# =========================
# Training
# =========================

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (MSE)
    recon = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon + beta * kl
    return loss, recon, kl


def save_samples(model, device, out_dir, step, num_samples=8):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)
        samples = model.decode(z)  # (N,1,H,W)
    samples = samples.cpu().numpy()

    for i in range(num_samples):
        img = samples[i, 0]
        img = np.clip(img, 0.0, 1.0)
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(
            os.path.join(out_dir, f"sample_step{step:06d}_{i:02d}.png")
        )


def main():
    img_dir = "mid_slices_T1"
    out_samples_dir = "vae_samples"
    ckpt_path = "vae_mid_slices_t1.pth"

    dataset = MidSliceT1Dataset(img_dir)

    # train/val split
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # infer image size from one sample
    x0 = dataset[0]  # (1,H,W)
    _, H, W = x0.shape
    model = ConvVAE(in_ch=1, latent_dim=64, img_shape=(1, H, W)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 500
    beta = 1.0
    best_val = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for x in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_recon += recon.item() * bs
            train_kl += kl.item() * bs
            global_step += 1

            if global_step % 200 == 0:
                save_samples(model, device, out_samples_dir, global_step, num_samples=8)

        train_loss /= len(train_loader.dataset)
        train_recon /= len(train_loader.dataset)
        train_kl /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                loss, recon, kl = vae_loss(recon_x, x, mu, logvar, beta=beta)
                bs = x.size(0)
                val_loss += loss.item() * bs
                val_recon += recon.item() * bs
                val_kl += kl.item() * bs

        val_loss /= len(val_loader.dataset)
        val_recon /= len(val_loader.dataset)
        val_kl /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- train loss: {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}) "
            f"- val loss: {val_loss:.4f} (recon {val_recon:.4f}, kl {val_kl:.4f})"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
