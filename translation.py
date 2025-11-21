import os
import glob
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------
# Dataset
# --------------------

class T1ToT2Dataset(Dataset):
    def __init__(self, t1_dir: str, t2_dir: str):
        self.pairs: List[Tuple[str, str]] = []

        t1_paths = sorted(glob.glob(os.path.join(t1_dir, "*_mid_T1.png")))
        for t1_path in t1_paths:
            name = os.path.basename(t1_path)
            subj_prefix = name.replace("_mid_T1.png", "")
            t2_name = f"{subj_prefix}_mid_T2.png"
            t2_path = os.path.join(t2_dir, t2_name)
            if os.path.exists(t2_path):
                self.pairs.append((t1_path, t2_path))

        if not self.pairs:
            raise RuntimeError("No paired T1/T2 slices found.")

        print(f"Found {len(self.pairs)} paired slices.")

    def __len__(self):
        return len(self.pairs)

    def _load_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
        arr = np.expand_dims(arr, axis=0)              # (1, H, W)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        t1_path, t2_path = self.pairs[idx]
        t1 = self._load_img(t1_path)
        t2 = self._load_img(t2_path)
        return t1, t2


# --------------------
# Simple UNet-like model
# --------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()

        self.down1 = DoubleConv(in_ch, base_ch)
        self.down2 = DoubleConv(base_ch, base_ch * 2)
        self.down3 = DoubleConv(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.conv3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.conv1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        # bottleneck
        xb = self.bottleneck(self.pool(x3))

        # decoder
        x = self.up3(xb)
        x = torch.cat([x3, x], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv1(x)

        x = self.out_conv(x)
        # predict in [0,1]
        x = torch.sigmoid(x)
        return x


# --------------------
# Training
# --------------------

def save_example(t1, t2_gt, t2_pred, out_dir, step):
    os.makedirs(out_dir, exist_ok=True)
    t1 = t1[0, 0].cpu().numpy()
    t2_gt = t2_gt[0, 0].cpu().numpy()
    t2_pred = t2_pred[0, 0].detach().cpu().numpy()


    def to_uint8(x):
        x = np.clip(x, 0.0, 1.0)
        return (x * 255).astype(np.uint8)

    Image.fromarray(to_uint8(t1)).save(os.path.join(out_dir, f"step{step:06d}_t1.png"))
    Image.fromarray(to_uint8(t2_gt)).save(os.path.join(out_dir, f"step{step:06d}_t2_gt.png"))
    Image.fromarray(to_uint8(t2_pred)).save(os.path.join(out_dir, f"step{step:06d}_t2_pred.png"))


def main():
    t1_dir = "mid_slices_T1"
    t2_dir = "mid_slices_T2"
    out_dir = "t1_to_t2_samples"
    ckpt_path = "t1_to_t2_unet.pth"

    dataset = T1ToT2Dataset(t1_dir, t2_dir)

    # simple split into train/val
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=1, out_ch=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    num_epochs = 50
    global_step = 0
    best_val = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for t1, t2 in train_loader:
            t1 = t1.to(device)  # (B,1,H,W)
            t2 = t2.to(device)

            optimizer.zero_grad()
            t2_pred = model(t1)
            loss = criterion(t2_pred, t2)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * t1.size(0)
            global_step += 1

            if global_step % 200 == 0:
                save_example(t1, t2, t2_pred, out_dir, global_step)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t1, t2 in val_loader:
                t1 = t1.to(device)
                t2 = t2.to(device)
                t2_pred = model(t1)
                loss = criterion(t2_pred, t2)
                val_loss += loss.item() * t1.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"- train L1: {train_loss:.4f} "
              f"- val L1: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
