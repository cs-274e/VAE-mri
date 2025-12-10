import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import csv

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

TEST_ROOT = "dataset/test"                # test split root
AGE_CSV = "dataset/meta_cleaned.csv"      # same CSV format as training
CKPT_PATH = "conditional_ddpm/train/best_models/ddpm_cond_t1_to_t2_3_corrected_2_age.pth"
IMAGE_SIZE = 192
BATCH_SIZE = 64

OUT_DIR = "conditional_ddpm/test/eval_outputs_corrected_2_age/trajectories"
NUM_SAVE = 16   # number of (real, pred) pairs to save from test

# Ages (in months) for trajectories from a single T1 slice
TRAJECTORY_AGES_MONTHS = [0.0, 4.0, 8.0, 12.0, 24.0, 36.0]
TRAJECTORY_OUT = "conditional_ddpm/eval_agecond/trajectory_example.png"


# =========================
# Dataset: Paired T1/T2 + AGE (same logic as training)
# =========================

class PairedNiftiSliceDataset(Dataset):
    """
    Expects:

      root_split_dir/
        s0001/t1.nii.gz, t2.nii.gz
        s0002/t1.nii.gz, t2.nii.gz
        ...

    Uses a semicolon-separated CSV with columns:

      image_id;myelinisation;age;age_corrected;doctor_predicted_age;diagnosis;group;diagnosis_clean

    and uses doctor_predicted_age (months) as conditioning.

    Returns:
      t1:  (1,H,W) in [-1,1]
      t2:  (1,H,W) in [-1,1]
      age: (1,) normalized to [0,1]
      (subj_idx, z)
    """

    def __init__(
        self,
        root_split_dir: str,
        age_csv: str,
        image_size: Optional[int] = None,
        min_nonzero_fraction: float = 0.0,
        crop_to_brain: bool = True,
    ):
        self.root = Path(root_split_dir)
        self.image_size = image_size
        self.min_nonzero_fraction = min_nonzero_fraction
        self.crop_to_brain = crop_to_brain

        # --- load ages ---
        self.age_map: Dict[str, float] = self._load_age_csv(age_csv)

        self.subjects: List[Path] = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        if not self.subjects:
            raise RuntimeError(f"No subject dirs found in {self.root}")

        self.nifti_imgs: List[dict] = []
        self.volume_stats: List[dict] = []
        self.subj_ages: List[float] = []
        self.index: List[Tuple[int, int]] = []  # (subj_idx, z)

        print(f"[Dataset] Scanning subjects in {self.root} ...")
        for subj_dir in self.subjects:
            subj_id = subj_dir.name  # e.g. "s0001"
            if subj_id not in self.age_map:
                print(f"  Warning: {subj_id} missing in age CSV, skipping.")
                continue
            age_months = float(self.age_map[subj_id])

            t1_path = subj_dir / "t1.nii.gz"
            t2_path = subj_dir / "t2.nii.gz"
            if not t1_path.exists() or not t2_path.exists():
                print(f"  Warning: missing T1 or T2 in {subj_dir}, skipping.")
                continue

            t1_img = nib.load(str(t1_path))
            t2_img = nib.load(str(t2_path))
            if t1_img.shape != t2_img.shape:
                print(f"  Warning: T1/T2 shapes differ in {subj_dir}, skipping.")
                continue
            if len(t1_img.shape) != 3:
                print(f"  Warning: expected 3D volume in {subj_dir}, skipping.")
                continue

            # Load for bbox + stats
            t1_vol = t1_img.get_fdata(dtype=np.float32)
            t2_vol = t2_img.get_fdata(dtype=np.float32)
            H, W, Z = t2_vol.shape

            brain_mask = t2_vol > 0
            if self.crop_to_brain and brain_mask.any():
                ys, xs, zs = np.where(brain_mask)
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
            else:
                y_min, y_max = 0, H - 1
                x_min, x_max = 0, W - 1

            if brain_mask.any():
                t1_vals = t1_vol[brain_mask]
                t2_vals = t2_vol[brain_mask]
            else:
                t1_vals = t1_vol.reshape(-1)
                t2_vals = t2_vol.reshape(-1)

            t1_vmin, t1_vmax = np.percentile(t1_vals, [1, 99])
            t2_vmin, t2_vmax = np.percentile(t2_vals, [1, 99])

            if t1_vmax <= t1_vmin:
                t1_vmax = t1_vmin + 1e-6
            if t2_vmax <= t2_vmin:
                t2_vmax = t2_vmin + 1e-6

            self.nifti_imgs.append({"t1": t1_img, "t2": t2_img})
            self.volume_stats.append(
                {
                    "t1": {"vmin": float(t1_vmin), "vmax": float(t1_vmax)},
                    "t2": {"vmin": float(t2_vmin), "vmax": float(t2_vmax)},
                    "bbox": (int(y_min), int(y_max), int(x_min), int(x_max)),
                    "shape": (H, W, Z),
                }
            )
            self.subj_ages.append(age_months)
            subj_idx = len(self.nifti_imgs) - 1

            for z in range(Z):
                slice_np = t2_vol[:, :, z]
                nonzero_fraction = np.count_nonzero(slice_np) / slice_np.size
                if nonzero_fraction >= self.min_nonzero_fraction:
                    self.index.append((subj_idx, z))

        if not self.index:
            raise RuntimeError("No valid slices found; check min_nonzero_fraction or data.")

        self.age_min = float(min(self.subj_ages))
        self.age_max = float(max(self.subj_ages))

        print(
            f"[Dataset] Subjects: {len(self.nifti_imgs)}, "
            f"slices: {len(self.index)}"
        )
        print(
            f"[Dataset] Age range (doctor_predicted_age, months): "
            f"[{self.age_min:.2f}, {self.age_max:.2f}]"
        )

    @staticmethod
    def _load_age_csv(age_csv: str) -> Dict[str, float]:
        age_map: Dict[str, float] = {}
        with open(age_csv, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                sid = row["image_id"]
                age_months = float(row["doctor_predicted_age"])
                age_map[sid] = age_months
        return age_map

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _normalize_with_stats(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, vmin, vmax)
        x = (x - vmin) / (vmax - vmin)
        x = x * 2.0 - 1.0
        return x

    @staticmethod
    def _resize_tensor(x: torch.Tensor, size: Optional[int]) -> torch.Tensor:
        if size is None:
            return x
        if x.shape[-2] == size and x.shape[-1] == size:
            return x
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)
        return x

    def __getitem__(self, idx: int):
        subj_idx, z = self.index[idx]
        imgs = self.nifti_imgs[subj_idx]
        stats = self.volume_stats[subj_idx]
        age_months = self.subj_ages[subj_idx]

        t1_img = imgs["t1"]
        t2_img = imgs["t2"]
        y_min, y_max, x_min, x_max = stats["bbox"]

        t1_slice = np.asarray(t1_img.dataobj[:, :, z], dtype=np.float32)
        t2_slice = np.asarray(t2_img.dataobj[:, :, z], dtype=np.float32)

        t1_slice = t1_slice[y_min : y_max + 1, x_min : x_max + 1]
        t2_slice = t2_slice[y_min : y_max + 1, x_min : x_max + 1]

        t1_slice = self._normalize_with_stats(
            t1_slice, stats["t1"]["vmin"], stats["t1"]["vmax"]
        )
        t2_slice = self._normalize_with_stats(
            t2_slice, stats["t2"]["vmin"], stats["t2"]["vmax"]
        )

        t1 = torch.from_numpy(t1_slice).unsqueeze(0)
        t2 = torch.from_numpy(t2_slice).unsqueeze(0)

        t1 = self._resize_tensor(t1, self.image_size)
        t2 = self._resize_tensor(t2, self.image_size)

        age_norm = (age_months - self.age_min) / (self.age_max - self.age_min + 1e-8)
        age = torch.tensor([age_norm], dtype=torch.float32)

        return t1, t2, age, (subj_idx, z)


# =========================
# Diffusion config (same as training)
# =========================

class DiffusionConfig:
    def __init__(self, T=400, beta_start=1e-4, beta_end=0.02, device="cpu"):
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
    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        self.cond_mlp = None
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, out_ch),
                nn.ReLU(inplace=True),
            )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, cond_emb=None):
        h = self.conv1(x)
        if self.cond_mlp is not None and cond_emb is not None:
            ce = self.cond_mlp(cond_emb)
            h = h + ce[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        return h


class UNetConditional(nn.Module):
    """
    Same as training:

      input:  (B, 2, H, W) [noisy_T2, T1]
      cond:   timestep t (B,) and age (B,1)
      output: (B, 1, H, W) predicted noise
    """

    def __init__(
        self,
        in_ch=2,
        base_ch=32,
        time_emb_dim=128,
        age_emb_dim=32,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(age_emb_dim, time_emb_dim),
        )

        cond_dim = time_emb_dim

        self.down1 = DoubleConv(in_ch, base_ch, cond_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_ch, base_ch * 2, cond_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_ch * 2, base_ch * 4, cond_dim)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8, cond_dim)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, cond_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, cond_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, cond_dim)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t, age):
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        if age.dim() == 1:
            age = age.unsqueeze(1)
        age_emb = self.age_mlp(age)

        cond_emb = t_emb + age_emb

        x1 = self.down1(x, cond_emb)
        x2 = self.down2(self.pool1(x1), cond_emb)
        x3 = self.down3(self.pool2(x2), cond_emb)

        xb = self.bottleneck(self.pool3(x3), cond_emb)

        x_up = self.up3(xb)
        x_up = torch.cat([x3, x_up], dim=1)
        x_up = self.dec3(x_up, cond_emb)

        x_up = self.up2(x_up)
        x_up = torch.cat([x2, x_up], dim=1)
        x_up = self.dec2(x_up, cond_emb)

        x_up = self.up1(x_up)
        x_up = torch.cat([x1, x_up], dim=1)
        x_up = self.dec1(x_up, cond_emb)

        out = self.out_conv(x_up)
        return out


# =========================
# Sampling: DDPM T2 | T1, age
# =========================

@torch.no_grad()
def sample_t2_from_t1_age(
    model: UNetConditional,
    diffusion: DiffusionConfig,
    t1: torch.Tensor,
    age: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Reverse diffusion to sample T2_hat given T1 and age.

    t1:  (B,1,H,W) in [-1,1]
    age: (B,1) normalized [0,1]
    returns: T2_hat (B,1,H,W) in [-1,1]
    """
    model.eval()
    B, _, H, W = t1.shape
    x = torch.randn(B, 1, H, W, device=device)
    T = diffusion.T

    betas = diffusion.betas
    alphas = diffusion.alphas
    alphas_cumprod = diffusion.alphas_cumprod

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        model_in = torch.cat([x, t1], dim=1)
        eps_theta = model(model_in, t_batch, age)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

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


def to_uint8(x: np.ndarray) -> np.ndarray:
    x = (x + 1.0) * 0.5
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


# =========================
# Main eval + trajectory
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Config] device      =", device)
    print("[Config] test_root   =", TEST_ROOT)
    print("[Config] age_csv     =", AGE_CSV)
    print("[Config] ckpt        =", CKPT_PATH)
    print("[Config] image_size  =", IMAGE_SIZE)
    print("[Config] batch_size  =", BATCH_SIZE)
    print("[Config] out_dir     =", OUT_DIR)

    print("[Stage] Building test dataset...")
    dataset = PairedNiftiSliceDataset(
        TEST_ROOT,
        age_csv=AGE_CSV,
        image_size=IMAGE_SIZE,
        min_nonzero_fraction=0.0,
        crop_to_brain=True,
    )
    print("[Stage] Test dataset ready.")

    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    print("[Stage] Test dataloader ready.")

    diffusion = DiffusionConfig(T=400, beta_start=1e-4, beta_end=0.02, device=device)
    model = UNetConditional(in_ch=2, base_ch=32, time_emb_dim=128, age_emb_dim=32).to(device)

    print(f"[Stage] Loading checkpoint from {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("[Stage] Checkpoint loaded.")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    mae_total = 0.0
    mse_total = 0.0
    n_pixels = 0
    saved = 0

    print("[Stage] Starting evaluation sampling (at true ages)...")
    with torch.no_grad():
        for batch_idx, (t1, t2, age, idx_info) in enumerate(
            tqdm(test_loader, desc="Testing")
        ):
            t1 = t1.to(device)
            t2 = t2.to(device)
            age = age.to(device)

            t2_hat = sample_t2_from_t1_age(model, diffusion, t1, age, device=device)

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

                real_uint8 = to_uint8(img_real)
                fake_uint8 = to_uint8(img_fake)

                Image.fromarray(real_uint8).save(out_dir / f"real_{saved:04d}.png")
                Image.fromarray(fake_uint8).save(out_dir / f"fake_{saved:04d}.png")

                saved += 1

    mae_mean = mae_total / float(n_pixels)
    mse_mean = mse_total / float(n_pixels)

    print("\n=== Evaluation results (true ages) ===")
    print(f"Mean MAE (per-pixel in [-1,1] space): {mae_mean:.6f}")
    print(f"Mean MSE (per-pixel in [-1,1] space): {mse_mean:.6f}")
    print(f"Saved {saved} example slice pairs to {out_dir}")

    # =========================
    # Trajectory visualization on a single example
    # =========================
    print("\n[Stage] Generating age trajectory for a single slice...")
    # take first slice from dataset
    t1_single, t2_single, age_single, info = dataset[0]
    t1_single = t1_single.unsqueeze(0).to(device)  # (1,1,H,W)

    age_min = dataset.age_min
    age_max = dataset.age_max

    gen_imgs = []
    for age_m in TRAJECTORY_AGES_MONTHS:
        age_norm = (age_m - age_min) / (age_max - age_min + 1e-8)
        age_tensor = torch.tensor([[age_norm]], dtype=torch.float32, device=device)  # (1,1)

        t2_hat = sample_t2_from_t1_age(model, diffusion, t1_single, age_tensor, device)
        img = t2_hat[0, 0].detach().cpu().numpy()
        gen_imgs.append(to_uint8(img))

    # also include the real T2 at the subject's doctor_predicted_age as first image
    real_uint8 = to_uint8(t2_single.numpy())
    all_imgs = [real_uint8] + gen_imgs

    # concatenate horizontally
    H, W = all_imgs[0].shape
    grid = np.zeros((H, W * len(all_imgs)), dtype=np.uint8)
    for i, im in enumerate(all_imgs):
        grid[:, i * W : (i + 1) * W] = im

    traj_path = Path(TRAJECTORY_OUT)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(traj_path)
    print(f"[Stage] Saved trajectory image to {traj_path}")
    print("Leftmost = real T2 at true age; others = generated T2 at ages (months):")
    print(TRAJECTORY_AGES_MONTHS)


if __name__ == "__main__":
    main()
