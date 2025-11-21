import os
import glob
import numpy as np
import nibabel as nib
import cv2  # pip install opencv-python
from tqdm import tqdm


# -----------------------
# CONFIG
# -----------------------
DATA_ROOT = "data"            # root folder with s0001, s0002, ...
OUT_T1 = "mid_slices_T1"      # output folder for T1 mid slices
OUT_T2 = "mid_slices_T2"      # output folder for T2 mid slices
DOWNSCALE_TO = (128, 128)     # (width, height) or None to keep original size


# -----------------------
# HELPERS
# -----------------------
def normalize_volume(vol):
    """Normalize a 3D volume to [0, 1] using 1st/99th percentile clipping."""
    vol = vol.astype(np.float32)
    vmin = np.percentile(vol, 1)
    vmax = np.percentile(vol, 99)
    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return vol


def save_slice(img2d, out_path):
    """Save a 2D float32 slice in [0,1] as a PNG."""
    img = img2d
    if DOWNSCALE_TO is not None:
        img = cv2.resize(img, DOWNSCALE_TO, interpolation=cv2.INTER_AREA)
    img_uint8 = (img * 255).astype(np.uint8)
    cv2.imwrite(out_path, img_uint8)


# -----------------------
# MAIN
# -----------------------
def main():
    os.makedirs(OUT_T1, exist_ok=True)
    os.makedirs(OUT_T2, exist_ok=True)

    subjects = sorted(glob.glob(os.path.join(DATA_ROOT, "s*")))
    print(f"Found {len(subjects)} subjects.")

    for subj_path in tqdm(subjects):
        subj_name = os.path.basename(subj_path)

        t1_path = os.path.join(subj_path, "t1.nii.gz")
        t2_path = os.path.join(subj_path, "t2.nii.gz")

        if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
            print(f"Skipping {subj_name}: missing t1.nii.gz or t2.nii.gz.")
            continue

        # Load & normalize volumes
        t1 = normalize_volume(nib.load(t1_path).get_fdata())
        t2 = normalize_volume(nib.load(t2_path).get_fdata())

        if t1.shape != t2.shape:
            print(f"Skipping {subj_name}: shape mismatch.")
            continue

        H, W, D = t1.shape
        mid = D // 2

        t1_slice = t1[:, :, mid]
        t2_slice = t2[:, :, mid]

        # Skip subjects whose mid slice is empty
        if np.max(t1_slice) < 0.05 and np.max(t2_slice) < 0.05:
            print(f"Skipping {subj_name}: mid slice empty.")
            continue

        # Save T1 & T2 slices separately
        out_t1 = os.path.join(OUT_T1, f"{subj_name}_mid_T1.png")
        out_t2 = os.path.join(OUT_T2, f"{subj_name}_mid_T2.png")

        save_slice(t1_slice, out_t1)
        save_slice(t2_slice, out_t2)

    print("Done!")
    print(f"T1 mid slices → {OUT_T1}")
    print(f"T2 mid slices → {OUT_T2}")


if __name__ == "__main__":
    main()
