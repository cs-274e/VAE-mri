import os
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image


def load_png_as_float(path: Path) -> np.ndarray:
    img = Image.open(path).convert("F")  # 32-bit float grayscale
    arr = np.array(img, dtype=np.float32)
    return arr

def main():
    root = Path("mri_slice_images/train")

    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}")

    t2_paths = sorted(root.glob("s*/t2_*.png"))
    if not t2_paths:
        raise SystemExit(f"No t2_*.png files found under {root}")

    print(f"Found {len(t2_paths)} T2 slices")

    global_min = float("inf")
    global_max = float("-inf")

    num_with_zero_min = 0
    num_without_zero_min = 0

    # collect some of the smallest non-zero values observed
    small_nonzero_values = []

    for p in t2_paths:
        arr = load_png_as_float(p)

        img_min = float(arr.min())
        img_max = float(arr.max())

        global_min = min(global_min, img_min)
        global_max = max(global_max, img_max)

        if img_min == 0.0:
            num_with_zero_min += 1
        else:
            num_without_zero_min += 1

        # collect a few smallest non-zero values for this image
        flat = arr.ravel()
        nonzero = flat[flat != 0]
        if nonzero.size > 0:
            # take up to 5 smallest non-zero values
            k = min(5, nonzero.size)
            smallest = np.partition(nonzero, k - 1)[:k]
            small_nonzero_values.extend(smallest.tolist())

    print("\n=== Global stats over all T2 slices ===")
    print(f"Global min value: {global_min}")
    print(f"Global max value: {global_max}")
    print(f"Slices with min == 0.0: {num_with_zero_min}")
    print(f"Slices with min > 0.0:  {num_without_zero_min}")

    if small_nonzero_values:
        # round to see mode-ish small values
        rounded = [round(v, 3) for v in small_nonzero_values]
        counts = Counter(rounded)
        most_common = counts.most_common(10)

        print("\nSmallest non-zero values observed (rounded, top 10):")
        for val, cnt in most_common:
            print(f"  value {val} : {cnt} occurrences")

        # also show the absolute smallest few
        unique_sorted = sorted(set(rounded))[:10]
        print("\nAbsolutely smallest distinct non-zero values (rounded):")
        for v in unique_sorted:
            print(f"  {v}")
    else:
        print("\nNo non-zero pixels found at all (this would be very strange).")


if __name__ == "__main__":
    main()

# $ python preparing_data/check_background.py
# Found 27333 T2 slices

# === Global stats over all T2 slices ===
# Global min value: 0.0
# Global max value: 255.0
# Slices with min == 0.0: 27333
# Slices with min > 0.0:  0

# Smallest non-zero values observed (rounded, top 10):
#   value 1.0 : 108312 occurrences
#   value 2.0 : 8234 occurrences
#   value 3.0 : 3971 occurrences
#   value 4.0 : 2518 occurrences
#   value 5.0 : 1631 occurrences
#   value 6.0 : 1306 occurrences
#   value 7.0 : 1043 occurrences
#   value 8.0 : 840 occurrences
#   value 9.0 : 705 occurrences
#   value 10.0 : 598 occurrences

# Absolutely smallest distinct non-zero values (rounded):
#   1.0
#   2.0
#   3.0
#   4.0
#   5.0
#   6.0
#   7.0
#   8.0
#   9.0
#   10.0