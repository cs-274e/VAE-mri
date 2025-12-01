import os
import shutil

# Base dataset directory
base_dir = "dataset"

# Target directories
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through folders s0001 to s0833
for i in range(1, 834):  # inclusive of 833
    folder_name = f"s{i:04d}"  # formats as s0001, s0002, ...
    src_path = os.path.join(base_dir, folder_name)

    if not os.path.exists(src_path):
        continue  # skip if folder doesn't exist

    # Decide destination
    if i <= 710:
        dest_path = os.path.join(train_dir, folder_name)
    else:
        dest_path = os.path.join(test_dir, folder_name)

    # Move folder
    shutil.move(src_path, dest_path)
    print(f"Moved {folder_name} → {dest_path}")
