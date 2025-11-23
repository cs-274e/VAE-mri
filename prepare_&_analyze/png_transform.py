import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def nii_to_png(input_path, output_path):
    """
    Convert a NIfTI (.nii or .nii.gz) file into PNG slices using SimpleITK.
    """
    img = sitk.ReadImage(input_path)
    data = sitk.GetArrayFromImage(img)  # shape: [slices, H, W]

    # Normalize to 0–255
    data = data.astype(np.float32)
    data -= data.min()
    if data.max() > 0:
        data /= data.max()
    data = (data * 255).astype(np.uint8)

    # Make output directory
    os.makedirs(output_path, exist_ok=True)

    # Save slices
    for i, slice_img in enumerate(data):
        out_file = os.path.join(output_path, f"slice_{i:04d}.png")
        plt.imsave(out_file, slice_img, cmap="gray")

    print(f"Saved {len(data)} slices → {output_path}")


def convert_dataset_to_png(dataset_folder="dataset"):
    output_root = dataset_folder + "_png"

    for subject in os.listdir(dataset_folder):
        subject_path = os.path.join(dataset_folder, subject)

        if not os.path.isdir(subject_path):
            continue

        print(f"Processing subject: {subject}")

        subject_output = os.path.join(output_root, subject)

        for file in os.listdir(subject_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                input_file = os.path.join(subject_path, file)

                file_name = file.replace(".nii.gz", "").replace(".nii", "")
                file_output = os.path.join(subject_output, file_name)

                nii_to_png(input_file, file_output)

    print("\nAll conversions completed!")
    print(f"PNG dataset saved in: {output_root}")


if __name__ == "__main__":
    convert_dataset_to_png("dataset")

