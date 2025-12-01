import os

def count_images_in_split(split_path):
    """
    Count folders and t1/t2 images inside a split folder.
    """
    results = {}
    total_images = 0

    # list all subfolders (subjects)
    subject_folders = [
        f for f in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, f))
    ]

    for folder in subject_folders:
        folder_path = os.path.join(split_path, folder)
        t1_count = 0
        t2_count = 0

        # list all files inside each subject folder
        for file in os.listdir(folder_path):
            if file.lower().startswith('t1'):
                t1_count += 1
            elif file.lower().startswith('t2'):
                t2_count += 1

        folder_total = t1_count + t2_count
        total_images += folder_total

        results[folder] = {
            "t1": t1_count,
            "t2": t2_count,
            "total": folder_total
        }

    return results, len(subject_folders), total_images


def generate_report(base_path="mri_slice_images", output_file="report.txt"):
    splits = ["train", "test"]
    report_lines = []

    for split in splits:
        split_path = os.path.join(base_path, split)

        if not os.path.exists(split_path):
            report_lines.append(f"Split '{split}' not found.\n")
            continue

        results, num_folders, total_images = count_images_in_split(split_path)

        report_lines.append(f"=== {split.upper()} SPLIT ===")
        report_lines.append(f"Number of subject folders: {num_folders}\n")

        for folder, counts in results.items():
            report_lines.append(
                f"{folder}: t1={counts['t1']}, "
                f"t2={counts['t2']}, total={counts['total']}"
            )

        report_lines.append(f"\nTOTAL IMAGES in {split}: {total_images}\n")
        report_lines.append("=" * 40 + "\n")

    # write to txt file
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved as {output_file}")


if __name__ == "__main__":
    generate_report()
