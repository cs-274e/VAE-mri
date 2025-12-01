import os
import shutil

def move_s_folders(base_path="."):
    # Path to the output folder
    dataset_path = os.path.join(base_path, "dataset")

    # Create "dataset" folder if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)

    # List all items in the base directory
    items = os.listdir(base_path)

    # Loop through all items and move folders starting with "s"
    for item in items:
        item_path = os.path.join(base_path, item)

        # Check if it's a folder and starts with "s" (case-insensitive)
        if os.path.isdir(item_path) and item.lower().startswith("s"):
            target_path = os.path.join(dataset_path, item)
            
            print(f"Moving: {item} → dataset/")
            shutil.move(item_path, target_path)

    print("Done! All 's*' folders have been moved to 'dataset'.")


if __name__ == "__main__":
    move_s_folders("/baldig/bioprojects2/uxue/VAE-mri")   # you can replace "." with a specific path
