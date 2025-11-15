import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(root_dir):
    """
    Recursively parse the LGG MRI dataset.
    Returns a list of dictionaries:
    [
      {
        "image_path": "...",
        "mask_path": "...",
        "image": np.ndarray,
        "mask": np.ndarray
      },
      ...
    ]
    """
    data = []

    # Walk through all subdirectories
    for subdir, dirs, files in os.walk(root_dir):
        # Only consider tif files
        tifs = [f for f in files if f.endswith(".tif")]

        # Create mapping: index -> file
        image_files = {f.replace("_mask", ""): f for f in tifs if "_mask" not in f}
        mask_files  = {f.replace("_mask", ""): f for f in tifs if "_mask" in f}

        # Pair image + mask files
        for key in sorted(image_files.keys()):
            if key in mask_files:
                img_path = os.path.join(subdir, image_files[key])
                mask_path = os.path.join(subdir, mask_files[key])

                # Load arrays
                img = np.array(Image.open(img_path))
                mask = np.array(Image.open(mask_path))

                data.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "image": img,
                    "mask": mask
                })

    return data


if __name__ == "__main__":
    root = "/home/tyler/projects/UNet/data/lgg-mri-segmentation/kaggle_3m"
    dataset = load_dataset(root)

    print(f"Loaded {len(dataset)} image/mask pairs.\n")

    # Example: print one sample
    sample = dataset[0]
    print("Image:", sample["image_path"])
    print("Mask:", sample["mask_path"])
    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)

    # ---- Split into tumor / non-tumor ----
    tumor = []
    no_tumor = []
    
    for item in dataset:
        if np.any(item["mask"] > 0):
            tumor.append(item)
        else:
            no_tumor.append(item)
    
    print("Tumor images:", len(tumor))
    print("No-tumor images:", len(no_tumor))
    
    # ---- Pick examples ----
    example_tumor = tumor[0]
    example_no_tumor = no_tumor[0]
    
    # ---- Visualize ----
    plt.figure(figsize=(10, 7))
    
    plt.subplot(2, 2, 1)
    plt.title("No Tumor – Image")
    plt.imshow(example_no_tumor["image"], cmap="gray")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.title("No Tumor – Mask")
    plt.imshow(example_no_tumor["mask"], cmap="gray")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.title("Tumor – Image")
    plt.imshow(example_tumor["image"], cmap="gray")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.title("Tumor – Mask")
    plt.imshow(example_tumor["mask"], cmap="gray")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
