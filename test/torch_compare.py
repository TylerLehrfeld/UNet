import torch
import torch.nn as nn
import torch.nn.functional as F
import neural_net
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage import convolve


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
        mask_files = {f.replace("_mask", ""): f for f in tifs if "_mask" in f}

        # Pair image + mask files
        for key in sorted(image_files.keys()):
            if key in mask_files:
                img_path = os.path.join(subdir, image_files[key])
                mask_path = os.path.join(subdir, mask_files[key])

                # Load arrays
                img = np.array(Image.open(img_path))
                mask = np.array(Image.open(mask_path))

                data.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "image": img,
                        "mask": mask,
                    }
                )

    return data


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

images = np.stack(
    [np.transpose(item["image"], (2, 0, 1)) for item in tumor]  # (H,W,C) → (C,H,W)
).astype(
    np.float32
)  # final shape [N, 3, 256, 256]
images = images / 255.0
mean = images.mean(axis=(0, 2, 3), keepdims=True)  # per-channel mean
std = images.std(axis=(0, 2, 3), keepdims=True)  # per-channel std
images = (images - mean) / (std + 1e-7)
masks = np.stack([item["mask"] for item in tumor]).astype(np.float32)  # (H,W)
if masks.max() > 1.0:
    masks = masks / 255.0
# final shape [N, 256, 256]
images = np.ascontiguousarray(images.copy())
masks = np.ascontiguousarray(masks.copy())
N, C, H, W = images.shape
shape = [float(N), float(C), float(H), float(W)]


# Small test image
image = torch.tensor(images[0][:1], dtype=torch.float32).reshape(1,1,256,256)
kernel = torch.tensor([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
bias = torch.tensor([0], dtype=torch.float32)

target = F.conv2d(image, kernel, bias=bias, stride=1,padding=1)
# Kernel and bias
kernel = torch.tensor([[0.0749, 0.4010, 0.0577],
                       [-0.1607, 0.1313, -0.1326],
                       [0.2074, 0.4494, 0.2920]], dtype=torch.float32).reshape(1, 1, 3, 3)
bias = torch.tensor([0.1970], dtype=torch.float32)

# Make them trainable
kernel = kernel.clone().requires_grad_(True)
bias = bias.clone().requires_grad_(True)


optimizer = torch.optim.Adam([kernel, bias], lr=0.35)

# Compute start loss
with torch.no_grad():
    start_output = F.conv2d(image, kernel, bias=bias, stride=1, padding=1)
    start_loss = F.mse_loss(start_output, target, reduction='sum') * 0.5
print(f"Start loss: {start_loss.item()}")

for t in range(100):
    optimizer.zero_grad()

    # Forward pass
    output = F.conv2d(image, kernel, bias=bias, stride=1, padding=1)
    if torch.isnan(output).any():
        print(f"NaN in output at iteration {t}")
        break

    
    # Loss
    loss = F.mse_loss(output, target, reduction='sum') * 0.5

    # Backward pass
    loss.backward()

    
    optimizer.step()

print(f"Final loss: {loss.item()}")
