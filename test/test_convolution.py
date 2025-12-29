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


def visualizeImage(image):
    num_channels = image.shape[0]
    if num_channels > 1:
        fig, axes = plt.subplots(1, num_channels)
        for i in range(num_channels):
            axes[i].imshow(image[i, :, :], cmap="gray")
            axes[i].set_title(f"Channel {i+1}")
        plt.show()
    else:
        fig, axes = plt.subplots(1)
        axes.imshow(image[0, :, :], cmap="gray")
        axes.set_title(f"Channel {1}")
        plt.show()

def mse_loss(y, y_hat):
    """
    y, y_hat: numpy arrays of same shape
    returns scalar loss
    """
    diff = y_hat - y
    return 0.5 * np.sum(diff * diff)

# get original image
image = images[0]
#image = np.random.randn(256, 256).astype(np.float32)
image = np.reshape(image[:1,:,:], (1, 256, 256))

# visualize image
# visualizeImage(image[:1, :, :])
# transform image using convolution

from neural_net import NeuralNet
from neural_net import LayerType
from neural_net import LayerDesc
from neural_net import ActivationType
from neural_net import Convolution

Convolution2D = Convolution()

# get kernel
kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel = np.reshape(kernel, (1, 1, 3, 3))

bias = np.array([0])

# apply transform
convolved_image = Convolution2D.convolution2D(image[:1, :, :], kernel, bias, 1, 1)
convolved_image = np.reshape(convolved_image, (1,256,256))
# visualize image
# visualizeImage(convolved_image)

# train


# kernel = np.reshape(kernel, (1, 1, 3, 3))
# bias = np.array([0])
# convolved_image = Convolution2D.convolution2D(image[:1, :, :], kernel, bias, 1, 1)

# convolved_image = np.reshape(convolved_image, (1, 256, 256))

kernel = np.array([[0.0749, 0.4010, 0.0577],
                       [-0.1607, 0.1313, -0.1326],
                       [0.2074, 0.4494, 0.2920]])
bias = np.array([0.1970])
kernel = np.reshape(kernel, (1, 1, 3, 3))

old_convolved_image = Convolution2D.convolution2D(image[:1, :, :], kernel, bias, 1, 1)
old_convolved_image = np.reshape(old_convolved_image, (1, 256, 256))
lr = .35
weights = Convolution2D.train(image[:1, :, :], convolved_image, kernel, bias, 1, 1,lr)
print(weights)
new_kernel = np.array(
    [
        [weights[0], weights[1], weights[2]],
        [weights[3], weights[4], weights[5]],
        [weights[6], weights[7], weights[8]],
    ]
)
new_kernel = np.reshape(new_kernel, (1, 1, 3, 3))
new_convolved_image = Convolution2D.convolution2D(
    image[:1, :, :], new_kernel, [weights[9]], 1, 1
)
new_convolved_image = np.reshape(new_convolved_image, (1, 256, 256))
# visualizeImage(new_convolved_image)
print(f"original loss: {mse_loss(convolved_image, old_convolved_image)}")
print(f"final loss: {mse_loss(convolved_image, new_convolved_image)}, LR: {lr}")
