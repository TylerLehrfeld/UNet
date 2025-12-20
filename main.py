import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    
    images = np.stack([
        np.transpose(item["image"], (2, 0, 1))  # (H,W,C) → (C,H,W)
        for item in tumor 
    ]).astype(np.float32)                        # final shape [N, 3, 256, 256]
    images = images / 255.0
    mean = images.mean(axis=(0, 2, 3), keepdims=True)  # per-channel mean
    std = images.std(axis=(0, 2, 3), keepdims=True)    # per-channel std
    images = (images - mean) / (std + 1e-7)
    masks = np.stack([
        item["mask"]                             # (H,W)
        for item in tumor 
    ]).astype(np.float32)    
    if masks.max() > 1.0:
        masks = masks / 255.0
    # final shape [N, 256, 256]
    images = np.ascontiguousarray(images.copy())
    masks = np.ascontiguousarray(masks.copy())

    N, C, H, W = images.shape
    shape = [float(N), float(C), float(H), float(W)]

# Split 80% train / 20% test
train_images, test_images, train_masks, test_masks = train_test_split(
    images, masks, test_size=0.15, random_state=42, shuffle=True
)
train_masks = np.ascontiguousarray(train_masks)
train_images = np.ascontiguousarray(train_images)

print(f"Train images: {train_images.shape[0]}")
print(f"Test images: {test_images.shape[0]}")

from neural_net import NeuralNet
from neural_net import LayerType
from neural_net import LayerDesc
from neural_net import ActivationType

UNET = NeuralNet()  # instantiate the C++ class

# assume strides and sizes tile image correctly without padding
max_pool_stride = 2
max_pool_size = 2
H1 = 256
H2 = int(H1 / max_pool_stride)
H3 = int(H2 / max_pool_stride)
H4 = int(H3 / max_pool_stride)

W1 = 256
W2 = int(W1 / max_pool_stride)
W3 = int(W2 / max_pool_stride)
W4 = int(W3 / max_pool_stride)
in_channels = 3
F1 = 64
F2 = int(2 * F1)
F3 = int(2 * F2)
F4 = int(2 * F3)

# keep encoder and decoder even
upsample_factor = max_pool_stride

UNET.create(
    [
        # parameters are in_height, in_width, in_channels, out_channels, kernel size, padding, and stride
        LayerDesc(LayerType.CONV_LAYER, [H1, W1, in_channels, F1, 3, 1, 1], []),  # 0
        # parameters are in_channels, in_height, in_width, size and stride
        LayerDesc(
            LayerType.MAX_POOL_LAYER, [F1, H1, W1, max_pool_size, max_pool_stride], [0]
        ),  # 1
        LayerDesc(LayerType.CONV_LAYER, [H2, W2, F1, F2, 3, 1, 1], [1]),  # 2
        LayerDesc(
            LayerType.MAX_POOL_LAYER, [F2, H2, W2, max_pool_size, max_pool_stride], [2]
        ),  # 3
        LayerDesc(LayerType.CONV_LAYER, [H3, W3, F2, F3, 3, 1, 1], [3]),  # 4
        LayerDesc(
            LayerType.MAX_POOL_LAYER, [F3, H3, W3, max_pool_size, max_pool_stride], [4]
        ),  # 5
        LayerDesc(LayerType.CONV_LAYER, [H4, W4, F3, F4, 3, 1, 1], [5]),  # 6
        # parameters are h_in, w_in, c_in, c_out, stride/upscale
        LayerDesc(
            LayerType.UPSAMPLING_LAYER, [H4, W4, F4, F3, upsample_factor], [6]
        ),  # 7
        # parameters are fully determined by parents. First parent is skip connection, second is convolution below
        LayerDesc(LayerType.ATTENTION_LAYER, [], [4, 6]),  # 8
        # parameters are fully determined by parents. First parent is from skip/attention, second is from convolution below
        LayerDesc(LayerType.CONCAT_LAYER, [], [8, 7]),  # 9
        LayerDesc(LayerType.CONV_LAYER, [H3, W3, F4, F2, 3, 1, 1], [9]),  # 10
        LayerDesc(
            LayerType.UPSAMPLING_LAYER, [H3, W3, F2, F2, upsample_factor], [10]
        ),  # 11
        LayerDesc(LayerType.ATTENTION_LAYER, [], [2, 10]),  # 12
        LayerDesc(LayerType.CONCAT_LAYER, [], [12, 11]),  # 13
        LayerDesc(LayerType.CONV_LAYER, [H2, W2, F3, F1, 3, 1, 1], [13]),  # 14
        LayerDesc(
            LayerType.UPSAMPLING_LAYER, [H2, W2, F1, F1, upsample_factor], [14]
        ),  # 15
        LayerDesc(LayerType.ATTENTION_LAYER, [], [0, 14]),  # 16
        LayerDesc(LayerType.CONCAT_LAYER, [], [16, 15]),  # 17
        LayerDesc(LayerType.CONV_LAYER, [H1, W1, F2, 1, 3, 1, 1], [17], ActivationType.SIGMOID),  # 18
        
    ]
)


learning_rate = 0.001
epochs = 30
batch_size = 4

UNET.train(train_images, train_masks, learning_rate, epochs, batch_size)
