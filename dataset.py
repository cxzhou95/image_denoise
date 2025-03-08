import os
import cv2
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def augmentation(img, crop_size):  # custom augmentation function (random crop, flips, rotations)
    h, w = img.shape[:2]
    # check the crop size
    assert crop_size <= h and crop_size <= w, 'crop size is too large'

    y1 = random.randint(0, h - crop_size)
    x1 = random.randint(0, w - crop_size)
    img = img[y1:y1 + crop_size, x1:x1 + crop_size]

    # geometric transformation
    if random.random() < 0.5:  # hflip
        img = img[:, ::-1]
    if random.random() < 0.5:  # vflip
        img = img[::-1, :]
    if random.random() < 0.5:  # rot90
        # img = img.transpose(1, 0, 2) # use this if the input is not the grayscale image
        img = img.transpose(1, 0)

    return np.ascontiguousarray(img)


def add_gaussian_noise(img, mean, std):
    noise = np.random.normal(mean, std, img.shape)
    noisy_image = img + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


class DenoisingDataset(Dataset):
    def __init__(self, image_dir, patch_size=40, mean=0, sigma=25, num_patches=1600):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
        self.patch_size = patch_size
        self.mean = mean / 255.0
        self.sigma = sigma / 255.0  # normalize noise level
        self.num_patches = num_patches  # each image will generate multiple patches

    def __len__(self):
        return len(self.image_paths) * self.num_patches

    def __getitem__(self, index):
        # load and preprocess image
        img_index = index % len(self.image_paths)  # ensure index loops over images
        img_path = self.image_paths[img_index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0  # normalize
        # apply augmentations
        patch = augmentation(img, self.patch_size)  # extract patch

        # add Gaussian noise
        noisy_patch = add_gaussian_noise(patch, self.mean, self.sigma)

        # patch shape is [H,W], need convert to [C,H,W]
        noisy_patch = torch.from_numpy(np.expand_dims(noisy_patch, axis=0)).float()
        patch = torch.from_numpy(np.expand_dims(patch, axis=0)).float()

        return noisy_patch, patch  # return (noisy image, clean image)