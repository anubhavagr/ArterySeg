import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from config import SHAPE

# Headings: Albumentations Transform
album_transform = A.Compose([
    A.RandomResizedCrop(height=SHAPE[0], width=SHAPE[1], always_apply=True, p=0.4),
    A.Flip(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.VerticalFlip(p=0.1),
        A.HorizontalFlip(p=0.1),
        A.GridDistortion(p=0.1),
    ], p=0.5),
    A.FromFloat(dtype=np.float32, always_apply=True)
], additional_targets={'image': 'image', 'image1': 'image'}, is_check_shapes=False)

# Headings: Stent Dataset
class StentDataset(Dataset):
    def __init__(self, input_dir, mask_dir, transform=None):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.files = os.listdir(input_dir)
        self.images = [fname for fname in self.files if fname.endswith('.npy')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.input_dir, image_name)
        mask_path_full = os.path.join(self.mask_dir, image_name)
        image = np.load(image_path, allow_pickle=True)
        mask = np.load(mask_path_full, allow_pickle=True)
        mask[mask > 0] = 255
        pair = album_transform(image=image, image1=mask)
        return torch.as_tensor(pair['image'] / 255.).unsqueeze(0), torch.as_tensor(pair['image1'])

