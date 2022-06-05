import math
import torch
import cv2
import albumentations.pytorch
import numpy as np
from pathlib import Path
from typing import List


class RgbdDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, file_names: List[str], augment: bool = False):
        self._file_names = file_names
        self._input_dir = f"{path}/input"
        self._input_rgb_dir = f"{path}/input_rgb"

        self._output_dir = f"{path}/ground_truth"
        self._augment = augment
        self.image_size = (240, 320)
        self.padded_image_size = (
            math.ceil(self.image_size[0] / 32) * 32,
            math.ceil(self.image_size[1] / 32) * 32
        )

        self.transforms = albumentations.Compose([
            albumentations.Resize(*self.image_size),
            albumentations.PadIfNeeded(*self.padded_image_size)
        ])

        self.augmentations = albumentations.Compose([
            albumentations.Resize(*self.image_size),
            albumentations.PadIfNeeded(*self.padded_image_size),
            albumentations.HorizontalFlip(),
            albumentations.Affine(scale=(0.95, 1.05), rotate=(-10, 10))
        ])

    def __getitem__(self, index: int):

        input_path = f"{self._input_dir}/{self._file_names[index]}"
        input_rgb_path = f"{self._input_rgb_dir}/{self._file_names[index]}"
        output_path = f"{self._output_dir}/{self._file_names[index]}"

        input_image = cv2.imread(str(input_path), cv2.CV_16UC1).astype(np.float) / 65536
        input_rgb_image = cv2.imread(str(input_rgb_path), cv2.CV_16UC1).astype(np.float) / 65536
        output_image = cv2.imread(str(output_path), cv2.CV_16UC1).astype(np.float) / 65536

        if self._augment:
            transformed = self.augmentations(image=input_image, rgb=input_rgb_image, mask=output_image)
            transformed_rgb = self.augmentations(image=input_rgb_image, mask=output_image)

        else:
            transformed = self.transforms(image=input_image, rgb=input_rgb_image, mask=output_image)
            transformed_rgb = self.transforms(image=input_rgb_image, mask=output_image)

        transformed_input = torch.from_numpy(transformed["image"]).unsqueeze(dim=0)
        transformed_input_rgb = torch.from_numpy(transformed_rgb["image"]).unsqueeze(dim=0)
        transformed_output = torch.from_numpy(transformed["mask"]).unsqueeze(dim=0)


        return torch.cat((transformed_input,transformed_input_rgb)), transformed_output

    def __len__(self):
        return len(self._file_names)
