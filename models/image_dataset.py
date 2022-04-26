import math
import torch
import cv2
import albumentations.pytorch
import numpy as np
from pathlib import Path
from typing import List


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, file_names: List[str], augment: bool = False):
        self._file_names = file_names
        self._input_dir = f"{path}/input"
        self._output_dir = f"{path}/ground_truth"
        self._augment = augment
        self.image_size = (480, 640)
        self.padded_image_size = (
            math.ceil(self.image_size[0] / 32) * 32,
            math.ceil(self.image_size[1] / 32) * 32
        )

        self.transforms = albumentations.Compose([
            albumentations.Resize(*self.image_size),
            albumentations.PadIfNeeded(*self.padded_image_size),
            albumentations.ToFloat(max_value=255),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

        self.augmentations = albumentations.Compose([
            albumentations.Resize(*self.image_size),
            albumentations.PadIfNeeded(*self.padded_image_size),
            albumentations.HorizontalFlip(),
            albumentations.Affine(scale=(0.95, 1.05), rotate=(-10, 10)),
            albumentations.ToFloat(max_value=255),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    def __getitem__(self, index: int):
        input_path = f"{self._input_dir}/{self._file_names[index]}"
        output_path = f"{self._output_dir}/{self._file_names[index]}"

        input_image = cv2.imread(str(input_path))
        output_image = cv2.imread(str(output_path))

        if self._augment:
            transformed = self.augmentations(image=input_image, mask=output_image)
        else:
            transformed = self.transforms(image=input_image, mask=output_image)

        transformed_input = transformed["image"].type(torch.float32)
        transformed_output = transformed["mask"].type(torch.float32)

        return transformed_input, transformed_output

    def __len__(self):
        return len(self._file_names)
