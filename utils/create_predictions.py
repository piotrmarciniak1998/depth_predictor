import os
import cv2
import torch
import albumentations.pytorch
import pytorch_lightning as pl
from pathlib import Path
from random import sample
from models.basic_model import BasicModel


DATASET = "1_20"
NUMBER_OF_PREDICTIONS = 100
RANDOMIZE = True


def predict():
    abs_path = os.path.abspath("../")
    dataset_path = f"{abs_path}/dataset/{DATASET}"
    dataset_ground_truth_path = f"{dataset_path}/ground_truth"
    dataset_input_path = f"{dataset_path}/input"
    predictions_output_path = f"{abs_path}/predictions"
    os.makedirs(predictions_output_path, exist_ok=True)

    input_transforms = albumentations.Compose([
            albumentations.Resize(480, 640),
            albumentations.ToFloat(max_value=255),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
    output_transforms = albumentations.Compose([
        albumentations.CenterCrop(480, 640),
        albumentations.Resize(480, 640, interpolation=cv2.INTER_NEAREST)
    ])

    device = torch.device('cuda')
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f"{abs_path}/checkpoints")
    model = BasicModel.load_from_checkpoint(model_checkpoint.best_model_path).to(
        device)  # wczytanie najlepszych wag z treningu
    model = model.eval()

    all_names = sorted([path.name for path in Path(dataset_ground_truth_path).iterdir()])
    if len(all_names) < NUMBER_OF_PREDICTIONS:
        print(f"{len(all_names)} files detected in input path, but {NUMBER_OF_PREDICTIONS} needed.")
        quit()

    if RANDOMIZE:
        input_names = sample(all_names, NUMBER_OF_PREDICTIONS)
    else:
        input_names = all_names[:100]

    for image_name in input_names:
        image = cv2.imread(f"{dataset_input_path}/{image_name}")
        image = input_transforms(image=image)['image'][None, ...]
        with torch.no_grad():
            prediction = model(image.to(device)).cpu().squeeze().numpy()

        cv2.imwrite(f"{predictions_output_path}/{image_name}", prediction)


if __name__ == '__main__':
    predict()
