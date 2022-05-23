import os
import cv2
import torch
import albumentations.pytorch
import pytorch_lightning as pl
from pathlib import Path
from random import sample
from models.basic_model import BasicModel


DATASET = "1_60"
NUMBER_OF_PREDICTIONS = 1500
RANDOMIZE = False


def predict():
    abs_path = os.path.abspath("../")
    dataset_path = f"{abs_path}/test_dataset/{DATASET}"
    dataset_ground_truth_path = f"{dataset_path}/ground_truth"
    dataset_input_path = f"{dataset_path}/input"
    predictions_output_path = f"{abs_path}/predictions1-60"
    os.makedirs(predictions_output_path, exist_ok=True)

    input_transforms = albumentations.Compose([
            albumentations.Resize(240, 320),
            albumentations.ToFloat(),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
    output_transforms = albumentations.Compose([
        albumentations.CenterCrop(240, 320),
        albumentations.Resize(2400, 320, interpolation=cv2.INTER_NEAREST)
    ])

    device = torch.device('cuda')
    model = BasicModel.load_from_checkpoint(f"{abs_path}/checkpoints/epoch=8-step=44333.ckpt").to(
        device)  # wczytanie najlepszych wag z treningu
    model = model.eval()

    all_names = sorted([path.name for path in Path(dataset_ground_truth_path).iterdir()])
    if len(all_names) < NUMBER_OF_PREDICTIONS:
        print(f"{len(all_names)} files detected in input path, but {NUMBER_OF_PREDICTIONS} needed.")
        quit()

    if RANDOMIZE:
        input_names = sample(all_names, NUMBER_OF_PREDICTIONS)
    else:
        input_names = all_names[:NUMBER_OF_PREDICTIONS]

    for image_name in input_names:

        file_info = image_name.rstrip(".png").split("_")
        file_type = file_info[1]
        if file_type == 'depth':
            image = cv2.imread(f"{dataset_input_path}/{image_name}")
            image = input_transforms(image=image)['image'][None, ...]
            with torch.no_grad():
                prediction = model(image.to(device)).cpu().squeeze().numpy()

            cv2.imwrite(f"{predictions_output_path}/{image_name}", prediction)


if __name__ == '__main__':
    predict()
