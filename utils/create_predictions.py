import os
import cv2
import torch
import albumentations.pytorch
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from models.basic_model import BasicModel


DATASET = "1_20"
USE_RGB = False
MODEL_NAME = "epoch=15-step=27311_1_20.ckpt"


def predict():
    abs_path = os.path.abspath("../")
    dataset_path = f"{abs_path}/test_dataset/{DATASET}"
    dataset_ground_truth_path = f"{dataset_path}/ground_truth"
    dataset_input_path = f"{dataset_path}/input"
    dataset_masks_path = f"{dataset_path}/masks"
    predictions_path = f"{abs_path}/predictions/{DATASET}"
    predictions_output_path = f"{predictions_path}/output"
    predictions_concatenated_path = f"{predictions_path}/concatenated"
    model_path = f"{abs_path}/checkpoints/{MODEL_NAME}"
    os.makedirs(predictions_output_path, exist_ok=True)
    os.makedirs(predictions_concatenated_path, exist_ok=True)

    input_transforms = albumentations.Compose([
        albumentations.Resize(240, 320),
        albumentations.ToFloat(),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    device = torch.device('cpu')
    model = BasicModel.load_from_checkpoint(model_path).to(device)
    model = model.eval()

    data = {
        "MAE": [],
        "MAE_target": []
    }

    input_names = sorted([path.name for path in Path(dataset_ground_truth_path).iterdir()])
    for i, image_name in enumerate(input_names):
        print(f"{round(100 * i / len(input_names))}%")
        if "_rgb_" in image_name and not USE_RGB:
            continue
        image_input = cv2.imread(f"{dataset_input_path}/{image_name}")
        image_ground_truth = cv2.imread(f"{dataset_ground_truth_path}/{image_name}")
        image_mask = cv2.imread(f"{dataset_masks_path}/{image_name}")

        input = input_transforms(image=image_input)['image'][None, ...]
        with torch.no_grad():
            prediction = model(input.to(device)).cpu().squeeze().numpy()

        cv2.imwrite(f"{predictions_output_path}/{image_name}", prediction)

        prediction_resized = cv2.resize(prediction, (640, 480))
        image_concatenated = np.concatenate((image_input, prediction_resized, image_ground_truth), axis=1)

        cv2.imwrite(f"{predictions_concatenated_path}/{image_name}", image_concatenated)
        prediction_resized = cv2.cvtColor(prediction_resized, cv2.COLOR_BGR2GRAY)
        image_ground_truth = cv2.cvtColor(image_ground_truth, cv2.COLOR_BGR2GRAY)
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
        image_mask = image_mask.astype(np.float32)
        image_ground_truth = image_ground_truth.astype(np.float32)

        mae_image = sum(abs(image_ground_truth - prediction_resized)) / len(image_ground_truth)
        data["MAE"].append(mae_image)
        image_ground_truth_circle = cv2.bitwise_and(image_ground_truth, image_mask)
        prediction_circle = cv2.bitwise_and(prediction_resized, image_mask)

        mae_target = sum(abs(image_ground_truth_circle - prediction_circle)) / len(image_ground_truth_circle)
        data["MAE_target"].append(mae_target)

    metrics = {}
    for metric in data.keys():
        value = np.sum(data[metric], axis=None) / len(data[metric])
        metrics[f"{metric}_abs"] = value
        metrics[f"{metric}_norm"] = value / 65536
        metrics[f"{metric}_%"] = value / 65536 * 100
        metrics[f"{metric}_m"] = value / 65535 * 4.00

    print(metrics)
    with open(f"{predictions_path}/info.txt", "w") as f:
        for metric in metrics.keys():
            f.write(f"{metric}: {metrics[metric]}\n")


if __name__ == '__main__':
    predict()
