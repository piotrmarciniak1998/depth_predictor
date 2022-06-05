import os
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pathlib import Path
from models.basic_dataset import BasicDataset
from models.basic_model import BasicModel

from models.mask_model import MaskModel
from models.mask_dataset import MaskDataset

from models.rgbd_dataset import RgbdDataset
from models.rgbd_model import RgbdModel

import albumentations

DATASET = "1_80_dc"
VAL_SIZE = 0.15
TEST_SIZE = 0.0
PATIENCE = 12
MAX_EPOCHS = 40

global model_checkpoint

def training():
    abs_path = os.path.abspath("../")
    dataset_path = f"{abs_path}/dataset/{DATASET}"
    dataset_ground_truth_path = f"{dataset_path}/ground_truth"

    train_names = sorted([path.name for path in Path(dataset_ground_truth_path).iterdir()])
    train_names, val_names = train_test_split(train_names, test_size=VAL_SIZE + TEST_SIZE, random_state=42)
    if TEST_SIZE > 0:
        val_names, test_names = train_test_split(val_names, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), random_state=42)
    else:
        test_names = []

    # train_dataset = RgbdDataset(Path(dataset_path), train_names)
    # val_dataset = RgbdDataset(Path(dataset_path), val_names)
    # test_dataset = RgbdDataset(Path(dataset_path), test_names)

    train_dataset = BasicDataset(Path(dataset_path), train_names)
    val_dataset = BasicDataset(Path(dataset_path), val_names)
    test_dataset = BasicDataset(Path(dataset_path), test_names)


    # train_dataset = MaskDataset(Path(dataset_path), train_names)
    # val_dataset = MaskDataset(Path(dataset_path), val_names)
    # test_dataset = MaskDataset(Path(dataset_path), test_names)
    model = BasicModel()
    # ckp_path = f"{abs_path}/checkpoints/epoch=5-step=6827-v1.ckpt"
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f"{abs_path}/checkpoints")
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=6 , num_workers=4)
    # trainer = pl.Trainer(callbacks=[model_checkpoint, early_stopping],
    #                      gpus=1,  resume_from_checkpoint=ckp_path, max_epochs=MAX_EPOCHS)
    trainer = pl.Trainer(callbacks=[model_checkpoint, early_stopping], gpus=1, max_epochs=MAX_EPOCHS)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    training()
