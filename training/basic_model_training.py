import os
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pathlib import Path
from models.image_dataset import ImageDataset
from models.basic_model import BasicModel


DATASET = "1_20"
VAL_SIZE = 0.15
TEST_SIZE = 0.0
PATIENCE = 10
MAX_EPOCHS = 100


abs_path = os.path.abspath("../")
dataset_path = f"{abs_path}/dataset/{DATASET}"
dataset_ground_truth_path = f"{dataset_path}/ground_truth/"
dataset_input_path = f"{dataset_path}/input/"

train_names = sorted([path.name for path in Path(dataset_ground_truth_path).iterdir()])
train_names, val_names = train_test_split(train_names, test_size=VAL_SIZE + TEST_SIZE, random_state=42)
if TEST_SIZE > 0:
    val_names, test_names = train_test_split(val_names, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), random_state=42)
else:
    test_names = []

train_dataset = ImageDataset(Path(abs_path), train_names)
val_dataset = ImageDataset(Path(abs_path), val_names)
test_dataset = ImageDataset(Path(abs_path), test_names)

model = BasicModel()
model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f"{abs_path}/checkpoints")
early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=2)

trainer = pl.Trainer(callbacks=[model_checkpoint, early_stopping], gpus=1, max_epochs=100)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
