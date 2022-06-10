import pytorch_lightning as pl
# import pandas as pd
import torch
import torchmetrics
from fastai.vision.learner import create_unet_model
from fastai.vision.models import resnet34
from piqa import ssim
import cv2
class MaskModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.img_size = (240, 320)
        encoder_model = resnet34
        model = create_unet_model(encoder_model, n_out=1, img_size=self.img_size, n_in=1)
        self.network = model
        self.loss_function = torch.nn.L1Loss()

#MAE SSIM
        # metrics = torchmetrics.MetricCollection([
        #     torchmetrics.Precision(num_classes=4, average="macro", mdmc_average="samplewise"),
        #     torchmetrics.Recall(num_classes=4, average="macro", mdmc_average="samplewise"),
        #     torchmetrics.F1Score(num_classes=4, average="macro", mdmc_average="samplewise"),
        #     torchmetrics.Accuracy(num_classes=4, average="macro", mdmc_average="samplewise")
        # ])
        # self.train_metrics = metrics.clone("train_")
        # self.val_metrics = metrics.clone("val_")
    def weighted_loss(self, input, output, mask, w1=0.90, w2=0.10, criterion=torch.nn.L1Loss()):
        loss = criterion(input * mask, output * mask) * w1 + criterion(input * (1 - mask), output * (1 - mask)) * w2
        return loss

    def forward(self, x):
        x = x.type(torch.float32)
        return self.network(x)
    def training_step(self, batch, batch_idx):
        input_image, ground_truth, mask = batch
        output_image = self(input_image)
        # loss = self.loss_function(output_image, ground_truth)
        loss = self.weighted_loss(output_image, ground_truth, mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_image, ground_truth, mask = batch
        # split_input = torch.chunk(input_image, 2, dim=1)
        output_image = self(input_image)
        loss = self.weighted_loss(output_image, ground_truth,mask)
        # loss = self.loss_function(output_image, ground_truth)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)