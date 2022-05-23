import pytorch_lightning as pl
# import pandas as pd
import torch
import torchmetrics
from fastai.vision.learner import create_unet_model
from fastai.vision.models import resnet34


class BasicModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.img_size = (240, 320)
        encoder_model = resnet34
        model = create_unet_model(encoder_model, n_out=3, img_size=self.img_size, n_in=3)
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

    def forward(self, x):
        x = x.type(torch.float32)
        return torch.permute(self.network(x), (0, 2, 3, 1))

    def training_step(self, batch, batch_idx):
        input_image, ground_truth = batch
        output_image = self(input_image)

        loss = self.loss_function(output_image, ground_truth)
        self.log("train_loss", loss)

        # output_image = torch.softmax(output_image, dim=1)
        # self.log_dict(self.train_metrics(output_image, ground_truth))

        return loss

    def validation_step(self, batch, batch_idx):
        input_image, ground_truth = batch
        output_image = self(input_image)

        loss = self.loss_function(output_image, ground_truth)
        self.log("val_loss", loss, prog_bar=True)

        # output_image = torch.softmax(output_image, dim=1)
        # self.log_dict(self.train_metrics(output_image, ground_truth))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
