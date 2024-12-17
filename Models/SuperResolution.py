
from Models.Model1_UNET2017 import UNet
import matplotlib.pyplot as plt
import torch

import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image import PeakSignalNoiseRatio as psnr



class SuperResolutionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Define your super-resolution model here
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        self.loss_fn = torch.nn.MSELoss()

        # Initialize variables to store epoch loss as lists
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.avg_train_loss = []
        self.avg_val_loss = []

        self.train_ssim = ssim(data_range=1.0)
        self.val_ssim = ssim(data_range=1.0)
        self.val_epoch_ssim = []
        self.train_epoch_ssim = []
        self.avg_train_ssim = []
        self.avg_val_ssim = []

        self.train_psnr = psnr(data_range=1.0)
        self.val_psnr = psnr(data_range=1.0)
        self.val_epoch_psnr = []
        self.train_epoch_psnr = []
        self.avg_train_psnr = []
        self.avg_val_psnr = []

    def forward(self, lr_image):
        return torch.sigmoid(self.model(lr_image))

    def training_step(self, batch, batch_idx):
        # Training logic here
        hr_image, lr_image = batch
        # Forward pass
        hr_prediction = self(lr_image)

        # Compute loss
        loss = self.loss_fn(hr_prediction, hr_image)
        self.train_epoch_loss.append(loss.item())
        # Compute SSIM
        train_ssim = self.train_ssim(hr_prediction, hr_image)
        self.train_epoch_ssim.append(train_ssim.item())
        # Compute PSNR
        train_psnr = self.train_psnr(hr_prediction, hr_image)
        self.train_epoch_psnr.append(train_psnr.item())

        # Log loss, ssim, psnr
        self.log("Train Loss MSE", loss, prog_bar=True)
        self.log("Train SSIM", train_ssim, prog_bar=True)
        self.log("Train PSNR", train_psnr, prog_bar=True)

        # Log high-resolution images
        if batch_idx % 1500 == 0:
            self.log_images(lr_image.cpu(), hr_prediction.cpu(), hr_image.cpu(), "Train")
        return loss

    def on_train_epoch_end(self):
        # Compute avg loss
        avg_train_loss = torch.tensor(self.train_epoch_loss).mean()
        self.avg_train_loss.append(avg_train_loss)
        self.train_epoch_loss = []

        # Compute avg SSIM
        avg_train_ssim = torch.tensor(self.train_epoch_ssim).mean()
        self.avg_train_ssim.append(avg_train_ssim)
        self.train_epoch_ssim = []

        # Compute avg PSNR
        avg_train_psnr = torch.tensor(self.train_epoch_psnr).mean()
        self.avg_train_psnr.append(avg_train_psnr)
        self.train_epoch_psnr = []

    def validation_step(self, batch, batch_idx):
        # Validation logic here
        hr_image, lr_image = batch

        # Forward pass
        hr_prediction = self(lr_image)

        # Compute loss
        loss = self.loss_fn(hr_prediction, hr_image)
        self.val_epoch_loss.append(loss.item())
        # Compute SSIM
        val_ssim = self.val_ssim(hr_prediction, hr_image)
        self.val_epoch_ssim.append(val_ssim.item())
        # Compute PSNR
        val_psnr = self.val_psnr(hr_prediction, hr_image)
        self.val_epoch_psnr.append(val_psnr.item())

        # Log loss, ssim, psnr
        self.log("Val loss MSE", loss, prog_bar=True)
        self.log("Val SSIM", val_ssim, prog_bar=True)
        self.log("Val PSNR", val_psnr, prog_bar=True)
        # Log high-resolution images
        if batch_idx % 1200 == 0:
            self.log_images(lr_image.cpu(), hr_prediction.cpu(), hr_image.cpu(), "Val")

        return loss

    def on_validation_epoch_end(self):
        # Compute avg loss
        avg_val_loss = torch.tensor(self.val_epoch_loss).mean()
        self.avg_val_loss.append(avg_val_loss)
        self.val_epoch_loss = []
        # Compute avg SSIM
        avg_val_ssim = torch.tensor(self.val_epoch_ssim).mean()
        self.avg_val_ssim.append(avg_val_ssim)
        self.val_epoch_ssim = []
        # Compute avg PSNR
        avg_val_psnr = torch.tensor(self.val_epoch_psnr).mean()
        self.avg_val_psnr.append(avg_val_psnr)
        self.val_epoch_psnr = []

    def log_images(self, lr_image, hr_prediction, hr_image, name):
        # Visualize and log high-resolution images

        fig, axis = plt.subplots(1, 3, figsize=(12, 4))

        # Detach the tensors before using them in imshow
        lr_image = lr_image.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        hr_prediction = hr_prediction.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        hr_image = hr_image.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()

        axis[0].imshow(lr_image[0], cmap="gray")
        axis[0].set_title("Low-Res Image")

        axis[1].imshow(hr_prediction[0], cmap="gray")
        axis[1].set_title("Generated High-Res Image")

        axis[2].imshow(hr_image[0], cmap="gray")
        axis[2].set_title("Ground Truth High-Res Image")
        plt.show()

        for ax in axis:
            ax.axis('off')

        self.logger.experiment.add_figure(name, fig, self.global_step)

    def configure_optimizers(self):
        return [self.optimizer]