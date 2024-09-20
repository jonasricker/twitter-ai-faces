import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torchvision.models import ResNet50_Weights, resnet50

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


class ResNet50(pl.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        no_down: bool = False,
    ) -> None:
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if no_down:  # stride 2 -> 1
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=1, padding=3, bias=False
            )
            self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.model.fc = nn.Linear(512 * 4, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.acc = lambda y_pred, y: balanced_accuracy_score(
            y_true=y.cpu(), y_pred=(y_pred.sigmoid() > 0.5).cpu()
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_pred = self(x).squeeze()
        return self.loss(y_pred, y.float()), self.acc(y_pred, y - float())

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, acc = self._step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer(self.model.parameters())
        scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_start(self) -> None:
        if self.trainer.optimizers[0].param_groups[0]["lr"] < 1e-6:
            self.trainer.should_stop = True
