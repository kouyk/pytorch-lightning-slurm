import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import cross_entropy
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        self.model.fc = nn.Identity()
        self.model.requires_grad_(False)

        metrics = MetricCollection({
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro")
        })
        self.metrics = nn.ModuleDict({
            "Train": metrics.clone(prefix="train_"),
            "val": metrics.clone(prefix="val_"),
            "test": metrics.clone(prefix="test_")
        })

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.model(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)

        self.log("train_loss", loss)
        metrics_output = self.metrics["Train"](pred, y)
        self.log_dict(metrics_output, prog_bar=True)

        return {"loss": loss, "predictions": pred, "targets": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)

        self.log("val_loss", loss)
        metrics_output = self.metrics["val"](pred, y)
        self.log_dict(metrics_output, prog_bar=True)

        return {"loss": loss, "predictions": pred, "targets": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)

        self.log("test_loss", loss)
        metrics_output = self.metrics["test"](pred, y)
        self.log_dict(metrics_output)

        return {"loss": loss, "predictions": pred, "targets": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
