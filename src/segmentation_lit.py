import numpy as np
import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchmetrics
from matplotlib import pyplot as plt
from torch import nn

import wandb

# from utils import vis
from .constants import class_names


class LitSegmentation(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.ignore_index = 255
        # self.class_names = ["Road", "Grass", "Vegetation", "Sky", "Obstacle"]
        num_classes = 40
        # self.class_names = [str(i) for i in range(num_classes)]
        self.class_names = class_names.NYU_V2_segmentation_classes
        assert len(self.class_names) == num_classes
        self.learning_rate = learning_rate

        self.segmenter = smp.DeepLabV3(
            # encoder_name="efficientnet-b1",
            encoder_weights="imagenet",
            # encoder_depth=3,
            # decoder_channels=(64, 32, 16),
            classes=num_classes,
            # activation="logsoftmax",
        )
        self.loss_fn = smp.losses.LovaszLoss(
            mode="multiclass", ignore_index=self.ignore_index
        )
        self.save_hyperparameters()
        # self.loss_fn = smp.losses.DiceLoss(
        #     mode="multiclass", ignore_index=self.ignore_index
        # )
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.set_metrics(num_classes)

    def set_metrics(self, num_classes):
        self.macro_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.classification.MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=self.ignore_index,
                    validate_args=False,
                ),
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=self.ignore_index,
                    validate_args=False,
                ),
            ],
            prefix="test/",
            postfix="_macro",
        )

        self.micro_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.classification.MulticlassAccuracy(
                    num_classes=num_classes,
                    average=None,
                    ignore_index=self.ignore_index,
                    validate_args=False,
                ),
                torchmetrics.classification.MulticlassJaccardIndex(
                    num_classes=num_classes,
                    average=None,
                    ignore_index=self.ignore_index,
                    validate_args=False,
                ),
                torchmetrics.classification.MulticlassConfusionMatrix(
                    num_classes,
                    ignore_index=self.ignore_index,
                    normalize="true",
                    validate_args=False,
                ),
            ],
            prefix="test/",
        )

    def training_step(self, batch, batch_idx):
        data, targets = batch
        targets = targets.long()
        predictions = self.segmenter(data)
        loss = self.loss_fn(predictions, targets)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        targets = targets.long()
        predictions = self.segmenter(data)
        loss = self.loss_fn(predictions, targets)

        self.log_dict({"val/loss": loss})

    def test_step(self, batch, batch_idx):
        data, targets = batch
        targets = targets.long()
        predictions = self.segmenter(data)

        output = self.macro_metrics(predictions, targets)
        self.log_dict(output)
        self.micro_metrics.update(predictions, targets)

    def test_epoch_end(self, outputs) -> None:
        # self._get_confusion_matrix(self.test_confusion_matrix_all, "normalized_all")
        micro_metrics = self.micro_metrics.compute()
        metric_names = ["test/MulticlassAccuracy", "test/MulticlassJaccardIndex"]
        self._log_micro_metrics(micro_metrics, metric_names)
        self._log_confusion_matrix(micro_metrics["test/MulticlassConfusionMatrix"])
        self.micro_metrics.reset()

    def _log_micro_metrics(self, micro_metrics, metric_names):
        frame = torch.vstack(
            [micro_metrics[name] for name in metric_names],
        ).cpu()
        micro_metrics_df = pd.DataFrame(
            frame, index=metric_names, columns=self.class_names
        )

        fig = px.line_polar(
            micro_metrics_df.iloc[1].T.reset_index(),
            r="test/MulticlassJaccardIndex",
            theta="index",
            line_close=True,
        )
        wandb.log({"Polar IOU": fig})
        self.logger.log_table(key="micro_metrics", dataframe=micro_metrics_df)

    def _log_confusion_matrix(self, cf_matrix):
        matrix_df = pd.DataFrame(cf_matrix.cpu(), self.class_names, self.class_names)
        self.logger.log_table(key="conf", dataframe=matrix_df)

        fig = px.imshow(matrix_df, template="seaborn")
        wandb.log({"Confusion Matrix": fig})

    def _split_n_drop(self, output, multiclass_metrics):
        splitted_metrics = self._split_micro_metric(
            {key: output[key] for key in output.keys() and multiclass_metrics}
        )
        [output.pop(item) for item in multiclass_metrics]
        return splitted_metrics

    def _split_micro_metric(
        self, metric_name_vals: dict[str, torch.Tensor]
    ) -> list[dict[str, float]]:
        result = []
        for metric_name, metric_vals in metric_name_vals.items():
            metric_name = metric_name.replace("/", "_")
            result.append(
                {
                    f"{metric_name}/{name}": val * 100
                    for name, val in zip(self.class_names, metric_vals.tolist())
                }
            )
        return result

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return opt
