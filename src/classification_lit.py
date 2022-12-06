import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchmetrics

import wandb

from . import classification_models, constants


class LitClassification(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.class_names = constants.SCENE_MERGED_IDS.keys()

        self.num_classes = len(self.class_names)
        self.ignore_index = None
        self.classifier = classification_models.ImagenetTransferLearning(
            self.num_classes
        )
        # print(self.classifier)
        # self.classifier.feature_extractor[:7].eval()
        # for param in self.classifier.feature_extractor[:7].parameters():
        #     param.requires_grad = False
        # self.loss_fn = smp.losses.LovaszLoss(
        #     mode="multiclass", ignore_index=self.ignore_index
        # )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.accuracy = torchmetrics.Accuracy(average="macro", num_classes=27)

        self.set_metrics(self.num_classes)

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
        # training_step defines the train loop.
        data, targets = batch
        preditions = self.classifier(data)
        loss = self.loss_fn(preditions, targets)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, targets = batch
        preditions = self.classifier(data)
        loss = self.loss_fn(preditions, targets)

        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        targets = targets.long()
        predictions = self.classifier(data)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     momentum=0.9,
        #     weight_decay=0.2,
        #     nesterov=True
        # )
        return optimizer
