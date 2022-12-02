import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class LitSegmentation(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.ignore_index = 255
        # self.class_names = ["Road", "Grass", "Vegetation", "Sky", "Obstacle"]
        num_classes = 40
        self.class_names = [str(i) for i in range(num_classes)]
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
        # self.loss_fn = smp.losses.DiceLoss(
        #     mode="multiclass", ignore_index=self.ignore_index
        # )
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.set_metrics(num_classes)

    def set_metrics(self, num_classes):
        metrics = torchmetrics.MetricCollection(
            [
                # torchmetrics.MetricCollection(
                #     [
                #         torchmetrics.classification.MulticlassAccuracy(
                #             num_classes=num_classes,
                #             average=None,
                #             ignore_index=self.ignore_index,
                #             validate_args=False,
                #         ),
                #         torchmetrics.classification.MulticlassJaccardIndex(
                #             num_classes=num_classes,
                #             average=None,
                #             ignore_index=self.ignore_index,
                #             validate_args=False,
                #         ),
                #     ],
                # ),
                torchmetrics.MetricCollection(
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
                    postfix="_macro",
                ),
            ]  # type:ignore
        )
        # self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        # self.test_confusion_matrix_all = (
        #     torchmetrics.classification.MulticlassConfusionMatrix(
        #         num_classes, ignore_index=self.ignore_index, normalize="all"
        #     )
        # )

        self.test_confusion_matrix_true = (
            torchmetrics.classification.MulticlassConfusionMatrix(
                num_classes, ignore_index=self.ignore_index, normalize="true"
            )
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
        # predictions = predictions.unsqueeze(1)
        # loss = self.loss_fn(preditions, targets, ignore_index=self.ignore_index)
        loss = self.loss_fn(predictions, targets)
        # output = self.val_metrics(predictions, targets)
        # multiclass_metrics = [
        #     "val/MulticlassAccuracy",
        #     "val/MulticlassJaccardIndex",
        # ]
        # splitted_metrics = self._split_n_drop(output, multiclass_metrics)
        # [self.log_dict(splitted_metric) for splitted_metric in splitted_metrics]
        # self.log_dict({key: val * 100 for key, val in output.items()})
        self.log_dict({"val/loss": loss})

    def test_step(self, batch, batch_idx):
        data, targets = batch
        targets = targets.long()
        predictions = self.segmenter(data)

        output = self.test_metrics(predictions, targets)
        # multiclass_metrics = [
        #     "test/MulticlassAccuracy",
        #     "test/MulticlassJaccardIndex",
        # ]
        # splitted_metrics = self._split_n_drop(output, multiclass_metrics)
        # [self.log_dict(splitted_metric) for splitted_metric in splitted_metrics]
        self.log_dict({key: val * 100 for key, val in output.items()})

        # self.test_confusion_matrix_all.update(predictions, targets)
        self.test_confusion_matrix_true.update(predictions, targets)

    def test_epoch_end(self, outputs) -> None:
        # self._get_confusion_matrix(self.test_confusion_matrix_all, "normalized_all")
        self._get_confusion_matrix(self.test_confusion_matrix_true, "normalized_true")

    def _get_confusion_matrix(self, confusion_metric, name):
        cf_matrix = confusion_metric.compute()
        matrix_df = pd.DataFrame(cf_matrix.cpu(), self.class_names, self.class_names)
        ax = sns.heatmap(matrix_df, annot=True)
        ax.set(xlabel="Actual", ylabel="Predited")
        self.logger.experiment.add_figure(name, ax.get_figure())
        confusion_metric.reset()

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
