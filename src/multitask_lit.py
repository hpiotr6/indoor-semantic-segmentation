import pytorch_lightning as pl
import torch
from torchmetrics import classification

from .constants import class_names

from . import metrics


def shared_step(model, batch):
    data, mask_targets, scene_targets = batch
    mask_logits, scene_logits = model(data)
    return mask_logits, mask_targets.long(), scene_logits, scene_targets


def freeze_nontrainable_params(model, trainable_params):
    for p in model.parameters():
        p.requires_grad = False
    for p in trainable_params:
        p.requires_grad = True


class LitMultitask(pl.LightningModule):
    def __init__(
        self, model, learning_rate, scene_weights=None, seg_weights=None
    ) -> None:
        super().__init__()
        self.ignore_seg_index = 255
        self.learning_rate = learning_rate

        self.model = model
        # self.segmentation_criterior = smp.losses.LovaszLoss(
        #     mode="multiclass", ignore_index=self.ignore_seg_index
        # )

        self.segmentation_criterior = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_seg_index,
            weight=seg_weights,
        )

        self.classification_criterior = torch.nn.CrossEntropyLoss(weight=scene_weights)

        self.seg_metrics = metrics.MetricsHelper(
            metrics=[
                classification.MulticlassJaccardIndex,
                classification.MulticlassAccuracy,
            ],
            stage="test",
            name="segmentation",
            class_names=class_names.NYU_V2_segmentation_13classes,
            ignore_index=self.ignore_seg_index,
        )
        self.scene_metrics = metrics.MetricsHelper(
            metrics=[
                classification.MulticlassF1Score,
                classification.MulticlassAccuracy,
            ],
            stage="test",
            name="classification",
            class_names=class_names.SCENE_MERGED_IDS.keys(),
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return [optimizer], [scheduler]

    def criterior(self, mask_logits, mask_targets, scene_logits, scene_targets):
        seg_loss = self.segmentation_criterior(mask_logits, mask_targets)
        scene_loss = self.classification_criterior(scene_logits, scene_targets)
        agg_loss = seg_loss + scene_loss
        return dict(seg_loss=seg_loss, scene_loss=scene_loss, loss=agg_loss)

    def training_step(self, batch):
        loss = self.criterior(*shared_step(self, batch))
        self.log_dict(
            {f"train/{loss_name}": loss_val for loss_name, loss_val in loss.items()}
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.criterior(*shared_step(self, batch))
        self.log_dict(
            {f"val/{loss_name}": loss_val for loss_name, loss_val in loss.items()}
        )
        return loss

    def test_step(self, batch, batch_idx):
        mask_logits, mask_targets, scene_logits, scene_targets = shared_step(
            self, batch
        )

        self.update_metrics(self.scene_metrics, scene_logits, scene_targets)
        self.update_metrics(self.seg_metrics, mask_logits, mask_targets)

    def test_epoch_end(self, outputs) -> None:
        self.log_output_end(self.scene_metrics)
        self.log_output_end(self.seg_metrics)

    def update_metrics(self, metric: metrics.MetricsHelper, logits, targets):
        metric.macro_metrics.update(logits, targets)
        metric.nonaverage_metrics.update(logits, targets)

    def log_output_end(self, metric: metrics.MetricsHelper):
        nonaverage_metrics = metric.nonaverage_metrics.compute()
        nonaverage_dict = {}
        for metric_name, tensor in nonaverage_metrics.items():
            for class_name, value in zip(metric.class_names, tensor.tolist()):
                nonaverage_dict[f"{metric_name}/{class_name}"] = value

        self.log_dict(nonaverage_dict)
        self.log_dict(metric.macro_metrics.compute())

        metric.macro_metrics.reset()
        metric.nonaverage_metrics.reset()
