import re
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torchmetrics import classification

from .constants import class_names

from . import metrics, multitask_model
from .classification_models import ImagenetTransferLearning


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
                classification.F1Score,
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

    # def test_step(self, batch, batch_idx):

    #     data, mask_targets, scene_targets = batch
    #     mask_logits, scene_logits = self(data)
    #     # TODO: czy faktycznie potrzebny long?
    #     scene_targets = scene_targets.long()

    #     scene_output = self.scene_metrics.macro_metrics(scene_logits, scene_targets)
    #     self.log_dict(scene_output)
    #     self.scene_metrics.nonaverage_metrics.update(scene_logits, scene_targets)

    #     seg_output = self.seg_metrics.macro_metrics(mask_logits, mask_targets)
    #     self.log_dict(seg_output)
    #     self.seg_metrics.nonaverage_metrics.update(mask_logits, mask_targets)

    # def test_epoch_end(self, outputs) -> None:
    #     # self._get_confusion_matrix(self.test_confusion_matrix_all, "normalized_all")
    #     scene_nonaverage_metrics = self.scene_metrics.nonaverage_metrics.compute()
    #     scene_matrix_df = self.scene_metrics.get_matrix_df(
    #         scene_nonaverage_metrics.get("test/classification_confusion_matrix")
    #     )
    #     self.scene_metrics.log_confusion_matrix(scene_matrix_df)
    #     self.scene_metrics.log_confusion_matrix_table(scene_matrix_df)

    #     self.scene_metrics.nonaverage_metrics.reset()

    #     seg_nonaverage_metrics = self.seg_metrics.nonaverage_metrics.compute()
    #     seg_matrix_df = self.seg_metrics.get_matrix_df(
    #         seg_nonaverage_metrics.get("test/segmentation_confusion_matrix")
    #     )
    #     self.seg_metrics.log_confusion_matrix(seg_matrix_df)
    #     self.seg_metrics.log_confusion_matrix_table(seg_matrix_df)

    #     self.seg_metrics.nonaverage_metrics.reset()


# class LitClassification(pl.LightningModule):
#     def __init__(self, learning_rate, weights=None) -> None:
#         super().__init__()
#         self.learning_rate = learning_rate

#         self.model = smp.Unet(
#             encoder_weights="imagenet",
#             classes=13,
#             aux_params={"classes": 7},
#         )
#         # self.model = ImagenetTransferLearning(7)
#         # for param in self.model.parameters():
#         #     param.requires_grad = False
#         self.criterior = torch.nn.CrossEntropyLoss(weight=weights)
#         self.test_metric = classification.MulticlassAccuracy(
#             num_classes=7,
#         )
#         # self.trainable_params = [
#         #     p
#         #     for name, p in self.model.named_parameters()
#         #     # if not bool(re.search("decoder|segmentation", name))
#         #     # if bool(re.search("classifier", name))
#         # ]
#         # freeze_nontrainable_params(self.model, self.trainable_params)
#         # self.metrics = metrics.WandbHelper(
#         #     stage="test",
#         #     name="classification",
#         #     class_names=class_names.SCENE_MERGED_IDS.keys(),
#         # )

#     # def forward(self, x):
#     #     return self.model(x)

#     def training_step(self, batch):
#         loss = self.criterior(*shared_step(self.model, batch)[2:])
#         self.log_dict({"train/loss": loss})
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self.criterior(*shared_step(self.model, batch)[2:])
#         self.log_dict({"val/loss": loss})
#         return loss

#     def test_step(self, batch, batch_idx):
#         scene_logits, scene_targets = shared_step(self.model, batch)[2:]
#         self.test_metric(scene_logits, scene_targets)
#         self.log("macro_acc", self.test_metric)

#         #     data, scene_targets = batch
#         #     scene_logits = self(data)
#         #     # TODO: czy faktycznie potrzebny long?
#         #     scene_targets = scene_targets.long()

#         # output = self.metrics.macro_metrics(scene_logits, scene_targets)
#         # self.log_dict(output)

#         # self.metrics.nonaverage_metrics.update(scene_logits, scene_targets)

#     # def test_epoch_end(self, outputs) -> None:
#     #     # self._get_confusion_matrix(self.test_confusion_matrix_all, "normalized_all")
#     #     nonaverage_metrics = self.metrics.nonaverage_metrics.compute()
#     #     matrix_df = self.metrics.get_matrix_df(
#     #         nonaverage_metrics.get("test/classification_confusion_matrix")
#     #     )
#     #     self.metrics.log_confusion_matrix(matrix_df)
#     #     self.metrics.log_confusion_matrix_table(matrix_df)

#     #     self.metrics.nonaverage_metrics.reset()

#     def configure_optimizers(self):
#         # parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
#         # # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
#         # scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         #     optimizer=optimizer,
#         #     max_lr=4e-5,
#         #     total_steps=50,
#         #     # epochs=10,
#         #     # steps_per_epoch=4,
#         #     div_factor=25,
#         #     pct_start=0.1,
#         #     anneal_strategy="cos",
#         #     final_div_factor=1e4,
#         # )
#         return optimizer


# class LitSegmentation(pl.LightningModule):
#     def __init__(self, learning_rate, weights=None) -> None:
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.ignore_seg_index = 255
#         self.model = multitask_model.MultitaskNet()

#         # self.criterior = smp.losses.LovaszLoss(
#         #     mode="multiclass", ignore_index=self.ignore_seg_index
#         # )

#         self.criterior = torch.nn.CrossEntropyLoss(
#             ignore_index=self.ignore_seg_index,
#             weight=weights,
#         )
#         # self.metrics = metrics.WandbHelper(
#         #     stage="test",
#         #     name="segmentation",
#         #     class_names=constants.NYU_V2_segmentation_13classes,
#         #     ignore_index=self.ignore_seg_index,
#         # )

#         self.trainable_params = [
#             p for name, p in self.model.named_parameters() if not "classifier" in name
#         ]
#         freeze_nontrainable_params(self.model, self.trainable_params)

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch):
#         loss = self.criterior(*shared_step(self, batch)[:2])

#         self.log_dict({"train/loss": loss})
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self.criterior(*shared_step(self, batch)[:2])
#         self.log_dict({"val/loss": loss})
#         return loss

#     # def test_step(self, batch, batch_idx):
#     #     data, mask_targets = batch
#     #     mask_logits = self(data)
#     #     # TODO: czy faktycznie potrzebny long?
#     #     # mask_logits = mask_logits.long()

#     #     # output = self.metrics.macro_metrics(mask_logits, mask_targets)
#     #     # self.log_dict(output)
#     #     self.metrics.nonaverage_metrics.update(mask_logits, mask_targets)

#     # def test_epoch_end(self, outputs) -> None:
#     #     # self._get_confusion_matrix(self.test_confusion_matrix_all, "normalized_all")
#     #     nonaverage_metrics = self.metrics.nonaverage_metrics.compute()
#     #     matrix_df = self.metrics.get_matrix_df(
#     #         nonaverage_metrics.get("test/segmentation_confusion_matrix")
#     #     )
#     #     self.metrics.log_confusion_matrix(matrix_df)
#     #     self.metrics.log_confusion_matrix_table(matrix_df)

#     #     self.metrics.nonaverage_metrics.reset()

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.trainable_params, lr=self.learning_rate)
#         # scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         #     optimizer=optimizer,
#         #     max_lr=self.learning_rate,
#         #     total_steps=self.trainer.max_epochs + 1,
#         #     # epochs=10,
#         #     # steps_per_epoch=4,
#         #     div_factor=25,
#         #     pct_start=0.1,
#         #     anneal_strategy="cos",
#         #     final_div_factor=1e4,
#         # )

#         return optimizer
