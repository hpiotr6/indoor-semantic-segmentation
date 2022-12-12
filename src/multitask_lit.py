import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from . import constants, metrics, multitask_model


class LitMultitask(pl.LightningModule):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.ignore_seg_index = 255
        self.learning_rate = learning_rate

        self.model = multitask_model.MultitaskNet()
        self.segmentation_criterior = smp.losses.LovaszLoss(
            mode="multiclass", ignore_index=self.ignore_seg_index
        )

        self.classification_criterior = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def criterior(self, mask_logits, mask_targets, scene_logits, scene_targets):
        seg_loss = self.segmentation_criterior(mask_logits, mask_targets)
        scene_loss = self.classification_criterior(scene_logits, scene_targets)
        agg_loss = seg_loss + scene_loss
        return dict(seg_loss=seg_loss, scene_loss=scene_loss, loss=agg_loss)

    def training_step(self, batch):
        data, mask_targets, scene_targets = batch
        mask_logits, scene_logits = self(data)
        loss = self.criterior(mask_logits, mask_targets, scene_logits, scene_targets)
        self.log_dict(
            {f"train/{loss_name}": loss_val for loss_name, loss_val in loss.items()}
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data, mask_targets, scene_targets = batch
        mask_logits, scene_logits = self(data)
        loss = self.criterior(mask_logits, mask_targets, scene_logits, scene_targets)
        self.log_dict(
            {f"val/{loss_name}": loss_val for loss_name, loss_val in loss.items()}
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]


class LitClassification(pl.LightningModule):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.model = multitask_model.MultitaskNet(stage="classification")
        self.criterior = torch.nn.CrossEntropyLoss()

        self.metrics = metrics.WandbHelper(
            stage="test",
            name="classification",
            class_names=constants.SCENE_MERGED_IDS.keys(),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        data, scene_targets = batch
        scene_logits = self(data)
        loss = self.criterior(scene_logits, scene_targets)
        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        data, scene_targets = batch
        scene_logits = self(data)
        loss = self.criterior(scene_logits, scene_targets)
        self.log_dict({"val/loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        data, scene_targets = batch
        scene_logits = self(data)
        # TODO: czy faktycznie potrzebny long?
        scene_targets = scene_targets.long()

        output = self.metrics.macro_metrics(scene_logits, scene_targets)
        self.log_dict(output)
        self.metrics.nonaverage_metrics.update(scene_logits, scene_targets)

    def test_epoch_end(self, outputs) -> None:
        # self._get_confusion_matrix(self.test_confusion_matrix_all, "normalized_all")
        nonaverage_metrics = self.metrics.nonaverage_metrics.compute()
        matrix_df = self.metrics.get_matrix_df(
            nonaverage_metrics.get("test/classification_confusion_matrix")
        )
        self.metrics.log_confusion_matrix(matrix_df)
        self.metrics.log_confusion_matrix_table(matrix_df)

        self.metrics.nonaverage_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LitSegmentation(pl.LightningModule):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.ignore_seg_index = 255
        self.model = multitask_model.MultitaskNet(stage="segmentation")

        self.criterior = smp.losses.LovaszLoss(
            mode="multiclass", ignore_index=self.ignore_seg_index
        )
        self.metrics = metrics.WandbHelper(
            stage="test",
            name="segmentation",
            class_names=constants.NYU_V2_segmentation_classes,
            ignore_index=self.ignore_seg_index,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        data, mask_targets = batch
        mask_logits = self(data)
        loss = self.criterior(mask_logits, mask_targets)
        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        data, mask_targets = batch
        mask_logits = self(data)
        loss = self.criterior(mask_logits, mask_targets)
        self.log_dict({"val/loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        data, mask_targets = batch
        mask_logits = self(data)
        # TODO: czy faktycznie potrzebny long?
        # mask_logits = mask_logits.long()

        # output = self.metrics.macro_metrics(mask_logits, mask_targets)
        # self.log_dict(output)
        self.metrics.nonaverage_metrics.update(mask_logits, mask_targets)

    def test_epoch_end(self, outputs) -> None:
        # self._get_confusion_matrix(self.test_confusion_matrix_all, "normalized_all")
        nonaverage_metrics = self.metrics.nonaverage_metrics.compute()
        matrix_df = self.metrics.get_matrix_df(
            nonaverage_metrics.get("test/segmentation_confusion_matrix")
        )
        self.metrics.log_confusion_matrix(matrix_df)
        self.metrics.log_confusion_matrix_table(matrix_df)

        self.metrics.nonaverage_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
