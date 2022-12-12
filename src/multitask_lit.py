import pytorch_lightning as pl
import torch
from . import multitask_model

import segmentation_models_pytorch as smp


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
        return optimizer
