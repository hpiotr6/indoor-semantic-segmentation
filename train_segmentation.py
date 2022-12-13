import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch.utils.data as data
import torchvision
from matplotlib import pyplot as plt

from src import dataset, transforms
from src.segmentation_lit import LitSegmentation
from utils import vis


def test_batch(train_batch):
    inputs, masks = next(iter(train_batch))
    vis.imshow_semantic(masks)
    plt.show()
    vis.imshow(inputs)
    plt.show()


pl.seed_everything(42, workers=True)

PROJECT_DIR = Path(__file__).resolve().parent
img_dir = os.path.join(PROJECT_DIR, "datasets", "train", "rgb")
mask_dir = os.path.join(PROJECT_DIR, "datasets", "train", "semantic_40")

img_dir_test = os.path.join(PROJECT_DIR, "datasets", "test", "rgb")
mask_dir_test = os.path.join(PROJECT_DIR, "datasets", "test", "semantic_40")

train_set = dataset.NYUv2SegmentationDataset(img_dir, mask_dir, transforms.t2)
val_set = dataset.NYUv2SegmentationDataset(img_dir_test, mask_dir_test, transforms.t2)
# val_set = data.Subset(val_set, list(range(20)))

batch_params = {"num_workers": 12, "pin_memory": False}
train_batch = data.DataLoader(train_set, batch_size=3, shuffle=True, **batch_params)
val_batch = data.DataLoader(val_set, batch_size=3, shuffle=False, **batch_params)
print(train_set[0][0].size())
# test_batch(train_batch)
# plt.imshow(train_set[0][1])
# plt.show()


logger = pl.loggers.WandbLogger(
    project="segmentation-09.12",
    # name="baseline",
    log_model=True,
)
early_stopping = pl.callbacks.EarlyStopping("val/loss")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=30,
    # max_steps=100,
    # val_check_interval=0.25,
    precision=16,
    log_every_n_steps=5,
    logger=logger,
    callbacks=[early_stopping],
    # accumulate_grad_batches=3,
    # overfit_batches=2,
)
model = LitSegmentation(1e-4)
trainer.fit(
    model=model,
    train_dataloaders=train_batch,
    val_dataloaders=[val_batch],
    # ckpt_path="/home/piotr/SensorsArticle2022/logs/23.11-max_real/version_0/checkpoints/epoch=19-step=300.ckpt",
)
trainer.test(model=model, dataloaders=val_batch, verbose=True)
