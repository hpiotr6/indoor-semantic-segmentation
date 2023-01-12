import os
from pathlib import Path
import re
from pytorch_lightning import callbacks
import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch
from src.constants import weights

from src import multitask_datamod, multitask_model
from src.multitask_lit import LitMultitask
from functools import partial


def freeze_layers(model, regex):
    for name, p in model.named_parameters():
        if not bool(regex(string=name)):
            p.requires_grad = True
        else:
            p.requires_grad = False


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    PROJECT_DIR = Path(__file__).resolve().parent
    DATASET_PATH = os.path.join(PROJECT_DIR, "datasets")
    dataloader_params = {"batch_size": 8, "num_workers": 12, "pin_memory": False}
    data_module = multitask_datamod.MultitaskDataModule(DATASET_PATH, dataloader_params)

    logger = loggers.TensorBoardLogger(
        save_dir="logs",
        name="only_segmentation",
        # version="",
    )

    lr_monitor = callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        auto_lr_find=True,
        accelerator="gpu",
        devices=1,
        max_epochs=150,
        precision=16,
        logger=logger,
        callbacks=[lr_monitor],
    )
    mlt_model = multitask_model.MultitaskNet()

    # regex = partial(re.search, pattern="seg|dec|enc")
    # freeze_layers(mlt_model, regex)

    model = LitMultitask(
        model=mlt_model,
        learning_rate=1e-4,
        scene_weights=weights.SCENE_WEIGHTS,
        seg_weights=weights.SEG_WEIGHTS,
    )

    # trainer.fit(
    #     model=model,
    #     datamodule=data_module
    #     # ckpt_path="/home/piotr/SensorsArticle2022/logs/23.11-max_real/version_0/checkpoints/epoch=19-step=300.ckpt",
    # )
    trainer.test(
        model=model,
        datamodule=data_module,
        verbose=True,
        ckpt_path="logs/only_segmentation/version_10/checkpoints/epoch=149-step=15000.ckpt",
    )
