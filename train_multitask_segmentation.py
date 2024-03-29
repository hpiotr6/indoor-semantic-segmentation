from pathlib import Path
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from src import multitask_datamod, transforms
from src.multitask_lit import LitSegmentation

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    PROJECT_DIR = Path(__file__).resolve().parent
    dataloader_params = {"batch_size": 8, "num_workers": 12, "pin_memory": False}
    data_module = multitask_datamod.SegmentationDataModule(
        PROJECT_DIR,
        kwargs=dataloader_params,
        # transforms=transforms.train_transform,
        num_classes=13,
    )

    logger = pl.loggers.WandbLogger(
        project="segmentation-19.12",
        # name="baseline",
        log_model=True,
    )

    lr_monitor = callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        auto_lr_find=True,
        accelerator="gpu",
        devices=1,
        max_epochs=50,
        # max_steps=100,
        # val_check_interval=0.25,
        precision=16,
        # log_every_n_steps=5,
        logger=logger,
        callbacks=[lr_monitor]
        # callbacks=[early_stopping],
        # accumulate_grad_batches=3,
        # overfit_batches=2,
    )
    weights = 1 / torch.Tensor(
        [
            11611728,
            8578042,
            1308322,
            3535427,
            8258474,
            23949931,
            35664840,
            31453241,
            4864604,
            5747842,
            7347520,
            1349733,
            57555778,
        ]
    )
    model = LitSegmentation(learning_rate=1e-4, weights=weights)
    trainer.fit(
        model=model,
        datamodule=data_module
        # ckpt_path="/home/piotr/SensorsArticle2022/logs/23.11-max_real/version_0/checkpoints/epoch=19-step=300.ckpt",
    )
    trainer.test(model=model, datamodule=data_module, verbose=True)
