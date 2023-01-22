from matplotlib import pyplot as plt
from utils import timer
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

    # logger = loggers.TensorBoardLogger(
    #     save_dir="logs",
    #     name="only_segmentation",
    #     version="softmax1d",
    # )

    checkpoint_callback = callbacks.ModelCheckpoint()
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="step")
    early_stopping = callbacks.EarlyStopping(monitor="val/loss", patience=5)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=50,
        precision=16,
        logger=False,
    )
    mlt_model = multitask_model.MultitaskNet()

    # regex = partial(re.search, pattern="seg|dec")
    # freeze_layers(mlt_model, regex)

    model = LitMultitask(
        model=mlt_model,
        learning_rate=1e-4,
        scene_weights=weights.SCENE_WEIGHTS,
        seg_weights=weights.SEG_WEIGHTS,
    )

    # lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
    # new_lr = lr_finder.suggestion()
    # model.learning_rate = new_lr

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # print(type(fig))

    # plt.savefig("my_fig.jpg")
    class Logger:
        def __init__(self, log_dir) -> None:
            self.log_dir = log_dir
    logger = Logger(os.path.join("inferences_times","multitask"))

    with timer.timer(logger):
        trainer.test(
            model=model,
            datamodule=data_module,
            verbose=True,
            # ckpt_path="logs/only_segmentation/version_10/checkpoints/epoch=149-step=15000.ckpt",
        )

        trainer.test(
            model=model,
            datamodule=data_module,
            verbose=True,
            # ckpt_path="logs/only_segmentation/version_10/checkpoints/epoch=149-step=15000.ckpt",
        )


    