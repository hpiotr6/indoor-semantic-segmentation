from src.multitask_lit import LitMultitask
from pathlib import Path
import pytorch_lightning as pl

from src import multitask_datamod

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    PROJECT_DIR = Path(__file__).resolve().parent
    dataloader_params = {"batch_size": 8, "num_workers": 12, "pin_memory": False}
    data_module = multitask_datamod.MultitaskDataModule(PROJECT_DIR, dataloader_params)

    logger = pl.loggers.WandbLogger(
        project="segmentation-09.12",
        # name="baseline",
        log_model=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=30,
        # max_steps=100,
        # val_check_interval=0.25,
        precision=16,
        # log_every_n_steps=5,
        logger=logger,
        # callbacks=[early_stopping],
        # accumulate_grad_batches=3,
        # overfit_batches=2,
    )
    model = LitMultitask(learning_rate=1e-3)
    trainer.fit(
        model=model,
        datamodule=data_module
        # ckpt_path="/home/piotr/SensorsArticle2022/logs/23.11-max_real/version_0/checkpoints/epoch=19-step=300.ckpt",
    )
    trainer.test(model=model, datamodule=data_module, verbose=True)
