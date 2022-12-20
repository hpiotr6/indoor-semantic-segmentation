from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

# from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping
from pytorch_lightning import callbacks, loggers

from src import multitask_datamod, transforms
from src.multitask_lit import LitClassification

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    PROJECT_DIR = Path(__file__).resolve().parent
    dataloader_params = {"batch_size": 8, "num_workers": 12, "pin_memory": False}
    data_module = multitask_datamod.ClassificationDataModule(
        PROJECT_DIR, kwargs=dataloader_params, transforms=transforms.train_transform
    )

    logger = loggers.WandbLogger(
        project="long-training",
        # name="baseline",
        log_model=True,
    )

    multiplicative = lambda epoch: 1.5
    backbone_finetuning = callbacks.BackboneFinetuning(
        14, multiplicative, verbose=True, backbone_initial_lr=0.00001
    )
    weights = 1 / torch.Tensor([66.0, 192.0, 114.0, 63.0, 92.0, 149.0, 119.0])
    model = LitClassification(learning_rate=1e-3, weights=weights)

    def freezer(trainer, pl_module):
        if pl_module.current_epoch == 0:
            # [mod.requires for mod in pl_module.modules() if mod.__module__=="torch.nn.modules.batchnorm"]
            for param in pl_module.model.backbone.parameters():
                param.requires_grad = False
        if pl_module.current_epoch == 12:
            for param in pl_module.model.backbone.encoder[7].parameters():
                param.requires_grad = True
        if pl_module.current_epoch == 20:
            for param in pl_module.model.backbone.encoder[6].parameters():
                param.requires_grad = True
        unfreeze_batchnorm(pl_module)
        # parameters = len(
        #     list(filter(lambda x: x.requires_grad, pl_module.model.parameters()))
        # )
        # print(parameters)

    def unfreeze_batchnorm(pl_module):
        for mod in pl_module.modules():
            if not mod.__module__ == "torch.nn.modules.batchnorm":
                continue
            for param in mod.parameters():
                param.requires_grad = True

    lb_callback = callbacks.LambdaCallback(on_train_epoch_start=freezer)
    checkpoint_clb = callbacks.ModelCheckpoint(monitor="val/loss", save_last=True)
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="step")
    # backbone_finetuning.freeze_before_training(model.model)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=49,
        # max_steps=100,
        # val_check_interval=0.25,
        precision=16,
        # log_every_n_steps=5,
        logger=logger,
        # callbacks=[lb_callback],
        # callbacks=[early_stopping],
        callbacks=[checkpoint_clb, lr_monitor],
        # auto_lr_find=True,
        # callbacks=[
        #     FTSCheckpoint(monitor="val/loss"),
        #     FTSEarlyStopping(monitor="val/loss"),
        #     FinetuningScheduler(
        #         # ft_schedule="/home/piotr/piotr/inz/indoor-semantic-segmentation/LitClassification_ft_schedule.yaml"
        #     ),
        # ],
        # accumulate_grad_batches=3,
        # log_every_n_steps=2,
        # overfit_batches=2,
    )
    # trainer = pl.Trainer()
    # lr_finder = trainer.tuner.lr_find(model)  # Run learning rate finder

    # fig = lr_finder.plot(suggest=True)  # Plot
    # fig.show()

    # model.hparams.lr = lr_finder.suggestion()

    # trainer.fit(
    #     model=model,
    #     datamodule=data_module
    #     # ckpt_path="/home/piotr/SensorsArticle2022/logs/23.11-max_real/version_0/checkpoints/epoch=19-step=300.ckpt",
    # )

    trainer.test(
        model=model,
        datamodule=data_module,
        verbose=True,
        ckpt_path="/mnt/c/Users/piotr.hondra/Documents/inz/indoor-semantic-segmentation/long-training/14z0ew2z/checkpoints/last.ckpt",
    )
