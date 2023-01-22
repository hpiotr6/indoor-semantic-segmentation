
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
from src.multitask_lit import LitClassif
from functools import partial
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def freeze_layers(model, regex):
    for name, p in model.named_parameters():
        if not bool(regex(string=name)):
            p.requires_grad = True
        else:
            p.requires_grad = False
pl.seed_everything(42, workers=True)

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = os.path.join(PROJECT_DIR, "datasets")
dataloader_params = {"batch_size": 8, "num_workers": 12, "pin_memory": False}
data_module = multitask_datamod.MultitaskDataModule(DATASET_PATH, dataloader_params)
checkpoint_seg = torch.load("/mnt/c/Users/piotr.hondra/Documents/inz/indoor-semantic-segmentation/logs_to_convert/segmentation_only/lr-1.268e-04/checkpoints/epoch=4-step=500.ckpt")
def objective(trial: optuna.trial.Trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-7,1e-4)
    output_shape = trial.suggest_int("adptshape", 1,15)

    logger = loggers.TensorBoardLogger(
        save_dir="logs-2",
        name="classification_serial_correct_freeze",
        # version="softmax1d",
    )
    checkpoint_path=os.path.join(logger.log_dir, "checkpoints")
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=checkpoint_path, monitor="val/scene_loss")
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="step")
    early_stopping = callbacks.EarlyStopping(monitor="val/scene_loss", patience=5)
    trainer = pl.Trainer(
        auto_lr_find=True,
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        precision=16,
        logger=logger,
        callbacks=[lr_monitor, early_stopping, checkpoint_callback,PyTorchLightningPruningCallback(trial, monitor="val/scene_loss")],
    )
    mlt_model = multitask_model.ClassifiFromSeg(out_shape=output_shape,)
        


    model = LitClassif(
        model=mlt_model,
        learning_rate=lr,
        scene_weights=weights.SCENE_WEIGHTS,
    )
    model.load_state_dict(checkpoint_seg["state_dict"], strict=False)

    regex = partial(re.search, pattern="enc|segm|dec")
    freeze_layers(model.model, regex)

    with timer.timer(logger):
        trainer.fit(
            model=model,
            datamodule=data_module,
        )

    result = trainer.callback_metrics["val/scene_loss"].item()
    filename = os.listdir(checkpoint_path)[0]
    trainer.test(
        model=model,
        datamodule=data_module,
        verbose=True,
        ckpt_path=os.path.join(checkpoint_path, filename)
    )
    with open(os.path.join(logger.log_dir, "opt_params.txt"), 'w') as f:
        f.write(str(trial.params))
    return result

if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    