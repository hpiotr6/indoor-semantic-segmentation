import os
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils import data
from . import dataset


class MultitaskDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.kwargs = kwargs

    def setup(self, stage: str):

        self.train_set = dataset.NYUv2MultitaskDataset(
            *self.get_dirs("train"), transform=None
        )

        self.val_set = dataset.NYUv2MultitaskDataset(
            *self.get_dirs("test"), transform=None
        )
        self.test_set = dataset.NYUv2MultitaskDataset(
            *self.get_dirs("test"), transform=None
        )

    def get_dirs(self, stage: str):
        if stage not in ["train", "test"]:
            raise KeyError()
        tasks = ["rgb", "semantic_40", "scene_class"]
        return [os.path.joins(self.data_dir, "datasets", stage, task) for task in tasks]

    def train_dataloader(self):
        return data.DataLoader(self.train_set, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.kwargs)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.kwargs)


PROJECT_DIR = Path(__file__).resolve().parent
dataloader_params = {"batch_size": 8, "num_workers": 12, "pin_memory": False}
MultitaskDataModule(PROJECT_DIR, dataloader_params)
