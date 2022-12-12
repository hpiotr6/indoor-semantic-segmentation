import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils import data

from . import dataset, transforms


class MultitaskDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.transforms = transforms.t2
        self.save_hyperparameters()

    def setup(self, stage: str):
        data_class = dataset.NYUv2MultitaskDataset
        self.train_set = data_class(*self.get_dirs("train"), transform=self.transforms)

        self.val_set = data_class(*self.get_dirs("test"), transform=self.transforms)
        self.test_set = data_class(*self.get_dirs("test"), transform=self.transforms)

    def get_dirs(self, stage: str):
        if stage not in ["train", "test"]:
            raise KeyError()
        tasks = ["rgb", "semantic_40", "scene_class"]
        return [os.path.join(self.data_dir, "datasets", stage, task) for task in tasks]

    def train_dataloader(self):
        return data.DataLoader(self.train_set, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.kwargs)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.kwargs)


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.transforms = transforms.t2
        self.save_hyperparameters()

    def setup(self, stage: str):
        data_class = dataset.NYUv2ClassificationDataset
        self.train_set = data_class(*self.get_dirs("train"), transform=self.transforms)

        self.val_set = data_class(*self.get_dirs("test"), transform=self.transforms)
        self.test_set = data_class(*self.get_dirs("test"), transform=self.transforms)

    def get_dirs(self, stage: str):
        if stage not in ["train", "test"]:
            raise KeyError()
        tasks = ["rgb", "scene_class"]
        return [os.path.join(self.data_dir, "datasets", stage, task) for task in tasks]

    def train_dataloader(self):
        return data.DataLoader(self.train_set, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.kwargs)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.kwargs)


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.transforms = transforms.t2
        self.save_hyperparameters()

    def setup(self, stage: str):
        data_class = dataset.NYUv2SegmentationDataset
        self.train_set = data_class(*self.get_dirs("train"), transform=self.transforms)

        self.val_set = data_class(*self.get_dirs("test"), transform=self.transforms)
        self.test_set = data_class(*self.get_dirs("test"), transform=self.transforms)

    def get_dirs(self, stage: str):
        if stage not in ["train", "test"]:
            raise KeyError()
        tasks = ["rgb", "semantic_40"]
        return [os.path.join(self.data_dir, "datasets", stage, task) for task in tasks]

    def train_dataloader(self):
        return data.DataLoader(self.train_set, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.kwargs)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.kwargs)
