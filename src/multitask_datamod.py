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

    def setup(self, stage: str):
        data_class = dataset.NYUv2MultitaskDataset
        self.train_set = data_class(*self.get_dirs("train"), transform=self.transforms)

        self.val_set = data_class(*self.get_dirs("test"), transform=self.transforms)
        self.test_set = data_class(*self.get_dirs("test"), transform=self.transforms)

    def get_dirs(self, stage: str):
        if stage not in ["train", "test"]:
            raise KeyError()
        tasks = ["rgb", "semantic_13", "scene_class"]
        return [os.path.join(self.data_dir, "datasets", stage, task) for task in tasks]

    def train_dataloader(self):
        return data.DataLoader(self.train_set, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.kwargs)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.kwargs)


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, transforms=transforms.t2, kwargs=None):
        super().__init__()
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.transforms = transforms

    def setup(self, stage: str):
        data_class = dataset.NYUv2ClassificationDataset
        self.train_set = data_class(*self.get_dirs("train"), transform=self.transforms)

        self.val_set = data_class(*self.get_dirs("test"), transform=transforms.t2)
        self.test_set = data_class(*self.get_dirs("test"), transform=transforms.t2)

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
    def __init__(
        self, data_dir: str, transforms=transforms.t2, num_classes=40, kwargs=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.transforms = transforms
        self.num_classes = num_classes

    def setup(self, stage: str):
        data_class = dataset.NYUv2SegmentationDataset
        self.train_set = data_class(*self.get_dirs("train"), transform=self.transforms)

        self.val_set = data_class(*self.get_dirs("test"), transform=transforms.t2)
        self.test_set = data_class(*self.get_dirs("test"), transform=transforms.t2)

    def get_dirs(self, stage: str):
        if stage not in ["train", "test"]:
            raise KeyError()
        tasks = ["rgb", f"semantic_{self.num_classes}"]
        return [os.path.join(self.data_dir, "datasets", stage, task) for task in tasks]

    def train_dataloader(self):
        return data.DataLoader(self.train_set, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, **self.kwargs)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, **self.kwargs)
