import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from PIL import Image

from . import constants


class NYUv2SegmentationDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(mask_path), dtype=np.float32)

        mask[mask == 0] = 255
        mask[mask == 40] = 0
        # assert len(np.unique(mask)) < 40, np.unique(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


class NYUv2ClassificationDataset(data.Dataset):
    def __init__(self, image_dir, scene_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.scene_ids = constants.SCENE_IDS

        self.images = sorted(os.listdir(image_dir))
        self.scenes = self._read_scenes(scene_dir)

    def _read_scenes(self, scene_dir) -> list:
        scenes = []
        for filename in os.listdir(scene_dir):
            file_path = os.path.join(scene_dir, filename)

            with open(file_path, "r") as f:
                scene = f.readline()
                scenes.append(scene)
        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path))
        scene = self.scene_ids[self.scenes[index]]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            scene = torch.tensor(scene)

        return image, scene


def test():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    TRAIN_RGB = os.path.join(PROJECT_DIR, "datasets", "train", "rgb")
    TRAIN_SEG = os.path.join(PROJECT_DIR, "datasets", "train", "semantic_40")
    dataset = NYUv2SegmentationDataset(image_dir=TRAIN_RGB, mask_dir=TRAIN_SEG)
    plt.imshow(dataset[0][1])
    plt.show()


if __name__ == "__main__":
    test()
