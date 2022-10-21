from PIL import Image
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class NYUv2Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None) -> None:
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        mask_path = os.path.join(self.mask_dir, self.imgs[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask


def test():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    TRAIN_RGB = os.path.join(PROJECT_DIR, "datasets", "train_rgb")
    TRAIN_SEG = os.path.join(PROJECT_DIR, "datasets", "train_seg13")
    dataset = NYUv2Dataset(img_dir=TRAIN_RGB, mask_dir=TRAIN_SEG)
    plt.imshow(dataset[0][1])
    plt.show()


if __name__ == "__main__":
    test()
