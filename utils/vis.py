import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

plt.rcParams["savefig.bbox"] = "tight"


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def visualize_batch(batch):
    x, y = batch
    show(make_grid(x))


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def imshow(out, title=None):
    inp = torchvision.utils.make_grid(out)
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    # plt.show()
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated


def imshow_semantic(masks):
    inp = torchvision.utils.make_grid(masks.unsqueeze(1))
    inp = inp.numpy().transpose((1, 2, 0))
    # inp = np.clip(inp, 0, 255)
    plt.imshow(inp)


def check_dataloader_distribution(train_batch):
    ys = []
    for batch in train_batch:
        x, y = batch
        ys.append(y)
    values, counts = np.unique(torch.hstack(ys), return_counts=True)
    pd.DataFrame(dict(values, counts))
