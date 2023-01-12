from PIL import Image
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
from src import dataset

from src.constants import weights
from src import transforms
from torchmetrics import classification
import segmentation_models_pytorch as smp
from torch.utils import data
from torch import nn

import timm


# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(f"Using device {torch.cuda.current_device()}")


def get_dirs(root: str, stage: str):
    if stage not in ["train", "test"]:
        raise KeyError()
    tasks = ["rgb", "semantic_13", "scene_class"]
    return [os.path.join(root, "datasets", stage, task) for task in tasks]


PROJECT_DIR = Path(__file__).resolve().parent
data_class = dataset.NYUv2MultitaskDataset
tfs = transforms.t2
train_set = data_class(*get_dirs(PROJECT_DIR, "train"), transform=tfs)
val_set = data_class(*get_dirs(PROJECT_DIR, "test"), transform=tfs)
test_set = data_class(*get_dirs(PROJECT_DIR, "test"), transform=tfs)

dataloader_params = {"batch_size": 8, "num_workers": 6, "pin_memory": False}
train_loader = data.DataLoader(train_set, shuffle=True, **dataloader_params)
val_loader = data.DataLoader(val_set, **dataloader_params)
test_loader = data.DataLoader(test_set, **dataloader_params)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.DeepLabV3(
    encoder_weights="imagenet",
    activation="softmax2d",
    classes=13,
    aux_params={"classes": 7, "activation": "softmax2d"},
).to(device)


# model = timm.create_model("resnet34", pretrained=True)
# model.fc = nn.Linear(in_features=512, out_features=7)
# model = model.to(device)

seg_test_metric = classification.MulticlassAccuracy(
    num_classes=13, ignore_index=255
).to(device)
scene_test_metric = classification.MulticlassAccuracy(num_classes=7).to(device)
# Load model and move to GPU

# Set up criterion and optimizer
seg_criterior = torch.nn.CrossEntropyLoss(
    weight=weights.SEG_WEIGHTS.to(device), ignore_index=255
)
scene_criterior = torch.nn.CrossEntropyLoss(weight=weights.SCENE_WEIGHTS.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
EPOCHS = 5
# Loop over data
# for epoch in range(EPOCHS):
#     model.train()
#     train_pbar = tqdm(train_loader, desc=f"Training {epoch}", leave=False)
#     for batch in train_pbar:
#         optimizer.zero_grad()
#         rgb, mask_targets, scene_targets = map(lambda x: x.to(device), batch)
#         mask_preds, scene_preds = model(rgb)
#         scene_loss = scene_criterior(scene_preds, scene_targets.long())
#         seg_loss = seg_criterior(mask_preds, mask_targets.long())
#         loss = scene_loss + seg_loss
#         loss.backward()
#         optimizer.step()
#         train_pbar.set_postfix(
#             {
#                 "seg_loss": seg_loss.item(),
#                 "scene_loss": scene_loss.item(),
#                 "loss": loss.item(),
#             }
#         )

#     with torch.no_grad():
#         model.eval()

#         val_pbar = tqdm(train_loader, desc=f"Validating {epoch}", leave=False)
#         for batch in val_pbar:
#             rgb, mask_targets, scene_targets = map(lambda x: x.to(device), batch)
#             mask_preds, scene_preds = model(rgb)
#             scene_loss = scene_criterior(scene_preds, scene_targets.long())
#             seg_loss = seg_criterior(mask_preds, mask_targets.long())

#             val_pbar.set_postfix(
#                 {
#                     "seg_loss": seg_loss.item(),
#                     "scene_loss": scene_loss.item(),
#                 }
#             )
#     train_pbar.close()
#     val_pbar.close()

# torch.save(model.state_dict(), "manual_model.pth")
model.load_state_dict(torch.load("manual_model.pth"))

image = np.asarray(Image.open("IMG-0650.jpg").convert("RGB"), dtype=np.float32)

transformed = transforms.infer(image=image)
tensor_image = transformed["image"].to(device)

with torch.no_grad():
    model.eval()
    mask_preds, scene_preds = model(tensor_image.unsqueeze(0))
    mask_pred = torch.argmax(mask_preds, dim=1)

# plt.subplot(121)
plt.figure()
plt.imshow(mask_pred.detach().cpu().permute(1, 2, 0).numpy())

plt.show()
# with torch.no_grad():
#     model.eval()
#     for batch in tqdm(test_loader):
#         rgb, mask_targets, scene_targets = map(lambda x: x.to(device), batch)
#         mask_preds, scene_preds = model(rgb)
#         scene_test_metric(scene_preds, scene_targets)
#         seg_test_metric(mask_preds, mask_targets)

# acc_seg = seg_test_metric.compute()
# seg_test_metric.reset()
# print("test_acc_seg", acc_seg.item())

# acc_scene = scene_test_metric.compute()
# scene_test_metric.reset()
# print("test_acc_scene", acc_scene.item())
