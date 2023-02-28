import os
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm

from src import multitask_datamod, multitask_model
from src.constants import weights
from src.multitask_lit import LitMultitask


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = os.path.join(PROJECT_DIR, "datasets")
dataloader_params = {"batch_size": 8, "pin_memory": False}
data_module = multitask_datamod.MultitaskDataModule(DATASET_PATH, dataloader_params)

data_module.setup("test")
test_loader = data_module.test_dataloader()
device = "cpu"
mlt_model = multitask_model.MultitaskNet()


model = Model(mlt_model)
torch.load("")

with torch.no_grad():
    model.eval()
    for batch in tqdm(test_loader):
        rgb, mask_targets, scene_targets = map(lambda x: x.to(device), batch)
        mask_preds, scene_preds = model(rgb)
