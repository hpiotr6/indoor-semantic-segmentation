import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(weights=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


class ImagenetTransferLearning(nn.Module):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        num_target_classes = 27
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_filters),
            nn.Dropout(p=0.25),
            nn.Linear(num_filters, out_features=512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=num_target_classes, bias=False),
        )

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return torch.sigmoid(x)
