import timm
import torch
import torch.nn as nn
import torchvision.models as models

# class ResNet(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         # Use a pretrained model
#         self.network = models.resnet18(weights=True)
#         # Replace last layer
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Linear(num_ftrs, n_classes)

#     def forward(self, xb):
#         return torch.sigmoid(self.network(xb))


class ImagenetTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # init a pretrained resnet
        # backbone = models.resnet34(weights=True)

        backbone = timm.create_model("resnet34", pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(num_filters),
            # nn.Dropout(p=0.25),
            # nn.Linear(num_filters, out_features=512, bias=False),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(512),
            # nn.Dropout(p=0.25),
            # nn.Linear(in_features=512, out_features=num_classes, bias=False),
            nn.Linear(in_features=num_filters, out_features=num_classes, bias=False),
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        # self.feature_extractor.eval()
        # with torch.no_grad():
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        # softmax!!!
        return self.softmax(x)


model = ImagenetTransferLearning(5)

# # print(model.feature_extractor[:7])
print(model)
