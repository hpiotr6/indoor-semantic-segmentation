import segmentation_models_pytorch as smp
import torch
from torch import nn


class MultitaskNet(nn.Module):
    def __init__(self, scene_classes=7, segmentation_classes=13) -> None:
        super().__init__()
        segmentator = smp.DeepLabV3(
            encoder_weights="imagenet",
            classes=segmentation_classes,
        )
        self.encoder = segmentator.encoder
        self.decoder = segmentator.decoder
        self.segmentation_head = segmentator.segmentation_head
        num_filters = self.encoder.out_channels[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(p=0.25),
            nn.Linear(num_filters, out_features=256, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=256, out_features=scene_classes, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        classification_label = self.classifier(features[-1])
        segmentation_mask = self.decoder(*features)
        segmentation_mask = self.segmentation_head(segmentation_mask)
        return segmentation_mask, classification_label

    #     representations = self.backbone(x)
    #     segmentation_mask = self.decoder(representations)
    #     classification_label = self.classifier(representations)
    #     return segmentation_mask, classification_label
