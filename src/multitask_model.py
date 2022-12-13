import segmentation_models_pytorch as smp
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = nn.Sequential(*list(encoder.children()))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, decoder, segmentation_head, segmentation_classes) -> None:
        super().__init__()
        self.decoder = nn.Sequential(*list(decoder.children()))
        segmentation_head[2].activation = nn.Softmax(dim=1)
        self.segmentation_head = nn.Sequential(*list(segmentation_head.children()))

    def forward(self, x):
        dec = self.decoder(x)
        res = self.segmentation_head(dec)
        return res


class MultitaskNet(nn.Module):
    def __init__(
        self, stage="multitask", scene_classes=7, segmentation_classes=27
    ) -> None:
        super().__init__()
        segmentator = smp.DeepLabV3(
            encoder_weights="imagenet",
            classes=segmentation_classes,
        )

        self.backbone = Encoder(segmentator.encoder)
        # self.decoder = Decoder(
        #     segmentator.decoder, segmentator.segmentation_head, segmentation_classes
        # )

        num_filters = segmentator.encoder.out_channels[-1]
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
        segmentator = None

        if stage == "classification":
            self.forward = self.forward_classification
        elif stage == "segmentation":
            self.forward = self.forward_segmentation
        elif stage == "multitask":
            self.forward = self.forward_multitask
        else:
            raise KeyError()

    def forward_multitask(self, x):
        representations = self.backbone(x)
        segmentation_mask = self.decoder(representations)
        classification_label = self.classifier(representations)
        return segmentation_mask, classification_label

    def forward_classification(self, x):
        representations = self.backbone(x)
        classification_label = self.classifier(representations)
        return classification_label

    def forward_segmentation(self, x):
        representations = self.backbone(x)
        segmentation_mask = self.decoder(representations)
        return segmentation_mask


# model = MultitaskNet()
# # print(model)
# randnum = torch.rand((3, 3, 480, 640))
# x = model(randnum)
