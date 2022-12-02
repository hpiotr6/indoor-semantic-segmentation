import albumentations as A
from albumentations.pytorch import ToTensorV2

t1 = A.Compose(
    [
        A.Resize(160, 256),
        A.augmentations.transforms.Normalize(),
        ToTensorV2(),
    ]
)
t2 = A.Compose(
    [
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
