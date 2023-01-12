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
        # A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
infer = A.Compose(
    [
        # A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.Resize(480, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# extra_transforms = (
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
#     transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3, fill=0),
# )

mean = (0.4302, 0.4575, 0.4539)
std = (0.2606, 0.2588, 0.2907)
train_transform = A.Compose(
    [
        # A.SmallestMaxSize(max_size=160),
        A.HueSaturationValue(),
        A.VerticalFlip(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        # A.RandomCrop(height=128, width=128),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.CoarseDropout(),
        ToTensorV2(),
    ]
)


transform_net = A.Compose(
    [
        # # A.RandomRotate90(),
        # A.Flip(),
        # # A.Transpose(),
        # A.OneOf(
        #     [
        #         A.MotionBlur(p=0.2),
        #         A.MedianBlur(blur_limit=3, p=0.3),
        #         A.Blur(blur_limit=3, p=0.1),
        #     ],
        #     p=0.2,
        # ),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # A.OneOf(
        #     [
        #         A.OpticalDistortion(p=0.3),
        #         A.GridDistortion(p=0.1),
        #     ],
        #     p=0.2,
        # ),
        # A.OneOf(
        #     [
        #         A.CLAHE(clip_limit=2),
        #         A.RandomBrightnessContrast(),
        #     ],
        #     p=0.3,
        # ),
        # A.HueSaturationValue(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# transformTrain1 = A.Compose(
#     [
#         A.RandomHorizontalFlip(),
#         # transforms.Resize([256, 256]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#         RandomErasing(),
#         ToTensorV2(),
#     ]
# )

# transformTrain2 = A.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply(extra_transforms, p=0.2),
#         transforms.Resize([256, 256]),
#         transforms.ToTensor(),
#         transforms.RandomErasing(scale=(0.02, 0.2)),
#     ]
# )

# # Transforms for test data
# transformTest = A.Compose(
#     [
#         # transforms.Resize([256, 256]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ]
# )
