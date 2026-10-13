"""
Image transforms shared by training, evaluation and the inference engine.

Keeping them in one place is what guarantees the serving path normalises
images exactly like the training path did.
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int, train: bool = False) -> transforms.Compose:
    """Squash to ``image_size`` square, normalise with ImageNet statistics.

    Eval/serving always squashes the *whole* image to a square so every one of
    the 49 attention cells maps onto a fixed region of the picture. The train
    pipeline squashes first too (so geometry matches), then applies a mild
    crop + colour jitter. There is deliberately no horizontal flip: captions
    mention "left"/"right" and a flip would silently make them wrong.
    """
    steps: list = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomResizedCrop(
                image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)
