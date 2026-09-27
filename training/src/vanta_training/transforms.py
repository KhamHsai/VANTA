"""Image preprocessing shared by training, evaluation, and prediction."""

from torchvision import transforms

from vanta_training.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_training_transform() -> transforms.Compose:
    """Return moderate augmentation suitable for object-centered trash images."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_inference_transform() -> transforms.Compose:
    """Return deterministic ImageNet preprocessing for validation and inference."""

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
