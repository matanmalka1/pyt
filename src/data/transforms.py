from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD


def get_transforms(split: str, img_size: int, augment: bool = True) -> transforms.Compose:
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if split == "train" and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])
