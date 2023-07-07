from torchvision import transforms


def setup_center_crop_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
        ]
    )


def setup_crop_flip_transform():
    return transforms.Compose(
        [
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
        ]
    )
