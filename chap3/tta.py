from torchvision import transforms


def setup_tta_transforms():
    return [
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.functional.hflip,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
            ]
        ),
    ]
