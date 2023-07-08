from torchvision import transforms


def setup_crop_flip_transform(input_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
        ]
    )
