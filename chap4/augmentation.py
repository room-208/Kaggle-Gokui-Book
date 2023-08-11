import albumentations as albu


def get_augmentations(input_size):
    train_transform = albu.Compose(
        [
            albu.RandomResizedCrop(
                input_size,
                input_size,
                scale=(0.6, 1.0),
                p=1.0,
            ),
            albu.Normalize(),
        ]
    )
    test_transform = albu.Compose(
        [
            albu.Resize(width=input_size, height=input_size),
            albu.Normalize(),
        ]
    )
    return train_transform, test_transform
