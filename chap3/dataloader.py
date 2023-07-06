import os

import numpy as np
from sklearn import model_selection
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def setup_train_val_split(labels, dryrun=True, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)
    spilitter = model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, random_state=seed
    )
    train_indices, val_indices = next(spilitter.split(x, y))

    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace=False)
        val_indices = np.random.choice(val_indices, 100, replace=False)

    return train_indices, val_indices


def setup_center_crop_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),
        ]
    )


def get_labels(dataset):
    assert not isinstance(dataset, Subset)
    return np.array([img[1] for img in dataset.imgs])


def setup_train_val_datasets(data_dir, dryrun=False):
    dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=setup_center_crop_transform(),
    )
    labels = get_labels(dataset)
    train_indices, val_indices = setup_train_val_split(labels, dryrun)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def setup_train_val_loaders(data_dir, batch_size, dryrun=False):
    train_dataset, val_dataset = setup_train_val_datasets(data_dir, dryrun)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
    )
    return train_loader, val_loader
