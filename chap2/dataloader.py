import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def create_dataloder_ver1():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataloader


def create_dataloder_ver2():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=0
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_val = torch.tensor(y_val, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader
