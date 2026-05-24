import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 2,
    seed: int = 42,
):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
    full_val   = datasets.CIFAR10(data_dir, train=True,  download=True, transform=eval_transform)
    test_set   = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    train_size = len(full_train) - val_size
    train_indices, val_indices = random_split(
        range(len(full_train)), [train_size, val_size], generator=generator
    )

    train_subset = torch.utils.data.Subset(full_train, train_indices.indices)
    val_subset   = torch.utils.data.Subset(full_val,   val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,     batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] Train: {len(train_subset):,} | Val: {len(val_subset):,} | Test: {len(test_set):,}")
    return train_loader, val_loader, test_loader