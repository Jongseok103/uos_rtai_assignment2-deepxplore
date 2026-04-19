import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_resnet50_for_cifar10() -> nn.Module:
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_resnet50_for_cifar10()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_test_loader(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=img_tensor.device).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0.0, 1.0)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
