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
    """실험 재현하려고 seed 맞춰두는 함수."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_resnet50_for_cifar10() -> nn.Module:
    """CIFAR-10 checkpoint에 맞는 ResNet50 구조 만드는 부분."""
    model = models.resnet50(weights=None)
    # ImageNet용 기본 stem은 32x32에 좀 과해서 CIFAR-10에 맞게 줄여둠.
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
    """checkpoint 읽어서 eval 모드로 준비하는 함수."""
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
    """실행기에 넘길 CIFAR-10 test loader 만드는 부분."""
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
    """normalize된 이미지를 시각화용으로 다시 돌려놓는 함수."""
    # 시각화할 때는 normalize를 풀어야 원래 이미지처럼 보임.
    mean = torch.tensor(CIFAR10_MEAN, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=img_tensor.device).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0.0, 1.0)


def ensure_parent_dir(path: str) -> None:
    """파일 저장 전에 상위 폴더 없으면 미리 만들어두는 함수."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
