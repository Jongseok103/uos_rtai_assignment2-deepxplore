import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


@dataclass
class TrainConfig:
    # 모델 두 개를 일부러 다르게 학습시켜야 disagreement가 잘 생긴다.
    model_name: str
    seed: int
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 15
    lr: float = 0.1
    weight_decay: float = 5e-4
    optimizer_name: str = "sgd"   # "sgd" or "adamw"
    use_strong_aug: bool = False
    save_path: str = ""


def set_seed(seed: int) -> None:
    """실험 다시 돌렸을 때 결과가 크게 안 흔들리게 seed 고정하는 함수임."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(use_strong_aug: bool):
    """CIFAR-10 학습/평가용 transform 만드는 부분."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if use_strong_aug:
        # model_b 쪽은 augmentation을 더 세게 줘서
        # model_a랑 decision boundary가 좀 다르게 나오게 해둠.
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def build_dataloaders(batch_size: int, num_workers: int, use_strong_aug: bool):
    """학습에 바로 쓸 train/test DataLoader 만드는 함수."""
    train_transform, test_transform = build_transforms(use_strong_aug)

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def build_resnet50_for_cifar10() -> nn.Module:
    """ResNet50을 CIFAR-10용으로 살짝 바꿔서 만드는 함수."""
    model = models.resnet50(weights=None)

    # CIFAR-10이 32x32라서 앞단 stem을 좀 가볍게 바꿔줌.
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


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    """설정값 보고 optimizer 고르는 부분."""
    # optimizer도 다르게 써서 두 모델 성향 차이가 조금 나게 함.
    if cfg.optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    if cfg.optimizer_name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """현재 모델 test accuracy 보는 함수."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_one_model(cfg: TrainConfig, device: torch.device):
    """모델 하나 학습시키고 best checkpoint 저장까지 담당하는 함수."""
    print(f"\n===== Training {cfg.model_name} =====")
    print(cfg)

    set_seed(cfg.seed)
    train_loader, test_loader = build_dataloaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_strong_aug=cfg.use_strong_aug,
    )

    model = build_resnet50_for_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    if cfg.optimizer_name.lower() == "sgd":
        # SGD일 때만 cosine scheduler 붙였음.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = None

    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)

        print(
            f"[{cfg.model_name}] "
            f"epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            # 마지막 epoch가 아니라 제일 잘 나온 시점 저장하는 방식임.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_test_acc": best_acc,
                },
                cfg.save_path,
            )

    print(f"Saved best {cfg.model_name} to: {cfg.save_path}")
    print(f"Best test acc: {best_acc:.4f}")


def main():
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # seed, optimizer, augmentation을 다르게 줘서
    # 두 모델이 같은 데이터라도 예측이 좀 갈리게 유도했음.
    cfg_a = TrainConfig(
        model_name="model_a",
        seed=42,
        epochs=15,
        batch_size=128,
        lr=0.1,
        optimizer_name="sgd",
        use_strong_aug=False,
        save_path="models/model_a.pth",
    )

    cfg_b = TrainConfig(
        model_name="model_b",
        seed=123,
        epochs=15,
        batch_size=128,
        optimizer_name="adamw",
        use_strong_aug=True,
        save_path="models/model_b.pth",
    )

    train_one_model(cfg_a, device)
    train_one_model(cfg_b, device)


if __name__ == "__main__":
    main()
