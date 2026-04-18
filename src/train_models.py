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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(use_strong_aug: bool):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if use_strong_aug:
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
    model = models.resnet50(weights=None)

    # CIFAR-10 (32x32)에 맞게 초기 stem을 가볍게 조정
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