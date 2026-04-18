import os
import csv
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_resnet50_for_cifar10()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_test_loader(batch_size: int = 128, num_workers: int = 2) -> DataLoader:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_and_collect(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[Dict]]:
    correct_a = 0
    correct_b = 0
    total = 0
    disagreements: List[Dict] = []

    global_index = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_a = model_a(images)
        logits_b = model_b(images)

        preds_a = logits_a.argmax(dim=1)
        preds_b = logits_b.argmax(dim=1)

        correct_a += (preds_a == labels).sum().item()
        correct_b += (preds_b == labels).sum().item()
        total += labels.size(0)

        batch_size = labels.size(0)
        for i in range(batch_size):
            if preds_a[i].item() != preds_b[i].item():
                disagreements.append({
                    "index": global_index + i,
                    "image": images[i].detach().cpu(),
                    "true_label": labels[i].item(),
                    "pred_a": preds_a[i].item(),
                    "pred_b": preds_b[i].item(),
                })

        global_index += batch_size

    acc_a = correct_a / total
    acc_b = correct_b / total
    return acc_a, acc_b, disagreements


def save_csv(disagreements: List[Dict], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_a", "pred_b"])
        for item in disagreements:
            writer.writerow([
                item["index"],
                CIFAR10_CLASSES[item["true_label"]],
                CIFAR10_CLASSES[item["pred_a"]],
                CIFAR10_CLASSES[item["pred_b"]],
            ])


def save_visualizations(disagreements: List[Dict], output_dir: str, max_save: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)

    num_to_save = min(max_save, len(disagreements))
    for i in range(num_to_save):
        item = disagreements[i]
        image = denormalize(item["image"]).permute(1, 2, 0).numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"idx={item['index']}\n"
            f"true={CIFAR10_CLASSES[item['true_label']]}\n"
            f"model_a={CIFAR10_CLASSES[item['pred_a']]}, "
            f"model_b={CIFAR10_CLASSES[item['pred_b']]}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"baseline_disagreement_{i+1:02d}.png"), dpi=200)
        plt.close()


def main():
    set_seed(42)

    os.makedirs("results", exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this on a GPU node if you trained the models there."
        )

    device = torch.device("cuda")
    print("Using device:", device)
    print("GPU name:", torch.cuda.get_device_name(0))

    model_a_path = "models/model_a.pth"
    model_b_path = "models/model_b.pth"

    if not os.path.exists(model_a_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_a_path}")
    if not os.path.exists(model_b_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_b_path}")

    model_a = load_model(model_a_path, device)
    model_b = load_model(model_b_path, device)

    test_loader = get_test_loader(batch_size=128, num_workers=2)

    acc_a, acc_b, disagreements = evaluate_and_collect(
        model_a=model_a,
        model_b=model_b,
        loader=test_loader,
        device=device,
    )

    print(f"Model A test accuracy: {acc_a:.4f}")
    print(f"Model B test accuracy: {acc_b:.4f}")
    print(f"Number of baseline disagreements: {len(disagreements)}")

    csv_path = "results/baseline_disagreements.csv"
    save_csv(disagreements, csv_path)
    save_visualizations(disagreements, output_dir="results", max_save=5)

    print(f"Saved CSV to: {csv_path}")
    print("Saved baseline disagreement visualizations to results/")


if __name__ == "__main__":
    main()