import os
import csv
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

from coverage import NeuronCoverage


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
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
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


def get_test_loader(batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0.0, 1.0)


@torch.no_grad()
def collect_seed_disagreements(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_seeds: int = 20,
) -> List[Dict]:
    seeds: List[Dict] = []
    global_index = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_a = model_a(images)
        logits_b = model_b(images)

        preds_a = logits_a.argmax(dim=1)
        preds_b = logits_b.argmax(dim=1)

        for i in range(labels.size(0)):
            if preds_a[i].item() != preds_b[i].item():
                seeds.append({
                    "index": global_index + i,
                    "image": images[i].detach(),
                    "label": labels[i].item(),
                    "pred_a": preds_a[i].item(),
                    "pred_b": preds_b[i].item(),
                })
                if len(seeds) >= max_seeds:
                    return seeds

        global_index += labels.size(0)

    return seeds


def clamp_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, epsilon: float) -> torch.Tensor:
    delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
    return x_orig + delta


def normalize_to_valid_range(x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
    # Because input is normalized CIFAR-10 tensor, we only use epsilon projection.
    # Final visualization is denormalized separately.
    return x_adv


def compute_coverage_gain(
    coverage_a: NeuronCoverage,
    coverage_b: NeuronCoverage,
    model_a: nn.Module,
    model_b: nn.Module,
    x: torch.Tensor,
) -> float:
    coverage_a.reset()
    coverage_b.reset()

    with torch.no_grad():
        _ = model_a(x)
        _ = model_b(x)

    cov_a = coverage_a.coverage_ratio()
    cov_b = coverage_b.coverage_ratio()
    return 0.5 * (cov_a + cov_b)


def objective_fn(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    target_a: int,
    target_b: int,
) -> torch.Tensor:
    """
    Encourage:
    - model A to stay confident on target_a
    - model B to stay confident on target_b
    - disagreement margin between their respective preferred logits
    """
    score_a = logits_a[0, target_a]
    score_b = logits_b[0, target_b]
    return score_a + score_b


def generate_adversarial_disagreement(
    model_a: nn.Module,
    model_b: nn.Module,
    coverage_a: NeuronCoverage,
    coverage_b: NeuronCoverage,
    seed: Dict,
    device: torch.device,
    epsilon: float = 0.03,
    alpha: float = 0.005,
    steps: int = 20,
) -> Dict:
    x_orig = seed["image"].unsqueeze(0).to(device)
    target_a = seed["pred_a"]
    target_b = seed["pred_b"]

    before_cov = compute_coverage_gain(coverage_a, coverage_b, model_a, model_b, x_orig)

    x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits_a = model_a(x_adv)
        logits_b = model_b(x_adv)

        loss = objective_fn(logits_a, logits_b, target_a, target_b)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv + alpha * grad.sign()
        x_adv = clamp_linf(x_adv, x_orig, epsilon)
        x_adv = normalize_to_valid_range(x_adv, x_orig)
        x_adv = x_adv.detach()

    with torch.no_grad():
        final_logits_a = model_a(x_adv)
        final_logits_b = model_b(x_adv)

        final_pred_a = final_logits_a.argmax(dim=1).item()
        final_pred_b = final_logits_b.argmax(dim=1).item()

    after_cov = compute_coverage_gain(coverage_a, coverage_b, model_a, model_b, x_adv)

    perturbation = (x_adv - x_orig).view(-1)
    linf = perturbation.abs().max().item()
    l2 = torch.norm(perturbation, p=2).item()

    return {
        "index": seed["index"],
        "x_orig": x_orig.squeeze(0).detach().cpu(),
        "x_adv": x_adv.squeeze(0).detach().cpu(),
        "true_label": seed["label"],
        "seed_pred_a": seed["pred_a"],
        "seed_pred_b": seed["pred_b"],
        "final_pred_a": final_pred_a,
        "final_pred_b": final_pred_b,
        "before_cov": before_cov,
        "after_cov": after_cov,
        "linf": linf,
        "l2": l2,
        "success": final_pred_a != final_pred_b,
    }


def save_result_figure(result: Dict, output_path: str) -> None:
    orig = denormalize(result["x_orig"]).permute(1, 2, 0).numpy()
    adv = denormalize(result["x_adv"]).permute(1, 2, 0).numpy()
    diff = (adv - orig)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(orig)
    axes[0].set_title(
        f"Original\ntrue={CIFAR10_CLASSES[result['true_label']]}\n"
        f"A={CIFAR10_CLASSES[result['seed_pred_a']]}, "
        f"B={CIFAR10_CLASSES[result['seed_pred_b']]}"
    )
    axes[0].axis("off")

    axes[1].imshow(adv)
    axes[1].set_title(
        f"Generated\nA={CIFAR10_CLASSES[result['final_pred_a']]}, "
        f"B={CIFAR10_CLASSES[result['final_pred_b']]}"
    )
    axes[1].axis("off")

    axes[2].imshow(diff)
    axes[2].set_title(
        f"Diff\nLinf={result['linf']:.4f}\n"
        f"cov {result['before_cov']:.4f} -> {result['after_cov']:.4f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_summary_csv(results: List[Dict], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "true_label",
            "seed_pred_a",
            "seed_pred_b",
            "final_pred_a",
            "final_pred_b",
            "before_cov",
            "after_cov",
            "linf",
            "l2",
            "success",
        ])
        for r in results:
            writer.writerow([
                r["index"],
                CIFAR10_CLASSES[r["true_label"]],
                CIFAR10_CLASSES[r["seed_pred_a"]],
                CIFAR10_CLASSES[r["seed_pred_b"]],
                CIFAR10_CLASSES[r["final_pred_a"]],
                CIFAR10_CLASSES[r["final_pred_b"]],
                f"{r['before_cov']:.6f}",
                f"{r['after_cov']:.6f}",
                f"{r['linf']:.6f}",
                f"{r['l2']:.6f}",
                int(r["success"]),
            ])


def main():
    set_seed(42)
    os.makedirs("results", exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run on a GPU node.")

    device = torch.device("cuda")
    print("Using device:", device)
    print("GPU name:", torch.cuda.get_device_name(0))

    model_a = load_model("models/model_a.pth", device)
    model_b = load_model("models/model_b.pth", device)

    loader = get_test_loader(batch_size=64, num_workers=2)

    coverage_a = NeuronCoverage(model_a, threshold=0.0)
    coverage_b = NeuronCoverage(model_b, threshold=0.0)

    seeds = collect_seed_disagreements(
        model_a=model_a,
        model_b=model_b,
        loader=loader,
        device=device,
        max_seeds=20,
    )

    if len(seeds) == 0:
        raise RuntimeError("No baseline disagreement seeds found.")

    print(f"Collected {len(seeds)} baseline disagreement seeds.")

    results: List[Dict] = []
    saved_count = 0

    for i, seed in enumerate(seeds):
        result = generate_adversarial_disagreement(
            model_a=model_a,
            model_b=model_b,
            coverage_a=coverage_a,
            coverage_b=coverage_b,
            seed=seed,
            device=device,
            epsilon=0.03,
            alpha=0.005,
            steps=20,
        )

        results.append(result)

        print(
            f"[{i+1}/{len(seeds)}] idx={result['index']} "
            f"success={result['success']} "
            f"A:{CIFAR10_CLASSES[result['seed_pred_a']]}->{CIFAR10_CLASSES[result['final_pred_a']]} "
            f"B:{CIFAR10_CLASSES[result['seed_pred_b']]}->{CIFAR10_CLASSES[result['final_pred_b']]} "
            f"cov {result['before_cov']:.4f}->{result['after_cov']:.4f}"
        )

        if saved_count < 5:
            output_path = os.path.join("results", f"generated_disagreement_{saved_count+1:02d}.png")
            save_result_figure(result, output_path)
            saved_count += 1

    save_summary_csv(results, "results/generated_disagreement_summary.csv")
    print("Saved summary CSV to results/generated_disagreement_summary.csv")
    print("Saved generated disagreement figures to results/")

    coverage_a.remove()
    coverage_b.remove()


if __name__ == "__main__":
    main()