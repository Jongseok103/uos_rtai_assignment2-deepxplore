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
    """실험 다시 돌려도 결과가 크게 안 달라지게 seed 고정하는 함수."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_resnet50_for_cifar10() -> nn.Module:
    """CIFAR-10용으로 살짝 수정한 ResNet50 만드는 부분."""
    model = models.resnet50(weights=None)
    # CIFAR-10 입력 크기에 맞게 stem 부분만 간단히 바꿔둠.
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """checkpoint 불러와서 바로 추론 가능한 상태로 만드는 함수."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_resnet50_for_cifar10()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_test_loader(batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    """seed 찾을 때 쓸 CIFAR-10 test loader 만드는 함수."""
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
    """저장용 그림 만들 때 이미지 색 범위 복원하는 함수."""
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
    """두 모델이 원래부터 다르게 보는 샘플들을 seed로 모으는 함수."""
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
            # 처음부터 예측이 다른 샘플만 골라야 뒤에서 disagreement 유지하기 편함.
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
    """perturbation이 epsilon 범위를 넘지 않게 잘라주는 함수."""
    delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
    return x_orig + delta


def normalize_to_valid_range(x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
    # 입력 자체가 이미 normalize된 상태라 여기서는 epsilon projection만 신경 쓰면 됨.
    # 시각화할 때는 아래에서 따로 denormalize함.
    return x_adv


def compute_coverage_gain(
    coverage_a: NeuronCoverage,
    coverage_b: NeuronCoverage,
    model_a: nn.Module,
    model_b: nn.Module,
    x: torch.Tensor,
) -> float:
    """입력 하나 넣었을 때 두 모델 평균 coverage가 얼마나 나오는지 계산함."""
    # 여기서는 두 모델 coverage 평균을 간단한 점수처럼 사용함.
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
    """seed에서 보이던 서로 다른 예측이 유지되도록 target logit을 키우는 함수."""
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
    """seed 하나를 조금씩 바꿔서 disagreement 유지되는 입력 만드는 핵심 함수."""
    x_orig = seed["image"].unsqueeze(0).to(device)
    target_a = seed["pred_a"]
    target_b = seed["pred_b"]

    before_cov = compute_coverage_gain(coverage_a, coverage_b, model_a, model_b, x_orig)

    x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits_a = model_a(x_adv)
        logits_b = model_b(x_adv)

        # seed에서 갖고 있던 서로 다른 예측을 계속 유지시키는 objective임.
        loss = objective_fn(logits_a, logits_b, target_a, target_b)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # sign gradient로 조금씩 움직이고 perturbation은 epsilon 안에 묶어둠.
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
    """원본/생성 결과/차이 이미지를 한 장으로 저장하는 함수."""
    orig = denormalize(result["x_orig"]).permute(1, 2, 0).numpy()
    adv = denormalize(result["x_adv"]).permute(1, 2, 0).numpy()
    # 차이 이미지는 그대로 보면 잘 안 보여서 보기 좋게 다시 스케일링함.
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
    """생성 결과 요약 CSV 저장하는 함수."""
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

    # baseline에서 이미 의견 갈리는 샘플을 seed로 쓰는 방식임.
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
