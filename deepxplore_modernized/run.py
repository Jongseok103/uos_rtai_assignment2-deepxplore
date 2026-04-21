from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import torch

# matplotlib 캐시 경로를 결과 폴더 아래로 고정해둠.
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))

import matplotlib.pyplot as plt

from deepxplore_modernized.common import (
    CIFAR10_CLASSES,
    denormalize,
    ensure_parent_dir,
    get_test_loader,
    load_model,
    set_seed,
)
from deepxplore_modernized.coverage import NeuronCoverageTracker


def parse_args() -> argparse.Namespace:
    """실행할 때 필요한 인자들 받는 부분."""
    parser = argparse.ArgumentParser(
        description="Modernized PyTorch DeepXplore-style disagreement generation for CIFAR-10 ResNet50 models."
    )
    parser.add_argument("--model-a", default="models/model_a.pth")
    parser.add_argument("--model-b", default="models/model_b.pth")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--coverage-threshold", type=float, default=0.0)
    parser.add_argument("--weight-diff", type=float, default=1.0)
    parser.add_argument("--weight-nc", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output-dir",
        default="results/deepxplore_modernized",
        help="Directory for generated figures and summaries.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """환경 보고 CPU/GPU 뭐 쓸지 정하는 함수."""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def collect_seed_disagreements(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_seeds: int,
) -> List[Dict]:
    """두 모델이 서로 다르게 예측한 test 샘플을 seed로 모으는 함수."""
    seeds: List[Dict] = []
    global_index = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds_a = model_a(images).argmax(dim=1)
        preds_b = model_b(images).argmax(dim=1)

        for batch_idx in range(labels.size(0)):
            # 두 모델 예측이 이미 다른 샘플만 seed로 사용함.
            if preds_a[batch_idx].item() == preds_b[batch_idx].item():
                continue
            seeds.append(
                {
                    "index": global_index + batch_idx,
                    "image": images[batch_idx].detach(),
                    "label": labels[batch_idx].item(),
                    "pred_a": preds_a[batch_idx].item(),
                    "pred_b": preds_b[batch_idx].item(),
                }
            )
            if len(seeds) >= max_seeds:
                return seeds

        global_index += labels.size(0)

    return seeds


def clamp_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, epsilon: float) -> torch.Tensor:
    """입력 변형이 허용 범위 epsilon 안에만 있게 잘라주는 함수."""
    delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
    return x_orig + delta


def confidence_margin(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    """target 클래스가 다른 클래스보다 얼마나 우세한지 보는 margin 계산 함수."""
    # target 클래스가 나머지보다 얼마나 앞서는지 보는 값임.
    target_logit = logits[0, target_idx]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[0, target_idx] = False
    other_max = logits.masked_fill(~mask, float("-inf")).max(dim=1).values[0]
    return target_logit - other_max


def compute_coverage_ratio(
    tracker_a: NeuronCoverageTracker,
    tracker_b: NeuronCoverageTracker,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    x: torch.Tensor,
) -> float:
    """입력 x 하나에 대한 두 모델 평균 coverage 계산하는 함수."""
    # 샘플 하나 기준으로 두 모델 coverage 평균을 계산함.
    tracker_a.reset_coverage()
    tracker_b.reset_coverage()
    tracker_a.reset_current_activations()
    tracker_b.reset_current_activations()
    with torch.no_grad():
        _ = model_a(x)
        _ = model_b(x)
    return 0.5 * (tracker_a.coverage_ratio() + tracker_b.coverage_ratio())


def optimize_seed(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    tracker_a: NeuronCoverageTracker,
    tracker_b: NeuronCoverageTracker,
    seed: Dict,
    epsilon: float,
    alpha: float,
    steps: int,
    weight_diff: float,
    weight_nc: float,
) -> Dict:
    """seed 하나 기준으로 disagreement 유지랑 coverage 증가를 같이 최적화하는 핵심 함수."""
    x_orig = seed["image"].unsqueeze(0)
    x_adv = x_orig.clone().detach()
    target_a = seed["pred_a"]
    target_b = seed["pred_b"]

    before_cov = compute_coverage_ratio(tracker_a, tracker_b, model_a, model_b, x_orig)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        tracker_a.reset_current_activations()
        tracker_b.reset_current_activations()

        logits_a = model_a(x_adv)
        logits_b = model_b(x_adv)

        uncovered_a = tracker_a.pick_uncovered_neuron()
        uncovered_b = tracker_b.pick_uncovered_neuron()

        # loss_diff는 기존 disagreement 유지용,
        # loss_nc는 새 뉴런 활성화 유도용임.
        loss_diff = confidence_margin(logits_a, target_a) + confidence_margin(logits_b, target_b)
        loss_nc = tracker_a.activation_term(uncovered_a) + tracker_b.activation_term(uncovered_b)
        final_objective = weight_diff * loss_diff + weight_nc * loss_nc

        grad = torch.autograd.grad(final_objective, x_adv)[0]
        # 구현은 단순하게 FGSM 스타일 sign update로 감.
        x_adv = x_adv + alpha * grad.sign()
        x_adv = clamp_linf(x_adv.detach(), x_orig, epsilon)

    with torch.no_grad():
        final_logits_a = model_a(x_adv)
        final_logits_b = model_b(x_adv)
        final_pred_a = final_logits_a.argmax(dim=1).item()
        final_pred_b = final_logits_b.argmax(dim=1).item()

    after_cov = compute_coverage_ratio(tracker_a, tracker_b, model_a, model_b, x_adv)
    perturbation = (x_adv - x_orig).reshape(-1)

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
        "linf": perturbation.abs().max().item(),
        "l2": torch.norm(perturbation, p=2).item(),
        "success": final_pred_a != final_pred_b,
    }


def save_result_figure(result: Dict, output_path: str) -> None:
    """원본/생성 이미지/perturbation 차이를 한 장에 저장하는 함수."""
    ensure_parent_dir(output_path)

    orig = denormalize(result["x_orig"]).permute(1, 2, 0).numpy()
    adv = denormalize(result["x_adv"]).permute(1, 2, 0).numpy()
    # perturbation은 잘 안 보여서 그림 저장할 때 0~1로 다시 맞춰줌.
    diff = adv - orig
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(orig)
    axes[0].set_title(
        f"Original\ntrue={CIFAR10_CLASSES[result['true_label']]}\n"
        f"A={CIFAR10_CLASSES[result['seed_pred_a']]}, B={CIFAR10_CLASSES[result['seed_pred_b']]}"
    )
    axes[0].axis("off")

    axes[1].imshow(adv)
    axes[1].set_title(
        f"Generated\nA={CIFAR10_CLASSES[result['final_pred_a']]}, B={CIFAR10_CLASSES[result['final_pred_b']]}"
    )
    axes[1].axis("off")

    axes[2].imshow(diff)
    axes[2].set_title(
        f"Diff\nLinf={result['linf']:.4f}\n"
        f"cov {result['before_cov']:.4f}->{result['after_cov']:.4f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_summary_csv(results: List[Dict], csv_path: str) -> None:
    """각 seed 결과를 CSV로 저장하는 함수."""
    ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
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
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["index"],
                    CIFAR10_CLASSES[result["true_label"]],
                    CIFAR10_CLASSES[result["seed_pred_a"]],
                    CIFAR10_CLASSES[result["seed_pred_b"]],
                    CIFAR10_CLASSES[result["final_pred_a"]],
                    CIFAR10_CLASSES[result["final_pred_b"]],
                    f"{result['before_cov']:.6f}",
                    f"{result['after_cov']:.6f}",
                    f"{result['linf']:.6f}",
                    f"{result['l2']:.6f}",
                    int(result["success"]),
                ]
            )


def print_run_summary(results: List[Dict], output_dir: str) -> None:
    """실행 끝난 뒤 핵심 통계 콘솔에 요약 출력하는 함수."""
    success_count = sum(int(item["success"]) for item in results)
    average_cov_gain = (
        sum(item["after_cov"] - item["before_cov"] for item in results) / len(results) if results else 0.0
    )
    print(f"Processed seeds: {len(results)}")
    print(f"Successful disagreements: {success_count}/{len(results)}")
    print(f"Average coverage gain: {average_cov_gain:.6f}")
    print(f"Artifacts saved under: {output_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    for checkpoint_path in (args.model_a, args.model_b):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    loader = get_test_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)
    tracker_a = NeuronCoverageTracker(model_a, threshold=args.coverage_threshold)
    tracker_b = NeuronCoverageTracker(model_b, threshold=args.coverage_threshold)

    # baseline disagreement를 먼저 모아야 DeepXplore-style 최적화 시작 가능함.
    seeds = collect_seed_disagreements(
        model_a=model_a,
        model_b=model_b,
        loader=loader,
        device=device,
        max_seeds=args.max_seeds,
    )
    if not seeds:
        raise RuntimeError("No baseline disagreement seeds found for the supplied models.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for idx, seed in enumerate(seeds, start=1):
        seed["image"] = seed["image"].to(device)
        # seed마다 perturbation은 따로 최적화함.
        result = optimize_seed(
            model_a=model_a,
            model_b=model_b,
            tracker_a=tracker_a,
            tracker_b=tracker_b,
            seed=seed,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
            weight_diff=args.weight_diff,
            weight_nc=args.weight_nc,
        )
        results.append(result)
        print(
            f"[{idx}/{len(seeds)}] idx={result['index']} success={int(result['success'])} "
            f"A:{CIFAR10_CLASSES[result['seed_pred_a']]}->{CIFAR10_CLASSES[result['final_pred_a']]} "
            f"B:{CIFAR10_CLASSES[result['seed_pred_b']]}->{CIFAR10_CLASSES[result['final_pred_b']]} "
            f"cov {result['before_cov']:.4f}->{result['after_cov']:.4f}"
        )
        if idx <= 5:
            # 보고서에 쓰기 쉽게 앞쪽 몇 장만 PNG로 저장해둠.
            save_result_figure(result, str(output_dir / f"generated_disagreement_{idx:02d}.png"))

    save_summary_csv(results, str(output_dir / "generated_disagreement_summary.csv"))
    print_run_summary(results, str(output_dir))

    tracker_a.remove()
    tracker_b.remove()


if __name__ == "__main__":
    main()
