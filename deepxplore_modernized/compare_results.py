from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    """비교할 CSV 경로들 받는 부분."""
    parser = argparse.ArgumentParser(description="Compare legacy src and modernized DeepXplore result summaries.")
    parser.add_argument("--baseline-csv", default="results/generated_disagreement_summary.csv")
    parser.add_argument(
        "--modernized-csv",
        default="results/deepxplore_modernized/generated_disagreement_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="results/deepxplore_modernized/comparison_summary.csv",
    )
    return parser.parse_args()


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    """CSV 읽어서 행 단위 dict 리스트로 바꿔주는 함수."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for comparison: {csv_path}")
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """실험 결과를 핵심 지표 몇 개로 요약하는 함수."""
    if not rows:
        return {
            "num_seeds": 0,
            "success_rate": 0.0,
            "avg_cov_gain": 0.0,
            "avg_linf": 0.0,
            "avg_l2": 0.0,
        }

    # 두 결과를 같은 기준으로 비교하려고 핵심 지표만 평균으로 정리했음.
    cov_gain = sum(float(row["after_cov"]) - float(row["before_cov"]) for row in rows)
    success_count = sum(int(row["success"]) for row in rows)
    return {
        "num_seeds": len(rows),
        "success_rate": success_count / len(rows),
        "avg_cov_gain": cov_gain / len(rows),
        "avg_linf": sum(float(row["linf"]) for row in rows) / len(rows),
        "avg_l2": sum(float(row["l2"]) for row in rows) / len(rows),
    }


def save_summary(output_csv: str, baseline: Dict[str, float], modernized: Dict[str, float]) -> None:
    """두 결과 요약을 한 CSV에 나란히 저장하는 함수."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "legacy_src", "modernized_external"])
        for metric in ("num_seeds", "success_rate", "avg_cov_gain", "avg_linf", "avg_l2"):
            writer.writerow([metric, baseline[metric], modernized[metric]])


def main() -> None:
    args = parse_args()
    baseline_rows = load_rows(args.baseline_csv)
    modernized_rows = load_rows(args.modernized_csv)

    baseline_summary = summarize(baseline_rows)
    modernized_summary = summarize(modernized_rows)
    save_summary(args.output_csv, baseline_summary, modernized_summary)

    print("Saved comparison summary to:", args.output_csv)
    for metric in ("num_seeds", "success_rate", "avg_cov_gain", "avg_linf", "avg_l2"):
        print(
            f"{metric}: legacy_src={baseline_summary[metric]} | modernized_external={modernized_summary[metric]}"
        )


if __name__ == "__main__":
    main()
