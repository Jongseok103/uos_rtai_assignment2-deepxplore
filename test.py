import os
import sys
import glob
import subprocess


def run_command(cmd):
    """명령 실행하고 실패하면 바로 예외 띄우는 함수."""
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def check_required_files():
    """실행 전에 필요한 파일들 있는지 확인하는 부분."""
    required = [
        "models/model_a.pth",
        "models/model_b.pth",
        "deepxplore_modernized/run.py",
        "deepxplore_modernized/coverage.py",
        "deepxplore_modernized/common.py",
    ]

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )


def summarize_results():
    """결과 CSV랑 이미지가 잘 나왔는지 간단히 요약해서 보여줌."""
    output_dir = os.path.abspath("results/deepxplore_modernized")
    generated_csv = os.path.join(output_dir, "generated_disagreement_summary.csv")

    generated_imgs = sorted(glob.glob(os.path.join(output_dir, "generated_disagreement_*.png")))

    print("\n" + "=" * 80)
    print("Result Summary")
    print("=" * 80)

    print(f"Generated CSV exists: {os.path.exists(generated_csv)}")
    print(f"Generated figures found: {len(generated_imgs)}")

    if os.path.exists(generated_csv):
        print(f"- {generated_csv}")

    for path in generated_imgs[:5]:
        print(f"- {path}")

    print("\nFinished. Check the results/ directory for saved outputs.")


def main():
    os.makedirs("results", exist_ok=True)

    check_required_files()

    output_dir = os.path.abspath("results/deepxplore_modernized")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: run the modernized DeepXplore workflow end-to-end.
    run_command(
        [
            sys.executable,
            "-m",
            "deepxplore_modernized.run",
            "--model-a",
            "models/model_a.pth",
            "--model-b",
            "models/model_b.pth",
            "--output-dir",
            output_dir,
        ]
    )

    # Step 2: 실행 끝난 뒤 결과 파일 잘 만들어졌는지 확인함.
    summarize_results()


if __name__ == "__main__":
    main()
