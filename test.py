import os
import sys
import glob
import subprocess


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def check_required_files():
    required = [
        "models/model_a.pth",
        "models/model_b.pth",
        "src/evaluate_two_models.py",
        "src/generate_disagreement.py",
        "src/coverage.py",
    ]

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )


def summarize_results():
    baseline_csv = "results/baseline_disagreements.csv"
    generated_csv = "results/generated_disagreement_summary.csv"

    baseline_imgs = sorted(glob.glob("results/baseline_disagreement_*.png"))
    generated_imgs = sorted(glob.glob("results/generated_disagreement_*.png"))

    print("\n" + "=" * 80)
    print("Result Summary")
    print("=" * 80)

    print(f"Baseline CSV exists: {os.path.exists(baseline_csv)}")
    print(f"Generated CSV exists: {os.path.exists(generated_csv)}")
    print(f"Baseline figures found: {len(baseline_imgs)}")
    print(f"Generated figures found: {len(generated_imgs)}")

    if os.path.exists(baseline_csv):
        print(f"- {baseline_csv}")
    if os.path.exists(generated_csv):
        print(f"- {generated_csv}")

    for path in baseline_imgs[:5]:
        print(f"- {path}")
    for path in generated_imgs[:5]:
        print(f"- {path}")

    print("\nFinished. Check the results/ directory for saved outputs.")


def main():
    os.makedirs("results", exist_ok=True)

    check_required_files()

    # Step 1: baseline disagreement evaluation
    run_command([sys.executable, "src/evaluate_two_models.py"])

    # Step 2: DeepXplore-style disagreement generation
    run_command([sys.executable, "src/generate_disagreement.py"])

    # Step 3: final summary
    summarize_results()


if __name__ == "__main__":
    main()