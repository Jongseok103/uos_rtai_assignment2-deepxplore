import os
import subprocess
import sys


def run_command(cmd):
    """테스트용 명령 실행하고 실패하면 예외 띄우는 함수."""
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def check_required_files():
    """smoke test 돌리기 전에 필요한 파일 있는지 확인함."""
    required = [
        "models/model_a.pth",
        "models/model_b.pth",
        "deepxplore_modernized/run.py",
        "deepxplore_modernized/coverage.py",
        "deepxplore_modernized/common.py",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main():
    check_required_files()
    # smoke test라 seed 수랑 step 수 작게 잡고 빠르게만 확인함.
    run_command(
        [
            sys.executable,
            "-m",
            "deepxplore_modernized.run",
            "--max-seeds",
            "1",
            "--steps",
            "3",
            "--num-workers",
            "0",
            "--output-dir",
            "results/deepxplore_modernized_smoke",
        ]
    )


if __name__ == "__main__":
    main()
