# RT-AI Assignment 2: DeepXplore on CIFAR-10 ResNet50 Models

## Overview
이 저장소는 CIFAR-10용 ResNet50 두 모델에 대해 DeepXplore 스타일의 differential testing을 수행하는 과제 저장소다. 현재 두 개의 실행 트랙을 유지한다.

- `src/`: 과제용으로 단순 재구성한 기존 PyTorch 실험
- `external/deepxplore/modernized`: 원본 DeepXplore의 실행 흐름을 더 가깝게 반영하도록 별도 추가한 현대화된 PyTorch 트랙

원본 `external/deepxplore`의 Keras/TensorFlow 1.x 코드는 참고용으로 그대로 남겨두고, 새로운 실험은 별도 모듈에서 독립적으로 실행한다.

## Repository Structure
- `models/model_a.pth`, `models/model_b.pth`: CIFAR-10 ResNet50 체크포인트
- `src/train_models.py`: 두 개의 CIFAR-10 ResNet50 학습 스크립트
- `src/evaluate_two_models.py`: 기존 baseline disagreement 수집
- `src/generate_disagreement.py`: 기존 단순 DeepXplore-style 입력 생성
- `external/deepxplore/modernized/run.py`: 현대화된 PyTorch DeepXplore 실행 진입점
- `external/deepxplore/modernized/coverage.py`: Conv2d/Linear neuron coverage 추적기
- `external/deepxplore/modernized/compare_results.py`: 기존 `src` 결과와 새 결과 비교 요약
- `test.py`: 기존 `src` 트랙 실행
- `test_modernized.py`: 현대화된 트랙 smoke 실행

## Environment Setup
기본 셸의 `python`은 현재 PyTorch가 설치되지 않은 환경일 수 있다. 현대화된 DeepXplore 트랙은 별도 PyTorch 환경에서 실행하는 것을 전제로 한다.

최소 의존성은 [external/deepxplore/requirements-modernized.txt](/Users/Jongseok/Library/Mobile%20Documents/com~apple~CloudDocs/Documents/2026%20UOS/Reliable%20and%20Trustworthy%20AI/Assignment2/rtai-assignment2-deepxplore/external/deepxplore/requirements-modernized.txt)에 정리했다.

예시:

```bash
python -m pip install -r external/deepxplore/requirements-modernized.txt
```

## Existing `src` Track
기존 과제 트랙은 다음처럼 실행한다.

```bash
python test.py
```

산출물은 `results/` 아래에 저장된다.

## Modernized DeepXplore Track
현대화된 트랙은 다음처럼 실행한다.

```bash
python -m external.deepxplore.modernized.run \
  --model-a models/model_a.pth \
  --model-b models/model_b.pth \
  --output-dir results/deepxplore_modernized
```

주요 인자:

- `--batch-size`, `--max-seeds`
- `--epsilon`, `--alpha`, `--steps`
- `--coverage-threshold`
- `--weight-diff`, `--weight-nc`
- `--device`
- `--output-dir`

기본 동작은 다음과 같다.

1. 두 개의 CIFAR-10 ResNet50 체크포인트를 로드한다.
2. 테스트셋에서 baseline disagreement seed를 수집한다.
3. 각 seed에 대해 disagreement 유지 + neuron coverage 증가 목적함수로 gradient ascent를 수행한다.
4. coverage 전후, perturbation norm, 예측 클래스 변화를 기록한다.
5. CSV와 시각화 PNG를 저장한다.

산출물은 기본적으로 `results/deepxplore_modernized/` 아래에 저장된다.

## Smoke Test
짧은 smoke 실행은 다음처럼 수행한다.

```bash
python test_modernized.py
```

이 스크립트는 seed 1개, 소수의 optimization step만 사용해 end-to-end 경로를 빠르게 확인한다.

## Result Comparison
기존 `src` 결과와 현대화된 트랙 결과를 비교하려면:

```bash
python -m external.deepxplore.modernized.compare_results
```

기본 출력 파일은 `results/deepxplore_modernized/comparison_summary.csv`다.

## Notes On Modernization
이번 현대화는 원본 Keras/TensorFlow 1.x 구현의 완전 복원이 아니다. 대신 다음을 명시적으로 선택했다.

- 모델 수는 3개가 아니라 현재 자산에 맞춰 2개로 고정
- Keras backend graph API 대신 PyTorch eager execution 사용
- `xrange`, `scipy.misc.imsave`, TF1 전제 코드는 포팅 대상에서 제외
- 입력 제약은 CIFAR-10 정규화 공간 기준 `L_inf` projection 사용
- 원본 `light`, `occl`, `blackout` 변환은 1차 포팅 범위에서 제외

즉, “원형에 더 가까운 실행 흐름”을 PyTorch/CIFAR-10 환경에 맞게 재구성한 트랙으로 이해하면 된다.
