# RT-AI Assignment 2: DeepXplore on CIFAR-10 ResNet50

## 개요
이 저장소는 CIFAR-10에 대해 학습한 ResNet50 두 개를 대상으로 DeepXplore 스타일의 differential testing을 수행한 과제 코드다.

원래 DeepXplore는 오래된 Keras/TensorFlow 1.x 환경을 기준으로 작성되어 있어서, 그대로는 현재 환경에서 실행하기 어려웠다. 그래서 이번 과제에서는 원본 저장소를 참고하되, CIFAR-10 + ResNet50 조합에 맞게 PyTorch 기반으로 다시 연결해서 실행할 수 있도록 수정했다.

## 사용한 모델
- `models/model_a.pth`
- `models/model_b.pth`

두 모델 모두 CIFAR-10용 ResNet50 체크포인트이며, 서로 다른 설정으로 학습된 모델을 사용했다.

## 디렉터리 설명
- `src/train_models.py`: CIFAR-10 ResNet50 학습 코드
- `src/evaluate_two_models.py`: 두 모델의 baseline disagreement 확인용 코드
- `src/generate_disagreement.py`: 초기에 구현했던 간단한 DeepXplore-style 생성 코드
- `external/deepxplore/modernized/`: 과제용으로 수정한 DeepXplore 실행 코드
- `test.py`: 현대화한 DeepXplore를 실행하는 스크립트
- `results/`: 실행 결과 저장 폴더

## 실행 환경
기본 `python` 환경에는 PyTorch가 없을 수 있어서, 별도 conda 환경에서 실행했다. 이 저장소에서 실제 실행에 사용한 환경은 `rtai-a2`였다.

최소 의존성은 [requirements-modernized.txt](/Users/Jongseok/Library%20Mobile%20Documents/com~apple~CloudDocs/Documents/2026%20UOS/Reliable%20and%20Trustworthy%20AI/Assignment2/rtai-assignment2-deepxplore/external/deepxplore/requirements-modernized.txt)에 정리했다.

예시:

```bash
python -m pip install -r external/deepxplore/requirements-modernized.txt
```

## 실행 방법
가장 간단한 실행 방법은 아래와 같다.

```bash
python test.py
```

`test.py`는 현대화한 DeepXplore 실행기를 호출해서 두 모델에 대해 disagreement-inducing input을 찾고, 결과를 저장한다.

직접 실행하려면 아래처럼 실행할 수 있다.

```bash
python -m external.deepxplore.modernized.run \
  --model-a models/model_a.pth \
  --model-b models/model_b.pth \
  --output-dir results/deepxplore_modernized
```

## 수정한 내용
원본 DeepXplore를 그대로 쓰지 않고, 아래와 같이 바꿨다.

- Keras/TensorFlow 1.x 기반 코드를 PyTorch 기준으로 다시 구성
- CIFAR-10 입력과 ResNet50 체크포인트를 직접 로드하도록 수정
- 두 모델 사이의 disagreement를 기준으로 seed를 수집하도록 변경
- gradient ascent로 disagreement를 유지하면서 neuron coverage가 늘어나도록 objective를 구성
- Conv2d / Linear layer 기준으로 neuron coverage를 계산하도록 구현
- 결과를 CSV와 PNG로 저장하도록 정리

완전히 원 논문 구현을 그대로 복원한 것은 아니고, 과제 환경에서 실행 가능하도록 DeepXplore의 핵심 아이디어를 가져와 재구성한 버전이다.

## 결과물
실행이 끝나면 결과는 `results/` 아래에 저장된다.

주요 결과물:
- disagreement 요약 CSV
- 생성된 disagreement input 시각화 PNG 5장 이상

실행 중 기록되는 값:
- disagreement-inducing input 개수
- 각 샘플의 예측 클래스 변화
- neuron coverage before / after
- perturbation norm

## 참고
원본 DeepXplore 저장소는 `external/deepxplore/`에 그대로 남겨두었다. 다만 실제 과제 실행은 `external/deepxplore/modernized/` 아래의 PyTorch 코드로 진행했다.
