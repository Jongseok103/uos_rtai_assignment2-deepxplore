from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class NeuronCoverage:
    """
    Conv2d / Linear layer 기준으로 neuron coverage를 추적하는 간단한 클래스.

    coverage 계산 방식:
    - Conv2d 출력은 channel별 spatial mean 사용
    - Linear 출력은 뉴런별 activation 그대로 사용
    - activation이 threshold를 한 번이라도 넘으면 covered로 봄
    """

    def __init__(self, model: nn.Module, threshold: float = 0.0):
        self.model = model
        self.threshold = threshold

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_names: List[str] = []

        self.covered: Dict[str, torch.Tensor] = {}
        self.total_neurons: int = 0

        self._register_hooks()

    def _register_hooks(self) -> None:
        """coverage 볼 Conv2d/Linear layer들에 forward hook 거는 부분."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 각 layer 출력은 hook으로 잡아서 coverage 계산에 사용함.
                self.layer_names.append(name)
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
            if not isinstance(output, torch.Tensor):
                return

            with torch.no_grad():
                if output.dim() == 4:
                    # Conv feature map은 channel별 평균 activation으로 요약해서 봄.
                    act = output.mean(dim=(2, 3))
                elif output.dim() == 2:
                    # Linear layer는 뉴런별 출력을 그대로 사용함.
                    act = output
                else:
                    return

                if layer_name not in self.covered:
                    num_neurons = act.shape[1]
                    self.covered[layer_name] = torch.zeros(num_neurons, dtype=torch.bool, device="cpu")
                    self.total_neurons += num_neurons

                active = (act > self.threshold).any(dim=0).detach().cpu()
                self.covered[layer_name] |= active

        return hook

    def reset(self) -> None:
        """현재까지 켜진 뉴런 표시 초기화하는 함수."""
        # 샘플 하나 기준으로 다시 보고 싶을 때 이전 기록 비우는 용도임.
        for layer_name in self.covered:
            self.covered[layer_name].zero_()

    def coverage_ratio(self) -> float:
        """전체 뉴런 중 몇 퍼센트가 한 번이라도 켜졌는지 반환함."""
        if self.total_neurons == 0:
            return 0.0
        covered_count = sum(mask.sum().item() for mask in self.covered.values())
        return covered_count / self.total_neurons

    def covered_count(self) -> int:
        return sum(mask.sum().item() for mask in self.covered.values())

    def summary(self) -> Dict[str, float]:
        """coverage 관련 통계를 한 번에 보기 좋게 묶어 반환함."""
        return {
            "covered_neurons": self.covered_count(),
            "total_neurons": self.total_neurons,
            "coverage_ratio": self.coverage_ratio(),
        }

    def remove(self) -> None:
        """다 쓴 hook 정리하는 함수."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
